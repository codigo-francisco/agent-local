"""Calculadora de VRAM: pesos del modelo + KV cache (depende del contexto) + margen.

KV cache = capas × cabezas_kv × (dim_clave + dim_valor) × contexto × bytes_por_elemento
"""

from __future__ import annotations

import ctypes
import os
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from ..config import AppConfig, ModelEntry

GB = 1024 ** 3
KV_BYTES = {"f16": 2.0, "q8_0": 34 / 32, "q4_0": 18 / 32}  # bloques de 32 con escala f16
OVERHEAD_GB = 0.6  # contexto CUDA + buffers de cómputo por proceso llama-server
DESKTOP_RESERVE_GB = 0.8  # lo que suele usar Windows/escritorio
FIT_MARGIN_GB = 0.3  # --fit deja 1 GiB libre; ya descontamos DESKTOP_RESERVE_GB, sumamos el resto


@dataclass(frozen=True)
class ArchInfo:
    layers: int
    kv_heads: int
    key_dim: int
    value_dim: int
    attn_interval: int = 1  # atención híbrida (Qwen3.5/3.6): solo 1 de cada N capas guarda KV
    moe: bool = False  # mezcla de expertos: sus expertos pueden ir a RAM sin hundir la velocidad

    @property
    def kv_layers(self) -> int:
        return -(-self.layers // max(1, self.attn_interval))


@dataclass
class GPUInfo:
    name: str
    total_gb: float
    used_gb: float
    free_gb: float
    driver: str


@dataclass
class ModelEstimate:
    name: str
    weights_gb: float
    kv_gb: float
    overhead_gb: float
    arch_known: bool
    weights_known: bool
    note: str = ""
    offload_gb: float = 0.0  # pesos que llama.cpp deja en RAM (modo automático)

    @property
    def total_gb(self) -> float:
        """Lo que ocupa en la GPU."""
        return self.weights_gb + self.kv_gb + self.overhead_gb - self.offload_gb


@dataclass
class UsagePlan:
    models: list[ModelEstimate]
    total_gb: float
    capacity_gb: float | None
    gpu: GPUInfo | None
    concurrent: bool
    suggestions: list[str]
    ram_offload_gb: float = 0.0

    @property
    def fits(self) -> bool | None:
        return None if self.capacity_gb is None else self.total_gb <= self.capacity_gb


def kv_cache_bytes(arch: ArchInfo, ctx: int, kv_type: str) -> float:
    return arch.kv_layers * arch.kv_heads * (arch.key_dim + arch.value_dim) * ctx * KV_BYTES.get(kv_type, 2.0)


def gpu_info() -> GPUInfo | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        return None
    if not out:
        return None
    parts = [p.strip() for p in out[0].split(",")]
    try:
        return GPUInfo(parts[0], float(parts[1]) / 1024, float(parts[2]) / 1024,
                       float(parts[3]) / 1024, parts[4])
    except (IndexError, ValueError):
        return None


def _field(reader, key: str):
    f = reader.fields.get(key)
    if f is None:
        return None
    try:
        return f.contents()
    except AttributeError:  # versiones antiguas del paquete gguf
        vals = [f.parts[i].tolist() for i in f.data]
        vals = [v[0] if isinstance(v, list) and len(v) == 1 else v for v in vals]
        if vals and isinstance(vals[0], list):  # cadena como lista de bytes
            return bytes(vals[0]).decode("utf-8", errors="replace")
        return vals[0] if len(vals) == 1 else vals


@lru_cache(maxsize=32)
def _read_arch_cached(path: str, mtime: float) -> ArchInfo | None:  # noqa: ARG001 - mtime invalida la caché
    try:
        from gguf import GGUFReader
        r = GGUFReader(path, "r")
        arch = _field(r, "general.architecture")
        if isinstance(arch, (bytes, bytearray)):
            arch = arch.decode()
        layers = _field(r, f"{arch}.block_count")
        heads = _field(r, f"{arch}.attention.head_count")
        kv_heads = _field(r, f"{arch}.attention.head_count_kv") or heads
        emb = _field(r, f"{arch}.embedding_length")
        heads = max(heads) if isinstance(heads, list) else heads
        kv_heads = max(kv_heads) if isinstance(kv_heads, list) else kv_heads
        key_dim = _field(r, f"{arch}.attention.key_length") or (emb // heads if emb and heads else None)
        value_dim = _field(r, f"{arch}.attention.value_length") or key_dim
        interval = _field(r, f"{arch}.full_attention_interval") or 1
        experts = _field(r, f"{arch}.expert_count") or 0
        if not (layers and kv_heads and key_dim):
            return None
        return ArchInfo(int(layers), int(kv_heads), int(key_dim), int(value_dim), int(interval),
                        int(experts) > 0)
    except Exception:  # noqa: BLE001 - archivo incompleto, formato raro, paquete ausente...
        return None


def read_gguf_arch(path: Path) -> ArchInfo | None:
    if not path.is_file():
        return None
    return _read_arch_cached(str(path), path.stat().st_mtime)


@lru_cache(maxsize=32)
def _tokenizer_cached(path: str, mtime: float) -> tuple | None:  # noqa: ARG001
    try:
        from gguf import GGUFReader
        r = GGUFReader(path, "r")
        tokens = r.fields.get("tokenizer.ggml.tokens")
        return (_field(r, "tokenizer.ggml.model"), _field(r, "tokenizer.ggml.pre"),
                len(tokens.data) if tokens is not None else None)
    except Exception:  # noqa: BLE001
        return None


THINKING_MARKERS = ("<think>", "reasoning_content", "enable_thinking", "<|channel|>")


@lru_cache(maxsize=32)
def _meta_cached(path: str, mtime: float) -> dict:  # noqa: ARG001
    try:
        from gguf import GGUFReader
        r = GGUFReader(path, "r")
        arch = _field(r, "general.architecture")
        template = _field(r, "tokenizer.chat_template") or ""
        return {"context_length": _field(r, f"{arch}.context_length"),
                "thinking": isinstance(template, str) and any(m in template for m in THINKING_MARKERS)}
    except Exception:  # noqa: BLE001
        return {}


def model_meta(path: Path) -> dict:
    """{'context_length': contexto de entrenamiento, 'thinking': si el modelo razona antes de responder}."""
    if not path.is_file():
        return {}
    return _meta_cached(str(path), path.stat().st_mtime)


def tokenizer_signature(path: Path) -> tuple | None:
    """(tipo, pre-tokenizador, nº de tokens): igual en modelos de la misma familia."""
    if not path.is_file():
        return None
    return _tokenizer_cached(str(path), path.stat().st_mtime)


def draft_compatible(main: ModelEntry, draft: ModelEntry) -> bool:
    """La decodificación especulativa exige el mismo vocabulario (misma familia de modelos)."""
    a, b = tokenizer_signature(main.path), tokenizer_signature(draft.path)
    return a is None or b is None or a == b


def file_gb(path: Path) -> float:
    """Tamaño del archivo del modelo en GiB (≈ lo que ocupan sus pesos en memoria)."""
    return path.stat().st_size / GB


def catalog_entry(catalog: list[dict], filename: str) -> dict | None:
    fn = Path(filename).name.lower()
    return next((c for c in catalog if str(c.get("file", "")).lower() == fn), None)


def arch_from_catalog(item: dict | None) -> ArchInfo | None:
    a = (item or {}).get("arch")
    if not a:
        return None
    return ArchInfo(a["layers"], a["kv_heads"], a["head_dim"], a.get("value_dim", a["head_dim"]),
                    a.get("attn_interval", 1), bool(a.get("moe", False)))


def estimate_entry(entry: ModelEntry, catalog: list[dict], ctx: int | None = None,
                   kv_type: str | None = None) -> ModelEstimate:
    ctx = ctx or entry.ctx
    kv_type = kv_type or entry.kv_type
    item = catalog_entry(catalog, entry.file)
    path = entry.path
    weights_known = path.is_file()
    # size_gb del catálogo está en GB decimales (como en Hugging Face); aquí todo va en GiB.
    weights = file_gb(path) if weights_known else float((item or {}).get("size_gb", 0)) * 1e9 / GB
    arch = read_gguf_arch(path) or arch_from_catalog(item)
    note = ""
    if arch:
        kv = kv_cache_bytes(arch, ctx, kv_type) / GB
    else:  # sin datos de arquitectura: aproximación conservadora
        kv = weights * 0.05 * (ctx / 4096) * KV_BYTES.get(kv_type, 2.0) / 2
        note = "arquitectura desconocida: KV aproximada"
    if not weights_known and not item:
        note = "archivo no descargado y fuera del catálogo: tamaño desconocido"
    return ModelEstimate(entry.name, weights, kv, OVERHEAD_GB, arch is not None, weights_known, note)


def plan_usage(cfg: AppConfig, catalog: list[dict], gpu: GPUInfo | None = None) -> UsagePlan:
    main = cfg.model(cfg.roles.get("main", ""))
    fast = cfg.model(cfg.roles.get("fast", ""))
    draft = cfg.model(cfg.roles.get("draft", ""))
    ests: list[ModelEstimate] = []
    if main:
        e = estimate_entry(main, catalog)
        if draft and draft_compatible(main, draft):  # el borrador vive en el proceso de main
            d = estimate_entry(draft, catalog, ctx=main.ctx, kv_type=main.kv_type)
            e.weights_gb += d.weights_gb
            e.kv_gb += d.kv_gb
            e.note = (e.note + "; " if e.note else "") + f"incluye borrador {draft.name}"
        ests.append(e)
    if fast and fast is not main:
        ests.append(estimate_entry(fast, catalog))
    concurrent = cfg.keep_loaded

    def gpu_total() -> float:
        return sum(e.total_gb for e in ests) if concurrent else max((e.total_gb for e in ests), default=0)

    total = gpu_total()
    capacity = gpu.total_gb - DESKTOP_RESERVE_GB if gpu else None
    suggestions: list[str] = []
    ram_offload = 0.0
    if capacity is not None and total > capacity and main and main.gpu_layers < 0:
        # Modo automático: llama.cpp (--fit) deja en RAM lo que no quepa en la GPU.
        e = ests[0]
        arch = read_gguf_arch(main.path) or arch_from_catalog(catalog_entry(catalog, main.file))
        ram_offload = e.offload_gb = min(e.weights_gb, total - capacity + FIT_MARGIN_GB)
        total = gpu_total()
        if arch and arch.moe:
            e.note = (e.note + "; " if e.note else "") + \
                f"≈{ram_offload:.1f} GB de expertos MoE en RAM (automático, algo más lento)"
        else:
            e.note = (e.note + "; " if e.note else "") + \
                f"≈{ram_offload:.1f} GB de capas en RAM: MUCHO más lento"
            suggestions.append(f"«{main.name}» no es MoE: con capas en la RAM irá muy lento. Mejor un "
                               "modelo o una cuantización más pequeños.")
        ram = ram_total_gb()
        if ram and ram_offload > ram * 0.6:
            suggestions.append(f"Irían ≈{ram_offload:.0f} GB a la RAM y tienes {ram:.0f} GB: el sistema "
                               "puede quedarse sin memoria. Usa una cuantización más pequeña.")
    elif capacity is not None and total > capacity and main:
        others = sum(e.total_gb for e in ests[1:]) if concurrent else 0
        arch = read_gguf_arch(main.path) or arch_from_catalog(catalog_entry(catalog, main.file))
        if arch:
            free_for_kv = capacity - others - ests[0].weights_gb - ests[0].overhead_gb
            per_token = kv_cache_bytes(arch, 1, main.kv_type) / GB
            max_ctx = int(free_for_kv / per_token) // 1024 * 1024 if per_token and free_for_kv > 0 else 0
            if max_ctx >= 2048:
                suggestions.append(f"Con esta configuración, el contexto máximo de «{main.name}» que "
                                   f"cabe es ≈{max_ctx // 1024}K tokens.")
        if main.kv_type == "f16":
            suggestions.append("Usa KV cache q8_0: ocupa la mitad y apenas afecta a la calidad.")
        if concurrent and len(ests) > 1:
            suggestions.append("Desactiva «mantener main y fast cargados»: se turnarán en la GPU "
                               "(más lento al cambiar, pero cabe).")
        suggestions.append("Elige una cuantización más pequeña del modelo (p. ej. Q4_K_S o Q3_K_M) "
                           "o un modelo con menos parámetros.")
    return UsagePlan(ests, total, capacity, gpu, concurrent, suggestions, ram_offload)


def ram_total_gb() -> float | None:
    if os.name != "nt":
        try:
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / GB
        except (ValueError, OSError, AttributeError):
            return None

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    st = MEMORYSTATUSEX()
    st.dwLength = ctypes.sizeof(st)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
        return None
    return st.ullTotalPhys / GB


def ctx_hint(cfg: AppConfig, catalog: list[dict], model: str, n_ctx: int) -> str | None:
    """Sugerencia para ContextExhausted: cuánto costaría duplicar el contexto."""
    entry = cfg.model(model)
    if not entry:
        return None
    arch = read_gguf_arch(entry.path) or arch_from_catalog(catalog_entry(catalog, entry.file))
    if not arch:
        return None
    new_ctx = min(n_ctx * 2, 131072)
    if new_ctx <= n_ctx:
        return None
    extra = (kv_cache_bytes(arch, new_ctx, entry.kv_type) - kv_cache_bytes(arch, n_ctx, entry.kv_type)) / GB
    gpu = gpu_info()
    free = f" (ahora hay {gpu.free_gb:.1f} GB libres)" if gpu else ""
    return (f"Subir el contexto de «{model}» de {n_ctx // 1024}K a {new_ctx // 1024}K costaría "
            f"≈{extra:.1f} GB más de VRAM{free}. Se cambia en Configuración.")
