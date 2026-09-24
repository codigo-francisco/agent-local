"""Calculadora de VRAM: pesos del modelo + KV cache (depende del contexto) + margen.

KV cache = capas × cabezas_kv × (dim_clave + dim_valor) × contexto × bytes_por_elemento
"""

from __future__ import annotations

import ctypes
import functools
import json
import os
import re
import subprocess
import threading
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

from ..config import GENERATED_DIR, AppConfig, ModelEntry, atomic_write

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
    where: str = "local"  # local | split | remote (ver ModelEntry.placement)

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
    ram_budget_gb: float | None = None  # RAM que se permite usar (ram_limit_pct de la total)
    remote_gb: float = 0.0  # VRAM de PCs remotas sumada a capacity_gb (modelos «repartidos»)
    remote_used_gb: float = 0.0  # lo que ocupan en las PCs remotas los modelos que van enteros allí
    remote_capacity_gb: float = 0.0  # VRAM utilizable medida en las PCs remotas

    @property
    def ram_ok(self) -> bool:
        return self.ram_budget_gb is None or self.ram_offload_gb <= self.ram_budget_gb

    @property
    def remote_ok(self) -> bool:
        return self.remote_used_gb <= self.remote_capacity_gb + 1e-6 or not self.remote_capacity_gb

    @property
    def fits(self) -> bool | None:
        return None if self.capacity_gb is None else \
            self.total_gb <= self.capacity_gb and self.ram_ok and self.remote_ok


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


# --- caché en disco de metadatos GGUF ----------------------------------------
# GGUFReader recorre todo el vocabulario: ~5 s por modelo. Sin caché en disco se repetía en cada
# arranque y congelaba la interfaz al pintar la calculadora de VRAM.
GGUF_CACHE_FILE = GENERATED_DIR / "gguf-cache.json"
_disk_cache: dict | None = None
_disk_lock = threading.Lock()


def _load_disk() -> dict:
    global _disk_cache
    if _disk_cache is None:
        try:
            data = json.loads(GGUF_CACHE_FILE.read_text(encoding="utf-8"))
            _disk_cache = data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            _disk_cache = {}
    return _disk_cache


def _persistent(kind: str, dump=lambda v: v, load=lambda v: v):
    """Cachea en disco el resultado de leer un .gguf; la clave incluye tamaño y fecha, así un
    archivo re-descargado se vuelve a leer. Los fallos no se cachean (descarga a medias)."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(path: str, mtime: float):
            try:
                size = os.path.getsize(path)
            except OSError:
                size = -1
            key = f"{kind}|{path}|{size}|{mtime}"
            with _disk_lock:
                cache = _load_disk()
                if key in cache:
                    try:
                        return load(cache[key])
                    except (TypeError, ValueError):
                        pass  # formato antiguo: se vuelve a leer
            value = fn(path, mtime)
            if value:
                with _disk_lock:
                    cache = _load_disk()
                    cache[key] = dump(value)
                    for old in list(cache)[:-200]:  # que no crezca sin límite
                        del cache[old]
                    try:
                        atomic_write(GGUF_CACHE_FILE, json.dumps(cache).encode("utf-8"))
                    except OSError:
                        pass
            return value
        return wrapper
    return deco


@lru_cache(maxsize=32)
@_persistent("arch", asdict, lambda v: ArchInfo(**v))
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
@_persistent("tokenizer", list, tuple)
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


def template_uses_tools(template: str) -> bool:
    """La plantilla de chat sabe recibir herramientas: el modelo las llama de forma nativa."""
    return isinstance(template, str) and re.search(r"\btools\b", template) is not None


@lru_cache(maxsize=32)
@_persistent("meta2")
def _meta_cached(path: str, mtime: float) -> dict:  # noqa: ARG001
    try:
        from gguf import GGUFReader
        r = GGUFReader(path, "r")
        arch = _field(r, "general.architecture")
        template = _field(r, "tokenizer.chat_template") or ""
        return {"context_length": _field(r, f"{arch}.context_length"),
                "thinking": isinstance(template, str) and any(m in template for m in THINKING_MARKERS),
                "tools": template_uses_tools(template), "architecture": str(arch or "")}
    except Exception:  # noqa: BLE001
        return {}


def model_meta(path: Path) -> dict:
    """{'context_length': contexto de entrenamiento, 'thinking': si razona antes de responder,
    'tools': si usa herramientas de forma nativa, 'architecture': arquitectura de llama.cpp}."""
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
        note = "no pude leer su estructura: la memoria de la conversación es aproximada"
    if not weights_known and not item:
        note = "no está descargado ni en el catálogo: no sé cuánto ocupa"
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
            e.note = (e.note + "; " if e.note else "") + f"incluye el modelo borrador {draft.name}, que acelera las respuestas"
        ests.append(e)
    if fast and fast is not main:
        ests.append(estimate_entry(fast, catalog))
    concurrent = cfg.keep_loaded
    remote_on = bool(cfg.rpc_endpoints())
    for e in ests:
        entry = cfg.model(e.name)
        e.where = entry.placement if remote_on and entry and entry.rpc else "local"
    local_ests = [e for e in ests if e.where != "remote"]
    remote_ests = [e for e in ests if e.where == "remote"]

    def gpu_total() -> float:
        return sum(e.total_gb for e in local_ests) if concurrent else \
            max((e.total_gb for e in local_ests), default=0)

    total = gpu_total()
    remote_avail = remote_capacity_gb(cfg)
    remote_used = sum(e.total_gb for e in remote_ests) if concurrent else \
        max((e.total_gb for e in remote_ests), default=0)
    split = any(e.where == "split" for e in local_ests)
    remote = max(0.0, remote_avail - remote_used) if split else 0.0
    capacity = gpu.total_gb - DESKTOP_RESERVE_GB + remote if gpu else None
    suggestions: list[str] = []
    if capacity is not None and total > capacity and main and not main.rpc and remote_avail > 0:
        suggestions.append(f"Tienes ≈{remote_avail:.0f} GB de VRAM en las PCs remotas sin usar. Pulsa "
                           f"«Recalcular», o pon «{main.name}» en «Repartido con PC remota», para sumarlos "
                           "y no tener que usar (tanto) la RAM.")
    elif remote_on and any(e.where != "local" for e in ests) and not remote_avail:
        suggestions.append("Aún no sé cuánta VRAM tiene la PC remota: pulsa «Probar» en «PCs remotas» "
                           "para medirla. Hasta entonces no la sumo.")
    if remote_avail and remote_used > remote_avail:
        suggestions.append(f"Lo que mandas entero a la PC remota necesita ≈{remote_used:.1f} GB y allí solo "
                           f"hay ≈{remote_avail:.1f} GB: baja su contexto o déjalo en esta PC.")
    ram_offload = 0.0
    if capacity is not None and total > capacity and main and main.gpu_layers < 0 \
            and ests[0].where != "remote":
        # Modo automático: llama.cpp (--fit) deja en RAM lo que no quepa en la GPU.
        e = ests[0]
        arch = read_gguf_arch(main.path) or arch_from_catalog(catalog_entry(catalog, main.file))
        ram_offload = e.offload_gb = min(e.weights_gb, total - capacity + FIT_MARGIN_GB)
        total = gpu_total()
        if arch and arch.moe:
            e.note = (e.note + "; " if e.note else "") + \
                ("es un modelo MoE: en cada palabra solo usa una parte, así que tener parte en la RAM "
                 "lo hace algo más lento, no mucho")
        else:
            e.note = (e.note + "; " if e.note else "") + \
                "no es un modelo MoE: con parte en la RAM irá MUCHO más lento"
            suggestions.append(f"«{main.name}» no cabe en la VRAM y, al no ser MoE, con parte en la RAM irá "
                               "muy lento. Mejor un modelo más pequeño o una versión más comprimida "
                               "(cuantización más baja).")
        budget = ram_budget_gb(cfg)
        if budget is not None and ram_offload > budget:
            suggestions.append(f"Harían falta ≈{ram_offload:.0f} GB de RAM y has permitido {budget:.0f} GB "
                               f"({cfg.ram_limit_pct}% de los {ram_total_gb():.0f} GB del PC): el PC puede "
                               "quedarse sin memoria. Usa una versión más comprimida del modelo o sube "
                               "el límite de RAM más abajo.")
    elif capacity is not None and total > capacity and main and ests[0].where != "remote":
        others = sum(e.total_gb for e in local_ests[1:]) if concurrent else 0
        arch = read_gguf_arch(main.path) or arch_from_catalog(catalog_entry(catalog, main.file))
        if arch:
            free_for_kv = capacity - others - ests[0].weights_gb - ests[0].overhead_gb
            per_token = kv_cache_bytes(arch, 1, main.kv_type) / GB
            max_ctx = int(free_for_kv / per_token) // 1024 * 1024 if per_token and free_for_kv > 0 else 0
            if max_ctx >= 2048:
                suggestions.append(f"Baja el contexto de «{main.name}»: con esta VRAM caben unos "
                                   f"{max_ctx // 1024}K tokens de conversación.")
        if main.kv_type == "f16":
            suggestions.append("Pon «KV cache» en q8_0: la conversación ocupa la mitad y la calidad "
                               "apenas cambia.")
        if concurrent and len(ests) > 1:
            suggestions.append("Desactiva «Mantener main y fast cargados a la vez»: se turnarán en la "
                               "VRAM (hay una espera al cambiar, pero cabe).")
        suggestions.append("O usa una versión más comprimida del modelo (cuantización Q4_K_S o Q3_K_M) "
                           "o un modelo más pequeño.")
    return UsagePlan(ests, total, capacity, gpu, concurrent, suggestions, ram_offload,
                     ram_budget_gb(cfg), remote, remote_used, remote_avail)


def remote_capacity_gb(cfg: AppConfig) -> float:
    """VRAM utilizable de las PCs remotas de la lista (activas y ya medidas con «Probar»): cada una
    reserva lo de su escritorio y el contexto CUDA / buffers de cómputo de su parte del modelo. Se
    suma a la de esta PC como VRAM total (llama.cpp reparte las capas entre todas)."""
    return sum(max(0.0, w.vram_gb - DESKTOP_RESERVE_GB - OVERHEAD_GB)
               for w in cfg.rpc_workers if w.enabled and w.host and w.vram_gb > 0)


def ram_budget_gb(cfg: AppConfig, ram: float | None = None) -> float | None:
    """RAM que pueden ocupar los modelos: `ram_limit_pct` (90 % por defecto) de la RAM total."""
    ram = ram if ram is not None else ram_total_gb()
    return ram * cfg.ram_limit_pct / 100 if ram else None


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
