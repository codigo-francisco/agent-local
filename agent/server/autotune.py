"""«Recalcular»: propone los parámetros adecuados para el hardware de esta máquina.

Mira la VRAM y la RAM y los modelos asignados (sus GGUF reales) y decide, para cada rol:
contexto, tipo de KV cache y capas en GPU; además si main y fast caben a la vez, los tokens de
salida y si el borrador es compatible. No cambia nada: devuelve una propuesta con el motivo de
cada cambio, que la interfaz muestra antes de aplicarla con `apply`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from ..config import AppConfig, ModelEntry
from . import vram

CTX_STEPS = [4096, 8192, 12288, 16384, 24576, 32768, 49152, 65536, 98304, 131072]
CTX_CAP = 131072  # más allá de 128K el prefill se vuelve lento y rara vez compensa
FAST_CTX = 16384  # a «fast» solo le llegan trozos de conversación para resumir
MIN_MAIN_CTX = 16384  # por debajo, un agente de programación se queda sin sitio enseguida
F16_MIN_CTX = 32768  # solo usamos KV f16 si aun así caben al menos 32K
MOE_KV_SHARE = 0.15  # en modo automático, parte de la VRAM de main que dedicamos al contexto


@dataclass
class Change:
    model: str | None  # None = ajuste general
    field: str
    old: object
    new: object
    reason: str


@dataclass
class TuneResult:
    config: AppConfig | None
    changes: list[Change] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    hardware: str = ""
    error: str | None = None


def _step_down(value: float, cap: int) -> int:
    """El mayor escalón de contexto que no pasa de `value` ni de `cap` (0 si ninguno)."""
    return max((s for s in CTX_STEPS if s <= value and s <= cap), default=0)


def _kv_gb(arch: vram.ArchInfo | None, ctx: int, kv: str, weights_gb: float) -> float:
    if arch:
        return vram.kv_cache_bytes(arch, ctx, kv) / vram.GB
    return weights_gb * 0.05 * (ctx / 4096) * vram.KV_BYTES.get(kv, 2.0) / 2  # igual que estimate_entry


def _max_ctx(arch, weights_gb: float, budget_gb: float, kv: str, cap: int) -> int:
    free = budget_gb - weights_gb - vram.OVERHEAD_GB
    if free <= 0:
        return 0
    per_token = _kv_gb(arch, 1, kv, weights_gb)
    return _step_down(free / per_token, cap) if per_token > 0 else _step_down(cap, cap)


def recalculate(cfg: AppConfig, catalog: list[dict], gpu: vram.GPUInfo | None = None,
                ram_gb: float | None = None) -> TuneResult:
    gpu = gpu or vram.gpu_info()
    ram_gb = ram_gb if ram_gb is not None else vram.ram_total_gb()
    if gpu is None:
        return TuneResult(None, error="No detecto una GPU NVIDIA (nvidia-smi). Sin GPU no puedo "
                                       "calcular el reparto de memoria.")
    new = copy.deepcopy(cfg)
    reasons: dict[tuple[str | None, str], str] = {}
    notes: list[str] = []
    hardware = (f"{gpu.name} · {gpu.total_gb:.1f} GB de VRAM"
                + (f" · {ram_gb:.0f} GB de RAM" if ram_gb else ""))

    def put(obj, attr: str, value, reason: str, model: str | None = None) -> None:
        if getattr(obj, attr) != value:
            setattr(obj, attr, value)
            reasons[(model, attr)] = reason

    main = new.model(new.roles.get("main", ""))
    if main is None or not main.path.is_file():
        return TuneResult(None, error="Asigna primero un modelo descargado al rol «main» (página Modelos).",
                          hardware=hardware)
    fast = new.model(new.roles.get("fast", ""))
    if fast is not None and (fast.name == main.name or not fast.path.is_file()):
        fast = None
    draft = new.model(new.roles.get("draft", ""))
    capacity = gpu.total_gb - vram.DESKTOP_RESERVE_GB
    put(new, "flash_attn", "on", "Necesaria para la KV cache cuantizada y más rápida en GPUs NVIDIA.")

    # --- borrador: solo si es de la misma familia que main ---------------
    draft_gb = 0.0
    if draft is not None:
        if not draft.path.is_file() or not vram.draft_compatible(main, draft):
            new.roles["draft"] = ""
            reasons[(None, "draft")] = (f"«{draft.name}» no es de la misma familia que «{main.name}» "
                                        "(vocabulario distinto): no puede acelerarlo.")
            draft = None
        else:
            draft_gb = vram.file_gb(draft.path) + 0.3  # pesos + su pequeña KV

    # --- fast: contexto modesto y entero en la GPU -----------------------
    fast_gb = 0.0
    if fast is not None:
        trained = vram.model_meta(fast.path).get("context_length") or CTX_CAP
        put(fast, "ctx", _step_down(min(FAST_CTX, trained), CTX_CAP),
            "Para resumir basta un contexto moderado.", fast.name)
        put(fast, "kv_type", "q8_0", "Mitad de memoria que f16 sin pérdida apreciable.", fast.name)
        put(fast, "gpu_layers", 99, "Es pequeño: cabe entero en la GPU.", fast.name)
        fast_gb = vram.estimate_entry(fast, catalog).total_gb

    # --- main --------------------------------------------------------------
    arch = vram.read_gguf_arch(main.path) or vram.arch_from_catalog(vram.catalog_entry(catalog, main.file))
    meta = vram.model_meta(main.path)
    cap = min(CTX_CAP, meta.get("context_length") or CTX_CAP)
    weights = vram.file_gb(main.path)
    moe = bool(arch and arch.moe)
    need_min = weights + vram.OVERHEAD_GB + _kv_gb(arch, MIN_MAIN_CTX, "q8_0", weights)
    budget_shared = capacity - fast_gb - draft_gb
    budget_alone = capacity - draft_gb

    if need_min <= budget_shared:
        mode, budget, keep = "gpu", budget_shared, True
        why_keep = "main y fast caben a la vez en la GPU: no hay esperas al resumir."
    elif moe:
        mode, budget, keep = "auto", budget_shared, True
        why_keep = "main es MoE y reparte sus expertos con la RAM; fast cabe a la vez en la GPU."
    elif need_min <= budget_alone and fast is not None:
        mode, budget, keep = "gpu", budget_alone, False
        why_keep = "main no cabe junto a fast: se turnarán en la GPU (un poco de espera al resumir)."
    else:
        mode, budget, keep = "offload", budget_alone, False
        why_keep = "main ni siquiera cabe solo en la GPU."
        notes.append(f"«{main.name}» no es MoE y no cabe en {gpu.total_gb:.0f} GB de VRAM: parte de sus "
                     "capas irá a la RAM y será MUCHO más lento. Mejor elige un modelo o una "
                     "cuantización más pequeños.")
    if fast is not None:
        put(new, "keep_loaded", keep, why_keep)

    if mode == "gpu":
        put(main, "gpu_layers", 99, "Cabe entero en la GPU: máxima velocidad.", main.name)
        ctx_f16 = _max_ctx(arch, weights, budget, "f16", cap)
        if ctx_f16 >= min(cap, F16_MIN_CTX):
            kv, ctx = "f16", ctx_f16
            kv_reason = "Hay VRAM de sobra: KV sin cuantizar, máxima precisión."
        else:
            kv, ctx = "q8_0", _max_ctx(arch, weights, budget, "q8_0", cap)
            kv_reason = "La mitad de memoria que f16 sin pérdida apreciable: permite más contexto."
            if ctx < 8192:
                kv, ctx = "q4_0", _max_ctx(arch, weights, budget, "q4_0", cap)
                kv_reason = "La VRAM va justa: KV q4_0 para conseguir un contexto usable."
        ctx = max(ctx, CTX_STEPS[0])
        ctx_reason = (f"El mayor que cabe en la VRAM libre ({budget:.1f} GB para main)"
                      + (f", limitado a {cap // 1024}K." if ctx == cap else "."))
    elif mode == "auto":
        put(main, "gpu_layers", -1, "MoE que no cabe entero: llama.cpp deja expertos en la RAM "
                                    "(pierde poca velocidad).", main.name)
        kv = "q8_0"
        kv_reason = "En modo automático, cada GB que ahorra la KV es un GB de expertos más en la GPU."
        ctx = max(MIN_MAIN_CTX, _step_down(max(0.5, budget * MOE_KV_SHARE) /
                                           max(_kv_gb(arch, 1, kv, weights), 1e-9), cap))
        ctx_reason = "Contexto amplio reservando la mayor parte de la VRAM para los expertos."
        offload = weights + vram.OVERHEAD_GB + _kv_gb(arch, ctx, kv, weights) - budget + vram.FIT_MARGIN_GB
        if ram_gb and offload > ram_gb * 0.6:
            notes.append(f"Irían ≈{offload:.0f} GB de expertos a la RAM y tienes {ram_gb:.0f} GB: "
                         "el sistema puede ir justo. Considera una cuantización más pequeña.")
        else:
            notes.append(f"≈{max(0.0, offload):.1f} GB de expertos de «{main.name}» vivirán en la RAM.")
    else:
        put(main, "gpu_layers", -1, "No cabe en la GPU: llama.cpp reparte capas con la RAM.", main.name)
        kv, ctx = "q8_0", MIN_MAIN_CTX
        kv_reason = "Menos memoria en la GPU para el contexto."
        ctx_reason = "Contexto mínimo razonable: la memoria no da para más."
    put(main, "kv_type", kv, kv_reason, main.name)
    put(main, "ctx", ctx, ctx_reason, main.name)

    # --- tokens de salida ---------------------------------------------------
    thinking = bool(meta.get("thinking"))
    out = min(16384 if thinking else 4096, ctx // 4)
    put(new, "max_output_tokens", out,
        "Razona antes de responder y el razonamiento también gasta tokens de salida." if thinking
        else "Suficiente para respuestas y ediciones largas; el resto queda para el historial.")

    changes = _diff(cfg, new, reasons)
    return TuneResult(new, changes, notes, hardware)


def _diff(old: AppConfig, new: AppConfig, reasons: dict) -> list[Change]:
    changes = []
    for attr in ("keep_loaded", "flash_attn", "max_output_tokens"):
        if getattr(old, attr) != getattr(new, attr):
            changes.append(Change(None, attr, getattr(old, attr), getattr(new, attr), reasons.get((None, attr), "")))
    if old.roles.get("draft") != new.roles.get("draft"):
        changes.append(Change(None, "draft", old.roles.get("draft"), new.roles.get("draft"),
                              reasons.get((None, "draft"), "")))
    for m_new in new.models:
        m_old = old.model(m_new.name)
        if m_old is None:
            continue
        for attr in ("ctx", "kv_type", "gpu_layers"):
            if getattr(m_old, attr) != getattr(m_new, attr):
                changes.append(Change(m_new.name, attr, getattr(m_old, attr), getattr(m_new, attr),
                                      reasons.get((m_new.name, attr), "")))
    return changes


def apply(cfg: AppConfig, proposal: AppConfig) -> None:
    """Copia la propuesta sobre la configuración viva (mismo objeto que usa toda la app)."""
    for attr in ("keep_loaded", "flash_attn", "max_output_tokens"):
        setattr(cfg, attr, getattr(proposal, attr))
    cfg.roles.update(proposal.roles)
    for m_new in proposal.models:
        m_old = cfg.model(m_new.name)
        if m_old is not None:
            m_old.ctx, m_old.kv_type, m_old.gpu_layers = m_new.ctx, m_new.kv_type, m_new.gpu_layers


def describe(value: object, attr: str) -> str:
    if attr == "gpu_layers":
        return "auto (GPU+RAM)" if value == -1 else ("todas" if value == 99 else str(value))
    if attr == "ctx":
        return f"{int(value) // 1024}K"
    if isinstance(value, bool):
        return "sí" if value else "no"
    if attr == "draft":
        return value or "— ninguno —"
    return str(value)


FIELD_LABELS = {"ctx": "Contexto", "kv_type": "KV cache", "gpu_layers": "Capas en GPU",
                "keep_loaded": "Main y fast a la vez", "flash_attn": "Flash attention",
                "max_output_tokens": "Tokens de salida", "draft": "Rol draft"}
