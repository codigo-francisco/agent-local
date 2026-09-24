"""«Recalcular»: propone la configuración óptima para el hardware disponible.

Mira la VRAM de esta PC y de las PCs remotas, la RAM y todos los modelos descargados, y decide:
- qué modelo va a cada rol (main, fast, draft);
- dónde corre cada uno: GPU local, GPU local + RAM, repartido con la PC remota, entero en la PC
  remota, o turnándose con el otro modelo;
- contexto, tipo de KV cache, capas en GPU, tokens de salida y demás.

No cambia nada: devuelve una propuesta con el motivo de cada cambio, que la interfaz muestra antes
de aplicarla con `apply`. La VRAM de esta PC y la de las PCs remotas conectadas se suman como VRAM
total: lo que no quepa en esa VRAM total va a la RAM (poco castigo en MoE, mucho en modelos densos).
"""

from __future__ import annotations

import copy
import itertools
import math
import re
from dataclasses import dataclass, field
from typing import Callable

from .. import config
from ..config import AppConfig, ModelEntry, name_from_file
from . import vram

CTX_STEPS = [4096, 8192, 12288, 16384, 24576, 32768, 49152, 65536, 98304, 131072]
CTX_CAP = 131072  # más allá de 128K el prefill se vuelve lento y rara vez compensa
FAST_CTX = 16384  # a «fast» solo le llegan trozos de conversación para resumir
MIN_MAIN_CTX = 16384  # por debajo, un agente de programación se queda sin sitio enseguida
F16_MIN_CTX = 32768  # solo usamos KV f16 si aun así caben al menos 32K
MOE_KV_SHARE = 0.15  # en modo automático, parte de la VRAM de main que dedicamos al contexto
FAST_MAX_PARAMS = 14  # «fast» debe ser pequeño: se carga junto a main o se turna con él

# Velocidad relativa según dónde corre (1 = entero en la VRAM local). Cuánto «cuesta» usar las PCs
# remotas depende de la preferencia de cálculo del usuario (AppConfig.tune_mode):
#   (repartido con la remota, main entero en la remota, fast entero en la remota)
NETWORK_FACTORS = {
    "vram": (1.0, 1.0, 1.0),  # VRAM total: la remota vale como la local; se usa antes que la RAM
    "speed": (0.65, 0.6, 0.47),  # lo más rápido, con medidas reales (fast remoto: 40 frente a 85 tok/s)
    "local": (0.01, 0.01, 0.01),  # esta PC primero (GPU y RAM); la remota solo si no cabe de otra forma
}
TURNS_FACTOR = 0.85  # main y fast se turnan: recarga al resumir
MODE_LABELS = {"local": "preferir lo local primero", "vram": "preferir la VRAM total",
               "speed": "optimizar la velocidad"}
NO_FAST_TERM = 0.3  # sin «fast», los resúmenes los hace main: procesa el historial mucho más lento

Progress = Callable[[str], None]


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


@dataclass
class _Model:
    entry: ModelEntry
    weights: float
    arch: vram.ArchInfo | None
    moe: bool
    tools: bool
    thinking: bool
    ctx_train: int
    params: float  # miles de millones (estimado)

    @property
    def name(self) -> str:
        return self.entry.name

    def kv(self, ctx: int, kv: str = "q8_0") -> float:
        return _kv_gb(self.arch, ctx, kv, self.weights)

    def need(self, ctx: int, extra: float = 0.0) -> float:
        """VRAM mínima para tenerlo entero en GPU con ese contexto (KV q8_0)."""
        return self.weights + vram.OVERHEAD_GB + self.kv(ctx) + extra


@dataclass
class _Plan:
    main: str  # gpu | ram | split | split_ram | remote
    fast: str  # none | local | remote | turns
    score: float
    main_budget: float  # VRAM para main (pesos + KV + margen) en su ubicación
    ram_gb: float = 0.0
    remote_budget: float = 0.0  # parte del presupuesto de main que está en la PC remota


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


def _params_from_name(text: str) -> float | None:
    """'Qwen3.6-35B-A3B' -> 35; 'gpt-oss-20b' -> 20 (el mayor «NB» del nombre: total, no activos)."""
    nums = [float(n) for n in re.findall(r"(?<![a-z0-9.])a?(\d+(?:\.\d+)?)b(?![a-z])", text.lower())]
    return max(nums) if nums else None


def _ram_factor(moe: bool, share: float) -> float:
    """Velocidad con `share` de los pesos en RAM: poco castigo en MoE, enorme en modelos densos."""
    return max(0.05, 1 - 0.5 * share) if moe else 1 / (1 + 8 * share)


# --- descubrir modelos -----------------------------------------------------------

def discover_models(new: AppConfig, catalog: list[dict]) -> list[ModelEntry]:
    """Añade a `new` los .gguf de models/ que aún no están configurados (con los valores por
    defecto del catálogo si están en él), salvo los que el usuario quitó a propósito."""
    added = []
    known = {m.path.name.lower() for m in new.models} | {f.lower() for f in new.ignored_files}
    models_dir = config.MODELS_DIR  # se lee al llamar (los tests lo redirigen)
    files = sorted(models_dir.glob("*.gguf")) if models_dir.is_dir() else []
    for f in files:
        low = f.name.lower()
        if low in known or "mmproj" in low or low.startswith("mtp-") or re.search(r"-\d{5}-of-\d{5}", low):
            continue  # proyectores de visión, cabezas MTP y trozos de modelos partidos no se sirven solos
        item = vram.catalog_entry(catalog, f.name)
        name = item["id"] if item else name_from_file(f.name)
        if new.model(name):
            continue
        entry = ModelEntry(name, f.name)
        for key, value in ((item or {}).get("defaults") or {}).items():
            if hasattr(entry, key) and key != "placement":
                setattr(entry, key, value)
        new.models.append(entry)
        known.add(low)
        added.append(entry)
    return added


def _inspect(entry: ModelEntry, catalog: list[dict]) -> _Model:
    item = vram.catalog_entry(catalog, entry.file)
    arch = vram.read_gguf_arch(entry.path) or vram.arch_from_catalog(item)
    meta = vram.model_meta(entry.path)
    weights = vram.file_gb(entry.path)
    # El catálogo manda: algunos modelos traen «tools» en la plantilla pero las escriben como texto.
    tools = bool(item["tools"]) if item and "tools" in item else bool(meta.get("tools"))
    moe = bool(arch and arch.moe) or bool(re.search(r"-a\d+(\.\d+)?b", entry.file.lower()))
    params = (_params_from_name(entry.file) or _params_from_name(str((item or {}).get("params", "")))
              or weights * 1.8)
    return _Model(entry, weights, arch, moe, tools, bool(meta.get("thinking")),
                  min(CTX_CAP, meta.get("context_length") or CTX_CAP), params)


# --- elegir roles -----------------------------------------------------------------

def _main_quality(m: _Model) -> float:
    return m.params ** 0.6 * (1.0 if m.tools else 0.35)


def _fast_quality(m: _Model) -> float:
    return min(m.params, 8) / 8 * (1.0 if m.tools else 0.7)


def _best_plan(main: _Model, fast: _Model | None, local: float, remote: float, ram: float | None,
               draft_gb: float = 0.0, mode: str = "vram") -> _Plan | None:
    """La mejor combinación de ubicaciones para main y fast (None si main no cabe de ninguna forma),
    según la preferencia de cálculo `mode` (ver NETWORK_FACTORS)."""
    network, remote_only, remote_fast = NETWORK_FACTORS.get(mode, NETWORK_FACTORS["vram"])
    need_m = main.need(MIN_MAIN_CTX, draft_gb)
    need_f = fast.need(min(FAST_CTX, fast.ctx_train)) if fast else 0.0
    ram_budget = ram if ram is not None else math.inf
    fast_opts = ["none"] if fast is None else ["local", "remote", "turns"]
    best: _Plan | None = None
    for f_mode, m_mode in itertools.product(fast_opts, ("gpu", "ram", "split", "split_ram", "remote")):
        lm = local - (need_f if f_mode == "local" else 0)
        rm = remote - (need_f if f_mode == "remote" else 0)
        if lm <= 0 or rm < 0 or (f_mode == "remote" and remote <= 0):
            continue
        spill, rb = 0.0, 0.0
        if m_mode == "gpu":
            if need_m > lm:
                continue
            speed, budget = 1.0, lm
        elif m_mode == "ram":
            spill = need_m - lm + vram.FIT_MARGIN_GB
            if spill <= vram.FIT_MARGIN_GB or spill > ram_budget or spill > main.weights:
                continue
            speed, budget = _ram_factor(main.moe, spill / main.weights), lm
        elif m_mode == "split":
            if rm <= 0 or need_m <= lm or need_m > lm + rm:
                continue
            speed, budget, rb = network, lm + rm, rm
        elif m_mode == "split_ram":
            spill = need_m - lm - rm + vram.FIT_MARGIN_GB
            if rm <= 0 or spill <= vram.FIT_MARGIN_GB or spill > ram_budget or spill > main.weights:
                continue
            speed, budget, rb = network * _ram_factor(main.moe, spill / main.weights), lm + rm, rm
        else:  # remote: solo si no cabe en local (la GPU local queda para fast)
            if rm <= 0 or need_m <= lm or need_m > rm:
                continue
            speed, budget, rb = remote_only, rm, rm
        if f_mode == "none":
            term = NO_FAST_TERM
        else:
            term = _fast_quality(fast) * (remote_fast if f_mode == "remote" else 1.0)
        score = speed * (0.85 + 0.15 * term) * (TURNS_FACTOR if f_mode == "turns" else 1.0)
        # Desempate: menos dependencia de la red y de la RAM.
        score -= 0.001 * (m_mode != "gpu") + 0.001 * (f_mode == "remote")
        if best is None or score > best.score:
            best = _Plan(m_mode, f_mode, score, budget, max(0.0, spill), rb)
    return best


def _choose_roles(models: list[_Model], local: float, remote: float, ram: float | None,
                  mode: str = "vram") -> tuple[_Model, _Model | None, _Plan]:
    """main: el más capaz que corra a buena velocidad; fast: el que mejor combine con él."""
    scored = []
    for m in models:
        plan = _best_plan(m, None, local, remote, ram, mode=mode)
        if plan:
            scored.append((_main_quality(m) * plan.score, m))
    if not scored:
        raise LookupError
    scored.sort(key=lambda t: (-t[0], t[1].name))
    main = scored[0][1]
    best: tuple[float, _Model | None, _Plan] | None = None
    for f in [None] + [m for m in models if m is not main and m.params <= FAST_MAX_PARAMS]:
        plan = _best_plan(main, f, local, remote, ram, mode=mode)
        if plan and (best is None or plan.score > best[0] + 1e-9):
            best = (plan.score, f, plan)
    assert best is not None  # main solo ya tenía plan
    return main, best[1], best[2]


def _choose_draft(main: _Model, models: list[_Model]) -> _Model | None:
    """Borrador para decodificación especulativa: mismo vocabulario y mucho más pequeño. En MoE
    apenas acelera (ya calculan poco por token), así que no se usa."""
    if main.moe:
        return None
    options = [m for m in models if m is not main and m.params <= main.params / 8
               and vram.draft_compatible(main.entry, m.entry)
               and vram.tokenizer_signature(main.entry.path) is not None]
    return min(options, key=lambda m: m.weights, default=None)


# --- recalcular -------------------------------------------------------------------

def recalculate(cfg: AppConfig, catalog: list[dict], gpu: vram.GPUInfo | None = None,
                ram_gb: float | None = None, progress: Progress | None = None,
                choose_roles: bool = True) -> TuneResult:
    say = progress or (lambda _msg: None)
    say("Leyendo la GPU de esta PC…")
    gpu = gpu or vram.gpu_info()
    say("Leyendo la memoria RAM…")
    ram_gb = ram_gb if ram_gb is not None else vram.ram_total_gb()
    if gpu is None:
        return TuneResult(None, error="No detecto una GPU NVIDIA (nvidia-smi). Sin GPU no puedo "
                                       "calcular el reparto de memoria.")
    new = copy.deepcopy(cfg)
    reasons: dict[tuple[str | None, str], str] = {}
    notes: list[str] = []
    remote_on = bool(new.rpc_endpoints())
    remote = vram.remote_capacity_gb(new) if remote_on else 0.0
    mode = cfg.tune_mode
    remote_unmeasured = remote_on and not remote
    hardware = (f"{gpu.name} · {gpu.total_gb:.1f} GB de VRAM"
                + (f" · {ram_gb:.0f} GB de RAM" if ram_gb else "")
                + (f" · PCs remotas: ≈{remote:.1f} GB de VRAM utilizable" if remote else "")
                + f" · preferencia: {MODE_LABELS.get(mode, mode)}")

    def put(obj, attr: str, value, reason: str, model: str | None = None) -> None:
        if getattr(obj, attr) != value:
            setattr(obj, attr, value)
            reasons[(model, attr)] = reason

    # --- modelos disponibles --------------------------------------------------
    added = discover_models(new, catalog) if choose_roles else []
    entries = [m for m in new.models if m.path.is_file()]
    infos: list[_Model] = []
    for i, entry in enumerate(entries, 1):
        say(f"Analizando modelos ({i}/{len(entries)}): {entry.name}…")
        infos.append(_inspect(entry, catalog))
    by_name = {m.name: m for m in infos}
    if added:
        notes.append("Encontré modelos descargados que no estaban configurados: "
                     + ", ".join(e.name for e in added) + ".")

    capacity = gpu.total_gb - vram.DESKTOP_RESERVE_GB
    ram_budget = vram.ram_budget_gb(cfg, ram_gb)
    say("Calculando el mejor reparto de modelos y memoria…")
    put(new, "flash_attn", "on", "Necesaria para la KV cache cuantizada y más rápida en GPUs NVIDIA.")

    # --- roles -------------------------------------------------------------------
    old_roles = dict(new.roles)
    if choose_roles:
        try:
            main, fast, plan = _choose_roles(infos, capacity, remote, ram_budget, mode)
        except LookupError:
            return TuneResult(None, error="Ningún modelo descargado cabe en esta PC (ni con la RAM ni "
                                          "con PCs remotas). Descarga uno más pequeño en «Modelos».",
                              hardware=hardware)
        new.roles["main"] = main.name
        new.roles["fast"] = fast.name if fast else ""
        if old_roles.get("main") != main.name:
            reasons[(None, "main")] = (f"El más capaz que corre bien aquí ({main.params:.0f}B"
                                       + (", MoE" if main.moe else "")
                                       + (", usa herramientas de forma nativa" if main.tools else "") + ").")
        if old_roles.get("fast") != new.roles["fast"]:
            reasons[(None, "fast")] = (f"Pequeño ({fast.params:.0f}B) para resumir sin quitarle "
                                       "sitio a main." if fast else
                                       "Ningún modelo pequeño aporta: main hará también los resúmenes.")
        draft = _choose_draft(main, infos)
        new.roles["draft"] = draft.name if draft else ""
        if old_roles.get("draft") != new.roles["draft"]:
            reasons[(None, "draft")] = (f"Misma familia que «{main.name}» y mucho más pequeño: "
                                        "acelera la generación." if draft else
                                        "Ninguno es de la misma familia que main"
                                        + (" (y en MoE apenas acelera)." if main.moe else "."))
    else:
        main = by_name.get(new.roles.get("main", ""))
        if main is None:
            return TuneResult(None, error="Asigna primero un modelo descargado al rol «main» (página "
                                          "Modelos).", hardware=hardware)
        fast = by_name.get(new.roles.get("fast", ""))
        if fast is main:
            fast = None
        draft = by_name.get(new.roles.get("draft", ""))
        if draft is not None and not vram.draft_compatible(main.entry, draft.entry):
            new.roles["draft"] = ""
            reasons[(None, "draft")] = (f"«{draft.name}» no es de la misma familia que «{main.name}» "
                                        "(vocabulario distinto): no puede acelerarlo.")
            draft = None
        plan = None
    draft_gb = (draft.weights + 0.3) if draft else 0.0  # pesos + su pequeña KV
    plan = _best_plan(main, fast, capacity, remote, ram_budget, draft_gb, mode) or plan
    if plan is None:  # ni con la RAM permitida: lo mínimo, avisando
        plan = _Plan("ram", "turns" if fast else "none", 0, capacity, main.need(MIN_MAIN_CTX) - capacity)
        notes.append(f"«{main.name}» no cabe ni usando el {cfg.ram_limit_pct}% de la RAM: el sistema "
                     "puede quedarse sin memoria. Usa una cuantización más pequeña.")

    # --- fast ---------------------------------------------------------------------
    if fast is not None:
        put(fast.entry, "ctx", _step_down(min(FAST_CTX, fast.ctx_train), CTX_CAP),
            "Para resumir basta un contexto moderado.", fast.name)
        put(fast.entry, "kv_type", "q8_0", "Mitad de memoria que f16 sin pérdida apreciable.", fast.name)
        put(fast.entry, "gpu_layers", 99, "Es pequeño: cabe entero en la GPU.", fast.name)
        if not remote_unmeasured:
            put(fast.entry, "placement", "remote" if plan.fast == "remote" else "local",
                "Entero en la VRAM de la PC remota: deja tu tarjeta gráfica para main."
                if plan.fast == "remote" else "Cabe en la VRAM de esta PC junto a main.", fast.name)
        put(new, "keep_loaded", plan.fast != "turns",
            "main y fast no caben a la vez en la VRAM: se turnarán (un poco de espera al resumir)."
            if plan.fast == "turns" else
            "fast va en la PC remota: los dos siguen cargados sin quitarse VRAM."
            if plan.fast == "remote" else "main y fast caben a la vez en la VRAM: no hay esperas al resumir.")

    # --- main ----------------------------------------------------------------------
    cap = main.ctx_train
    budget = plan.main_budget - draft_gb
    m = main.entry
    if not remote_unmeasured:
        placement = {"split": "split", "split_ram": "split", "remote": "remote"}.get(plan.main, "local")
        put(m, "placement", placement, {
            "local": "Cabe en la VRAM de esta PC." if plan.main == "gpu" else
                     "Usa toda la VRAM de esta PC y el resto va a la RAM.",
            "split": f"No cabe en tu tarjeta gráfica: sumando la VRAM de la PC remota (≈{plan.remote_budget:.0f} "
                     "GB) cabe" + (" entero." if plan.main == "split" else " con menos RAM."),
            "remote": "Cabe entero en la VRAM de la PC remota y deja tu tarjeta libre para fast.",
        }[placement], m.name)
    elif m.rpc or (fast and fast.entry.rpc):
        notes.append("Pulsa «Probar» en «PCs remotas» para medir su VRAM: sin ese dato no puedo contar "
                     "con ella al recalcular.")
    if plan.main in ("gpu", "split", "remote"):
        put(m, "gpu_layers", 99, "Cabe entero en VRAM: máxima velocidad.", m.name)
        ctx_f16 = _max_ctx(main.arch, main.weights, budget, "f16", cap)
        if ctx_f16 >= min(cap, F16_MIN_CTX):
            kv, ctx = "f16", ctx_f16
            kv_reason = "Hay VRAM de sobra: KV sin cuantizar, máxima precisión."
        else:
            kv, ctx = "q8_0", _max_ctx(main.arch, main.weights, budget, "q8_0", cap)
            kv_reason = "La mitad de memoria que f16 sin pérdida apreciable: permite más contexto."
            if ctx < 8192:
                kv, ctx = "q4_0", _max_ctx(main.arch, main.weights, budget, "q4_0", cap)
                kv_reason = "La VRAM va justa: KV q4_0 para conseguir un contexto usable."
        ctx = max(ctx, CTX_STEPS[0])
        ctx_reason = (f"El mayor que cabe en la VRAM libre ({budget:.1f} GB para main)"
                      + (f", limitado a {cap // 1024}K." if ctx == cap else "."))
    elif main.moe:
        put(m, "gpu_layers", -1, "No cabe entero en la VRAM: llama.cpp mete lo que cabe y deja el resto "
                                 "en la RAM. Al ser MoE, apenas pierde velocidad.", m.name)
        kv = "q8_0"
        kv_reason = ("Comprimir la memoria de la conversación deja más VRAM para el modelo (menos RAM, "
                     "más rápido).")
        ctx = max(MIN_MAIN_CTX, _step_down(max(0.5, budget * MOE_KV_SHARE) /
                                           max(main.kv(1), 1e-9), cap))
        ctx_reason = "Contexto amplio, dejando la mayor parte de la VRAM para el propio modelo."
        offload = main.need(ctx) - budget + vram.FIT_MARGIN_GB
        if ram_budget is not None and offload > ram_budget:
            notes.append(f"Harían falta ≈{offload:.0f} GB de RAM para «{m.name}» y has permitido "
                         f"{ram_budget:.0f} GB ({cfg.ram_limit_pct}% de tus {ram_gb:.0f} GB): el PC puede ir "
                         "justo de memoria. Usa una versión más comprimida del modelo o sube el límite de RAM.")
        else:
            notes.append(f"«{m.name}» no cabe entero en la VRAM"
                         + (" (ni sumando la PC remota)" if remote else "")
                         + f": ≈{max(0.0, offload):.1f} GB irán a la RAM del PC. Al ser MoE, va algo más "
                           "lento, no mucho.")
    else:
        put(m, "gpu_layers", -1, "No cabe en la VRAM: llama.cpp mete lo que cabe y el resto va a la RAM.",
            m.name)
        kv, ctx = "q8_0", MIN_MAIN_CTX
        kv_reason = "Comprimir la memoria de la conversación deja más VRAM para el modelo."
        ctx_reason = "Contexto mínimo razonable: la memoria no da para más."
        notes.append(f"«{m.name}» no es MoE y no cabe en la VRAM"
                     + (" ni sumando la PC remota" if remote else "") + ": parte irá a la RAM del PC y "
                     "será MUCHO más lento. Mejor elige un modelo más pequeño o una versión más comprimida.")
    put(m, "kv_type", kv, kv_reason, m.name)
    put(m, "ctx", ctx, ctx_reason, m.name)
    if plan.main != "gpu" or plan.fast == "remote":
        if m.rpc or (fast and fast.entry.rpc):
            notes.append("La primera carga manda los pesos a la PC remota por la red (≈3 min por cada "
                         "20 GB con cable Gigabit); las siguientes usan su caché y son más rápidas.")

    # --- modelos sin rol: que no se queden apuntando a la PC remota -------------------
    for other in infos:
        if other is not main and other is not fast and other.entry.placement != "local" \
                and not remote_unmeasured:
            put(other.entry, "placement", "local", "Sin rol: no hace falta reservarle la PC remota.",
                other.name)

    # --- tokens de salida ---------------------------------------------------------
    out = min(16384 if main.thinking else 4096, ctx // 4)
    put(new, "max_output_tokens", out,
        "Razona antes de responder y el razonamiento también gasta tokens de salida." if main.thinking
        else "Suficiente para respuestas y ediciones largas; el resto queda para el historial.")

    changes = _diff(cfg, new, reasons)
    return TuneResult(new, changes, notes, hardware)


ROLE_FIELDS = ("main", "fast", "draft")
MODEL_FIELDS = ("placement", "ctx", "kv_type", "gpu_layers")


def _diff(old: AppConfig, new: AppConfig, reasons: dict) -> list[Change]:
    changes = []
    for role in ROLE_FIELDS:
        if (old.roles.get(role) or "") != (new.roles.get(role) or ""):
            changes.append(Change(None, role, old.roles.get(role), new.roles.get(role),
                                  reasons.get((None, role), "")))
    for attr in ("keep_loaded", "flash_attn", "max_output_tokens"):
        if getattr(old, attr) != getattr(new, attr):
            changes.append(Change(None, attr, getattr(old, attr), getattr(new, attr), reasons.get((None, attr), "")))
    for m_new in new.models:
        m_old = old.model(m_new.name)
        if m_old is None:
            if m_new.name in new.roles.values():
                changes.append(Change(m_new.name, "added", None, m_new.file,
                                      "Descargado pero sin configurar: lo añado."))
            continue
        for attr in MODEL_FIELDS:
            if getattr(m_old, attr) != getattr(m_new, attr):
                changes.append(Change(m_new.name, attr, getattr(m_old, attr), getattr(m_new, attr),
                                      reasons.get((m_new.name, attr), "")))
    return changes


def apply(cfg: AppConfig, proposal: AppConfig) -> None:
    """Copia la propuesta sobre la configuración viva (mismo objeto que usa toda la app)."""
    for attr in ("keep_loaded", "flash_attn", "max_output_tokens"):
        setattr(cfg, attr, getattr(proposal, attr))
    for m_new in proposal.models:
        m_old = cfg.model(m_new.name)
        if m_old is None:
            if m_new.name in proposal.roles.values():  # descubierto y con rol: se añade
                cfg.models.append(copy.deepcopy(m_new))
            continue
        for attr in MODEL_FIELDS:
            setattr(m_old, attr, getattr(m_new, attr))
    cfg.roles.update(proposal.roles)


PLACEMENT_LABELS = {"local": "esta PC", "split": "repartido con PC remota", "remote": "PC remota"}


def describe(value: object, attr: str) -> str:
    if attr == "gpu_layers":
        return "auto (GPU+RAM)" if value == -1 else ("todas" if value == 99 else str(value))
    if attr == "ctx":
        return f"{int(value) // 1024}K"
    if attr == "placement":
        return PLACEMENT_LABELS.get(str(value), str(value))
    if isinstance(value, bool):
        return "sí" if value else "no"
    if attr in ROLE_FIELDS:
        return value or "— ninguno —"
    if attr == "added":
        return str(value) if value else "—"
    return str(value)


FIELD_LABELS = {"ctx": "Contexto", "kv_type": "KV cache", "gpu_layers": "Capas en GPU",
                "keep_loaded": "Main y fast a la vez", "flash_attn": "Flash attention",
                "max_output_tokens": "Tokens de salida", "main": "Rol main", "fast": "Rol fast",
                "draft": "Rol draft", "placement": "Dónde corre", "added": "Añadido a la configuración"}
