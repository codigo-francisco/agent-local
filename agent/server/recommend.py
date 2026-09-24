"""«Recargar recomendaciones»: modelos nuevos de Hugging Face que servirían para programar con el
agente en ESTE hardware (GPU local, PCs remotas y RAM).

Consulta la API pública de Hugging Face (repos GGUF de cuantizadores conocidos), se queda con los
que llaman herramientas de forma nativa y cuya arquitectura entiende nuestro llama.cpp, elige para
cada uno la cuantización que mejor equilibra calidad y velocidad aquí, y los ordena. Los resultados
se guardan en generated/recommendations.json para mostrarlos sin volver a consultar.
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable

import httpx

from ..config import GENERATED_DIR, MODELS_DIR, AppConfig, ModelEntry, atomic_write
from . import autotune, swapconfig, vram

HF_API = "https://huggingface.co/api/models"
CACHE_FILE = GENERATED_DIR / "recommendations.json"
# Cuantizadores con GGUF fiables, en orden de preferencia si publican el mismo modelo.
AUTHORS = ("unsloth", "bartowski", "lmstudio-community", "ggml-org")
QUERIES = [{"sort": "trendingScore"}, {"sort": "downloads"}, {"sort": "trendingScore", "search": "coder"},
           {"sort": "createdAt"}]
EXPAND = ("gguf", "downloads", "likes", "lastModified", "createdAt", "pipeline_tag", "tags")
TEXT_PIPELINES = {"text-generation", "image-text-to-text", "any-to-any", None}
SKIP_WORDS = ("embed", "rerank", "guard", "abliterated", "uncensored", "-base-", "mmproj", "ocr", "tts")
CODE_WORDS = ("coder", "code", "devstral", "swe", "agent")
MAX_REPOS = 30  # repos que se inspeccionan a fondo (una petición por repo para ver sus archivos)
SHOW = 12

# Calidad relativa de cada cuantización (lo que se pierde frente al original).
QUANT_QUALITY = [
    (r"bf16|f16", 1.0), (r"mxfp4", 0.84), (r"q8_0|q8_k_xl", 0.97), (r"q6_k", 0.93),
    (r"q5_k_(m|l|xl)", 0.89), (r"q5_k_s|q5_0|q5_1", 0.87), (r"q4_k_(m|l|xl)", 0.82),
    (r"iq4_(nl|xs)|q4_k_s", 0.79), (r"q4_0|q4_1", 0.76), (r"q3_k_(m|l|xl)|iq3_m", 0.66),
    (r"iq3_(xxs|xs|s)|q3_k_s", 0.58),
]

Progress = Callable[[str], None]
# Puntuación de _best_plan para un modelo solo (sin «fast») entero en la GPU: la normaliza a 1.
SOLO_BASE = 0.85 + 0.15 * autotune.NO_FAST_TERM


@dataclass
class Recommendation:
    repo: str
    file: str
    size_gb: float
    params_b: float
    moe: bool
    architecture: str
    quant: str
    placement: str  # explicación de dónde correría
    speed: float  # velocidad relativa estimada (1 = entero en tu GPU)
    score: float
    downloads: int = 0
    likes: int = 0
    created: str = ""
    thinking: bool = False
    coding: bool = False
    downloaded: bool = False
    configured: bool = False
    base_model: str = ""
    ram_gb: float = 0.0  # parte del modelo que iría a la RAM (0 = cabe entero en VRAM)

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{self.repo}"

    @property
    def name(self) -> str:
        return self.repo.split("/")[-1].removesuffix("-GGUF").removesuffix("-gguf")


@dataclass
class Report:
    items: list[Recommendation] = field(default_factory=list)
    hardware: str = ""
    updated: float = 0.0
    error: str = ""
    inspected: int = 0
    # Modelos revisados a fondo con sus archivos: permiten rehacer el cálculo al cambiar el hardware
    # (p. ej. al añadir o quitar una PC remota) sin volver a consultar Hugging Face.
    candidates: list[dict] = field(default_factory=list)


# --- utilidades --------------------------------------------------------------------

def quant_of(filename: str) -> tuple[str, float] | None:
    low = filename.lower()
    for pattern, quality in QUANT_QUALITY:
        m = re.search(rf"(?:ud-)?({pattern})(?=[._-])", low)
        if m:
            return m.group(0).upper(), quality
    return None


@lru_cache(maxsize=1)
def _llama_dll_bytes() -> bytes:
    exe = swapconfig.find_binary("llama-server")
    for p in ((exe.parent / "llama.dll") if exe else None, (exe.parent / "libllama.so") if exe else None):
        if p and p.is_file():
            try:
                return p.read_bytes()
            except OSError:
                pass
    return b""


def arch_supported(arch: str) -> bool | None:
    """¿Conoce nuestro llama.cpp esta arquitectura? (None si no se puede saber)."""
    data = _llama_dll_bytes()
    if not data or not arch:
        return None
    return b"\x00" + arch.encode() + b"\x00" in data


def _is_moe(repo: str, arch: str) -> bool:
    return "moe" in arch or arch in ("gpt-oss", "qwen3next", "llama4", "deepseek2", "glm4moe") \
        or re.search(r"-a\d+(\.\d+)?b", repo.lower()) is not None


def _base_model(tags: list[str], repo: str) -> str:
    for t in tags or []:
        if t.startswith("base_model:quantized:"):
            return t.split(":", 2)[2].lower()
    for t in tags or []:
        if t.startswith("base_model:") and t.count(":") == 1:
            return t.split(":", 1)[1].lower()
    return repo.split("/")[-1].lower().removesuffix("-gguf")


def hardware(cfg: AppConfig, gpu: vram.GPUInfo | None, ram_gb: float | None) -> tuple[float, float, float | None, str]:
    local = (gpu.total_gb - vram.DESKTOP_RESERVE_GB) if gpu else 0.0
    remote = vram.remote_capacity_gb(cfg) if cfg.rpc_endpoints() else 0.0
    ram = vram.ram_budget_gb(cfg, ram_gb)
    gpu_text = f"{gpu.name} ({gpu.total_gb:.0f} GB)" if gpu else "sin GPU NVIDIA"
    if remote:  # la VRAM remota se suma: el modelo la ve como una sola
        gpu_text = f"≈{local + remote:.1f} GB de VRAM en total ({gpu_text} + PCs remotas ≈{remote:.1f} GB)"
    text = gpu_text + (f" + {ram:.0f} GB de RAM para modelos" if ram else "")         + f" · preferencia: {autotune.MODE_LABELS.get(cfg.tune_mode, cfg.tune_mode)}"
    return local, remote, ram, text


def _fit(size_gb: float, params: float, moe: bool, local: float, remote: float, ram: float | None,
         mode: str = "vram") -> tuple[float, float, str] | None:
    """(velocidad relativa, GB en RAM, dónde correría) o None si no cabe, según la preferencia de
    cálculo: «vram» suma la VRAM de las PCs remotas a la local como si fuera una sola GPU; «local»
    usa esta PC (GPU y RAM) primero; «speed» elige lo más rápido (la red también cuesta)."""
    probe = autotune._Model(ModelEntry("x", "x.gguf"), size_gb / 1.0737, None, moe, True, False,
                            autotune.CTX_CAP, params)
    need = probe.need(autotune.MIN_MAIN_CTX)
    if mode == "vram":
        plan = autotune._best_plan(probe, None, local + remote, 0.0, ram)
        if plan is None:
            return None
        speed = min(1.0, plan.score / SOLO_BASE + 0.002)  # 1 = entero en VRAM
        return speed, plan.ram_gb, fit_text(need, plan.ram_gb, moe, local, remote)
    plan = autotune._best_plan(probe, None, local, remote, ram, mode=mode)
    if plan is None:
        return None
    # Velocidad real estimada (con el coste de la red) aunque el orden lo marque la preferencia.
    factor = autotune.NETWORK_FACTORS["speed"]
    real = {"gpu": 1.0, "ram": 1.0, "split": factor[0], "split_ram": factor[0], "remote": factor[1]}
    ram_speed = autotune._ram_factor(moe, plan.ram_gb / probe.weights) if plan.ram_gb else 1.0
    speed = min(1.0, real[plan.main] * ram_speed + 0.002)
    uses_remote = plan.main in ("split", "split_ram", "remote")
    return speed, plan.ram_gb, fit_text(need, plan.ram_gb, moe, local, remote, uses_remote,
                                        whole_remote=plan.main == "remote")


def fit_text(need_gb: float, ram_gb: float, moe: bool, local: float, remote: float,
             uses_remote: bool | None = None, whole_remote: bool = False) -> str:
    """Frase para el usuario: dónde correría un modelo que necesita `need_gb` de VRAM."""
    in_vram = need_gb - ram_gb
    if uses_remote is None:
        uses_remote = remote > 0 and in_vram > local + 0.05
    if whole_remote:
        return "Cabe entero en la VRAM de la PC remota y deja tu tarjeta libre"
    if ram_gb <= 0:
        if uses_remote:
            return (f"Cabe entero en la VRAM sumando la PC remota (≈{in_vram - local:.0f} GB van allí): "
                    "va rápido")
        return "Cabe entero en tu tarjeta gráfica (VRAM): lo más rápido"
    vram_part = "toda la VRAM (tu tarjeta + la PC remota)" if uses_remote else "toda tu tarjeta gráfica"
    speed = ("algo más lento, porque es MoE y solo usa una parte cada vez" if moe
             else "MUCHO más lento, porque no es MoE")
    return f"Usa {vram_part} y deja ≈{ram_gb:.0f} GB en la RAM del PC: {speed}"


# --- consulta a Hugging Face ---------------------------------------------------------

def _get(client: httpx.Client, url: str, params=None):
    r = client.get(url, params=params)
    r.raise_for_status()
    return r.json()


def _list_repos(client: httpx.Client, say: Progress) -> list[dict]:
    seen: dict[str, dict] = {}
    for author in AUTHORS:
        say(f"Buscando modelos nuevos de {author} en Hugging Face…")
        for q in QUERIES:
            params = [("author", author), ("filter", "gguf"), ("direction", "-1"), ("limit", "40"),
                      *[("expand[]", e) for e in EXPAND], *q.items()]
            try:
                for repo in _get(client, HF_API, params):
                    seen.setdefault(repo["id"], repo)
            except (httpx.HTTPError, ValueError, KeyError):
                continue
    return list(seen.values())


def _candidate(repo: dict) -> dict | None:
    rid = repo.get("id", "")
    low = rid.lower()
    g = repo.get("gguf") or {}
    arch = str(g.get("architecture") or "")
    if repo.get("pipeline_tag") not in TEXT_PIPELINES or any(w in low for w in SKIP_WORDS):
        return None
    if not vram.template_uses_tools(g.get("chat_template") or ""):
        return None  # sin herramientas nativas no sirve como agente
    if arch_supported(arch) is False:
        return None  # tu llama.cpp no sabría cargarlo
    params = (g.get("total") or 0) / 1e9 or autotune._params_from_name(rid) or 0
    if params < 1.5:
        return None
    return {"repo": rid, "arch": arch, "params": params, "moe": _is_moe(rid, arch),
            "thinking": any(m in (g.get("chat_template") or "") for m in vram.THINKING_MARKERS),
            "downloads": int(repo.get("downloads") or 0), "likes": int(repo.get("likes") or 0),
            "created": str(repo.get("createdAt") or "")[:10], "base": _base_model(repo.get("tags"), rid),
            "coding": any(w in low for w in CODE_WORDS)}


def _files(client: httpx.Client, repo: str) -> list[tuple[str, float]]:
    """Archivos .gguf de un solo trozo en la raíz del repo, con su tamaño en GB decimales."""
    try:
        tree = _get(client, f"{HF_API}/{repo}/tree/main")
    except (httpx.HTTPError, ValueError):
        return []
    out = []
    for f in tree:
        path = f.get("path", "")
        low = path.lower()
        # Fuera: trozos de modelos partidos, proyectores de visión y cabezas MTP (no son el modelo).
        if (f.get("type") == "file" and low.endswith(".gguf") and "/" not in path
                and "mmproj" not in low and not low.startswith("mtp-")
                and not re.search(r"-\d{5}-of-\d{5}", low)):
            out.append((path, (f.get("size") or 0) / 1e9))
    return out


def _best_quant(c: dict, files: list[tuple[str, float]], local: float, remote: float,
                ram: float | None, mode: str = "vram") -> Recommendation | None:
    best: tuple[float, Recommendation] | None = None
    for path, size in files:
        q = quant_of(path)
        if not q or size <= 0:
            continue
        label, quality = q
        if label.endswith("MXFP4") and c["arch"] == "gpt-oss":
            quality = 1.0  # gpt-oss se publica así: es su precisión original
        if size < c["params"] * 0.3:
            continue  # < 2,4 bits por peso: archivo auxiliar o incompleto, no el modelo
        fit = _fit(size, c["params"], c["moe"], local, remote, ram, mode)
        if fit is None:
            continue
        speed, ram_gb, where = fit
        value = speed * quality ** 0.5
        if best is None or value > best[0]:
            rec =Recommendation(c["repo"], path, round(size, 2), round(c["params"], 1), c["moe"],
                                 c["arch"], label, where, round(speed, 2), 0.0, c["downloads"],
                                 c["likes"], c["created"], c["thinking"], c["coding"], base_model=c["base"],
                                 ram_gb=round(ram_gb, 1))
            best = (value, rec)
    if best is None:
        return None
    rec = best[1]
    # Capacidad (parámetros), velocidad y calidad de la cuantización; algo de ventaja a los
    # especializados en código y a los recientes y populares.
    age_days = 9999.0
    try:
        age_days = (time.time() - time.mktime(time.strptime(rec.created, "%Y-%m-%d"))) / 86400
    except ValueError:
        pass
    rec.score = round(rec.params_b ** 0.6 * best[0] * (1.15 if rec.coding else 1.0)
                      * (1.1 if age_days < 180 else 1.0) * (1 + min(rec.downloads, 1_000_000) / 5_000_000), 3)
    return rec


def refresh(cfg: AppConfig, gpu: vram.GPUInfo | None = None, ram_gb: float | None = None,
            say: Progress | None = None, transport: httpx.BaseTransport | None = None) -> Report:
    say = say or (lambda _m: None)
    say("Leyendo el hardware disponible…")
    gpu = gpu or vram.gpu_info()
    ram_gb = ram_gb if ram_gb is not None else vram.ram_total_gb()
    local, remote, ram, hw = hardware(cfg, gpu, ram_gb)
    if not gpu:
        return Report(hardware=hw, error="No detecto una GPU NVIDIA: sin ella no puedo saber qué cabe.")
    report = Report(hardware=hw, updated=time.time())
    try:
        with httpx.Client(timeout=httpx.Timeout(20.0, connect=10.0), follow_redirects=True,
                          transport=transport) as client:
            repos = _list_repos(client, say)
            if not repos:
                return Report(hardware=hw, error="No pude consultar Hugging Face. ¿Hay conexión a internet?")
            say(f"Filtrando {len(repos)} repos: herramientas nativas y arquitecturas de tu llama.cpp…")
            cands = [c for c in map(_candidate, repos) if c]
            # Un modelo base, un repo (el del cuantizador preferido).
            by_base: dict[str, dict] = {}
            for c in sorted(cands, key=lambda c: AUTHORS.index(c["repo"].split("/")[0])
                            if c["repo"].split("/")[0] in AUTHORS else 99):
                by_base.setdefault(c["base"], c)
            # Primero los que pueden caber (por parámetros): no pedir archivos de un 400B.
            max_params = (local + remote + (ram or 0)) * 2.2
            pool = sorted((c for c in by_base.values() if c["params"] <= max_params),
                          key=lambda c: (-c["params"] ** 0.6 * (1.15 if c["coding"] else 1)
                                         - c["downloads"] / 2e6))[:MAX_REPOS]
            done = {"n": 0}

            def inspect(c: dict) -> dict:
                files = _files(client, c["repo"])
                done["n"] += 1
                say(f"Eligiendo la cuantización que cabe ({done['n']}/{len(pool)}): {c['repo']}…")
                return {"c": c, "files": files}

            with ThreadPoolExecutor(max_workers=6) as ex:
                report.candidates = [x for x in ex.map(inspect, pool) if x["files"]]
    except httpx.HTTPError as e:
        return Report(hardware=hw, error=f"Error consultando Hugging Face: {e}")
    report.inspected = len(repos)
    _rank(report, cfg, local, remote, ram)
    save(report)
    say("Listo.")
    return report


def _rank(report: Report, cfg: AppConfig, local: float, remote: float, ram: float | None) -> None:
    recs = [r for r in (_best_quant(x["c"], [tuple(f) for f in x["files"]], local, remote, ram,
                                    cfg.tune_mode) for x in report.candidates) if r]
    local_files = {p.name.lower() for p in MODELS_DIR.glob("*.gguf")} if MODELS_DIR.is_dir() else set()
    configured = {m.path.name.lower() for m in cfg.models}
    for r in recs:
        r.downloaded = r.file.lower() in local_files
        r.configured = r.file.lower() in configured
    report.items = sorted(recs, key=lambda r: -r.score)[:SHOW]


def recompute(report: Report, cfg: AppConfig, gpu: vram.GPUInfo | None = None,
              ram_gb: float | None = None) -> Report:
    """Rehace las recomendaciones con el hardware actual (VRAM local + PCs remotas de la lista)
    usando los modelos ya revisados: instantáneo y sin internet."""
    gpu = gpu or vram.gpu_info()
    ram_gb = ram_gb if ram_gb is not None else vram.ram_total_gb()
    local, remote, ram, hw = hardware(cfg, gpu, ram_gb)
    if not gpu or not report.candidates:
        return report
    new = Report(hardware=hw, updated=report.updated, inspected=report.inspected,
                 candidates=report.candidates)
    _rank(new, cfg, local, remote, ram)
    save(new)
    return new


# --- caché en disco ------------------------------------------------------------------

def save(report: Report, path: Path | None = None) -> None:
    path = path or CACHE_FILE  # se lee al llamar (los tests lo redirigen)
    data = {"hardware": report.hardware, "updated": report.updated, "inspected": report.inspected,
            "items": [asdict(r) for r in report.items], "candidates": report.candidates}
    try:
        atomic_write(path, json.dumps(data, ensure_ascii=False, indent=1).encode("utf-8"))
    except OSError:
        pass


def load(path: Path | None = None) -> Report | None:
    path = path or CACHE_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        items = [Recommendation(**r) for r in data.get("items", [])]
    except (OSError, ValueError, TypeError):
        return None
    local_files = {p.name.lower() for p in MODELS_DIR.glob("*.gguf")} if MODELS_DIR.is_dir() else set()
    for r in items:  # pudo descargarse desde la última consulta
        r.downloaded = r.file.lower() in local_files
    cands = [x for x in data.get("candidates") or [] if isinstance(x, dict) and "c" in x and "files" in x]
    return Report(items, data.get("hardware", ""), float(data.get("updated") or 0), "",
                  int(data.get("inspected") or 0), cands)
