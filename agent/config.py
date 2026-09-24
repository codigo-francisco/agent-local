"""Configuración de la aplicación: rutas del proyecto y `config/models.yaml`."""

from __future__ import annotations

import os
import re
import tempfile
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
MODELS_DIR = ROOT / "models"
BIN_DIR = ROOT / "bin"
GENERATED_DIR = ROOT / "generated"
CONFIG_FILE = CONFIG_DIR / "models.yaml"
CATALOG_FILE = CONFIG_DIR / "catalog.yaml"

KV_TYPES = ("f16", "q8_0", "q4_0")
ROLES = ("main", "fast", "draft")


def atomic_write(path: Path, data: bytes) -> None:
    """Escribe en un temporal de la misma carpeta y lo renombra: si la app se cierra a mitad, el
    archivo queda como estaba o completo, nunca truncado."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        if path.exists():  # conservar permisos (p. ej. scripts ejecutables)
            try:
                os.chmod(tmp, path.stat().st_mode)
            except OSError:
                pass
        for attempt in range(5):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:  # Windows: otro programa (antivirus, editor) lo tiene abierto
                if attempt == 4:
                    raise
                time.sleep(0.1 * (attempt + 1))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


@dataclass
class ModelEntry:
    """Un modelo que llama-swap puede servir."""

    name: str  # identificador en la API (campo `model`)
    file: str  # nombre del .gguf dentro de models/
    ctx: int = 16384
    kv_type: str = "q8_0"
    gpu_layers: int = 99
    extra_args: str = ""
    # Dónde corre (llama.cpp RPC): "local" (esta PC), "split" (repartido entre esta PC y las
    # remotas) o "remote" (entero en las PCs remotas, deja libre la GPU local).
    placement: str = "local"

    @property
    def path(self) -> Path:
        p = Path(self.file)
        return p if p.is_absolute() else MODELS_DIR / p

    @property
    def rpc(self) -> bool:
        """¿Usa las PCs remotas (repartido o entero)?"""
        return self.placement in ("split", "remote")


PLACEMENTS = ("local", "split", "remote")
# Preferencia de cálculo: "local" = esta PC (GPU y RAM) primero, las remotas solo si no cabe;
# "vram" = VRAM total (local + remotas) antes que la RAM; "speed" = lo más rápido para el modelo.
TUNE_MODES = ("local", "vram", "speed")


RPC_DEFAULT_PORT = 50052


@dataclass
class RpcWorker:
    """Otra PC que presta su GPU con `ggml-rpc-server` (llama.cpp RPC)."""

    host: str
    port: int = RPC_DEFAULT_PORT
    enabled: bool = True
    vram_gb: float = 0.0  # VRAM total medida con «Probar» (0 = desconocida)
    name: str = ""
    devices: int = 1  # GPUs que ofrece (llama.cpp las numera RPC0, RPC1… en orden)

    @property
    def endpoint(self) -> str:
        return f"{self.host}:{self.port}"


@dataclass
class AppConfig:
    port: int = 8080
    endpoint: str = "http://127.0.0.1:8080/v1"
    workspace: str = str(Path.home())
    roles: dict[str, str] = field(default_factory=lambda: {"main": "", "fast": "", "draft": ""})
    models: list[ModelEntry] = field(default_factory=list)
    keep_loaded: bool = True  # main y fast cargados a la vez (grupo de llama-swap)
    flash_attn: str = "on"
    ram_limit_pct: int = 90  # % de la RAM total que pueden ocupar las capas/expertos que no caben en la GPU
    max_steps: int = 25
    max_output_tokens: int = 4096
    max_tool_output: int = 12000  # caracteres
    command_timeout: int = 120  # segundos
    confirm: str = "ask"  # "ask" | "auto"
    always_allow: list[str] = field(default_factory=list)  # herramientas aprobadas «siempre»
    rpc_workers: list[RpcWorker] = field(default_factory=list)  # PCs remotas (llama.cpp RPC)
    # .gguf de models/ que el usuario quitó de la configuración: no se vuelven a añadir solos.
    ignored_files: list[str] = field(default_factory=list)
    # Cómo reparten «Recalcular» y las recomendaciones la memoria (ver TUNE_MODES).
    tune_mode: str = "vram"

    # --- consultas -------------------------------------------------------
    def model(self, name: str) -> ModelEntry | None:
        return next((m for m in self.models if m.name == name), None)

    def rpc_endpoints(self) -> list[str]:
        """«host:puerto» de las PCs remotas activas, en el formato de `llama-server --rpc`."""
        return [w.endpoint for w in self.rpc_workers if w.enabled and w.host]

    def rpc_device_names(self) -> list[str]:
        """Nombres de las GPUs remotas para `--device` (RPC0, RPC1…, en el orden de `--rpc`)."""
        n = sum(max(1, w.devices) for w in self.rpc_workers if w.enabled and w.host)
        return [f"RPC{i}" for i in range(n)]

    def resolve(self, model_or_role: str | None) -> str:
        """Convierte un rol ("main", "fast") en el nombre del modelo."""
        key = model_or_role or "main"
        if key in self.roles:
            name = self.roles.get(key) or ""
            if not name:
                raise KeyError(f"El rol '{key}' no tiene modelo asignado.")
            return name
        return key

    def ctx_for(self, name: str) -> int:
        m = self.model(name)
        return m.ctx if m else 8192

    # --- validación ------------------------------------------------------
    def sanitize(self) -> list[str]:
        """Corrige tipos y valores fuera de rango (un YAML editado a mano con `port: "abc"` o
        `ctx: 0` fallaría lejos de su causa). Devuelve una descripción de cada corrección."""
        fixes: list[str] = []

        def clamp(obj, attr: str, lo: int, hi: int, default: int, label: str) -> None:
            value = getattr(obj, attr)
            try:
                n = int(value)
            except (TypeError, ValueError):
                n = default
            else:
                n = min(hi, max(lo, n))
            if n != value:
                fixes.append(f"{label}: {value!r} → {n}")
            setattr(obj, attr, n)

        def choice(obj, attr: str, allowed: tuple, default: str, label: str) -> None:
            value = getattr(obj, attr)
            if value not in allowed:
                fixes.append(f"{label}: {value!r} → {default!r}")
                setattr(obj, attr, default)

        clamp(self, "port", 1024, 65535, 8080, "port")
        clamp(self, "max_steps", 1, 500, 25, "max_steps")
        clamp(self, "max_output_tokens", 64, 131072, 4096, "max_output_tokens")
        clamp(self, "max_tool_output", 1000, 1_000_000, 12000, "max_tool_output")
        clamp(self, "command_timeout", 1, 3600, 120, "command_timeout")
        clamp(self, "ram_limit_pct", 10, 100, 90, "ram_limit_pct")
        choice(self, "confirm", ("ask", "auto"), "ask", "confirm")
        choice(self, "flash_attn", ("on", "auto", "off"), "on", "flash_attn")
        choice(self, "tune_mode", TUNE_MODES, "vram", "tune_mode")
        self.keep_loaded = bool(self.keep_loaded)
        for attr in ("endpoint", "workspace"):
            if not isinstance(getattr(self, attr), str):
                fixes.append(f"{attr}: {getattr(self, attr)!r} no es texto")
                setattr(self, attr, getattr(AppConfig(), attr))
        self.roles = {k: v if isinstance(v, str) else "" for k, v in self.roles.items()}
        self.always_allow = [t for t in self.always_allow if isinstance(t, str)]
        if not isinstance(self.ignored_files, list):
            self.ignored_files = []
        self.ignored_files = [f for f in self.ignored_files if isinstance(f, str) and f]
        seen: set[str] = set()
        models = []
        for m in self.models:
            if not isinstance(m.name, str) or not isinstance(m.file, str) or not m.name or not m.file \
                    or m.name in seen:
                fixes.append(f"modelo descartado (sin nombre/archivo o repetido): {m.name!r}")
                continue
            seen.add(m.name)
            clamp(m, "ctx", 512, 1_048_576, 16384, f"{m.name}.ctx")
            clamp(m, "gpu_layers", -1, 999, 99, f"{m.name}.gpu_layers")
            choice(m, "kv_type", KV_TYPES, "q8_0", f"{m.name}.kv_type")
            if not isinstance(m.extra_args, str):
                m.extra_args = ""
            choice(m, "placement", PLACEMENTS, "local", f"{m.name}.placement")
            models.append(m)
        self.models = models
        workers = []
        seen_rpc: set[str] = set()
        for w in self.rpc_workers:
            host = w.host.strip() if isinstance(w.host, str) else ""
            if not host:  # fila añadida en la GUI y dejada en blanco: no es un error
                continue
            w.host = host
            clamp(w, "port", 1, 65535, RPC_DEFAULT_PORT, f"rpc {host}.port")
            if w.endpoint in seen_rpc:
                fixes.append(f"PC remota repetida descartada: {w.endpoint}")
                continue
            seen_rpc.add(w.endpoint)
            try:
                w.vram_gb = max(0.0, float(w.vram_gb))
            except (TypeError, ValueError):
                w.vram_gb = 0.0
            w.enabled = bool(w.enabled)
            clamp(w, "devices", 1, 16, 1, f"rpc {host}.devices")
            if not isinstance(w.name, str):
                w.name = ""
            workers.append(w)
        self.rpc_workers = workers
        return fixes

    # --- persistencia ----------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "AppConfig":
        if not isinstance(data, dict):
            raise TypeError("la configuración debe ser un diccionario YAML")
        data = dict(data)
        entry_fields = {f.name for f in fields(ModelEntry)}  # ignora claves desconocidas
        models = []
        for m in data.pop("models", []) or []:
            if not isinstance(m, dict):
                continue
            if m.get("rpc") is True and "placement" not in m:  # formato anterior: rpc: true
                m = {**m, "placement": "split"}
            models.append(ModelEntry(**{k: v for k, v in m.items() if k in entry_fields}))
        worker_fields = {f.name for f in fields(RpcWorker)}
        workers = [RpcWorker(**{k: v for k, v in w.items() if k in worker_fields})
                   for w in data.pop("rpc_workers", []) or [] if isinstance(w, dict) and "host" in w]
        known = {f.name for f in fields(cls)}
        cfg = cls(**{k: v for k, v in data.items() if k in known})
        cfg.models = models
        cfg.rpc_workers = workers
        if not isinstance(cfg.roles, dict):
            cfg.roles = {}
        if not isinstance(cfg.always_allow, list):
            cfg.always_allow = []
        for role in ROLES:
            cfg.roles.setdefault(role, "")
        return cfg

    def save(self, path: Path = CONFIG_FILE) -> None:
        text = yaml.safe_dump(self.to_dict(), allow_unicode=True, sort_keys=False)
        atomic_write(path, text.encode("utf-8"))


def default_config() -> AppConfig:
    return AppConfig(
        roles={"main": "qwen2.5-coder-14b", "fast": "qwen2.5-coder-3b", "draft": ""},
        models=[
            ModelEntry("qwen2.5-coder-14b", "Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf", ctx=24576),
            ModelEntry("qwen2.5-coder-3b", "Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf", ctx=16384),
        ],
    )


# Aviso de la última carga (p. ej. config rota que se apartó): la GUI lo muestra en Configuración.
load_warning: str = ""


def load_config(path: Path = CONFIG_FILE) -> AppConfig:
    """Nunca impide arrancar: si el archivo está roto se aparta como .bak y se usan valores por
    defecto, dejando la explicación en `load_warning`."""
    global load_warning
    load_warning = ""
    if not path.exists():
        cfg = default_config()
        cfg.save(path)
        return cfg
    try:
        with open(path, encoding="utf-8") as f:
            cfg = AppConfig.from_dict(yaml.safe_load(f) or {})
        fixes = cfg.sanitize()
        if fixes:
            load_warning = f"Corregí valores no válidos de {path.name}: " + "; ".join(fixes) + "."
        return cfg
    except (yaml.YAMLError, TypeError, ValueError, AttributeError, UnicodeDecodeError) as e:
        backup = path.with_name(f"{path.name}.bak-{time.strftime('%Y%m%d-%H%M%S')}")
        try:
            os.replace(path, backup)
            where = f"Guardé el archivo roto como {backup.name}"
        except OSError:
            where = "No pude apartar el archivo roto"
        load_warning = (f"{path.name} no se pudo leer ({type(e).__name__}: {e}). {where} y cargué "
                        "la configuración por defecto.")
        cfg = default_config()
        try:
            cfg.save(path)
        except OSError:
            pass
        return cfg


def load_catalog(path: Path = CATALOG_FILE) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return (yaml.safe_load(f) or {}).get("models", [])


def name_from_file(filename: str) -> str:
    """'Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf' -> 'qwen2.5-coder-7b-instruct-q4_k_m'."""
    stem = Path(filename).stem.lower()
    return re.sub(r"[^a-z0-9._-]+", "-", stem).strip("-")
