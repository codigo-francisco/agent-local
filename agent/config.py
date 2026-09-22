"""Configuración de la aplicación: rutas del proyecto y `config/models.yaml`."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
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


@dataclass
class ModelEntry:
    """Un modelo que llama-swap puede servir."""

    name: str  # identificador en la API (campo `model`)
    file: str  # nombre del .gguf dentro de models/
    ctx: int = 16384
    kv_type: str = "q8_0"
    gpu_layers: int = 99
    extra_args: str = ""

    @property
    def path(self) -> Path:
        p = Path(self.file)
        return p if p.is_absolute() else MODELS_DIR / p


@dataclass
class AppConfig:
    port: int = 8080
    endpoint: str = "http://127.0.0.1:8080/v1"
    workspace: str = str(Path.home())
    roles: dict[str, str] = field(default_factory=lambda: {"main": "", "fast": "", "draft": ""})
    models: list[ModelEntry] = field(default_factory=list)
    keep_loaded: bool = True  # main y fast cargados a la vez (grupo de llama-swap)
    flash_attn: str = "on"
    max_steps: int = 25
    max_output_tokens: int = 4096
    max_tool_output: int = 12000  # caracteres
    command_timeout: int = 120  # segundos
    confirm: str = "ask"  # "ask" | "auto"

    # --- consultas -------------------------------------------------------
    def model(self, name: str) -> ModelEntry | None:
        return next((m for m in self.models if m.name == name), None)

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

    # --- persistencia ----------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "AppConfig":
        data = dict(data or {})
        models = [ModelEntry(**m) for m in data.pop("models", []) or []]
        known = {f for f in cls.__dataclass_fields__}  # ignora claves desconocidas
        cfg = cls(**{k: v for k, v in data.items() if k in known})
        cfg.models = models
        for role in ROLES:
            cfg.roles.setdefault(role, "")
        return cfg

    def save(self, path: Path = CONFIG_FILE) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, allow_unicode=True, sort_keys=False)


def default_config() -> AppConfig:
    return AppConfig(
        roles={"main": "qwen2.5-coder-14b", "fast": "qwen2.5-coder-3b", "draft": ""},
        models=[
            ModelEntry("qwen2.5-coder-14b", "Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf", ctx=24576),
            ModelEntry("qwen2.5-coder-3b", "Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf", ctx=16384),
        ],
    )


def load_config(path: Path = CONFIG_FILE) -> AppConfig:
    if not path.exists():
        cfg = default_config()
        cfg.save(path)
        return cfg
    with open(path, encoding="utf-8") as f:
        return AppConfig.from_dict(yaml.safe_load(f) or {})


def load_catalog(path: Path = CATALOG_FILE) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return (yaml.safe_load(f) or {}).get("models", [])


def name_from_file(filename: str) -> str:
    """'Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf' -> 'qwen2.5-coder-7b-instruct-q4_k_m'."""
    stem = Path(filename).stem.lower()
    return re.sub(r"[^a-z0-9._-]+", "-", stem).strip("-")
