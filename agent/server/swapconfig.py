"""Genera generated/llama-swap.yaml a partir de config/models.yaml (nunca se edita a mano)."""

from __future__ import annotations

import re
import shlex
import shutil
from pathlib import Path

import yaml

from ..config import BIN_DIR, GENERATED_DIR, AppConfig, ModelEntry, atomic_write
from . import vram

SWAP_FILE = GENERATED_DIR / "llama-swap.yaml"


def find_binary(name: str) -> Path | None:
    """Busca primero en bin/ (también en subcarpetas del zip) y luego en el PATH."""
    exe = f"{name}.exe"
    for candidate in (BIN_DIR / exe, BIN_DIR / name):
        if candidate.is_file():
            return candidate
    if BIN_DIR.is_dir():
        found = next(iter(sorted(BIN_DIR.rglob(exe))), None)
        if found:
            return found
    which = shutil.which(name)
    return Path(which) if which else None


def _q(p: Path | str) -> str:
    s = str(p).replace("\\", "/")
    return f'"{s}"' if " " in s else s


draft_compatible = vram.draft_compatible


def uses_rpc(cfg: AppConfig, entry: ModelEntry) -> bool:
    """El modelo se reparte con PCs remotas (salvo que ya lleve su propio --rpc en los args extra)."""
    return bool(entry.rpc and cfg.rpc_endpoints()
                and not re.search(r"(^|\s)--rpc(\s|=|$)", entry.extra_args))


def server_command(cfg: AppConfig, entry: ModelEntry, server_exe: Path | str,
                   fit_reserve_mib: int = 0) -> str:
    kv = entry.kv_type
    fa = cfg.flash_attn
    if kv != "f16" and fa == "off":
        kv = "f16"  # la KV cuantizada requiere flash attention
    args = [
        _q(server_exe), "--host", "127.0.0.1", "--port", "${PORT}",
        "-m", _q(entry.path), "--jinja",
        "-c", str(entry.ctx), "-np", "1", "-fa", fa,
    ]
    if entry.gpu_layers >= 0:
        args += ["-ngl", str(entry.gpu_layers)]
    else:
        # Automático: sin -ngl, --fit (activo por defecto) reparte GPU/RAM; en MoE manda expertos a
        # la RAM. El margen deja sitio a los modelos que se cargan a la vez (p. ej. «fast»).
        args += ["--fit-target", str(1024 + fit_reserve_mib)]
    if kv != "f16":
        args += ["-ctk", kv, "-ctv", kv]
    draft = cfg.model(cfg.roles.get("draft", ""))
    if (draft and entry.name == cfg.roles.get("main") and draft.path.is_file()
            and draft_compatible(entry, draft)):
        args += ["-md", _q(draft.path), "-ngld", "99"]
    if uses_rpc(cfg, entry):
        # Las GPUs de las PCs remotas se suman a la local y llama.cpp reparte las capas entre todas;
        # con «remote» solo se usan las remotas y la GPU local queda libre para otro modelo.
        args += ["--rpc", ",".join(cfg.rpc_endpoints())]
        if entry.placement == "remote" and not re.search(r"(^|\s)(-dev|--device)(\s|=|$)", entry.extra_args):
            args += ["--device", ",".join(cfg.rpc_device_names())]
    if entry.extra_args.strip():
        args += shlex.split(entry.extra_args, posix=True)
    return " ".join(args)


def build(cfg: AppConfig) -> tuple[dict, list[str]]:
    """Devuelve (config de llama-swap, avisos)."""
    warnings: list[str] = []
    server = find_binary("llama-server")
    if server is None:
        warnings.append("No encontré llama-server; descárgalo a bin/ (ver «Requisitos»).")
        server = BIN_DIR / "llama-server.exe"
    draft_name = cfg.roles.get("draft")
    models: dict[str, dict] = {}
    for entry in cfg.models:
        if entry.name == draft_name and entry.name not in (cfg.roles.get("main"), cfg.roles.get("fast")):
            continue  # el borrador no se sirve solo: va dentro del proceso de main
        if not entry.path.is_file():
            warnings.append(f"«{entry.name}»: falta el archivo {entry.path.name}; no se servirá.")
            continue
        if entry.kv_type != "f16" and cfg.flash_attn == "off":
            warnings.append(f"«{entry.name}»: la KV cache {entry.kv_type} requiere flash attention; "
                            "se usará f16.")
        draft = cfg.model(draft_name or "")
        if (entry.name == cfg.roles.get("main") and draft and draft.path.is_file()
                and not draft_compatible(entry, draft)):
            warnings.append(f"El borrador «{draft.name}» no es de la misma familia que «{entry.name}» "
                            "(vocabulario distinto): se desactiva la decodificación especulativa.")
        if entry.rpc and not cfg.rpc_endpoints():
            warnings.append(f"«{entry.name}» está marcado para usar la PC remota, pero no hay ninguna "
                            "activa: se servirá solo en esta PC.")
        reserve = 0
        fast = cfg.model(cfg.roles.get("fast", ""))
        if (cfg.keep_loaded and entry.name == cfg.roles.get("main") and fast
                and fast.name != entry.name and fast.path.is_file()
                and not (fast.placement == "remote" and uses_rpc(cfg, fast))):  # fast no ocupa esta GPU
            reserve = int(vram.estimate_entry(fast, []).total_gb * 1024)
        models[entry.name] = {"cmd": server_command(cfg, entry, server, reserve),
                              "proxy": "http://127.0.0.1:${PORT}"}
    # logToStdout=both: la salida de llama-server (p. ej. "out of memory") llega a nuestro registro.
    # Con PC remota, la primera carga manda los pesos por la red (≈3 min por cada 20 GB a 1 Gbps).
    remote = any(uses_rpc(cfg, e) for e in cfg.models if e.name in models)
    data: dict = {"healthCheckTimeout": 900 if remote else 600, "logLevel": "info",
                  "logToStdout": "both", "models": models}
    members = [m for m in (cfg.roles.get("main"), cfg.roles.get("fast")) if m and m in models]
    members = list(dict.fromkeys(members))
    if cfg.keep_loaded and len(members) > 1:
        data["groups"] = {"agent": {"swap": False, "exclusive": True, "members": members}}
    if not models:
        warnings.append("No hay ningún modelo descargado que servir.")
    return data, warnings


def write(cfg: AppConfig, path: Path = SWAP_FILE) -> list[str]:
    data, warnings = build(cfg)
    text = ("# Generado automáticamente por agent-local a partir de config/models.yaml.\n"
            "# No lo edites: se sobrescribe al guardar la configuración.\n"
            + yaml.safe_dump(data, allow_unicode=True, sort_keys=False, width=1000))
    atomic_write(path, text.encode("utf-8"))
    return warnings
