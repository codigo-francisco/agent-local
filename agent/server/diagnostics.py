"""Autochequeo: comprueba, en orden, todo lo que suele romper la app y dice cómo arreglarlo.
También empaqueta logs y configuración (sin secretos) en un .zip para pedir ayuda."""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

from .. import config
from ..config import CONFIG_FILE, GENERATED_DIR, MODELS_DIR, AppConfig
from . import rpc, swapconfig, vram
from .manager import ServerManager, port_in_use


@dataclass
class Check:
    name: str
    status: str  # ok | warn | fail
    detail: str = ""
    fix: str = ""


def _ok(name: str, detail: str = "") -> Check:
    return Check(name, "ok", detail)


async def run_checks(cfg: AppConfig, catalog: list[dict], manager: ServerManager, llm=None,
                     mcp=None) -> list[Check]:
    checks: list[Check] = []
    add = checks.append

    # Configuración
    if config.load_warning:
        add(Check("Configuración", "warn", config.load_warning,
                  "Revisa los valores en «Configuración» y guarda."))
    else:
        add(_ok("Configuración", f"{CONFIG_FILE.name} se leyó bien."))

    # Binarios
    for name in ("llama-swap", "llama-server"):
        exe = swapconfig.find_binary(name)
        add(_ok(name, str(exe)) if exe else
            Check(name, "fail", "No está en bin/ ni en el PATH.", "Descárgalo en «Requisitos»."))

    # Carpeta del proyecto
    ws = Path(cfg.workspace)
    if not ws.is_dir():
        add(Check("Carpeta del proyecto", "fail", f"No existe: {ws}", "Elige otra en el chat."))
    else:
        try:
            with tempfile.NamedTemporaryFile(dir=ws, prefix=".agent-check-"):
                pass
            add(_ok("Carpeta del proyecto", f"{ws} (con permiso de escritura)"))
        except OSError as e:
            add(Check("Carpeta del proyecto", "fail", f"No puedo escribir en {ws}: {e}",
                      "Elige una carpeta tuya o revisa los permisos."))
        if ws.resolve() == Path.home().resolve():
            add(Check("Alcance del proyecto", "warn",
                      "La carpeta del proyecto es todo tu perfil de usuario.",
                      "Elige la carpeta concreta del proyecto: el agente verá y podrá editar menos."))

    # Modelos
    main = cfg.model(cfg.roles.get("main", ""))
    if main is None:
        add(Check("Modelo principal", "fail", "El rol «main» no tiene modelo.", "Asígnalo en «Modelos»."))
    for role in ("main", "fast"):
        entry = cfg.model(cfg.roles.get(role, ""))
        if entry is None:
            continue
        if entry.path.is_file():
            add(_ok(f"Archivo de «{role}»", f"{entry.path.name} ({entry.path.stat().st_size / 1e9:.1f} GB)"))
        else:
            add(Check(f"Archivo de «{role}»", "fail", f"Falta {entry.path}",
                      "Descárgalo en «Modelos» o corrige el nombre del archivo."))

    # VRAM
    gpu = await asyncio.to_thread(vram.gpu_info)  # nvidia-smi tarda: no bloquear la interfaz
    plan = vram.plan_usage(cfg, catalog, gpu)
    if plan.fits is None:
        add(Check("VRAM", "warn", "No detecté una GPU NVIDIA (nvidia-smi).",
                  "Sin GPU funcionará en CPU, mucho más lento."))
    elif plan.fits:
        add(_ok("VRAM", f"Cabe: {gpu.total_gb:.1f} GB de GPU."))
    else:
        add(Check("VRAM", "fail", "La configuración no cabe en la GPU.",
                  " ".join(plan.suggestions) or "Baja el contexto o usa KV cache q8_0/q4_0."))

    # Espacio en disco
    try:
        free = shutil.disk_usage(MODELS_DIR if MODELS_DIR.exists() else GENERATED_DIR.parent).free / 1e9
        add(_ok("Espacio en disco", f"{free:.0f} GB libres") if free > 5 else
            Check("Espacio en disco", "warn", f"Solo {free:.1f} GB libres.",
                  "Libera espacio: las descargas y los logs pueden fallar."))
    except OSError:
        pass

    # Servidor
    if manager.crashed:
        add(Check("llama-swap", "fail", manager.crashed, "Mira el registro en «Servidor» y arráncalo."))
    info = await manager.status(cfg)
    if info["reachable"]:
        add(_ok("Servidor de modelos", f"Responde en {cfg.endpoint} ({len(info['models'])} modelos)."))
        if llm is not None and main is not None and main.name in info["models"]:
            n = await llm.tokenize(main.name, "hola")
            add(_ok("Conteo exacto de tokens", "/tokenize disponible") if n else
                Check("Conteo exacto de tokens", "warn", "El modelo no está cargado o no expone /tokenize.",
                      "Se estima el contexto; es normal hasta el primer mensaje."))
    elif not manager.running and port_in_use(cfg.port):
        add(Check("Servidor de modelos", "fail",
                  f"El puerto {cfg.port} está ocupado pero no responde como llama-swap.",
                  "Cierra el programa que lo usa o cambia el puerto en «Configuración»."))
    else:
        add(Check("Servidor de modelos", "warn", "No está arrancado.", "Arráncalo en «Servidor»."))

    # PCs remotas (RPC)
    for w in cfg.rpc_workers:
        name = f"PC remota {w.endpoint}"
        if not w.enabled:
            add(Check(name, "warn", "Desactivada en «Configuración»."))
        elif manager.running:
            # Con el servidor en marcha puede estar sirviendo un modelo (atiende a un cliente a la
            # vez): basta con saber que escucha.
            ok = await asyncio.to_thread(rpc.reachable, w.host, w.port)
            add(_ok(name, "Responde.") if ok else
                Check(name, "fail", "No responde.", "Abre start-worker.bat en esa PC y revisa IP y firewall."))
        else:
            res = await asyncio.to_thread(rpc.probe, w)
            add(_ok(name, res.summary()) if res.ok else
                Check(name, "fail", res.error, "Sigue el LEEME.txt del paquete de la otra PC."))

    # MCP
    if mcp is not None:
        for s in mcp.summary():
            if s["status"] == "conectado":
                add(_ok(f"MCP {s['name']}", f"{len(s['tools'])} herramientas"))
            elif s["status"] == "desactivado":
                add(Check(f"MCP {s['name']}", "warn", "Desactivado en mcp.json."))
            elif s["status"] == "conectando":  # arranque en curso (puede tardar hasta 45 s)
                add(Check(f"MCP {s['name']}", "warn", "Aún conectando.",
                          "Vuelve a comprobar en unos segundos."))
            else:
                add(Check(f"MCP {s['name']}", "fail", (s["error"] or s["status"])[:500],
                          "Revisa mcp.json en «Configuración» y pulsa «Guardar y reconectar»."))
        if mcp.load_error:
            add(Check("mcp.json", "fail", mcp.load_error, "Corrige el JSON en «Configuración»."))
    return checks


def _redact(obj):
    """Quita valores de cabeceras y variables de entorno (tokens, claves)."""
    if isinstance(obj, dict):
        return {k: ({h: "***" for h in v} if k in ("headers", "env") and isinstance(v, dict)
                    else _redact(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_redact(v) for v in obj]
    return obj


def export_bundle(checks: list[Check], dest_dir: Path = GENERATED_DIR) -> Path:
    """Zip con el informe, logs y configuración sin secretos."""
    dest = dest_dir / f"diagnostico-{time.strftime('%Y%m%d-%H%M%S')}.zip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    report = "\n".join(f"[{c.status.upper():4}] {c.name}: {c.detail}" + (f"\n       → {c.fix}" if c.fix else "")
                       for c in checks)
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("informe.txt", report)
        for folder in (GENERATED_DIR / "logs", GENERATED_DIR / "mcp-logs"):
            for f in sorted(folder.glob("*"))[:20] if folder.is_dir() else []:
                if f.is_file():
                    z.write(f, f"{folder.name}/{f.name}")
        for f in (CONFIG_FILE, swapconfig.SWAP_FILE):
            if f.is_file():
                z.write(f, f.name)
        mcp_file = CONFIG_FILE.parent / "mcp.json"
        if mcp_file.is_file():
            try:
                data = _redact(json.loads(mcp_file.read_text(encoding="utf-8")))
                z.writestr("mcp.json", json.dumps(data, ensure_ascii=False, indent=2))
            except (OSError, ValueError):
                z.writestr("mcp.json", "(no se pudo leer)")
    return dest
