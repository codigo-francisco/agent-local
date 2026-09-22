"""Comprobaciones del sistema. Cada una explica por qué hace falta y cómo solucionarla."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import httpx

from ..config import MODELS_DIR, ROOT, AppConfig
from . import swapconfig
from .vram import gpu_info

LLAMA_CPP_RELEASES = "https://github.com/ggml-org/llama.cpp/releases"
LLAMA_SWAP_RELEASES = "https://github.com/mostlygeek/llama-swap/releases"
PYTHON_DOWNLOAD = "https://www.python.org/downloads/windows/"
NVIDIA_DRIVERS = "https://www.nvidia.com/Download/index.aspx"


@dataclass
class Check:
    key: str
    title: str
    status: str  # ok | warn | error
    detail: str
    why: str
    fix: str = ""
    link: str = ""


def _version(exe: str) -> str:
    try:
        p = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=15,
                           errors="replace")
        text = (p.stdout + p.stderr).strip().splitlines()
        return next((l for l in text if "version" in l.lower()), text[-1] if text else "")
    except (OSError, subprocess.SubprocessError):
        return ""


def _cuda_devices(exe: str) -> list[str]:
    """Dispositivos CUDA que ve llama.cpp (`--list-devices`), p. ej. ['CUDA0: NVIDIA GeForce ...']."""
    try:
        p = subprocess.run([exe, "--list-devices"], capture_output=True, text=True, timeout=30,
                           errors="replace")
    except (OSError, subprocess.SubprocessError):
        return []
    return [l.strip() for l in (p.stdout + p.stderr).splitlines() if l.strip().startswith("CUDA")]


def _cuda_dll_deps(dll: Path) -> list[str]:
    """DLLs de CUDA que referencia ggml-cuda.dll (cublas64_13.dll, cudart64_12.dll...)."""
    try:
        data = dll.read_bytes()
    except OSError:
        return []
    return sorted({m.decode() for m in re.findall(rb"(?:cublas|cublasLt|cudart)64_\d+\.dll", data)})


def check_python() -> Check:
    v = sys.version_info
    ok = v >= (3, 11)
    return Check("python", "Python 3.11 o superior", "ok" if ok else "error",
                 f"Python {v.major}.{v.minor}.{v.micro}",
                 "La aplicación y el agente están escritos en Python.",
                 "" if ok else "Instala Python 3.12 (winget install Python.Python.3.12).",
                 "" if ok else PYTHON_DOWNLOAD)


def check_gpu() -> Check:
    gpu = gpu_info()
    why = ("Los modelos corren mucho más rápido en la GPU. La VRAM (memoria de la tarjeta) limita "
           "qué modelos y cuánto contexto puedes usar.")
    if gpu is None:
        return Check("gpu", "GPU NVIDIA y driver", "warn", "No se encontró nvidia-smi.", why,
                     "Sin GPU NVIDIA el agente funciona en CPU, pero será muy lento. Si tienes una, "
                     "instala o actualiza el driver.", NVIDIA_DRIVERS)
    status = "ok" if gpu.total_gb >= 7.5 else "warn"
    detail = (f"{gpu.name} — {gpu.total_gb:.1f} GB de VRAM ({gpu.free_gb:.1f} GB libres), "
              f"driver {gpu.driver}")
    fix = "" if status == "ok" else "Con menos de 8 GB usa modelos de 7B o menores."
    return Check("gpu", "GPU NVIDIA y driver", status, detail, why, fix)


def check_llama_server() -> Check:
    why = ("llama-server (de llama.cpp) es el programa que carga el modelo .gguf en la GPU y lo "
           "expone como API local.")
    fix = ("En las releases de llama.cpp descarga dos zips para Windows con CUDA: "
           "«llama-…-bin-win-cuda-12.x-x64.zip» y «cudart-llama-bin-win-cuda-12.x-x64.zip». "
           "Descomprime ambos dentro de la carpeta bin/ del proyecto.")
    exe = swapconfig.find_binary("llama-server")
    if exe is None:
        return Check("llama_server", "llama.cpp (llama-server)", "error",
                     f"No está en {ROOT / 'bin'} ni en el PATH.", why, fix, LLAMA_CPP_RELEASES)
    version = _version(str(exe))
    cuda_dll = next(iter(exe.parent.glob("ggml-cuda*.dll")), None)
    if cuda_dll is None:
        return Check("llama_server", "llama.cpp (llama-server)", "warn",
                     f"{exe} {version} — parece una build sin CUDA (no hay ggml-cuda.dll).", why,
                     "Descarga la build «win-cuda» para usar la GPU; con la build CPU irá lento.",
                     LLAMA_CPP_RELEASES)
    devices = _cuda_devices(str(exe))
    if not devices:
        # ggml-cuda.dll existe pero no carga: casi siempre faltan las DLLs de su versión de CUDA.
        missing = [d for d in _cuda_dll_deps(cuda_dll) if not (exe.parent / d).exists()]
        detail = (f"llama.cpp no detecta la GPU: ggml-cuda.dll no se puede cargar"
                  + (f" porque faltan {', '.join(missing)}." if missing else "."))
        major = next((re.search(r"64_(\d+)", d).group(1) for d in missing if re.search(r"64_(\d+)", d)), None)
        fix = (f"Tu build de llama.cpp es para CUDA {major}: descarga de la MISMA release el zip "
               f"«cudart-llama-bin-win-cuda-{major}.x-x64.zip» y descomprímelo en bin/. "
               "El zip de llama y el de cudart deben ser de la misma versión de CUDA."
               if major else "Actualiza el driver de NVIDIA y comprueba que los zips de llama y "
                             "cudart son de la misma versión de CUDA.")
        return Check("llama_server", "llama.cpp (llama-server)", "error", detail, why, fix,
                     LLAMA_CPP_RELEASES)
    return Check("llama_server", "llama.cpp (llama-server)", "ok",
                 f"{version} — GPU detectada: {devices[0]}", why)


def check_llama_swap() -> Check:
    why = ("llama-swap es un proxy delante de llama-server: da una única dirección para todos los "
           "modelos y los carga o descarga de la GPU según cuál pidas.")
    exe = swapconfig.find_binary("llama-swap")
    if exe is None:
        return Check("llama_swap", "llama-swap", "error", "No encontrado.", why,
                     "Descarga «llama-swap_…_windows_amd64.zip» de sus releases y descomprímelo en bin/.",
                     LLAMA_SWAP_RELEASES)
    return Check("llama_swap", "llama-swap", "ok", f"{exe} — {_version(str(exe))}", why)


def check_models(cfg: AppConfig) -> Check:
    why = "Los modelos son archivos .gguf con los pesos de la red neuronal."
    files = sorted(MODELS_DIR.glob("*.gguf")) if MODELS_DIR.is_dir() else []
    main = cfg.model(cfg.roles.get("main", ""))
    if not files:
        return Check("models", "Modelos descargados", "error", f"No hay archivos .gguf en {MODELS_DIR}.",
                     why, "Descarga al menos uno desde la página «Modelos».")
    if main is None or not main.path.is_file():
        return Check("models", "Modelos descargados", "warn",
                     f"{len(files)} archivo(s), pero el rol «main» no tiene un modelo descargado.",
                     why, "Asigna un modelo descargado al rol «main» en la página «Modelos».")
    fast = cfg.model(cfg.roles.get("fast", ""))
    note = "" if fast and fast.path.is_file() else \
        " Sin modelo «fast» los resúmenes de contexto los hará el modelo principal."
    return Check("models", "Modelos descargados", "ok" if not note else "warn",
                 f"{len(files)} archivo(s). Principal: {main.name}.{note}", why)


def check_disk() -> Check:
    free = shutil.disk_usage(ROOT).free / 1024 ** 3
    status = "ok" if free > 20 else "warn"
    return Check("disk", "Espacio en disco", status, f"{free:.0f} GB libres en {ROOT.drive or ROOT}",
                 "Cada modelo ocupa entre 0,5 y 20 GB.",
                 "" if status == "ok" else "Libera espacio antes de descargar modelos grandes.")


def check_server(cfg: AppConfig) -> Check:
    why = "El agente habla con los modelos a través de este servidor local."
    try:
        r = httpx.get(cfg.endpoint.rstrip("/") + "/models", timeout=1.5)
        if r.status_code == 200:
            return Check("server", "Servidor de modelos", "ok", f"Respondiendo en {cfg.endpoint}", why)
    except httpx.HTTPError:
        pass
    return Check("server", "Servidor de modelos", "warn", f"No arrancado ({cfg.endpoint}).", why,
                 "Arráncalo desde la página «Servidor».")


def check_all(cfg: AppConfig) -> list[Check]:
    return [check_python(), check_gpu(), check_llama_server(), check_llama_swap(),
            check_models(cfg), check_disk(), check_server(cfg)]
