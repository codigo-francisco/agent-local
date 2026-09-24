"""PCs remotas: otra PC presta su GPU con `ggml-rpc-server` (backend RPC de llama.cpp).

El llama-server de esta PC la usa como un dispositivo más (`--rpc host:puerto`) y reparte las capas
entre ambas GPUs. Los pesos viajan por la red: la otra PC no necesita el modelo ni esta app, solo los
binarios de la MISMA versión de llama.cpp, que `build_worker_package` empaqueta en un .zip.
"""

from __future__ import annotations

import os
import re
import socket
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from ..config import GENERATED_DIR, RPC_DEFAULT_PORT, RpcWorker
from . import swapconfig
from .requirements import _cuda_dll_deps

PACKAGE_FILE = GENERATED_DIR / "rpc-worker.zip"
RPC_BINARIES = ("ggml-rpc-server", "rpc-server")  # nombre actual y el de builds antiguas
MIB = 1024 ** 2
_DEVICE_RE = re.compile(r"^\s*(\S+?):\s*(.*?)\s*\((\d+) MiB,\s*(\d+) MiB free\)\s*$")
_FLAGS = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


@dataclass
class Device:
    name: str
    description: str
    total_gb: float
    free_gb: float


@dataclass
class ProbeResult:
    ok: bool
    devices: list[Device]
    error: str = ""

    @property
    def total_gb(self) -> float:
        return sum(d.total_gb for d in self.devices)

    @property
    def free_gb(self) -> float:
        return sum(d.free_gb for d in self.devices)

    def summary(self) -> str:
        if not self.ok:
            return self.error
        names = ", ".join(d.description or d.name for d in self.devices)
        return f"{names}: {self.total_gb:.1f} GB de VRAM ({self.free_gb:.1f} GB libres)"


def parse_devices(output: str) -> list[Device]:
    """Líneas de `llama-server --list-devices`: «CUDA0: NVIDIA … (16375 MiB, 15111 MiB free)»."""
    devices = []
    for line in output.splitlines():
        m = _DEVICE_RE.match(line)
        if m:
            devices.append(Device(m.group(1), m.group(2), int(m.group(3)) * MIB / 1024 ** 3,
                                  int(m.group(4)) * MIB / 1024 ** 3))
    return devices


def reachable(host: str, port: int, timeout: float = 2.0) -> bool:
    """¿Acepta conexiones? No valida que sea un rpc-server: solo que el puerto está abierto."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _list_devices(exe: Path, rpc: str | None, timeout: float) -> tuple[str, int]:
    args = [str(exe)] + (["--rpc", rpc] if rpc else []) + ["--list-devices"]
    p = subprocess.run(args, capture_output=True, text=True, timeout=timeout, errors="replace",
                       stdin=subprocess.DEVNULL, creationflags=_FLAGS)
    return p.stdout + p.stderr, p.returncode


@lru_cache(maxsize=4)
def _local_device_names(exe: Path) -> frozenset[str]:
    try:
        out, _ = _list_devices(exe, None, 30)
    except (OSError, subprocess.SubprocessError):
        return frozenset()
    return frozenset(d.name for d in parse_devices(out))


def probe(worker: RpcWorker, timeout: float = 20.0) -> ProbeResult:
    """Conecta con la PC remota a través de llama-server y lee sus GPUs y su VRAM."""
    where = worker.endpoint
    if not reachable(worker.host, worker.port, timeout=3.0):
        return ProbeResult(False, [], (
            f"{where} no responde. Comprueba que en la otra PC está abierto start-worker.bat, que la "
            "IP es la correcta (ipconfig) y que ejecutaste permitir-firewall.bat como administrador."))
    exe = swapconfig.find_binary("llama-server")
    if exe is None:
        return ProbeResult(False, [], "No encontré llama-server en bin/ (página «Requisitos»).")
    try:
        out, code = _list_devices(exe, where, timeout)
    except subprocess.TimeoutExpired:
        return ProbeResult(False, [], (
            f"{where} acepta la conexión pero no contesta. El trabajador atiende a un cliente a la vez: "
            "puede estar ocupado sirviendo un modelo (para el servidor y vuelve a probar)."))
    except OSError as e:
        return ProbeResult(False, [], f"No pude ejecutar llama-server: {e}")
    local = _local_device_names(exe)
    remote = [d for d in parse_devices(out) if d.name not in local]
    if remote:
        return ProbeResult(True, remote)
    low = out.lower()
    if ("version" in low and "mismatch" in low) or "protocol" in low:
        error = (f"{where} tiene otra versión de llama.cpp. Genera de nuevo el paquete en esta PC y "
                 "reemplaza la carpeta de la otra PC.")
    elif "failed to connect" in low:
        error = f"llama-server no pudo conectar con {where} (¿firewall?)."
    else:
        last = next((l for l in reversed(out.strip().splitlines()) if l.strip()), "")
        error = f"{where} no ofrece ninguna GPU (código {code}). {last[:200]}"
    return ProbeResult(False, [], error)


def refresh_workers(workers: list[RpcWorker], server_running: bool,
                    say=lambda _msg: None) -> list[str]:
    """Vuelve a medir la VRAM de las PCs remotas activas (actualiza vram_gb y devices). Con el
    servidor en marcha no se sondea: el trabajador atiende a un cliente a la vez y se colgaría.
    Devuelve avisos legibles."""
    warnings = []
    active = [w for w in workers if w.enabled and w.host]
    if active and server_running:
        say("Servidor en marcha: uso la VRAM remota medida la última vez…")
        return warnings
    for w in active:
        say(f"Consultando la PC remota {w.name or w.endpoint}…")
        res = probe(w)
        if res.ok:
            w.vram_gb, w.devices = round(res.total_gb, 1), len(res.devices)
        else:
            w.vram_gb = 0.0
            warnings.append(f"{w.endpoint}: {res.error} La dejo fuera del cálculo.")
    return warnings


def local_ip(towards: str | None = None) -> str:
    """IP de esta PC en la red local (la que usaría para llegar a `towards`). No envía nada."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect((towards or "192.168.0.1", 9))
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"


# --- paquete para la otra PC -----------------------------------------------------

def _start_bat(port: int) -> str:
    return (
        "@echo off\r\n"
        "rem Trabajador RPC de agent-local: presta la GPU de esta PC a la PC principal.\r\n"
        'cd /d "%~dp0"\r\n'
        "title agent-local - trabajador RPC (puerto " + str(port) + ")\r\n"
        "echo Escuchando en el puerto " + str(port) + ". Deja esta ventana abierta mientras uses el agente.\r\n"
        "echo Para parar: cierra la ventana o pulsa Ctrl+C.\r\n"
        "echo.\r\n"
        # -c: caché local de los pesos; la siguiente carga del mismo modelo no los vuelve a pedir por la red.
        f"ggml-rpc-server.exe -H 0.0.0.0 -p {port} -c\r\n"
        "echo.\r\n"
        "echo El trabajador se ha cerrado.\r\n"
        "pause\r\n"
    )


def _firewall_bat(port: int, main_ip: str) -> str:
    remote = f" remoteip={main_ip}" if main_ip and not main_ip.startswith("127.") else ""
    return (
        "@echo off\r\n"
        "rem Ejecutar UNA vez como administrador (clic derecho > Ejecutar como administrador).\r\n"
        "rem Abre el puerto solo en redes privadas y solo para la PC principal.\r\n"
        "net session >nul 2>&1\r\n"
        "if errorlevel 1 (\r\n"
        "  echo Hay que ejecutarlo como administrador: clic derecho ^> Ejecutar como administrador.\r\n"
        "  pause\r\n"
        "  exit /b 1\r\n"
        ")\r\n"
        'netsh advfirewall firewall delete rule name="agent-local RPC" >nul 2>&1\r\n'
        f'netsh advfirewall firewall add rule name="agent-local RPC" dir=in action=allow '
        f"protocol=TCP localport={port}{remote} profile=private\r\n"
        "echo.\r\n"
        "echo Listo. Comprueba que tu red esta marcada como Privada (Configuracion ^> Red e Internet).\r\n"
        "pause\r\n"
    )


def _readme(port: int, main_ip: str) -> str:
    return f"""TRABAJADOR RPC DE AGENT-LOCAL
=============================

Esta carpeta convierte esta PC en una "GPU remota" para la PC principal ({main_ip}).
No hace falta instalar Python, ni la app, ni descargar modelos: los pesos llegan por la red.

1. Actualiza el driver de NVIDIA.
2. Si al abrir start-worker.bat dice que falta MSVCP140.dll o VCRUNTIME140.dll, instala
   "Microsoft Visual C++ Redistributable" (x64):  winget install Microsoft.VCRedist.2015+.x64
3. Marca la red como Privada y ejecuta UNA vez permitir-firewall.bat como administrador.
4. Doble clic en start-worker.bat y deja la ventana abierta (puerto {port}).
5. Averigua la IP de esta PC con  ipconfig  (Dirección IPv4) y escríbela en la PC principal:
   Configuración > PCs remotas > Añadir PC > Probar.

SEGURIDAD: el protocolo RPC de llama.cpp no tiene contraseña ni cifrado. Úsalo solo en tu red de
casa. El script del firewall solo deja entrar a la PC principal; no abras este puerto en el router.

Estos binarios son de la misma versión de llama.cpp que la PC principal. Si actualizas llama.cpp
allí, genera otra vez el paquete y reemplaza esta carpeta (versiones distintas no se entienden).
"""


def _package_files(bin_dir: Path, exe: Path) -> list[Path]:
    """El servidor RPC, las DLL de ggml y solo las DLL de CUDA de la versión que usa ggml-cuda."""
    files = {exe}
    files.update(p for p in bin_dir.glob("ggml*.dll"))
    files.update(p for p in bin_dir.glob("libomp*.dll"))
    files.update(p for p in bin_dir.glob("LICENSE*"))
    cuda = next(iter(bin_dir.glob("ggml-cuda*.dll")), None)
    majors = {m.group(1) for d in (_cuda_dll_deps(cuda) if cuda else [])
              if (m := re.search(r"64_(\d+)\.dll", d))}
    for p in bin_dir.glob("*.dll"):
        m = re.match(r"(?:cublas|cublasLt|cudart)64_(\d+)\.dll$", p.name, re.I)
        if m and (not majors or m.group(1) in majors):
            files.add(p)
    return sorted(files)


def build_worker_package(main_ip: str, dest: Path = PACKAGE_FILE,
                         port: int = RPC_DEFAULT_PORT) -> Path:
    """Crea el .zip con todo lo que necesita la otra PC. Tarda: las DLL de CUDA pesan ~0,5 GB."""
    exe = next((e for e in (swapconfig.find_binary(n) for n in RPC_BINARIES) if e), None)
    if exe is None:
        raise FileNotFoundError("No encontré ggml-rpc-server.exe en bin/. Viene en el mismo zip de "
                                "llama.cpp que llama-server: descárgalo de nuevo (página «Requisitos»).")
    bin_dir = exe.parent
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{dest.name}.", suffix=".tmp", dir=dest.parent)
    os.close(fd)
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
            for f in _package_files(bin_dir, exe):
                z.write(f, f"agent-rpc/{'ggml-rpc-server.exe' if f == exe else f.name}")
            z.writestr("agent-rpc/start-worker.bat", _start_bat(port))
            z.writestr("agent-rpc/permitir-firewall.bat", _firewall_bat(port, main_ip))
            z.writestr("agent-rpc/LEEME.txt", _readme(port, main_ip).replace("\n", "\r\n"))
        os.replace(tmp, dest)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return dest
