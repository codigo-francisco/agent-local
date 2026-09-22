"""Arranca y para llama-swap, captura su registro y consulta su estado."""

from __future__ import annotations

import os
import socket
import subprocess
import threading
from collections import deque

import httpx

from ..config import AppConfig
from ..core.tools import kill_tree
from . import swapconfig


class ServerError(Exception):
    pass


def port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


class ServerManager:
    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self._lines: deque[str] = deque(maxlen=3000)
        self._count = 0  # líneas totales recibidas (para lecturas incrementales)
        self._lock = threading.Lock()

    # --- registro --------------------------------------------------------
    def _add(self, line: str) -> None:
        with self._lock:
            self._lines.append(line)
            self._count += 1

    def lines_since(self, index: int) -> tuple[list[str], int]:
        with self._lock:
            new = min(self._count - index, len(self._lines))
            return (list(self._lines)[-new:] if new > 0 else []), self._count

    def log_tail(self, chars: int = 4000) -> str:
        with self._lock:
            return "\n".join(self._lines)[-chars:]

    # --- ciclo de vida ---------------------------------------------------
    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, cfg: AppConfig) -> list[str]:
        if self.running:
            return []
        exe = swapconfig.find_binary("llama-swap")
        if exe is None:
            raise ServerError("No encontré llama-swap. Descárgalo a la carpeta bin/ "
                              "(página «Requisitos»).")
        if port_in_use(cfg.port):
            raise ServerError(f"El puerto {cfg.port} ya está en uso. Puede que haya otro llama-swap "
                              "o servidor abierto: ciérralo o cambia el puerto en Configuración.")
        warnings = swapconfig.write(cfg)
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self._add(f"$ {exe} --config {swapconfig.SWAP_FILE} --listen 127.0.0.1:{cfg.port}")
        self.proc = subprocess.Popen(
            [str(exe), "--config", str(swapconfig.SWAP_FILE), "--listen", f"127.0.0.1:{cfg.port}"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            creationflags=flags,
        )
        threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start()
        return warnings

    def _reader(self, proc: subprocess.Popen) -> None:
        assert proc.stdout is not None
        for raw in iter(proc.stdout.readline, b""):
            self._add(raw.decode("utf-8", errors="replace").rstrip())
        code = proc.wait()
        self._add(f"[llama-swap terminó con código {code}]")

    def stop(self) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is None:
            kill_tree(self.proc.pid)  # también mata los llama-server hijos (liberan la VRAM)
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    # --- estado ----------------------------------------------------------
    @staticmethod
    def _root(cfg: AppConfig) -> str:
        e = cfg.endpoint.rstrip("/")
        return e[:-3] if e.endswith("/v1") else e

    async def status(self, cfg: AppConfig) -> dict:
        """{'reachable': bool, 'models': [...], 'running': [{'model', 'state'}]}"""
        root = self._root(cfg)
        info: dict = {"reachable": False, "models": [], "running": []}
        async with httpx.AsyncClient(timeout=2.0) as client:
            try:
                r = await client.get(f"{root}/v1/models")
                info["reachable"] = r.status_code == 200
                info["models"] = [m["id"] for m in r.json().get("data", [])]
            except (httpx.HTTPError, ValueError, KeyError):
                return info
            try:
                r = await client.get(f"{root}/running")
                if r.status_code == 200:
                    info["running"] = r.json().get("running", [])
            except (httpx.HTTPError, ValueError):
                pass
        return info

    async def unload_all(self, cfg: AppConfig) -> bool:
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                r = await client.get(f"{self._root(cfg)}/unload")
                return r.status_code == 200
            except httpx.HTTPError:
                return False

    async def preload(self, cfg: AppConfig, model: str) -> tuple[bool, str]:
        """Fuerza la carga pidiendo /health del modelo a través de llama-swap."""
        async with httpx.AsyncClient(timeout=httpx.Timeout(900.0, connect=3.0)) as client:
            try:
                r = await client.get(f"{self._root(cfg)}/upstream/{model}/health")
                return r.status_code == 200, r.text[:500]
            except httpx.HTTPError as e:
                return False, str(e)
