"""Estado compartido de la aplicación (una sola instancia: es una app local de un usuario)."""

from __future__ import annotations

import asyncio
import traceback
from pathlib import Path
from typing import Callable

from nicegui import Client, ui

from ..config import AppConfig, ModelEntry, load_catalog, load_config
from ..core.llm import LLMClient
from ..server import swapconfig, vram
from ..server.downloader import DownloadState
from ..server.manager import ServerManager

DEFAULT_CTX = {"main": 24576, "fast": 16384, "draft": 4096}


class AppState:
    def __init__(self) -> None:
        self.cfg: AppConfig = load_config()
        self.catalog: list[dict] = load_catalog()
        self.manager = ServerManager()
        self.downloads: dict[str, DownloadState] = {}
        self.llm = LLMClient(self.cfg.endpoint, log_provider=self.manager.log_tail)
        self.navigate: Callable[[str], None] = lambda tab: None  # lo fija app.py
        self._listeners: list[tuple[Client, Callable[[], None]]] = []

    def on_config_saved(self, callback: Callable[[], None]) -> None:
        """Avisa a `callback` al guardar la configuración, mientras su página siga abierta."""
        self._listeners.append((ui.context.client, callback))

    def _notify_config_saved(self) -> None:
        alive = []
        for client, callback in self._listeners:
            if client.is_deleted:  # página cerrada o recargada: se descarta
                continue
            alive.append((client, callback))
            try:
                with client:
                    callback()
            except Exception:  # noqa: BLE001 - un fallo en una página no debe bloquear a las demás
                traceback.print_exc()
        self._listeners = alive

    def set_workspace(self, path: str) -> str | None:
        """Cambia la carpeta del proyecto y guarda. Devuelve un error legible o None."""
        raw = (path or "").strip().strip('"')
        if not raw:
            return "Indica una carpeta."
        p = Path(raw).expanduser()
        if not p.is_dir():
            return f"No existe la carpeta: {raw}"
        new = str(p.resolve())
        if new != str(Path(self.cfg.workspace).resolve()):
            self.cfg.workspace = new
            self.save_config()
        return None

    # --- modelos y roles -------------------------------------------------
    def ensure_entry(self, name: str, file: str, role: str | None = None,
                     defaults: dict | None = None) -> ModelEntry:
        entry = self.cfg.model(name)
        if entry is None:
            entry = ModelEntry(name, file, ctx=DEFAULT_CTX.get(role or "main", 16384))
            for key, value in (defaults or {}).items():  # recomendados del catálogo
                if hasattr(entry, key):
                    setattr(entry, key, value)
            self.cfg.models.append(entry)
        return entry

    def assign_role(self, role: str, name: str) -> None:
        self.cfg.roles[role] = name

    def remove_model(self, name: str) -> None:
        self.cfg.models = [m for m in self.cfg.models if m.name != name]
        for role, value in self.cfg.roles.items():
            if value == name:
                self.cfg.roles[role] = ""

    # --- persistencia ----------------------------------------------------
    def save_config(self) -> list[str]:
        self.cfg.endpoint = self.cfg.endpoint.strip() or f"http://127.0.0.1:{self.cfg.port}/v1"
        self.cfg.save()
        warnings = swapconfig.write(self.cfg)
        old = self.llm
        self.llm = LLMClient(self.cfg.endpoint, log_provider=self.manager.log_tail)
        try:
            asyncio.get_running_loop().create_task(old.aclose())
        except RuntimeError:
            pass
        self._notify_config_saved()
        return warnings

    def ctx_hint(self, model: str, n_ctx: int) -> str | None:
        return vram.ctx_hint(self.cfg, self.catalog, model, n_ctx)

    async def shutdown(self) -> None:
        self.manager.stop()
        await self.llm.aclose()


_state: AppState | None = None


def get_state() -> AppState:
    global _state
    if _state is None:
        _state = AppState()
    return _state
