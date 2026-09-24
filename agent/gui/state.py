"""Estado compartido de la aplicación (una sola instancia: es una app local de un usuario)."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Callable

from nicegui import Client, ui

from .. import config
from ..config import GENERATED_DIR, AppConfig, ModelEntry, load_catalog, load_config
from ..core.llm import LLMClient
from ..core.mcp_tools import MCPManager
from ..server import autotune, swapconfig, vram
from ..server.downloader import DownloadState
from ..server.manager import ServerManager

DEFAULT_CTX = {"main": 24576, "fast": 16384, "draft": 4096}
log = logging.getLogger("agent.gui")


SESSIONS_DIR = GENERATED_DIR / "sessions"
CHECKPOINTS_DIR = GENERATED_DIR / "checkpoints"


class AppState:
    def __init__(self, safe_mode: bool = False) -> None:
        self.safe_mode = safe_mode  # --safe: sin MCP, sin modo automático ni permisos «siempre»
        self.cfg: AppConfig = load_config()
        self.load_warning = config.load_warning  # config rota apartada al arrancar
        if self.load_warning:
            log.warning(self.load_warning)
        self.catalog: list[dict] = load_catalog()
        self.manager = ServerManager()
        self.downloads: dict[str, DownloadState] = {}
        self.llm = LLMClient(self.cfg.endpoint, log_provider=self.manager.log_tail)
        self.mcp = MCPManager()
        self.mcp_starting = False
        self.navigate: Callable[[str], None] = lambda tab: None  # lo fija app.py
        self._listeners: list[tuple[Client, Callable[[], None]]] = []
        self._tab_listeners: list[tuple[Client, str, Callable[[], None]]] = []
        self._remote_listeners: list[tuple[Client, Callable[[], None]]] = []
        # Modelos copiados o descargados con la app cerrada: que aparezcan ya en Configuración.
        if autotune.discover_models(self.cfg, self.catalog):
            try:
                self.cfg.save()
                swapconfig.write(self.cfg)
            except OSError:
                log.exception("No se pudo guardar la configuración con los modelos nuevos")

    def on_config_saved(self, callback: Callable[[], None]) -> None:
        """Avisa a `callback` al guardar la configuración, mientras su página siga abierta."""
        self._listeners.append((ui.context.client, callback))

    def on_remote_changed(self, callback: Callable[[], None]) -> None:
        """Avisa a `callback` cuando cambia la lista de PCs remotas (se añade, se quita, se activa o
        desactiva, o se mide con «Probar»): la VRAM total cambia y hay que rehacer los cálculos."""
        self._remote_listeners.append((ui.context.client, callback))

    def remote_changed(self) -> None:
        alive = []
        for client, callback in self._remote_listeners:
            if client.is_deleted:
                continue
            alive.append((client, callback))
            try:
                with client:
                    callback()
            except Exception:  # noqa: BLE001
                log.exception("Fallo al avisar del cambio de PCs remotas")
        self._remote_listeners = alive

    def on_tab_shown(self, key: str, callback: Callable[[], None]) -> None:
        """Avisa a `callback` cada vez que el usuario abre la pestaña `key` (p. ej. para refrescar
        listas que pudieron cambiar mientras tanto)."""
        self._tab_listeners.append((ui.context.client, key, callback))

    def tab_shown(self, key: str) -> None:
        if key in ("models", "settings"):
            self.sync_downloaded_models()  # p. ej. un .gguf copiado a mano con la app abierta
        alive = []
        for client, k, callback in self._tab_listeners:
            if client.is_deleted:
                continue
            alive.append((client, k, callback))
            if k == key:
                try:
                    with client:
                        callback()
                except Exception:  # noqa: BLE001
                    log.exception("Fallo al refrescar la pestaña %s", key)
        self._tab_listeners = alive

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
                log.exception("Fallo al avisar a una página del cambio de configuración")
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
    def sync_downloaded_models(self, forget: str | None = None) -> list[str]:
        """Añade a la configuración los .gguf de models/ que aún no están (guarda y avisa a las
        páginas). `forget`: archivo recién descargado que antes se había quitado a propósito."""
        if forget:
            self.cfg.ignored_files = [f for f in self.cfg.ignored_files if f.lower() != forget.lower()]
        added = autotune.discover_models(self.cfg, self.catalog)
        if added or forget:
            self.save_config()
        return [e.name for e in added]

    def ensure_entry(self, name: str, file: str, role: str | None = None,
                     defaults: dict | None = None) -> ModelEntry:
        self.cfg.ignored_files = [f for f in self.cfg.ignored_files if f.lower() != Path(file).name.lower()]
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
        entry = self.cfg.model(name)
        if entry and entry.path.is_file() and entry.path.parent == config.MODELS_DIR \
                and entry.path.name not in self.cfg.ignored_files:
            self.cfg.ignored_files.append(entry.path.name)  # que no vuelva a añadirse solo
        self.cfg.models = [m for m in self.cfg.models if m.name != name]
        for role, value in self.cfg.roles.items():
            if value == name:
                self.cfg.roles[role] = ""

    # --- persistencia ----------------------------------------------------
    def save_config(self) -> list[str]:
        self.cfg.endpoint = self.cfg.endpoint.strip() or f"http://127.0.0.1:{self.cfg.port}/v1"
        self.cfg.save()
        warnings = swapconfig.write(self.cfg)
        # Solo se cambia de cliente si cambió el endpoint. Y el viejo se cierra cuando termine lo
        # que esté recibiendo: cerrarlo ya cortaba la respuesta en curso (p. ej. al terminar una
        # descarga, que guarda la configuración, en mitad de un turno del agente).
        if self.llm.endpoint != self.cfg.endpoint.rstrip("/"):
            old = self.llm
            self.llm = LLMClient(self.cfg.endpoint, log_provider=self.manager.log_tail)
            try:
                asyncio.get_running_loop().create_task(old.close_when_idle())
            except RuntimeError:
                pass
        self._notify_config_saved()
        return warnings

    def persist(self) -> None:
        """Guardado ligero (p. ej. permisos): no regenera llama-swap ni recrea el cliente del modelo,
        así es seguro en mitad de una tarea."""
        self.cfg.save()
        self._notify_config_saved()

    def revoke_always(self, tool: str | None = None) -> None:
        """Quita un permiso de «siempre» (o todos si tool es None)."""
        self.cfg.always_allow = [t for t in self.cfg.always_allow if tool is not None and t != tool]
        self.persist()

    def ctx_hint(self, model: str, n_ctx: int) -> str | None:
        return vram.ctx_hint(self.cfg, self.catalog, model, n_ctx)

    async def restart_mcp(self) -> None:
        if self.safe_mode:
            return
        self.mcp_starting = True
        try:
            await self.mcp.start()
        finally:
            self.mcp_starting = False

    async def shutdown(self) -> None:
        self.manager.stop()
        await self.mcp.stop()
        await self.llm.aclose()


_state: AppState | None = None


def get_state(safe_mode: bool = False) -> AppState:
    global _state
    if _state is None:
        _state = AppState(safe_mode)
    return _state
