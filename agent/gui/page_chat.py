"""Página «Chat»: conversación con el agente, diffs con aprobación y medidor de contexto."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from nicegui import app, background_tasks, ui

from ..core import sessions
from ..core.events import (AgentError, ContextUsage, Done, Event, FilesChanged, Notice,
                           ReasoningDelta, TextDelta, TextRewrite, ToolRequest, ToolResult)
from ..core.loop import Agent
from .state import CHECKPOINTS_DIR, SESSIONS_DIR, AppState
from .widgets import workspace_picker

TOOL_ICONS = {"list_files": "folder", "read_file": "description", "search": "search",
              "edit_file": "edit", "write_file": "note_add", "run_command": "terminal"}
ACTION_LABELS = {"server": ("Ir a Servidor", "dns"), "settings": ("Ir a Configuración", "tune"),
                 "new_chat": ("Nueva conversación", "add_comment")}


def _k(n: int) -> str:
    return f"{n / 1000:.1f}K" if n >= 1000 else str(n)


def _folder_key(path: str | Path) -> str:
    try:
        return str(Path(path).resolve()).lower()
    except OSError:
        return str(path).lower()


def _when(ts: float) -> str:
    """Fecha corta y legible: «hoy 14:05», «ayer 09:12», «12/09 18:30», «12/09/2025»."""
    if not ts:
        return "?"
    t, now = time.localtime(ts), time.localtime()
    days = (time.mktime(now[:3] + (0, 0, 0) + now[6:]) - time.mktime(t[:3] + (0, 0, 0) + t[6:])) // 86400
    if days == 0:
        return time.strftime("hoy %H:%M", t)
    if days == 1:
        return time.strftime("ayer %H:%M", t)
    return time.strftime("%d/%m %H:%M" if t.tm_year == now.tm_year else "%d/%m/%Y", t)


class ChatView:
    def __init__(self, state: AppState):
        self.state = state
        self.agent = Agent(state.cfg, state.llm, self.on_event, self.approve, hint=state.ctx_hint,
                           mcp=state.mcp, persist=state.persist, sessions_dir=SESSIONS_DIR,
                           checkpoints_dir=CHECKPOINTS_DIR, safe_mode=state.safe_mode)
        self.cancel: asyncio.Event | None = None
        self.pending: dict[str, asyncio.Future] = {}
        self.tool_cards: dict[str, dict] = {}
        self.md: ui.markdown | None = None
        self.md_text = ""
        self.md_last = 0.0
        self.reasoning: ui.label | None = None
        state.on_config_saved(self.config_changed)

    # --- construcción ----------------------------------------------------
    def build(self) -> None:
        with ui.row().classes("w-full no-wrap items-start gap-3"):
            with ui.column().classes("w-72 shrink-0 gap-2"):
                self.sidebar()
            with ui.column().classes("grow min-w-0 gap-2"):
                self.chat_area()

    def chat_area(self) -> None:
        with ui.row().classes("w-full items-center gap-3"):
            ui.label("Chat").classes("text-2xl font-bold")
            self.model_select = ui.select(self.model_options(), value="main", label="Modelo") \
                .classes("w-72").tooltip("Puedes cambiar de modelo en cualquier momento")
            ui.space()
            with ui.column().classes("gap-0 w-80"):
                self.ctx_label = ui.label("Contexto: —").classes("text-xs text-gray-500")
                self.ctx_bar = ui.linear_progress(value=0, show_value=False).classes("w-full")
                with self.ctx_bar:
                    self.ctx_tip = ui.tooltip("")
        workspace_picker(self.state, "📁 Carpeta del proyecto")
        self.scroll = ui.scroll_area().classes("w-full border rounded").style("height: calc(100vh - 290px)")
        with self.scroll:
            self.messages = ui.column().classes("w-full gap-2 p-2")
        with self.messages:
            self.welcome()
        with ui.row().classes("w-full items-end no-wrap"):
            self.input = ui.textarea(placeholder="Pide algo: «corre los tests y arregla lo que falle»… "
                                                 "(Enter envía, Shift+Enter salto de línea)") \
                .props("autogrow outlined dense").classes("grow")
            self.input.on("keydown", self.send, js_handler=(
                "(e) => { if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) "
                "{ e.preventDefault(); emit(); } }"))
            self.send_btn = ui.button(icon="send", on_click=self.send).props("round")
            self.stop_btn = ui.button(icon="stop", color="negative", on_click=self.stop).props("round")
            self.stop_btn.set_visibility(False)

    def welcome(self) -> None:
        ui.markdown(
            "👋 Soy tu agente local. Trabajo **dentro de la carpeta del proyecto** configurada arriba: "
            "puedo listar, leer y buscar archivos libremente; para **editar archivos o ejecutar "
            "comandos te pediré confirmación** mostrando el cambio exacto.\n\n"
            "Consejos: pide tareas concretas; si el contexto se llena lo compacto automáticamente y "
            "te aviso. Tus conversaciones se guardan solas: las tienes a la izquierda para retomarlas."
        ).classes("text-sm text-gray-600 dark:text-gray-400")

    # --- conversaciones (multichat) -----------------------------------------
    def sidebar(self) -> None:
        with ui.row().classes("w-full items-center no-wrap"):
            ui.label("Conversaciones").classes("text-lg font-semibold")
            ui.space()
            ui.button(icon="add_comment", on_click=self.new_chat).props("flat round dense") \
                .tooltip("Nueva conversación")
            ui.button(icon="file_upload", on_click=self.import_chat).props("flat round dense") \
                .tooltip("Importar una conversación exportada (.json)")
        self.search = ui.input(placeholder="Buscar…", on_change=lambda: self.chat_list.refresh()) \
            .props("dense outlined clearable").classes("w-full")
        # content-style: sin él, el contenido crece con el título más largo y tapa el botón ⋮.
        with ui.scroll_area().classes("w-full border rounded").style("height: calc(100vh - 200px)") \
                .props('content-style="width: 100%" content-active-style="width: 100%"'):
            self.chat_list = ui.refreshable(self._chat_list)
            self.chat_list()
        # Al volver al chat: pudo haber conversaciones nuevas (importadas, CLI, otra ventana).
        self.state.on_tab_shown("chat", lambda: None if self.agent.running else self.chat_list.refresh())

    def _chat_list(self) -> None:
        # Todas las conversaciones de todas las carpetas, agrupadas por carpeta de proyecto (la
        # actual primero). Antes solo salían las de la carpeta actual y parecía que las demás se
        # habían borrado.
        items = sessions.list_sessions(SESSIONS_DIR)
        query = (self.search.value or "").strip().lower()
        if query:
            items = [s for s in items if query in s.title.lower()
                     or any(query in str(m.get("content") or "").lower() for m in s.history
                            if m.get("role") == "user")]
        if not items:
            ui.label("Aún no hay conversaciones." if not query else "Sin resultados.") \
                .classes("text-sm text-gray-500 p-2")
        here = _folder_key(self.agent.toolbox.workspace)
        groups: dict[str, list[sessions.Session]] = {}
        for s in items:
            groups.setdefault(_folder_key(s.workspace), []).append(s)
        for key in sorted(groups, key=lambda k: (k != here, k)):
            folder = Path(groups[key][0].workspace)
            with ui.row().classes("w-full items-center no-wrap gap-1 px-2 pt-2"):
                ui.icon("folder_open" if key == here else "folder", size="16px",
                        color="primary" if key == here else "grey")
                ui.label(f"{folder.name or folder} ({len(groups[key])})").classes(
                    "text-xs font-semibold truncate " + ("text-primary" if key == here else "text-gray-500")) \
                    .tooltip(str(folder) + (" · carpeta actual" if key == here else
                                            " · al abrir una de estas, el agente cambia a esta carpeta"))
            for s in groups[key]:
                self._chat_row(s)

    def _chat_row(self, s: sessions.Session) -> None:
        """Una conversación de la lista: clic para abrirla, ⋮ para renombrar/exportar/borrar."""
        current = s.id == self.agent.session_id
        turns = sum(1 for m in s.history if m.get("role") == "user")
        row_classes = ("w-full items-center no-wrap gap-1 px-2 py-1 rounded cursor-pointer "
                       "hover:bg-gray-100 dark:hover:bg-gray-800")
        with ui.row().classes(row_classes + (" bg-indigo-50 dark:bg-indigo-950" if current else "")) \
                .on("click", lambda s=s: self.open_session(s.id)):
            with ui.column().classes("gap-0 grow min-w-0"):
                ui.label(s.title).classes("text-sm truncate w-full" + (" font-semibold" if current else ""))
                ui.label(f"{_when(s.updated)} · {turns} mensaje{'s' if turns != 1 else ''}") \
                    .classes("text-xs text-gray-500")
            with ui.button(icon="more_vert").props("flat round dense size=sm") \
                    .on("click.stop", lambda: None):
                with ui.menu():
                    ui.menu_item("Renombrar", on_click=lambda s=s: self.rename_chat(s))
                    ui.menu_item("Exportar…", on_click=lambda s=s: self.export_chat(s))
                    ui.menu_item("Eliminar", on_click=lambda s=s: self.delete_chat(s))

    def open_session(self, session_id: str) -> None:
        """Retoma una conversación guardada (sobrevive a cerrar la app o a un fallo)."""
        if session_id == self.agent.session_id:
            return
        if self.agent.running:
            ui.notify("Detén la tarea actual antes de cambiar de conversación.", type="warning")
            return
        saved = sessions.get(SESSIONS_DIR, session_id)
        if saved is None:
            ui.notify("No encontré esa conversación (¿se borró?).", type="warning")
            self.chat_list.refresh()
            return
        if _folder_key(saved.workspace) != _folder_key(self.agent.toolbox.workspace):
            # Es de otro proyecto: el agente vuelve a esa carpeta para seguir donde lo dejó.
            if Path(saved.workspace).is_dir():
                error = self.state.set_workspace(saved.workspace)
                if error:
                    ui.notify(error, type="negative")
                    return
                self.sync_workspace()
                ui.notify(f"Carpeta del proyecto: {saved.workspace}")
            else:
                ui.notify(f"La carpeta de esa conversación ({saved.workspace}) ya no existe: la abro en "
                          "la carpeta actual.", type="warning", multi_line=True)
        self.agent.restore(saved)
        self.messages.clear()
        self.reset_context_bar()
        with self.messages:
            for m in self.agent.history:
                content = m.get("content") or ""
                if m["role"] == "user" and content:
                    with ui.chat_message(name="Tú", sent=True).classes("self-end max-w-3xl"):
                        ui.markdown(content)
                elif m["role"] == "assistant":
                    if content:
                        with ui.chat_message(name="Agente").classes("max-w-4xl w-full"):
                            ui.markdown(content).classes("w-full")
                        self.copy_button(content)
                    for tc in m.get("tool_calls") or []:
                        ui.label(f"🔧 {tc['function']['name']}").classes("text-xs font-mono text-gray-500")
            ui.label(f"↺ Conversación «{saved.title}» retomada: sigue donde lo dejaste.") \
                .classes("text-sm text-gray-500")
        self.chat_list.refresh()
        self.scroll_down()

    def rename_chat(self, s: sessions.Session) -> None:
        with ui.dialog() as dialog, ui.card().classes("min-w-96"):
            ui.label("Renombrar conversación").classes("text-lg font-semibold")
            name = ui.input("Nombre", value=s.title).classes("w-full").props("autofocus")

            def accept() -> None:
                sessions.rename(SESSIONS_DIR, s.id, name.value or "")
                dialog.close()
                self.chat_list.refresh()

            name.on("keydown.enter", accept)
            with ui.row().classes("w-full justify-end"):
                ui.button("Cancelar", on_click=dialog.close).props("flat")
                ui.button("Guardar", icon="check", on_click=accept)
        dialog.open()

    def delete_chat(self, s: sessions.Session) -> None:
        if self.agent.running and s.id == self.agent.session_id:
            ui.notify("Detén la tarea actual antes de borrar esta conversación.", type="warning")
            return
        with ui.dialog() as dialog, ui.card():
            ui.label(f"¿Eliminar «{s.title}»?").classes("text-lg font-semibold")
            ui.label("Se borra del disco y no se puede recuperar (salvo que la hayas exportado). "
                     "Los archivos que cambió el agente no se tocan.").classes("text-sm")

            def accept() -> None:
                sessions.delete(SESSIONS_DIR, s.id)
                dialog.close()
                if s.id == self.agent.session_id:
                    self.new_chat()
                self.chat_list.refresh()
                ui.notify("Conversación eliminada.")

            with ui.row().classes("w-full justify-end"):
                ui.button("Cancelar", on_click=dialog.close).props("flat")
                ui.button("Eliminar", icon="delete", color="negative", on_click=accept)
        dialog.open()

    async def export_chat(self, s: sessions.Session) -> None:
        saved = sessions.get(SESSIONS_DIR, s.id) or s
        data, filename = sessions.export_bytes(saved), sessions.export_filename(saved)
        window = getattr(app.native, "main_window", None)
        if window is None:  # navegador: descarga normal
            ui.download.content(data, filename, "application/json")
            return
        import webview
        kind = getattr(getattr(webview, "FileDialog", None), "SAVE", None) or webview.SAVE_DIALOG
        result = await window.create_file_dialog(kind, save_filename=filename,
                                                 file_types=("Conversación (*.json)",))
        path = result[0] if isinstance(result, (list, tuple)) else result
        if not path:
            return
        try:
            Path(path).write_bytes(data)
        except OSError as e:
            ui.notify(f"No pude guardar el archivo: {e}", type="negative")
            return
        ui.notify(f"Conversación exportada a {path}", type="positive")

    async def import_chat(self) -> None:
        if self.agent.running:
            ui.notify("Detén la tarea actual antes de importar una conversación.", type="warning")
            return
        window = getattr(app.native, "main_window", None)
        if window is not None:
            import webview
            kind = getattr(getattr(webview, "FileDialog", None), "OPEN", None) or webview.OPEN_DIALOG
            result = await window.create_file_dialog(kind, file_types=("Conversación (*.json)",))
            if result:
                try:
                    self.finish_import(Path(result[0]).read_bytes())
                except OSError as e:
                    ui.notify(f"No pude leer el archivo: {e}", type="negative")
            return
        with ui.dialog() as dialog, ui.card():
            ui.label("Importar conversación").classes("text-lg font-semibold")
            ui.label("Elige un archivo .json exportado desde agent-local. Se añadirá a esta carpeta de "
                     "proyecto.").classes("text-sm")

            async def uploaded(e) -> None:
                dialog.close()
                self.finish_import(await e.file.read())

            ui.upload(auto_upload=True, on_upload=uploaded, max_file_size=50_000_000) \
                .props("accept=.json").classes("w-full")
            ui.button("Cancelar", on_click=dialog.close).props("flat")
        dialog.open()

    def finish_import(self, raw: bytes) -> None:
        try:
            s = sessions.import_bytes(SESSIONS_DIR, raw, str(self.agent.toolbox.workspace))
        except sessions.SessionImportError as e:
            ui.notify(str(e), type="negative", multi_line=True)
            return
        ui.notify(f"Conversación «{s.title}» importada.", type="positive")
        self.open_session(s.id)

    def undo_card(self, ev: FilesChanged) -> None:
        with ui.row().classes("items-center no-wrap text-sm") as row:
            ui.icon("difference", color="primary")
            ui.label(f"Archivos cambiados: {', '.join(ev.files)}").classes("text-gray-600 dark:text-gray-400")

            def undo() -> None:
                if self.agent.running:
                    ui.notify("Espera a que termine la tarea actual.", type="warning")
                    return
                result = self.agent.undo(ev.checkpoint_id)
                button.delete()
                with row:
                    ui.label(f"↶ {result}").classes("text-gray-500")

            button = ui.button("Deshacer", icon="undo", on_click=undo).props("flat dense") \
                .tooltip("Devuelve estos archivos a como estaban antes del turno (los comandos "
                         "ejecutados no se deshacen)")

    def model_options(self) -> dict[str, str]:
        cfg = self.state.cfg
        opts = {r: f"{r} → {cfg.roles[r]}" for r in ("main", "fast") if cfg.roles.get(r)}
        opts.update({m.name: m.name for m in cfg.models if m.name != cfg.roles.get("draft")})
        return opts or {"main": "main (sin asignar)"}

    def sync_workspace(self) -> None:
        """Alinea el agente con la carpeta configurada; si cambió, empieza conversación nueva."""
        target = Path(self.state.cfg.workspace).resolve()
        if self.agent.toolbox.workspace == target:
            return
        self.agent.set_workspace(str(target))
        self.messages.clear()
        self.reset_context_bar()
        with self.messages:
            self.welcome()
            with ui.row().classes("items-center no-wrap text-sm"):
                ui.icon("folder_open", color="primary")
                ui.label(f"Ahora trabajo en {target}. Empecé una conversación nueva (las anteriores "
                         "siguen en la lista de la izquierda).") \
                    .classes("text-gray-600 dark:text-gray-400")
        self.chat_list.refresh()  # la lista muestra las conversaciones de la carpeta nueva

    def config_changed(self) -> None:
        self.agent.llm = self.state.llm
        if not self.agent.running:
            self.sync_workspace()
        self.agent.auto_approve = self.state.cfg.confirm == "auto" and not self.state.safe_mode
        self.agent._n_ctx.clear()  # el contexto pudo cambiar al reiniciar el servidor
        options = self.model_options()
        current = self.model_select.value
        self.model_select.set_options(options, value=current if current in options else "main")

    # --- acciones --------------------------------------------------------
    async def send(self) -> None:
        text = (self.input.value or "").strip()
        if not text or self.agent.running:
            return
        self.sync_workspace()  # por si la carpeta cambió mientras había una tarea en marcha
        self.input.set_value("")
        with self.messages:
            with ui.chat_message(name="Tú", sent=True).classes("self-end max-w-3xl"):
                ui.markdown(text)
        self.md = None
        self.set_running(True)
        self.cancel = asyncio.Event()
        background_tasks.create(self.agent.run(text, self.model_select.value, self.cancel))
        self.scroll_down()

    def stop(self) -> None:
        if self.cancel:
            self.cancel.set()
        for fut in self.pending.values():
            if not fut.done():
                fut.set_result("no")

    def new_chat(self) -> None:
        if self.agent.running:
            ui.notify("Detén la tarea actual antes de empezar otra conversación.", type="warning")
            return
        self.agent.reset()
        self.messages.clear()
        with self.messages:
            self.welcome()
        self.reset_context_bar()
        self.chat_list.refresh()

    def reset_context_bar(self) -> None:
        self.ctx_bar.set_value(0)
        self.ctx_label.set_text("Contexto: —")

    def set_running(self, running: bool) -> None:
        self.send_btn.set_visibility(not running)
        self.stop_btn.set_visibility(running)
        self.input.set_enabled(not running)

    def scroll_down(self) -> None:
        self.scroll.scroll_to(percent=1.0)

    # --- eventos del agente ----------------------------------------------
    def flush_text(self, force: bool = True) -> None:
        if self.md is not None and (force or time.monotonic() - self.md_last > 0.12):
            self.md.set_content(self.md_text)
            self.md_last = time.monotonic()
            self.scroll_down()

    def close_text(self) -> None:
        self.flush_text()
        if self.md is not None and self.md_text.strip():
            with self.messages:
                self.copy_button(self.md_text)
        self.md, self.md_text, self.reasoning = None, "", None

    def copy_button(self, text: str) -> None:
        """Botón para copiar una respuesta entera (el texto también se puede seleccionar a mano)."""
        with ui.row().classes("w-full max-w-4xl justify-end -mt-1"):
            ui.button("Copiar", icon="content_copy", on_click=lambda t=text: self.copy_text(t)) \
                .props("flat dense no-caps size=sm").classes("text-gray-500") \
                .tooltip("Copia la respuesta completa (en Markdown)")

    @staticmethod
    def copy_text(text: str) -> None:
        ui.clipboard.write(text)
        ui.notify("Respuesta copiada al portapapeles.", type="positive")

    async def on_event(self, ev: Event) -> None:
        with self.messages:
            if isinstance(ev, TextDelta):
                if self.md is None:
                    with ui.chat_message(name="Agente").classes("max-w-4xl w-full"):
                        self.md = ui.markdown("").classes("w-full")
                    self.md_text = ""
                self.md_text += ev.text
                self.flush_text(force=False)
            elif isinstance(ev, TextRewrite):
                if self.md is not None:
                    self.md_text = ev.text
                    if not ev.text.strip():  # solo había la llamada: fuera la burbuja vacía
                        self.md.parent_slot.parent.delete()
                        self.md = None
                    else:
                        self.flush_text()
            elif isinstance(ev, ReasoningDelta):
                if self.reasoning is None:
                    with ui.expansion("Razonando…", icon="psychology").classes("text-xs text-gray-500 w-full"):
                        self.reasoning = ui.label("").classes("whitespace-pre-wrap")
                self.reasoning.set_text(self.reasoning.text + ev.text)
            elif isinstance(ev, ToolRequest):
                self.close_text()
                self.tool_request(ev)
            elif isinstance(ev, ToolResult):
                self.tool_result(ev)
            elif isinstance(ev, ContextUsage):
                frac = min(1.0, ev.used / max(1, ev.budget))
                self.ctx_bar.set_value(frac)
                self.ctx_bar.props(f"color={'positive' if frac < 0.6 else 'warning' if frac < 0.85 else 'negative'}")
                self.ctx_label.set_text(f"Contexto: {_k(ev.used)} / {_k(ev.budget)} tokens útiles "
                                        f"(ventana {_k(ev.limit)})")
                self.ctx_tip.set_text(" · ".join(f"{k}: {_k(v)}" for k, v in ev.breakdown.items()))
            elif isinstance(ev, Notice):
                self.close_text()
                with ui.row().classes("items-center no-wrap text-sm"):
                    ui.icon("info" if ev.level == "info" else "warning",
                            color="primary" if ev.level == "info" else "warning")
                    ui.label(ev.text).classes("text-gray-600 dark:text-gray-400")
            elif isinstance(ev, AgentError):
                self.close_text()
                self.error_card(ev)
            elif isinstance(ev, FilesChanged):
                self.close_text()
                self.undo_card(ev)
            elif isinstance(ev, Done):
                self.close_text()
                if ev.reason == "cancelled":
                    ui.label("⏹ Tarea detenida.").classes("text-sm text-gray-500")
                self.set_running(False)
                # El agente guarda la conversación justo después de Done: refrescar la lista luego.
                ui.timer(0.5, self.chat_list.refresh, once=True)
        if not isinstance(ev, (TextDelta, ReasoningDelta)):  # el texto ya hace scroll al refrescar
            self.scroll_down()

    def tool_request(self, ev: ToolRequest) -> None:
        short = ", ".join(f"{k}={str(v)[:50]}" for k, v in ev.args.items()
                          if k not in ("content", "old", "new"))
        with ui.card().classes("w-full max-w-4xl p-2") as card:
            with ui.row().classes("items-center no-wrap w-full"):
                ui.icon(TOOL_ICONS.get(ev.name, "build"), size="20px")
                ui.label(f"{ev.name}({short})").classes("font-mono text-sm grow truncate")
                status = ui.spinner(size="sm")
            if ev.needs_approval:
                ui.code(ev.preview, language="diff" if ev.name != "run_command" else "powershell") \
                    .classes("w-full text-xs max-h-96 overflow-auto")
                buttons = ui.row()
            else:
                buttons = None
                if ev.name in ("edit_file", "write_file", "run_command"):
                    with ui.expansion("Ver cambio", icon="difference").classes("w-full text-xs"):
                        ui.code(ev.preview).classes("w-full text-xs")
            result_box = ui.column().classes("w-full")
        self.tool_cards[ev.call_id] = {"card": card, "status": status, "result": result_box,
                                       "buttons": buttons}
        if ev.needs_approval and buttons is not None:
            fut: asyncio.Future = asyncio.get_running_loop().create_future()
            self.pending[ev.call_id] = fut

            def decide(choice: str) -> None:
                if not fut.done():
                    fut.set_result(choice)

            with buttons:
                ui.button("Aprobar", icon="check", color="positive", on_click=lambda: decide("yes")).props("dense")
                ui.button("Rechazar", icon="close", color="negative", on_click=lambda: decide("no")).props("dense")
                ui.button("Aprobar en esta sesión", icon="schedule", on_click=lambda: decide("session")) \
                    .props("dense flat").tooltip(f"No volver a preguntar por «{ev.name}» hasta cerrar la app")
                ui.button("Aprobar siempre", icon="done_all", on_click=lambda: decide("always")) \
                    .props("dense flat").tooltip(f"No volver a preguntar por «{ev.name}». Puedes quitar "
                                                 "este permiso en Configuración → Permisos")

    async def approve(self, ev: ToolRequest) -> str:
        fut = self.pending.get(ev.call_id)
        if fut is None:
            return "yes"
        try:
            choice = await fut
        finally:
            self.pending.pop(ev.call_id, None)
        box = self.tool_cards.get(ev.call_id, {}).get("buttons")
        if box is not None:
            box.clear()
            with box:
                ui.label({"yes": "✔ aprobado", "session": f"✔ aprobado «{ev.name}» en esta sesión",
                          "always": f"✔ aprobado «{ev.name}» siempre", "no": "✖ rechazado"}
                         .get(choice, choice)).classes("text-xs text-gray-500")
        return choice

    def tool_result(self, ev: ToolResult) -> None:
        card = self.tool_cards.get(ev.call_id)
        if not card:
            return
        card["status"].delete()
        with card["result"]:
            with ui.expansion("Resultado" if ev.ok else "Error", icon="check" if ev.ok else "error_outline") \
                    .classes("w-full text-xs" + ("" if ev.ok else " text-negative")) as exp:
                ui.code(ev.output).classes("w-full text-xs max-h-80 overflow-auto")
            if not ev.ok:
                exp.set_value(True)

    def error_card(self, ev: AgentError) -> None:
        with ui.card().classes("w-full max-w-4xl border-l-4 border-red-500"):
            with ui.row().classes("items-center"):
                ui.icon("error", color="negative", size="24px")
                ui.label(ev.title).classes("font-semibold text-negative")
            ui.label(ev.cause).classes("text-sm")
            if ev.suggestions:
                ui.label("Qué puedes hacer:").classes("text-sm font-medium mt-1")
                for s in ev.suggestions:
                    ui.label(f"• {s}").classes("text-sm")
            if ev.detail:
                with ui.expansion("Detalle técnico", icon="code").classes("w-full text-xs"):
                    ui.code(ev.detail).classes("w-full text-xs max-h-80 overflow-auto")
            if ev.action in ACTION_LABELS:
                label, icon = ACTION_LABELS[ev.action]
                action = self.new_chat if ev.action == "new_chat" else \
                    (lambda a=ev.action: self.state.navigate(a))
                ui.button(label, icon=icon, on_click=action).props("flat")


def build(state: AppState) -> None:
    ChatView(state).build()
