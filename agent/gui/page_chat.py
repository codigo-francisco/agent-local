"""Página «Chat»: conversación con el agente, diffs con aprobación y medidor de contexto."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from nicegui import background_tasks, ui

from ..core.events import (AgentError, ContextUsage, Done, Event, Notice, ReasoningDelta, TextDelta,
                           TextRewrite, ToolRequest, ToolResult)
from ..core.loop import Agent
from .state import AppState
from .widgets import workspace_picker

TOOL_ICONS = {"list_files": "folder", "read_file": "description", "search": "search",
              "edit_file": "edit", "write_file": "note_add", "run_command": "terminal"}
ACTION_LABELS = {"server": ("Ir a Servidor", "dns"), "settings": ("Ir a Configuración", "tune"),
                 "new_chat": ("Nueva conversación", "add_comment")}


def _k(n: int) -> str:
    return f"{n / 1000:.1f}K" if n >= 1000 else str(n)


class ChatView:
    def __init__(self, state: AppState):
        self.state = state
        self.agent = Agent(state.cfg, state.llm, self.on_event, self.approve, hint=state.ctx_hint)
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
        with ui.row().classes("w-full items-center gap-3"):
            ui.label("Chat").classes("text-2xl font-bold")
            self.model_select = ui.select(self.model_options(), value="main", label="Modelo") \
                .classes("w-72").tooltip("Puedes cambiar de modelo en cualquier momento")
            ui.button("Nueva conversación", icon="add_comment", on_click=self.new_chat).props("flat")
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
            "te aviso."
        ).classes("text-sm text-gray-600 dark:text-gray-400")

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
        with self.messages:
            self.welcome()
            with ui.row().classes("items-center no-wrap text-sm"):
                ui.icon("folder_open", color="primary")
                ui.label(f"Ahora trabajo en {target}. Empecé una conversación nueva.") \
                    .classes("text-gray-600 dark:text-gray-400")

    def config_changed(self) -> None:
        self.agent.llm = self.state.llm
        if not self.agent.running:
            self.sync_workspace()
        self.agent.auto_approve = self.state.cfg.confirm == "auto"
        self.agent._n_ctx.clear()  # el contexto pudo cambiar al reiniciar el servidor
        self.model_select.set_options(self.model_options(), value="main")

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
        self.md, self.md_text, self.reasoning = None, "", None

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
            elif isinstance(ev, Done):
                self.close_text()
                if ev.reason == "cancelled":
                    ui.label("⏹ Tarea detenida.").classes("text-sm text-gray-500")
                self.set_running(False)
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
                ui.button("Aprobar siempre", icon="done_all", on_click=lambda: decide("always")) \
                    .props("dense flat").tooltip("No volver a preguntar en esta sesión")

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
                ui.label({"yes": "✔ aprobado", "always": "✔ aprobado (siempre)", "no": "✖ rechazado"}[choice]) \
                    .classes("text-xs text-gray-500")
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
