"""Ventana principal: navegación lateral entre las páginas."""

from __future__ import annotations

import argparse
import asyncio
import sys
import webbrowser

import httpx
from nicegui import app, background_tasks, ui

from .. import log
from ..server.manager import port_in_use
from . import (page_chat, page_diagnostics, page_learn, page_models, page_requirements, page_server,
               page_settings)
from .security import LocalOnly
from .state import get_state

TITLE = "Agente local"

PAGES = [
    ("requirements", "Requisitos", "checklist", page_requirements.build),
    ("models", "Modelos", "view_in_ar", page_models.build),
    ("settings", "Configuración", "tune", page_settings.build),
    ("server", "Servidor", "dns", page_server.build),
    ("chat", "Chat", "forum", page_chat.build),
    ("learn", "Aprende", "school", page_learn.build),
    ("diagnostics", "Diagnóstico", "troubleshoot", page_diagnostics.build),
]


@ui.page("/")
def index() -> None:
    state = get_state()
    ui.colors(primary="#4f46e5")
    ui.add_css(".nicegui-content { padding: 0; } .q-tab-panel { padding: 16px 24px; }")

    with ui.header().classes("items-center bg-primary text-white py-2 px-4"):
        ui.icon("smart_toy", size="28px")
        ui.label(TITLE).classes("text-lg font-semibold")
        if state.safe_mode:
            ui.badge("modo seguro", color="amber").tooltip("Arrancada con --safe")
        ui.space()
        ui.label().bind_text_from(state.manager, "running",
                                  lambda r: "● servidor en marcha" if r else "○ servidor parado") \
            .classes("text-sm opacity-90")

    with ui.left_drawer(value=True, fixed=True).classes("bg-gray-50 dark:bg-gray-900 p-0").props("width=200 breakpoint=600"):
        with ui.tabs(on_change=lambda e: state.tab_shown(e.value)) \
                .props("vertical inline-label align=left").classes("w-full") as tabs:
            for key, label, icon, _ in PAGES:
                ui.tab(key, label=label, icon=icon)

    state.navigate = lambda key: tabs.set_value(key)

    with ui.tab_panels(tabs, value="requirements").classes("w-full").props("keep-alive"):
        for key, _, _, build in PAGES:
            with ui.tab_panel(key):
                build(state)


def main() -> None:
    parser = argparse.ArgumentParser(description="GUI del agente local")
    parser.add_argument("--browser", action="store_true",
                        help="abrir en el navegador en lugar de ventana nativa")
    parser.add_argument("--no-open", action="store_true",
                        help="con --browser, no abrir el navegador automáticamente")
    parser.add_argument("--port", type=int, default=8765, help="puerto de la GUI (no del modelo)")
    parser.add_argument("--safe", action="store_true",
                        help="modo seguro: sin MCP, sin modo automático ni permisos «siempre»")
    args = parser.parse_args()
    log.setup()

    if port_in_use(args.port):
        # ¿Es otra instancia de esta app? Entonces se abre esa en lugar de fallar al escuchar.
        url = f"http://127.0.0.1:{args.port}"
        try:
            ours = TITLE in httpx.get(url, timeout=2.0).text
        except httpx.HTTPError:
            ours = False
        if ours:
            print(f"La app ya está abierta; la muestro en el navegador ({url}).")
            webbrowser.open(url)
            return
        sys.exit(f"El puerto {args.port} lo usa otro programa. Arranca con --port <otro>.")

    state = get_state(safe_mode=args.safe)
    app.add_middleware(LocalOnly, port=args.port)  # la GUI ejecuta comandos: solo esta máquina

    async def startup() -> None:
        asyncio.get_running_loop().set_exception_handler(log.asyncio_handler)
        background_tasks.create(state.restart_mcp())  # sin retrasar la ventana

    app.on_startup(startup)
    app.on_shutdown(state.shutdown)
    log.log.info("GUI arrancando (puerto %s%s)", args.port, ", modo seguro" if args.safe else "")
    # pywebview desactiva por defecto la selección de texto en la ventana nativa: sin esto no se
    # puede copiar nada del chat.
    app.native.window_args["text_select"] = True
    ui.run(title=TITLE, native=not args.browser,
           window_size=None if args.browser else (1400, 900), reload=False,
           host="127.0.0.1", port=args.port,  # solo local: la GUI puede ejecutar comandos
           favicon="🤖", show=args.browser and not args.no_open, dark=None)
