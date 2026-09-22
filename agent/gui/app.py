"""Ventana principal: navegación lateral entre las páginas."""

from __future__ import annotations

import argparse

from nicegui import app, ui

from . import page_chat, page_learn, page_models, page_requirements, page_server, page_settings
from .state import get_state

PAGES = [
    ("requirements", "Requisitos", "checklist", page_requirements.build),
    ("models", "Modelos", "view_in_ar", page_models.build),
    ("settings", "Configuración", "tune", page_settings.build),
    ("server", "Servidor", "dns", page_server.build),
    ("chat", "Chat", "forum", page_chat.build),
    ("learn", "Aprende", "school", page_learn.build),
]


@ui.page("/")
def index() -> None:
    state = get_state()
    ui.colors(primary="#4f46e5")
    ui.add_css(".nicegui-content { padding: 0; } .q-tab-panel { padding: 16px 24px; }")

    with ui.header().classes("items-center bg-primary text-white py-2 px-4"):
        ui.icon("smart_toy", size="28px")
        ui.label("Agente local").classes("text-lg font-semibold")
        ui.space()
        ui.label().bind_text_from(state.manager, "running",
                                  lambda r: "● servidor en marcha" if r else "○ servidor parado") \
            .classes("text-sm opacity-90")

    with ui.left_drawer(value=True, fixed=True).classes("bg-gray-50 dark:bg-gray-900 p-0").props("width=200 breakpoint=600"):
        with ui.tabs().props("vertical inline-label align=left").classes("w-full") as tabs:
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
    args = parser.parse_args()

    state = get_state()
    app.on_shutdown(state.shutdown)
    ui.run(title="Agente local", native=not args.browser,
           window_size=None if args.browser else (1400, 900), reload=False,
           host="127.0.0.1", port=args.port,  # solo local: la GUI puede ejecutar comandos
           favicon="🤖", show=args.browser and not args.no_open, dark=None)
