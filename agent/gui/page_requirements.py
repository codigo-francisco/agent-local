"""Página «Requisitos»: semáforo del sistema con por qué y cómo solucionarlo."""

from __future__ import annotations

import webbrowser

from nicegui import run, ui

from ..server.requirements import Check, check_all
from .state import AppState

STYLE = {
    "ok": ("check_circle", "positive", "Correcto"),
    "warn": ("warning", "warning", "Atención"),
    "error": ("cancel", "negative", "Falta"),
}


def _card(c: Check) -> None:
    icon, color, label = STYLE[c.status]
    with ui.card().classes("w-full"):
        with ui.row().classes("items-center w-full no-wrap"):
            ui.icon(icon, color=color, size="28px")
            with ui.column().classes("gap-0 grow"):
                ui.label(c.title).classes("font-semibold")
                ui.label(c.detail).classes("text-sm text-gray-600 dark:text-gray-400 break-all")
            ui.badge(label, color=color)
        with ui.expansion("¿Por qué hace falta?", icon="help_outline").classes("w-full text-sm"):
            ui.label(c.why)
        if c.fix:
            with ui.row().classes("items-center w-full"):
                ui.icon("build", size="18px").classes("text-gray-500")
                ui.label(c.fix).classes("text-sm grow")
                if c.link:
                    ui.button("Abrir descarga", icon="open_in_new",
                              on_click=lambda url=c.link: webbrowser.open(url)).props("flat dense")


def build(state: AppState) -> None:
    ui.label("Requisitos").classes("text-2xl font-bold")
    ui.markdown(
        "El agente funciona **sin internet y sin API externa**: todo corre en tu PC. Para eso "
        "necesita tres piezas: **llama.cpp** (ejecuta el modelo en la GPU), **llama-swap** "
        "(gestiona varios modelos detrás de una sola dirección) y al menos un **modelo .gguf**."
    ).classes("text-gray-700 dark:text-gray-300")

    async def refresh() -> None:
        container.clear()
        with container:
            ui.spinner(size="lg")
        checks = await run.io_bound(check_all, state.cfg)
        container.clear()
        with container:
            for c in checks:
                _card(c)
        errors = sum(c.status == "error" for c in checks)
        warns = sum(c.status == "warn" for c in checks)
        summary.set_text("✅ Todo listo." if not errors and not warns else
                         f"{errors} requisito(s) sin cumplir y {warns} aviso(s). Resuélvelos de arriba "
                         "abajo: cada paso depende del anterior.")

    with ui.row():
        ui.button("Volver a comprobar", icon="refresh", on_click=refresh)
        ui.button("Ir a Modelos", icon="arrow_forward",
                  on_click=lambda: state.navigate("models")).props("flat")
    summary = ui.label().classes("font-medium")
    container = ui.column().classes("w-full gap-3")
    ui.timer(0.2, refresh, once=True)
