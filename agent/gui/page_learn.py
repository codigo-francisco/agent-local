"""Página «Aprende»: explicaciones de los conceptos (contenido en learn.md)."""

from __future__ import annotations

from pathlib import Path

from nicegui import ui

from .state import AppState

LEARN_FILE = Path(__file__).with_name("learn.md")


def build(state: AppState) -> None:  # noqa: ARG001 - misma firma que el resto de páginas
    ui.label("Aprende").classes("text-2xl font-bold")
    ui.label("Conceptos básicos para entender y ajustar tu agente local.").classes("text-gray-600")
    text = LEARN_FILE.read_text(encoding="utf-8")
    sections = [s for s in text.split("\n## ") if s.strip()]
    for i, section in enumerate(sections):
        title, _, body = section.removeprefix("## ").partition("\n")
        with ui.expansion(title.strip(), value=i == 0).classes("w-full border rounded"):
            ui.markdown(body.strip(), extras=["fenced-code-blocks", "tables"]).classes("text-sm")
