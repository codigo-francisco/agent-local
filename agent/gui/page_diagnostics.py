"""Página «Diagnóstico»: autochequeo con arreglos sugeridos y exportación para pedir ayuda."""

from __future__ import annotations

from nicegui import run, ui

from ..server import diagnostics
from .widgets import open_folder
from .state import AppState

ICONS = {"ok": ("check_circle", "positive"), "warn": ("warning", "warning"), "fail": ("cancel", "negative")}


def build(state: AppState) -> None:
    last: dict = {"checks": []}

    ui.label("Diagnóstico").classes("text-2xl font-bold")
    ui.markdown("Comprueba en orden lo que suele fallar (binarios, modelos, VRAM, servidor, MCP…) "
                "y te dice cómo arreglarlo.").classes("text-gray-700 dark:text-gray-300")
    with ui.row():
        check_btn = ui.button("Comprobar", icon="troubleshoot")
        export_btn = ui.button("Exportar informe (.zip)", icon="archive").props("flat") \
            .tooltip("Informe + logs + configuración, sin cabeceras ni variables de entorno de MCP")
    results = ui.column().classes("w-full gap-1")

    async def do_check() -> None:
        check_btn.props("loading")
        try:
            checks = await diagnostics.run_checks(state.cfg, state.catalog, state.manager,
                                                  state.llm, state.mcp)
        except Exception as e:  # noqa: BLE001 - el diagnóstico no debe romper la página
            ui.notify(f"El diagnóstico falló: {type(e).__name__}: {e}", type="negative")
            return
        finally:
            check_btn.props(remove="loading")
        last["checks"] = checks
        results.clear()
        with results:
            for c in checks:
                icon, color = ICONS.get(c.status, ICONS["warn"])
                with ui.row().classes("items-start no-wrap w-full"):
                    ui.icon(icon, color=color, size="20px")
                    with ui.column().classes("gap-0"):
                        ui.label(c.name).classes("font-medium text-sm")
                        if c.detail:
                            ui.label(c.detail).classes("text-xs text-gray-600 dark:text-gray-400 whitespace-pre-wrap")
                        if c.fix:
                            ui.label(f"→ {c.fix}").classes("text-xs")
        fails = sum(c.status == "fail" for c in checks)
        ui.notify("Todo en orden." if not fails else f"{fails} problema(s) encontrado(s).",
                  type="positive" if not fails else "warning")

    async def do_export() -> None:
        if not last["checks"]:
            await do_check()
        path = await run.io_bound(diagnostics.export_bundle, last["checks"])
        ui.notify(f"Guardado en {path}", type="positive", multi_line=True)
        open_folder(path.parent)

    check_btn.on_click(do_check)
    export_btn.on_click(do_export)
