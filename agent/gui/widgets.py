"""Componentes compartidos entre páginas."""

from __future__ import annotations

from nicegui import app, ui

from .state import AppState


def workspace_picker(state: AppState, label: str = "Carpeta del proyecto (workspace)") -> ui.input:
    """Campo de carpeta que se aplica al momento (Enter, al salir del campo o con «Elegir…»),
    valida que exista y se mantiene sincronizado con el resto de páginas."""
    with ui.row().classes("w-full items-end no-wrap"):
        field = ui.input(label, value=state.cfg.workspace).classes("grow") \
            .tooltip("Se aplica al pulsar Enter o al salir del campo. Cambiarla empieza una "
                     "conversación nueva.")

        def commit() -> None:
            value = (field.value or "").strip()
            if value == state.cfg.workspace:
                return
            error = state.set_workspace(value)
            if error:
                ui.notify(error, type="negative")
                field.set_value(state.cfg.workspace)
            else:
                ui.notify(f"Carpeta del proyecto: {state.cfg.workspace}", type="positive")

        async def pick() -> None:
            window = getattr(app.native, "main_window", None)
            if window is None:
                ui.notify("El selector solo funciona en la ventana nativa; escribe la ruta y pulsa Enter.")
                return
            import webview
            kind = getattr(getattr(webview, "FileDialog", None), "FOLDER", None) or webview.FOLDER_DIALOG
            result = await window.create_file_dialog(kind, directory=state.cfg.workspace)
            if result:
                field.set_value(result[0])
                commit()

        field.on("keydown.enter", commit)
        field.on("blur", commit)
        ui.button("Elegir…", icon="folder_open", on_click=pick).props("flat")
    state.on_config_saved(lambda: field.set_value(state.cfg.workspace))
    return field
