"""Página «Modelos»: catálogo explicado, descargas y asignación de roles."""

from __future__ import annotations

import webbrowser

from nicegui import background_tasks, ui

from ..config import MODELS_DIR, ModelEntry, name_from_file
from ..server import vram
from ..server.downloader import DownloadError, DownloadState, download, hf_page
from .state import DEFAULT_CTX, AppState

ROLE_INFO = {
    "main": ("Principal", "Lleva el agente: decide, usa herramientas y escribe código. El más capaz que quepa."),
    "fast": ("Rápido", "Tareas auxiliares, sobre todo resumir la conversación cuando el contexto se llena."),
    "draft": ("Borrador", "Opcional. Acelera al principal con decodificación especulativa (misma familia)."),
}


def _fmt_bytes(n: float) -> str:
    return f"{n / 1024 ** 3:.2f} GB" if n >= 1024 ** 3 else f"{n / 1024 ** 2:.0f} MB"


def build(state: AppState) -> None:
    progress_widgets: dict[str, tuple[ui.linear_progress, ui.label]] = {}

    ui.label("Modelos").classes("text-2xl font-bold")
    ui.markdown(
        "Cada modelo es un archivo **.gguf**. El agente usa hasta tres **roles**; asigna un modelo "
        "a cada uno. Tras cambiar roles, **reinicia el servidor** para aplicarlo. La VRAM estimada "
        "incluye los pesos y la memoria del contexto por defecto del rol."
    ).classes("text-gray-700 dark:text-gray-300")

    @ui.refreshable
    def roles_view() -> None:
        with ui.row().classes("w-full gap-3"):
            for role, (label, desc) in ROLE_INFO.items():
                name = state.cfg.roles.get(role) or ""
                entry = state.cfg.model(name) if name else None
                ok = entry is not None and entry.path.is_file()
                with ui.card().classes("grow basis-60"):
                    with ui.row().classes("items-center w-full no-wrap"):
                        ui.badge(role, color="primary")
                        ui.label(label).classes("font-semibold")
                        ui.space()
                        if name:
                            ui.button(icon="close", on_click=lambda r=role: unassign(r)) \
                                .props("flat round dense size=sm") \
                                .tooltip(f"Quitar «{name}» del rol {role}")
                    ui.label(name or "— sin asignar —").classes(
                        "font-mono text-sm " + ("" if ok or not name else "text-negative"))
                    if name and not ok:
                        ui.label("archivo no descargado").classes("text-xs text-negative")
                    ui.label(desc).classes("text-xs text-gray-500")

    def assign(role: str, item: dict | None = None, filename: str | None = None) -> None:
        if item is not None:
            state.ensure_entry(item["id"], item["file"], role, item.get("defaults"))
            state.assign_role(role, item["id"])
        elif filename:
            name = name_from_file(filename)
            state.ensure_entry(name, filename, role)
            state.assign_role(role, name)
        warnings = state.save_config()
        ui.notify(f"Asignado a «{role}». Reinicia el servidor para aplicarlo.", type="positive")
        for w in warnings:
            ui.notify(w, type="warning")
        roles_view.refresh()
        catalog_view.refresh()
        local_view.refresh()

    def unassign(role: str) -> None:
        name = state.cfg.roles.get(role) or ""
        if not name:
            return
        state.assign_role(role, "")
        state.save_config()  # avisa a las demás páginas, que refrescan sus vistas
        if role == "main":
            ui.notify(f"«{name}» ya no es el modelo principal. El chat no funcionará hasta que "
                      "asignes otro a «main».", type="warning", multi_line=True)
        else:
            ui.notify(f"Quitado «{name}» del rol {role}. Reinicia el servidor para aplicarlo.")

    def held_roles(model_name: str) -> list[str]:
        return [r for r in ROLE_INFO if state.cfg.roles.get(r) == model_name]

    def role_chips(model_name: str) -> None:
        """Una etiqueta por cada rol que ocupa el modelo, con ✕ para quitárselo."""
        for r in held_roles(model_name):
            ui.chip(r, icon="star" if r == "main" else None, removable=True,
                    on_value_change=lambda e, r=r: None if e.value else unassign(r)) \
                .props("dense").tooltip(f"Quitar del rol {r}")

    def start_download(repo: str, filename: str) -> None:
        current = state.downloads.get(filename)
        if current and current.status == "descargando":
            return
        st = DownloadState(repo, filename)
        state.downloads[filename] = st
        client = ui.context.client  # la tarea corre fuera del contexto del clic

        async def job() -> None:
            error = None
            try:
                await download(st)
            except DownloadError as e:
                error = str(e)
            with client:
                if error:
                    ui.notify(error, type="negative", multi_line=True, timeout=15000)
                elif st.status == "completado":
                    ui.notify(f"Descargado {filename}", type="positive")
                catalog_view.refresh()
                local_view.refresh()

        background_tasks.create(job())
        catalog_view.refresh()
        local_view.refresh()

    def progress_block(filename: str) -> None:
        st = state.downloads.get(filename)
        if st and st.status == "descargando":
            bar = ui.linear_progress(value=st.fraction, show_value=False).classes("w-full")
            lbl = ui.label().classes("text-xs text-gray-500")
            progress_widgets[filename] = (bar, lbl)
            ui.button("Cancelar", icon="close", on_click=lambda: st.cancel.set()).props("flat dense")
        elif st and st.status == "error":
            ui.label(st.error).classes("text-xs text-negative")

    @ui.refreshable
    def catalog_view() -> None:
        progress_widgets.clear()
        for item in state.catalog:
            path = MODELS_DIR / item["file"]
            downloaded = path.is_file()
            role = item.get("role", "main")
            defaults = item.get("defaults", {})
            est_ctx = defaults.get("ctx", DEFAULT_CTX.get(role, 16384))
            est = vram.estimate_entry(ModelEntry(item["id"], item["file"], ctx=est_ctx,
                                                 kv_type=defaults.get("kv_type", "q8_0")), state.catalog)
            with ui.card().classes("w-full"):
                with ui.row().classes("items-center w-full"):
                    ui.label(item["name"]).classes("text-lg font-semibold")
                    ui.badge(item.get("params", ""), color="grey-7")
                    ui.badge(item.get("quant", ""), color="grey-7")
                    ui.badge(f"{item.get('size_gb', '?')} GB", color="grey-7")
                    ui.badge("usa herramientas ✓" if item.get("tools") else "sin herramientas",
                             color="positive" if item.get("tools") else "grey-5")
                    ui.badge(f"rol sugerido: {role}", color="primary").props("outline")
                    ui.space()
                    if downloaded:
                        ui.label("✔ descargado").classes("text-positive text-sm")
                ui.markdown(item.get("summary", "")).classes("text-sm")
                with ui.row().classes("gap-1"):
                    for g in item.get("good_for", []):
                        ui.chip(g, icon="check").props("dense outline")
                ui.label(f"Memoria estimada con {est_ctx // 1024}K de contexto: "
                         f"≈{est.total_gb:.1f} GB ({est.weights_gb:.1f} pesos + {est.kv_gb:.1f} contexto + "
                         f"{est.overhead_gb:.1f} margen) · licencia {item.get('license', '?')}") \
                    .classes("text-xs text-gray-500")
                progress_block(item["file"])
                with ui.row().classes("items-center"):
                    if not downloaded:
                        ui.button("Descargar", icon="download",
                                  on_click=lambda i=item: start_download(i["repo"], i["file"]))
                    role_chips(item["id"])
                    held = held_roles(item["id"])
                    for r in ("main", "fast", "draft"):
                        if r in held:
                            continue
                        ui.button(f"Usar como {r}", on_click=lambda r=r, i=item: assign(r, i)) \
                            .props("flat dense" if r != role else "dense outline") \
                            .set_enabled(downloaded)
                    ui.button("Hugging Face", icon="open_in_new",
                              on_click=lambda i=item: webbrowser.open(hf_page(i["repo"]))).props("flat dense")

    @ui.refreshable
    def local_view() -> None:
        catalog_files = {c["file"] for c in state.catalog}
        for filename, st in state.downloads.items():  # descargas manuales en curso
            if filename not in catalog_files and st.status in ("descargando", "error"):
                ui.label(f"{filename} ({st.repo})").classes("font-mono text-sm")
                progress_block(filename)
        known = {m.file.lower() for m in state.cfg.models} | {c["file"].lower() for c in state.catalog}
        files = sorted(MODELS_DIR.glob("*.gguf")) if MODELS_DIR.is_dir() else []
        extra = [f for f in files if f.name.lower() not in known]
        if not extra:
            ui.label("No hay otros .gguf en models/.").classes("text-sm text-gray-500")
        for f in extra:
            with ui.row().classes("items-center w-full"):
                ui.icon("description")
                ui.label(f"{f.name} ({_fmt_bytes(f.stat().st_size)})").classes("font-mono text-sm grow")
                role_chips(name_from_file(f.name))
                held = held_roles(name_from_file(f.name))
                for r in ("main", "fast", "draft"):
                    if r not in held:
                        ui.button(f"Usar como {r}", on_click=lambda r=r, f=f: assign(r, filename=f.name)) \
                            .props("flat dense")

    def tick() -> None:
        for filename, (bar, lbl) in list(progress_widgets.items()):
            st = state.downloads.get(filename)
            if not st or bar.is_deleted:
                progress_widgets.pop(filename, None)
                continue
            bar.set_value(st.fraction)
            speed = f" · {st.speed / 1024 ** 2:.1f} MB/s" if st.speed else ""
            lbl.set_text(f"{_fmt_bytes(st.done)} de {_fmt_bytes(st.total)} ({st.fraction:.0%}){speed}")

    ui.label("Roles actuales").classes("text-lg font-semibold mt-2")
    roles_view()
    ui.label("Catálogo recomendado para tu equipo").classes("text-lg font-semibold mt-4")
    catalog_view()
    ui.label("Otros modelos en la carpeta models/").classes("text-lg font-semibold mt-4")
    local_view()
    with ui.expansion("Descargar otro modelo de Hugging Face", icon="add").classes("w-full mt-2"):
        ui.markdown("Busca un repo con archivos **GGUF** (p. ej. de *bartowski* o *unsloth*) y copia el "
                    "nombre exacto del archivo. Para 16 GB de VRAM, busca modelos de hasta ~14B en Q4_K_M. "
                    "Comprueba en su página que el modelo soporta *tool calling*.").classes("text-sm")
        with ui.row().classes("w-full items-end"):
            repo_in = ui.input("Repositorio", placeholder="bartowski/Qwen2.5-Coder-7B-Instruct-GGUF") \
                .classes("grow")
            file_in = ui.input("Archivo", placeholder="Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf").classes("grow")

            def custom() -> None:
                if repo_in.value.strip() and file_in.value.strip().endswith(".gguf"):
                    start_download(repo_in.value.strip(), file_in.value.strip())
                    ui.notify("Descarga iniciada: el progreso aparece en «Otros modelos».")
                else:
                    ui.notify("Indica el repositorio y un archivo .gguf", type="warning")

            ui.button("Descargar", icon="download", on_click=custom)
    ui.timer(0.5, tick)
    state.on_config_saved(lambda: (roles_view.refresh(), catalog_view.refresh(), local_view.refresh()))
