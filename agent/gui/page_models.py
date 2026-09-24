"""Página «Modelos»: catálogo explicado, descargas y asignación de roles."""

from __future__ import annotations

import time
import webbrowser

from nicegui import background_tasks, run, ui

from ..config import MODELS_DIR, ModelEntry, name_from_file
from ..server import autotune, recommend, rpc, vram
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
    # Barras de progreso por archivo (el mismo puede salir en el catálogo y en recomendaciones).
    progress_widgets: dict[str, list[tuple[ui.linear_progress, ui.label]]] = {}

    ui.label("Modelos").classes("text-2xl font-bold")
    ui.markdown(
        "Cada modelo es un archivo **.gguf**. El agente usa hasta tres **roles**; asigna un modelo "
        "a cada uno. Tras cambiar roles, **reinicia el servidor** para aplicarlo. En cada modelo verás "
        "cuánta **VRAM** (memoria de la tarjeta gráfica) necesita y si cabe en tu equipo: tu tarjeta, "
        "más la de las PCs remotas (se suman) y, si hace falta, la **RAM** del PC (más lento)."
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
            name = entry_name(filename)
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

    def entry_name(filename: str) -> str:
        """Nombre con el que está configurado ese archivo (o el que tendría al añadirlo)."""
        entry = next((m for m in state.cfg.models if m.path.name.lower() == filename.lower()), None)
        return entry.name if entry else name_from_file(filename)

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
                    # Que aparezca ya en Configuración → Modelos y roles (y en los selectores de rol).
                    state.sync_downloaded_models(forget=filename)
                    ui.notify(f"Descargado {filename}. Ya está en Configuración: asígnale un rol o "
                              "pulsa «Recalcular».", type="positive", multi_line=True)
                catalog_view.refresh()
                local_view.refresh()
                recommendations_view.refresh()

        background_tasks.create(job())
        catalog_view.refresh()
        local_view.refresh()
        recommendations_view.refresh()

    def progress_block(filename: str) -> None:
        st = state.downloads.get(filename)
        if st and st.status == "descargando":
            bar = ui.linear_progress(value=st.fraction, show_value=False).classes("w-full")
            lbl = ui.label().classes("text-xs text-gray-500")
            progress_widgets.setdefault(filename, []).append((bar, lbl))
            ui.button("Cancelar", icon="close", on_click=lambda: st.cancel.set()).props("flat dense")
        elif st and st.status == "error":
            ui.label(st.error).classes("text-xs text-negative")

    hw_box: dict = {"hw": None}  # (VRAM local, VRAM remota, RAM permitida) en GB

    def fit_for(need_gb: float, weights_gb: float, item: dict) -> str:
        """Dónde correría en este equipo (frase), con la misma cuenta y la misma preferencia de
        cálculo que las recomendaciones; "" si aún no se conoce el hardware."""
        if not hw_box["hw"]:
            return ""
        local, remote, ram = hw_box["hw"]
        moe = bool((vram.arch_from_catalog(item) or vram.ArchInfo(1, 1, 1, 1)).moe)
        params = (autotune._params_from_name(str(item.get("params", "")))
                  or autotune._params_from_name(item["file"]) or weights_gb * 1.8)
        fit = recommend._fit(weights_gb * 1.0737, params, moe, local, remote, ram, state.cfg.tune_mode)
        if fit is None:
            return f"NO cabe: necesita ≈{need_gb:.0f} GB y no hay tanta VRAM ni RAM permitida"
        return fit[2]

    @ui.refreshable
    def catalog_view() -> None:
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
                ui.label(f"Necesita ≈{est.total_gb:.1f} GB de VRAM con {est_ctx // 1024}K de contexto: "
                         f"{est.weights_gb:.1f} GB el modelo + {est.kv_gb:.1f} GB la conversación + "
                         f"{est.overhead_gb:.1f} GB de margen · licencia {item.get('license', '?')}") \
                    .classes("text-xs text-gray-500")
                where = fit_for(est.total_gb, est.weights_gb, item)
                if where:
                    ui.label(f"En tu equipo: {where}.").classes("text-xs")
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
        # Todos los descargados que no son del catálogo (configurados o no: si no, los descargados
        # desde «Recomendaciones» desaparecían de aquí al añadirse a la configuración).
        known = {c["file"].lower() for c in state.catalog}
        files = sorted(MODELS_DIR.glob("*.gguf")) if MODELS_DIR.is_dir() else []
        extra = [f for f in files if f.name.lower() not in known]
        if not extra:
            ui.label("No hay otros .gguf en models/.").classes("text-sm text-gray-500")
        for f in extra:
            name = entry_name(f.name)
            with ui.row().classes("items-center w-full"):
                ui.icon("description")
                ui.label(f"{f.name} ({_fmt_bytes(f.stat().st_size)})").classes("font-mono text-sm grow")
                if state.cfg.model(name) is None:
                    ui.badge("no configurado", color="grey-6").props("outline") \
                        .tooltip("Lo quitaste de la configuración: asígnale un rol para volver a usarlo")
                role_chips(name)
                held = held_roles(name)
                for r in ("main", "fast", "draft"):
                    if r not in held:
                        ui.button(f"Usar como {r}", on_click=lambda r=r, f=f: assign(r, filename=f.name)) \
                            .props("flat dense")

    def tick() -> None:
        for filename, widgets in list(progress_widgets.items()):
            st = state.downloads.get(filename)
            alive = [(bar, lbl) for bar, lbl in widgets if not bar.is_deleted]
            if not st or not alive:
                progress_widgets.pop(filename, None)
                continue
            progress_widgets[filename] = alive
            speed = f" · {st.speed / 1024 ** 2:.1f} MB/s" if st.speed else ""
            for bar, lbl in alive:
                bar.set_value(st.fraction)
                lbl.set_text(f"{_fmt_bytes(st.done)} de {_fmt_bytes(st.total)} ({st.fraction:.0%}){speed}")

    # --- recomendaciones de Hugging Face ------------------------------------------
    rec_box: dict = {"report": recommend.load(), "running": False, "msg": ""}

    @ui.refreshable
    def recommendations_view() -> None:
        report: recommend.Report | None = rec_box["report"]
        if rec_box["running"]:
            with ui.row().classes("items-center no-wrap"):
                ui.spinner(size="md")
                rec_step = ui.label(rec_box["msg"] or "Recargando recomendaciones…").classes("text-sm")
            ui.timer(0.2, lambda: rec_step.set_text(rec_box["msg"]))
            return
        if report is None:
            ui.label("Pulsa «Recargar recomendaciones» para buscar en Hugging Face modelos nuevos que "
                     "sirvan para programar con el agente en tu hardware.").classes("text-sm text-gray-500")
            return
        if report.error:
            ui.label(report.error).classes("text-sm text-negative")
            return
        when = time.strftime("%d/%m/%Y %H:%M", time.localtime(report.updated)) if report.updated else "?"
        ui.label(f"Para {report.hardware} · {report.inspected} repos revisados el {when}.") \
            .classes("text-xs text-gray-500")
        for r in report.items:
            with ui.card().classes("w-full p-3"):
                with ui.row().classes("items-center w-full gap-2"):
                    ui.label(r.name).classes("font-semibold")
                    ui.badge(f"{r.params_b:g}B" + (" MoE" if r.moe else ""), color="grey-7")
                    ui.badge(r.quant, color="grey-7")
                    ui.badge(f"descarga {r.size_gb:.1f} GB", color="grey-7")
                    ui.badge("usa herramientas ✓", color="positive")
                    if r.coding:
                        ui.badge("código", color="primary").props("outline")
                    if r.thinking:
                        ui.badge("razona", color="primary").props("outline")
                    ui.space()
                    downloaded = (MODELS_DIR / r.file).is_file()
                    if downloaded:
                        ui.label("✔ descargado").classes("text-positive text-sm")
                speed = ("rápida" if r.speed >= 0.9 else "buena" if r.speed >= 0.7
                         else "lenta" if r.speed >= 0.4 else "muy lenta")
                ui.label(f"En tu equipo: {r.placement}.").classes("text-xs")
                ui.label(f"Velocidad estimada: {speed} ({r.speed:.0%} de lo que iría entero en VRAM) · "
                         f"publicado {r.created or '?'} · {r.downloads:,} descargas".replace(",", ".")) \
                    .classes("text-xs text-gray-500")
                progress_block(r.file)
                with ui.row().classes("items-center"):
                    if not downloaded:
                        ui.button("Descargar", icon="download",
                                  on_click=lambda r=r: start_download(r.repo, r.file)).props("dense")
                    ui.label(f"{r.repo} · {r.file}").classes("text-xs font-mono text-gray-500")
                    ui.button("Hugging Face", icon="open_in_new",
                              on_click=lambda r=r: webbrowser.open(r.url)).props("flat dense")
        ui.label("Tras descargar uno, pulsa «Recalcular» en Configuración para asignarle rol y memoria.") \
            .classes("text-xs text-gray-500")

    async def reload_recommendations() -> None:
        if rec_box["running"]:
            return
        rec_box.update(running=True, msg="Leyendo el hardware disponible…")
        reload_btn.props("loading")
        recommendations_view.refresh()

        def say(msg: str) -> None:
            rec_box["msg"] = msg

        try:
            for warning in await run.io_bound(rpc.refresh_workers, state.cfg.rpc_workers,
                                              state.manager.running, say):
                ui.notify(warning, type="warning", multi_line=True)
            report = await run.io_bound(recommend.refresh, state.cfg, None, None, say)
        finally:
            rec_box["running"] = False
            reload_btn.props(remove="loading")
        rec_box["report"] = report
        recommendations_view.refresh()
        if report.error:
            ui.notify(report.error, type="negative", multi_line=True)
        else:
            ui.notify(f"{len(report.items)} modelos recomendados para tu hardware.", type="positive")

    def remote_list_changed() -> None:
        """Se añadió, quitó o midió una PC remota: la VRAM total cambió. Se rehacen las
        recomendaciones con los modelos ya revisados (sin volver a consultar Hugging Face)."""
        report = rec_box["report"]
        if report is None or rec_box["running"]:
            return
        client = ui.context.client
        if not report.candidates:  # caché de una versión anterior: hay que consultar de nuevo
            async def full_reload() -> None:
                with client:
                    await reload_recommendations()

            background_tasks.create(full_reload())
            return

        async def job() -> None:
            new = await run.io_bound(recommend.recompute, report, state.cfg)
            rec_box["report"] = new
            with client:
                recommendations_view.refresh()

        background_tasks.create(job())

    state.on_remote_changed(remote_list_changed)

    ui.label("Roles actuales").classes("text-lg font-semibold mt-2")
    roles_view()
    with ui.row().classes("items-center w-full mt-4"):
        ui.label("Recomendaciones para tu hardware").classes("text-lg font-semibold")
        ui.space()
        reload_btn = ui.button("Recargar recomendaciones", icon="travel_explore",
                               on_click=reload_recommendations).props("outline") \
            .tooltip("Busca en Hugging Face modelos GGUF recientes que llaman herramientas de forma "
                     "nativa y que tu llama.cpp sabe cargar, y elige la cuantización que mejor cabe en tu "
                     "GPU, tus PCs remotas y tu RAM.")
    recommendations_view()
    ui.label("Catálogo recomendado para tu equipo").classes("text-lg font-semibold mt-4")
    catalog_view()
    ui.label("Otros modelos descargados (fuera del catálogo)").classes("text-lg font-semibold mt-4")
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
    async def load_hardware() -> None:
        gpu = await run.io_bound(vram.gpu_info)
        hw_box["hw"] = recommend.hardware(state.cfg, gpu, vram.ram_total_gb())[:3] if gpu else None
        catalog_view.refresh()

    def remote_list_changed_catalog() -> None:
        if hw_box["hw"]:
            background_tasks.create(load_hardware())

    ui.timer(0.5, tick)
    ui.timer(0.3, load_hardware, once=True)
    state.on_remote_changed(remote_list_changed_catalog)
    state.on_config_saved(lambda: (roles_view.refresh(), catalog_view.refresh(), local_view.refresh()))
