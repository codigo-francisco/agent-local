"""Página «Servidor»: arrancar/parar llama-swap, modelos cargados, VRAM y registro en vivo."""

from __future__ import annotations

from nicegui import run, ui

from ..log import LOG_DIR
from ..server import rpc, vram
from ..server.manager import ServerError
from .state import AppState
from .widgets import open_folder

def build(state: AppState) -> None:
    cfg = state.cfg
    log_index = {"i": 0}

    ui.label("Servidor").classes("text-2xl font-bold")
    ui.markdown(
        "**llama-swap** escucha en un único puerto y arranca un proceso **llama-server** por modelo "
        "cuando se le pide. El primer mensaje tras arrancar tarda más: el modelo se está copiando a "
        "la VRAM (de segundos a un minuto)."
    ).classes("text-gray-700 dark:text-gray-300")

    with ui.card().classes("w-full"):
        with ui.row().classes("items-center w-full"):
            status_icon = ui.icon("circle", size="18px")
            status_label = ui.label().classes("font-semibold")
            ui.space()
            vram_label = ui.label().classes("text-sm text-gray-500")
        vram_bar = ui.linear_progress(value=0, show_value=False).classes("w-full")
        running_label = ui.label().classes("text-sm")
        remote_label = ui.label().classes("text-sm")
        with ui.row().classes("items-center no-wrap w-full text-negative") as crash_row:
            ui.icon("report", color="negative")
            crash_label = ui.label().classes("text-sm")
        crash_row.set_visibility(False)
        with ui.row():
            start_btn = ui.button("Arrancar", icon="play_arrow")
            stop_btn = ui.button("Parar", icon="stop", color="negative")
            restart_btn = ui.button("Reiniciar", icon="restart_alt").props("flat")
            preload_btn = ui.button("Cargar modelo principal ahora", icon="download_for_offline").props("flat")
            unload_btn = ui.button("Liberar VRAM", icon="eject").props("flat") \
                .tooltip("Descarga todos los modelos de la GPU sin parar el servidor")

    with ui.row().classes("items-center w-full mt-2"):
        ui.label("Registro").classes("text-lg font-semibold")
        ui.space()
        ui.button("Abrir carpeta de logs", icon="folder_open",
                  on_click=lambda: open_folder(LOG_DIR)).props("flat dense") \
            .tooltip("Registro de la app (agent.log) y de los servidores MCP, para diagnosticar fallos")
    log = ui.log(max_lines=2000).classes("w-full h-96 font-mono text-xs")

    def do_start() -> None:
        try:
            warnings = state.manager.start(cfg)
        except ServerError as e:
            ui.notify(str(e), type="negative", multi_line=True, timeout=10000)
            return
        for w in warnings:
            ui.notify(w, type="warning", multi_line=True)
        ui.notify("Servidor arrancando…")

    async def do_stop() -> None:
        await run.io_bound(state.manager.stop)
        ui.notify("Servidor parado; VRAM liberada.")

    async def do_restart() -> None:
        await run.io_bound(state.manager.stop)
        do_start()

    async def do_preload() -> None:
        main = cfg.roles.get("main")
        if not main:
            ui.notify("No hay modelo principal asignado.", type="warning")
            return
        preload_btn.props("loading")
        ok, detail = await state.manager.preload(cfg, main)
        preload_btn.props(remove="loading")
        if ok:
            ui.notify(f"«{main}» cargado en la GPU.", type="positive")
        else:
            ui.notify(f"No se pudo cargar «{main}». Mira el registro. {detail[:200]}",
                      type="negative", multi_line=True, timeout=10000)

    async def do_unload() -> None:
        ok = await state.manager.unload_all(cfg)
        ui.notify("Modelos descargados de la GPU." if ok else "No se pudo (¿servidor parado?).")

    start_btn.on_click(do_start)
    stop_btn.on_click(do_stop)
    restart_btn.on_click(do_restart)
    preload_btn.on_click(do_preload)
    unload_btn.on_click(do_unload)

    def pull_log() -> None:
        lines, log_index["i"] = state.manager.lines_since(log_index["i"])
        for line in lines:
            log.push(line)

    async def poll_status() -> None:
        info = await state.manager.status(cfg)
        own = state.manager.running
        if info["reachable"]:
            status_icon.props("color=positive")
            status_label.set_text("En marcha" + ("" if own else " (iniciado fuera de esta app)"))
        elif own:
            status_icon.props("color=warning")
            status_label.set_text("Arrancando…")
        else:
            status_icon.props("color=grey")
            status_label.set_text("Parado")
        crash_row.set_visibility(bool(state.manager.crashed))
        crash_label.set_text(state.manager.crashed or "")
        running = [f"{r.get('model')} ({r.get('state', '?')})" for r in info["running"]]
        running_label.set_text("Cargados en GPU: " + (", ".join(running) if running else "ninguno"))
        start_btn.set_enabled(not own and not info["reachable"])
        stop_btn.set_enabled(own)
        restart_btn.set_enabled(own)
        gpu = await run.io_bound(vram.gpu_info)
        if gpu:
            vram_bar.set_value(gpu.used_gb / gpu.total_gb)
            vram_label.set_text(f"VRAM: {gpu.used_gb:.1f} / {gpu.total_gb:.1f} GB")

    async def poll_remote() -> None:
        # Lento a propósito: el trabajador RPC atiende a un cliente a la vez y encola el resto.
        workers = [w for w in cfg.rpc_workers if w.enabled]
        remote_label.set_visibility(bool(workers))
        if not workers:
            return
        states = []
        for w in workers:
            ok = await run.io_bound(rpc.reachable, w.host, w.port)
            states.append(f"{w.name or w.endpoint} {'conectada' if ok else 'NO responde'}")
        remote_label.set_text("PCs remotas: " + ", ".join(states))

    ui.timer(0.5, pull_log)
    ui.timer(3.0, poll_status)
    ui.timer(0.5, poll_remote, once=True)
    ui.timer(30.0, poll_remote)
