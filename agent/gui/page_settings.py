"""Página «Configuración»: agente, modelos (contexto, KV cache, capas) y calculadora de VRAM."""

from __future__ import annotations

import asyncio
import json

from nicegui import background_tasks, run, ui

from ..config import KV_TYPES, ROLES, TUNE_MODES, RpcWorker
from ..core.mcp_tools import MCP_FILE
from ..server import autotune, rpc, vram
from .state import AppState
from .widgets import open_folder, workspace_picker

CTX_OPTIONS = [2048, 4096, 8192, 12288, 16384, 24576, 32768, 49152, 65536, 98304, 131072]
KV_LABELS = {"f16": "f16 (máxima calidad, ocupa el doble)",
             "q8_0": "q8_0 (recomendado: la mitad, casi igual de bueno)",
             "q4_0": "q4_0 (ocupa lo mínimo, algo menos preciso)"}
MCP_TEMPLATE = """{
  "mcpServers": {
  }
}
"""
COLORS = ["#4f46e5", "#0891b2", "#16a34a", "#d97706", "#db2777"]
PLACEMENT_OPTIONS = {"local": "Esta PC", "split": "Repartido con PC remota", "remote": "PC remota"}
PART_NAMES = {"pesos": "el modelo", "contexto": "la conversación (contexto)", "margen": "margen de trabajo"}
TUNE_MODE_INFO = {
    "local": ("Preferir lo local primero",
              "Usa tu tarjeta gráfica y, si no cabe, la RAM de esta PC. Las PCs remotas solo se usan "
              "cuando el modelo no cabe aquí ni con la RAM."),
    "vram": ("Preferir la VRAM total",
             "Suma la VRAM de las PCs remotas a la de tu tarjeta como si fuera una sola, y la usa antes "
             "que la RAM, aunque la red la haga algo más lenta."),
    "speed": ("Optimizar la velocidad",
              "Elige lo que haga responder más rápido al modelo principal, según medidas reales: por "
              "ejemplo, un modelo MoE va más rápido con una parte en la RAM que repartido por la red."),
}


def build(state: AppState) -> None:
    cfg = state.cfg
    gpu_box: dict = {"gpu": None}

    ui.label("Configuración").classes("text-2xl font-bold")
    if state.load_warning:
        with ui.card().classes("w-full border-l-4 border-amber-500"):
            with ui.row().classes("items-center no-wrap"):
                ui.icon("warning", color="warning")
                ui.label(state.load_warning).classes("text-sm")
    if state.safe_mode:
        with ui.card().classes("w-full border-l-4 border-sky-500"):
            with ui.row().classes("items-center no-wrap"):
                ui.icon("shield", color="primary")
                ui.label("Modo seguro: sin servidores MCP, sin modo automático y sin permisos "
                         "«siempre» (siguen guardados; vuelven al arrancar sin --safe).").classes("text-sm")

    # --- calculadora de VRAM ----------------------------------------------
    @ui.refreshable
    def calculator() -> None:
        plan = vram.plan_usage(cfg, state.catalog, gpu_box["gpu"])
        with ui.card().classes("w-full"):
            with ui.row().classes("items-center w-full"):
                ui.icon("memory", size="24px")
                ui.label("¿Caben tus modelos? (calculadora de memoria)").classes("text-lg font-semibold")
                ui.space()
                if plan.fits is None:
                    ui.badge("sin GPU detectada", color="grey")
                else:
                    ui.badge("cabe" if plan.fits else "NO cabe",
                             color="positive" if plan.fits else "negative")
            cap = plan.capacity_gb or max(plan.total_gb, 1)
            scale = max(cap, plan.total_gb)
            with ui.element("div").classes("w-full h-7 rounded flex overflow-hidden bg-gray-200 dark:bg-gray-700"):
                i = 0
                for m in plan.models:
                    for part, value in (("pesos", m.weights_gb - m.offload_gb), ("contexto", m.kv_gb),
                                        ("margen", m.overhead_gb)):
                        width = 100 * value / scale
                        opacity = {"pesos": 1.0, "contexto": 0.65, "margen": 0.35}[part]
                        ui.element("div").style(
                            f"width:{width}%;background:{COLORS[i % len(COLORS)]};opacity:{opacity}"
                        ).tooltip(f"{m.name} · {PART_NAMES[part]}: {value:.1f} GB")
                    i += 1
            with ui.row().classes("gap-4 text-xs text-gray-500"):
                for label, opacity in (("el modelo", 1.0), ("la conversación (contexto)", 0.65),
                                       ("margen de trabajo", 0.35)):
                    with ui.row().classes("items-center gap-1 no-wrap"):
                        ui.element("div").classes("w-3 h-3 rounded-sm") \
                            .style(f"background:{COLORS[0]};opacity:{opacity}")
                        ui.label(label)
                ui.label("· un color por modelo")
            gpu = gpu_box["gpu"]
            # Resumen en una frase: cuánto hace falta y cuánto hay.
            if gpu and plan.capacity_gb:
                verdict = "caben" if plan.fits else "NO caben"
                ui.label(f"Tus modelos necesitan {plan.total_gb:.1f} GB de VRAM (memoria de la tarjeta "
                         f"gráfica) y tienes {plan.capacity_gb:.1f} GB disponibles: {verdict}.") \
                    .classes("text-sm font-medium")
                local_gb = gpu.total_gb - vram.DESKTOP_RESERVE_GB
                remote = (f" + {plan.remote_gb:.1f} GB de las PCs remotas (se suman como si fueran una "
                          "sola tarjeta)" if plan.remote_gb else "")
                ui.label(f"VRAM disponible: {local_gb:.1f} GB de tu {gpu.name} (tiene {gpu.total_gb:.1f} "
                         f"GB; se dejan {vram.DESKTOP_RESERVE_GB} GB para Windows){remote}.") \
                    .classes("text-sm")
            else:
                ui.label(f"Tus modelos necesitan {plan.total_gb:.1f} GB de VRAM (memoria de la tarjeta "
                         "gráfica). No detecto la GPU, así que no sé si caben.").classes("text-sm")
            if plan.ram_offload_gb:
                limit = (f" Puedes usar hasta {plan.ram_budget_gb:.0f} GB de RAM para modelos"
                         + ("." if plan.ram_ok else ": lo supera, el PC puede quedarse sin memoria.")
                         if plan.ram_budget_gb else "")
                ui.label(f"Lo que no cabe en la VRAM ({plan.ram_offload_gb:.1f} GB del modelo) se queda "
                         f"en la RAM (memoria normal del PC): así funciona, pero más lento.{limit}") \
                    .classes("text-sm")
            if len(plan.models) > 1:
                ui.label("Modelos cargados a la vez: main y fast siempre listos, sin esperas."
                         if plan.concurrent else
                         "Modelos de uno en uno: se turnan en la VRAM (cabe más, pero hay una espera al "
                         "cambiar de uno a otro).").classes("text-sm")
            for m in plan.models:
                where = {"split": " (repartido con la PC remota)", "remote": " (en la PC remota)"} \
                    .get(m.where, "")
                with ui.column().classes("gap-0 mt-1"):
                    ui.label(f"{m.name}{where}: ocupa {m.total_gb:.1f} GB de VRAM") \
                        .classes("text-sm font-mono font-semibold")
                    in_ram = (f", de los que {m.offload_gb:.1f} GB van a la RAM" if m.offload_gb else "")
                    for line in (f"El modelo en sí: {m.weights_gb:.1f} GB{in_ram}.",
                                 f"La conversación (lo que el modelo «recuerda», el contexto): "
                                 f"{m.kv_gb:.1f} GB. Crece con el tamaño de contexto.",
                                 f"Margen de trabajo de llama.cpp: {m.overhead_gb:.1f} GB."):
                        ui.label(f"• {line}").classes("text-xs text-gray-500 ml-2")
                    if m.note:
                        ui.label(f"• {m.note[0].upper() + m.note[1:]}.").classes("text-xs text-gray-500 ml-2")
            for s in plan.suggestions:
                with ui.row().classes("items-center no-wrap"):
                    ui.icon("lightbulb", color="warning", size="18px")
                    ui.label(s).classes("text-sm")

    async def load_gpu() -> None:
        gpu_box["gpu"] = await run.io_bound(vram.gpu_info)
        calculator.refresh()

    def changed(*_) -> None:
        calculator.refresh()

    # --- modelos -------------------------------------------------------------
    @ui.refreshable
    def models_view() -> None:
        names = [m.name for m in cfg.models]
        with ui.row().classes("w-full gap-4"):
            for role in ROLES:
                options = {"": "— ninguno —", **{n: n for n in names}}
                ui.select(options, label=f"Rol {role}", value=cfg.roles.get(role) or "",
                          on_change=lambda e, r=role: (cfg.roles.__setitem__(r, e.value or ""), changed())) \
                    .classes("min-w-60")
        if not cfg.models:
            ui.label("No hay modelos configurados: asígnalos desde la página «Modelos».").classes("text-gray-500")
        for m in cfg.models:
            with ui.card().classes("w-full"):
                with ui.row().classes("items-center w-full"):
                    ui.label(m.name).classes("font-semibold font-mono")
                    ui.label(m.file + ("" if m.path.is_file() else "  (no descargado)")) \
                        .classes("text-xs " + ("text-gray-500" if m.path.is_file() else "text-negative"))
                    ui.space()
                    ui.button(icon="delete", on_click=lambda m=m: (state.remove_model(m.name),
                                                                   models_view.refresh(), changed())) \
                        .props("flat dense round").tooltip("Quitar de la configuración (no borra el archivo)")
                with ui.row().classes("w-full items-end gap-4"):
                    ui.select({c: f"{c // 1024}K" for c in CTX_OPTIONS} | ({m.ctx: f"{m.ctx}"} if m.ctx not in CTX_OPTIONS else {}),
                              label="Contexto (tokens)", value=m.ctx,
                              on_change=lambda e, m=m: (setattr(m, "ctx", int(e.value)), changed())) \
                        .classes("w-36").tooltip("Cuánto texto tiene en cuenta el modelo a la vez (1 token ≈ 4 letras): "
                                             "la conversación, los archivos leídos… Más contexto = recuerda "
                                             "más, pero ocupa más VRAM.")
                    ui.select({k: KV_LABELS[k] for k in KV_TYPES}, label="KV cache (memoria de la conversación)",
                              value=m.kv_type,
                              on_change=lambda e, m=m: (setattr(m, "kv_type", e.value), changed()))                         .classes("w-80").tooltip("Cómo se guarda en la VRAM lo que el modelo «recuerda» de "
                                                 "la conversación. Comprimirlo (q8_0) deja sitio para más "
                                                 "contexto sin notar diferencia.")
                    ui.number("Capas en GPU (-1 = auto)", value=m.gpu_layers, min=-1, max=999, step=1,
                              format="%d",
                              on_change=lambda e, m=m: (setattr(m, "gpu_layers", int(e.value if e.value is not None else -1)),
                                                        changed())) \
                        .classes("w-44").tooltip("Cuánto del modelo va a la VRAM (tarjeta gráfica). 99 = todo, "
                                                 "lo más rápido. -1 = automático: llama.cpp pone en la VRAM lo "
                                                 "que cabe y el resto en la RAM (ideal para modelos MoE).")
                    ui.input("Args extra de llama-server", value=m.extra_args,
                             on_change=lambda e, m=m: setattr(m, "extra_args", e.value or "")) \
                        .classes("grow").tooltip("Ej.: --n-cpu-moe 20 para modelos MoE que no caben")
                    ui.select(PLACEMENT_OPTIONS, label="Dónde corre", value=m.placement,
                              on_change=lambda e, m=m: (setattr(m, "placement", e.value or "local"),
                                                        changed())).classes("w-52") \
                        .tooltip("Esta PC: su GPU (y RAM). Repartido: suma la GPU de las PCs remotas "
                                 "(para modelos que no caben). PC remota: entero allí, deja tu GPU "
                                 "libre para el otro modelo. «Recalcular» lo decide por ti.")

    # --- agente -------------------------------------------------------------
    def agent_section() -> None:
        with ui.card().classes("w-full"):
            ui.label("Agente").classes("text-lg font-semibold")
            workspace_picker(state)
            with ui.row().classes("w-full items-end gap-4"):
                # bind_value: el campo refleja cualquier cambio de la config (p. ej. «Recalcular»).
                ui.toggle({"ask": "Pedir confirmación", "auto": "Automático"}).bind_value(cfg, "confirm") \
                    .tooltip("Automático: edita archivos y ejecuta comandos sin preguntar")
                for attr, label, lo, hi, tip in (
                    ("max_steps", "Máx. pasos por tarea", 1, 200, "Límite de llamadas al modelo por petición."),
                    ("max_output_tokens", "Tokens de salida", 256, 32768,
                     "Máximo por respuesta. Se reserva del contexto: más salida = menos historial."),
                    ("max_tool_output", "Máx. caracteres por resultado", 1000, 100000,
                     "Resultados más largos se recortan por el medio."),
                    ("command_timeout", "Timeout de comandos (s)", 5, 3600, ""),
                ):
                    ui.number(label, min=lo, max=hi, step=1, format="%d") \
                        .bind_value(cfg, attr, forward=lambda v, lo=lo: int(v) if v is not None else lo) \
                        .classes("w-48").tooltip(tip)

    def server_section() -> None:
        with ui.card().classes("w-full"):
            ui.label("Servidor").classes("text-lg font-semibold")
            with ui.row().classes("w-full items-end gap-4"):
                endpoint = ui.input("Endpoint (API compatible con OpenAI)", value=cfg.endpoint,
                                    on_change=lambda e: setattr(cfg, "endpoint", e.value or "")).classes("grow") \
                    .tooltip("Cambia esto para usar otro backend, p. ej. Ollama: http://127.0.0.1:11434/v1")

                def set_port(e) -> None:
                    cfg.port = int(e.value or 8080)
                    endpoint.set_value(f"http://127.0.0.1:{cfg.port}/v1")

                ui.number("Puerto de llama-swap", value=cfg.port, min=1024, max=65535, format="%d",
                          on_change=set_port).classes("w-44")
                ui.switch("Mantener main y fast cargados a la vez", on_change=changed) \
                    .bind_value(cfg, "keep_loaded") \
                    .tooltip("Evita recargas al resumir, pero suma la VRAM de ambos")
                ui.select({"on": "on", "auto": "auto", "off": "off"}, label="Flash attention") \
                    .bind_value(cfg, "flash_attn") \
                    .classes("w-36").tooltip("Necesaria para KV cache cuantizada (q8_0/q4_0)")
            ram = vram.ram_total_gb()
            with ui.row().classes("w-full items-center no-wrap gap-4 mt-2"):
                ram_label = ui.label().classes("text-sm w-80 shrink-0")

                def set_ram(value) -> None:
                    cfg.ram_limit_pct = int(value or 90)
                    gb = f" (≈{ram * cfg.ram_limit_pct / 100:.0f} de tus {ram:.0f} GB)" if ram else ""
                    ram_label.set_text(f"Límite de RAM para modelos: {cfg.ram_limit_pct} %{gb}")
                    changed()

                ui.slider(min=10, max=100, step=5, value=cfg.ram_limit_pct,
                          on_change=lambda e: set_ram(e.value)).props("label").classes("grow") \
                    .tooltip("Si un modelo no cabe en la VRAM (tarjeta gráfica), lo que sobra se queda en la "
                             "RAM (memoria normal del PC). Este límite evita que se coma toda la memoria: "
                             "la calculadora y «Recalcular» no pasarán de aquí. El resto queda para Windows "
                             "y tus programas (90 % por defecto).")
                set_ram(cfg.ram_limit_pct)

    # --- permisos «siempre» ----------------------------------------------------
    @ui.refreshable
    def permissions_view() -> None:
        tools = list(cfg.always_allow)
        if not tools:
            ui.label("Ninguna herramienta tiene permiso permanente: todas piden confirmación "
                     "(salvo las de solo lectura y los servidores MCP con autoApprove).") \
                .classes("text-sm text-gray-500")
            return
        with ui.row().classes("gap-2"):
            for tool in tools:
                ui.chip(tool, icon="done_all", removable=True,
                        on_value_change=lambda e, t=tool: None if e.value else revoke(t)) \
                    .props("outline").tooltip("Quitar: volverá a pedir confirmación")
        ui.button("Quitar todos", icon="remove_done", on_click=lambda: revoke(None)) \
            .props("flat dense color=negative")

    def revoke(tool: str | None) -> None:
        state.revoke_always(tool)
        ui.notify("Permiso quitado: volverá a pedir confirmación." if tool else
                  "Quitados todos los permisos permanentes.", type="positive")

    def permissions_section() -> None:
        with ui.card().classes("w-full"):
            with ui.row().classes("items-center"):
                ui.icon("verified_user", size="24px")
                ui.label("Permisos permanentes").classes("text-lg font-semibold")
            ui.label("Herramientas que aprobaste con «Aprobar siempre» en el chat. Los permisos «en "
                     "esta sesión» se olvidan al cerrar la app.").classes("text-sm")
            permissions_view()

    # --- PCs remotas (llama.cpp RPC) -------------------------------------------
    probe_results: dict[int, tuple[bool, str]] = {}  # id(worker) -> (ok, mensaje) del último «Probar»
    rpc_box: dict = {}  # tarjeta (contexto estable para notificar) y botón del paquete

    def remote_list_changed() -> None:
        """La lista de PCs remotas cambió: la VRAM total cambia en todas las páginas (calculadora,
        recomendaciones…)."""
        state.remote_changed()

    def set_host(w: RpcWorker, value: str | None) -> None:
        host = (value or "").strip()
        if host != w.host:
            had_vram = w.vram_gb > 0
            w.host, w.vram_gb = host, 0.0  # otra PC: la VRAM medida ya no vale
            if had_vram:
                remote_list_changed()
            probe_results.pop(id(w), None)

    async def probe_worker(w: RpcWorker) -> None:
        if not w.host:
            ui.notify("Escribe primero la IP de la otra PC.", type="warning")
            return
        probe_results[id(w)] = (True, "Probando…")
        rpc_view.refresh()  # borra el botón que lanzó el evento: a partir de aquí, contexto de la tarjeta
        res = await run.io_bound(rpc.probe, w)
        with rpc_box["card"]:
            if res.ok:
                w.vram_gb, w.devices = round(res.total_gb, 1), len(res.devices)
                probe_results[id(w)] = (True, res.summary())
                ui.notify(f"Conectado con {res.summary()}. Pulsa «Guardar».", type="positive",
                          multi_line=True)
            else:
                probe_results[id(w)] = (False, res.error)
            rpc_view.refresh()
            remote_list_changed()

    def add_worker() -> None:
        cfg.rpc_workers.append(RpcWorker(host=""))
        rpc_view.refresh()

    def remove_worker(w: RpcWorker) -> None:
        cfg.rpc_workers.remove(w)
        rpc_view.refresh()
        remote_list_changed()

    @ui.refreshable
    def rpc_view() -> None:
        if not cfg.rpc_workers:
            ui.label("No hay PCs remotas configuradas.").classes("text-sm text-gray-500")
        for w in cfg.rpc_workers:
            with ui.row().classes("w-full items-end no-wrap gap-4"):
                ui.switch(value=w.enabled,
                          on_change=lambda e, w=w: (setattr(w, "enabled", bool(e.value)), remote_list_changed())) \
                    .tooltip("Usar esta PC")
                ui.input("IP de la otra PC", value=w.host, placeholder="192.168.1.50",
                         on_change=lambda e, w=w: set_host(w, e.value)).classes("w-48")
                ui.number("Puerto", value=w.port, min=1, max=65535, step=1, format="%d",
                          on_change=lambda e, w=w: setattr(w, "port", int(e.value or 50052))).classes("w-28")
                ui.input("Nombre (opcional)", value=w.name,
                         on_change=lambda e, w=w: setattr(w, "name", e.value or "")).classes("w-40")
                ui.label(f"{w.vram_gb:.1f} GB de VRAM" if w.vram_gb else "VRAM sin medir") \
                    .classes("text-sm " + ("" if w.vram_gb else "text-gray-500"))
                ui.space()
                ui.button("Probar", icon="network_check", on_click=lambda w=w: probe_worker(w)).props("flat") \
                    .tooltip("Conecta con la otra PC y mide su VRAM (con el servidor parado)")
                ui.button(icon="delete", on_click=lambda w=w: remove_worker(w)).props("flat dense round") \
                    .tooltip("Quitar esta PC")
            result = probe_results.get(id(w))
            if result:
                ok, msg = result
                ui.label(msg).classes("text-xs ml-14 " + ("text-gray-500" if ok else "text-negative"))

    async def make_package() -> None:
        first = next((w.host for w in cfg.rpc_workers
                      if w.host and not w.host.startswith("127.") and w.host != "localhost"), None)
        main_ip = rpc.local_ip(first)
        button = rpc_box["package"]
        button.props("loading")
        ui.notify("Generando el paquete (≈1 min: las DLL de CUDA pesan ~0,5 GB)…")
        try:
            path = await run.io_bound(rpc.build_worker_package, main_ip)
        except OSError as e:  # incluye FileNotFoundError: falta ggml-rpc-server.exe
            ui.notify(f"No pude generar el paquete: {e}", type="negative", multi_line=True)
            return
        finally:
            button.props(remove="loading")
        ui.notify(f"Paquete listo: {path.name} ({path.stat().st_size / 1e9:.1f} GB). Cópialo a la otra "
                  "PC, descomprímelo y sigue el LEEME.txt.", type="positive", multi_line=True,
                  timeout=10000)
        open_folder(path.parent)

    def rpc_section() -> None:
        with ui.card().classes("w-full") as card:
            rpc_box["card"] = card
            with ui.row().classes("items-center w-full"):
                ui.icon("lan", size="24px")
                ui.label("PCs remotas (RPC)").classes("text-lg font-semibold")
                ui.space()
                ui.label(f"Esta PC: {rpc.local_ip()}").classes("text-sm text-gray-500")
            ui.markdown(
                "Suma la GPU de otra PC de tu red para modelos que no caben en esta. En la otra PC "
                "solo corre `ggml-rpc-server` de llama.cpp (sin Python, sin modelos: los pesos viajan "
                "por la red). **1)** Genera el paquete y ábrelo allí siguiendo su `LEEME.txt`. "
                "**2)** Añade su IP y pulsa **Probar**. **3)** Activa «PC remota» en el modelo (o "
                "pulsa **Recalcular**), **Guardar** y reinicia el servidor. Mejor con cable Gigabit."
            ).classes("text-sm")
            with ui.row().classes("items-center no-wrap"):
                ui.icon("gpp_maybe", color="warning")
                ui.label("El protocolo RPC no tiene contraseña ni cifrado: úsalo solo en tu red de casa "
                         "y no abras su puerto en el router.").classes("text-sm")
            rpc_view()
            with ui.row():
                ui.button("Añadir PC", icon="add", on_click=add_worker).props("flat")
                rpc_box["package"] = ui.button("Generar paquete para la otra PC", icon="archive",
                                               on_click=make_package).props("outline") \
                    .tooltip("Crea generated/rpc-worker.zip con ggml-rpc-server y las DLL de TU versión "
                             "de llama.cpp (deben coincidir en ambas PCs), más los .bat de arranque y "
                             "firewall.")

    # --- servidores MCP -------------------------------------------------------
    STATUS_ICON ={"conectado": ("check_circle", "positive"), "error": ("error", "negative"),
                   "conectando": ("sync", "primary"), "desactivado": ("block", "grey")}

    @ui.refreshable
    def mcp_status() -> None:
        if state.mcp_starting:
            with ui.row().classes("items-center"):
                ui.spinner(size="sm")
                ui.label("Conectando servidores MCP…").classes("text-sm")
        if state.mcp.load_error:
            ui.label(state.mcp.load_error).classes("text-sm text-negative")
        servers = state.mcp.summary()
        if not servers and not state.mcp_starting:
            ui.label("No hay servidores MCP configurados.").classes("text-sm text-gray-500")
        for s in servers:
            icon, color = STATUS_ICON.get(s["status"], ("help", "grey"))
            with ui.row().classes("items-center w-full no-wrap"):
                ui.icon(icon, color=color)
                ui.label(s["name"]).classes("font-mono font-semibold")
                ui.label(s["status"] + (f" · {len(s['tools'])} herramientas" if s["tools"] else "")) \
                    .classes("text-sm")
                if s["auto"]:
                    ui.badge("sin confirmación", color="warning").props("outline")
            if s["error"]:
                ui.label(s["error"]).classes("text-xs text-negative ml-8 break-all")
            if s["tools"]:
                ui.label(", ".join(s["tools"])).classes("text-xs text-gray-500 ml-8")

    def mcp_section() -> None:
        with ui.card().classes("w-full"):
            with ui.row().classes("items-center w-full"):
                ui.icon("extension", size="24px")
                ui.label("Servidores MCP").classes("text-lg font-semibold")
            ui.markdown(
                "Conecta herramientas externas mediante el **Model Context Protocol**. Mismo formato "
                "que Claude Desktop o Cursor: copia la configuración de la documentación del servidor. "
                "Usa `command`/`args`/`env` para servidores locales o `url`/`headers` para remotos. "
                "Añade `\"autoApprove\": true` para no pedir confirmación, o `\"disabled\": true` "
                "para desactivarlo. Se guarda en `config/mcp.json`."
            ).classes("text-sm")
            try:
                current = MCP_FILE.read_text(encoding="utf-8") if MCP_FILE.exists() else MCP_TEMPLATE
            except OSError:
                current = MCP_TEMPLATE
            editor = ui.textarea(value=current).props("outlined autogrow input-style='font-family: monospace'") \
                .classes("w-full font-mono text-xs")

            async def save_and_reconnect() -> None:
                try:
                    json.loads(editor.value or "{}")
                except ValueError as e:
                    ui.notify(f"JSON inválido: {e}", type="negative", multi_line=True)
                    return
                MCP_FILE.parent.mkdir(parents=True, exist_ok=True)
                MCP_FILE.write_text(editor.value, encoding="utf-8")
                mcp_status.refresh()
                task = background_tasks.create(state.restart_mcp())
                await asyncio.sleep(0.1)
                mcp_status.refresh()
                await task
                mcp_status.refresh()
                ok = sum(s["status"] == "conectado" for s in state.mcp.summary())
                tools = sum(len(s["tools"]) for s in state.mcp.summary())
                ui.notify(f"MCP: {ok} servidor(es) conectados, {tools} herramientas disponibles.",
                          type="positive" if ok else "warning")

            with ui.row():
                ui.button("Guardar y reconectar", icon="sync", on_click=save_and_reconnect)
            mcp_status()
            was_starting = {"v": state.mcp_starting}

            def poll_mcp() -> None:
                # Refresca mientras conecta y una vez más al terminar (si no, se queda en «conectando»).
                if state.mcp_starting or was_starting["v"]:
                    mcp_status.refresh()
                was_starting["v"] = state.mcp_starting

            ui.timer(1.0, poll_mcp)

    def save() -> None:
        warnings = state.save_config()
        ui.notify("Configuración guardada. Si el servidor está en marcha, reinícialo para aplicar "
                  "cambios de modelos o contexto.", type="positive", multi_line=True)
        for w in warnings:
            ui.notify(w, type="warning")

    # --- preferencia de cálculo ----------------------------------------------
    def tune_section() -> None:
        with ui.card().classes("w-full"):
            with ui.row().classes("items-center w-full gap-3"):
                ui.icon("tune", size="22px")
                ui.label("Preferencia de cálculo").classes("font-semibold")
                toggle = ui.toggle({k: TUNE_MODE_INFO[k][0] for k in TUNE_MODES}, value=cfg.tune_mode) \
                    .props("no-caps")
            explain = ui.label(TUNE_MODE_INFO[cfg.tune_mode][1]).classes("text-sm text-gray-500")

            def set_mode(e) -> None:
                if e.value == cfg.tune_mode:
                    return
                cfg.tune_mode = e.value
                explain.set_text(TUNE_MODE_INFO[e.value][1])
                state.persist()  # se guarda al momento
                state.remote_changed()  # las recomendaciones se rehacen con la nueva preferencia
                ui.notify(f"Preferencia: {TUNE_MODE_INFO[e.value][0]}. Pulsa «Recalcular» para aplicarla "
                          "a tus modelos.", multi_line=True)

            toggle.on_value_change(set_mode)

    async def recalculate() -> None:
        status = {"msg": "Preparando…"}
        with ui.dialog().props("persistent") as wait, ui.card().classes("items-center gap-3 p-8 min-w-96"):
            ui.spinner(size="xl")
            ui.label("Recalculando la configuración óptima…").classes("text-lg font-semibold")
            step = ui.label(status["msg"]).classes("text-sm text-gray-500")
        ticker = ui.timer(0.2, lambda: step.set_text(status["msg"]))
        wait.open()
        recalc_btn.props("loading")
        try:
            def say(msg: str) -> None:
                status["msg"] = msg

            # La VRAM de las PCs remotas puede haber cambiado: se mide antes de decidir.
            for warning in await run.io_bound(rpc.refresh_workers, cfg.rpc_workers,
                                              state.manager.running, say):
                ui.notify(warning, type="warning", multi_line=True)
            result = await run.io_bound(autotune.recalculate, cfg, state.catalog, None, None, say)
        finally:
            ticker.cancel()
            wait.close()
            recalc_btn.props(remove="loading")
        rpc_view.refresh()
        if result.error:
            ui.notify(result.error, type="warning", multi_line=True)
            return
        if not result.changes:
            ui.notify(f"Tu configuración ya es la adecuada para {result.hardware}.", type="positive",
                      multi_line=True)
            return
        with ui.dialog() as dialog, ui.card().classes("w-full max-w-4xl"):
            ui.label("Configuración recomendada para tu equipo").classes("text-lg font-semibold")
            ui.label(result.hardware).classes("text-sm text-gray-500")
            columns = [{"name": k, "label": l, "field": k, "align": "left"} for k, l in
                       (("modelo", "Modelo"), ("param", "Parámetro"), ("actual", "Actual"),
                        ("nuevo", "Nuevo"), ("motivo", "Motivo"))]
            rows = [{"modelo": c.model or "general", "param": autotune.FIELD_LABELS.get(c.field, c.field),
                     "actual": autotune.describe(c.old, c.field), "nuevo": autotune.describe(c.new, c.field),
                     "motivo": c.reason} for c in result.changes]
            ui.table(columns=columns, rows=rows).classes("w-full").props("dense flat wrap-cells")
            for note in result.notes:
                with ui.row().classes("items-center no-wrap"):
                    ui.icon("info", color="primary", size="18px")
                    ui.label(note).classes("text-sm")

            def accept() -> None:
                autotune.apply(cfg, result.config)
                dialog.close()
                warnings = state.save_config()  # refresca esta página y las demás
                ui.notify("Parámetros aplicados. Reinicia el servidor para que tomen efecto.",
                          type="positive")
                for w in warnings:
                    ui.notify(w, type="warning", multi_line=True)

            with ui.row().classes("w-full justify-end"):
                ui.button("Cancelar", on_click=dialog.close).props("flat")
                ui.button("Aplicar", icon="check", on_click=accept)
        dialog.open()

    with ui.row():
        ui.button("Guardar", icon="save", on_click=save)
        recalc_btn = ui.button("Recalcular", icon="auto_fix_high", on_click=recalculate).props("outline") \
            .tooltip("Elige qué modelo descargado va a cada rol, dónde corre cada uno (esta GPU, RAM, "
                     "PC remota) y su contexto, KV cache y capas, según tu hardware. Te muestra la "
                     "propuesta antes de aplicarla.")
        ui.button("Ir a Servidor", icon="arrow_forward", on_click=lambda: state.navigate("server")).props("flat")
    tune_section()
    calculator()
    ui.label("Modelos y roles").classes("text-lg font-semibold mt-2")
    models_view()
    agent_section()
    permissions_section()
    server_section()
    rpc_section()
    mcp_section()
    ui.timer(0.3, load_gpu, once=True)
    state.on_config_saved(lambda: (models_view.refresh(), calculator.refresh(), permissions_view.refresh(),
                                   rpc_view.refresh()))
    state.on_remote_changed(calculator.refresh)  # VRAM total distinta: rehacer la calculadora
