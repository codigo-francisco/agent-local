"""Página «Configuración»: agente, modelos (contexto, KV cache, capas) y calculadora de VRAM."""

from __future__ import annotations

from nicegui import run, ui

from ..config import KV_TYPES, ROLES
from ..server import autotune, vram
from .state import AppState
from .widgets import workspace_picker

CTX_OPTIONS = [2048, 4096, 8192, 12288, 16384, 24576, 32768, 49152, 65536, 98304, 131072]
KV_LABELS = {"f16": "f16 (máxima calidad, el doble de VRAM)", "q8_0": "q8_0 (recomendado)",
             "q4_0": "q4_0 (mínima VRAM, algo menos precisa)"}
COLORS = ["#4f46e5", "#0891b2", "#16a34a", "#d97706", "#db2777"]


def build(state: AppState) -> None:
    cfg = state.cfg
    gpu_box: dict = {"gpu": None}

    ui.label("Configuración").classes("text-2xl font-bold")

    # --- calculadora de VRAM ----------------------------------------------
    @ui.refreshable
    def calculator() -> None:
        plan = vram.plan_usage(cfg, state.catalog, gpu_box["gpu"])
        with ui.card().classes("w-full"):
            with ui.row().classes("items-center w-full"):
                ui.icon("memory", size="24px")
                ui.label("Calculadora de VRAM").classes("text-lg font-semibold")
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
                        ).tooltip(f"{m.name} · {part}: {value:.2f} GB")
                    i += 1
            gpu = gpu_box["gpu"]
            cap_txt = f" de {plan.capacity_gb:.1f} GB utilizables ({gpu.name}, {gpu.total_gb:.1f} GB " \
                      f"menos ~{vram.DESKTOP_RESERVE_GB} GB para Windows)" if gpu and plan.capacity_gb else ""
            mode = "a la vez" if plan.concurrent else "de uno en uno (se turnan)"
            ram_txt = f" Además ≈{plan.ram_offload_gb:.1f} GB irán a la RAM." if plan.ram_offload_gb else ""
            ui.label(f"Total estimado en GPU: {plan.total_gb:.1f} GB{cap_txt}. Modelos cargados {mode}."
                     f"{ram_txt}").classes("text-sm")
            for m in plan.models:
                note = f" — {m.note}" if m.note else ""
                off = f" − {m.offload_gb:.2f} GB en RAM" if m.offload_gb else ""
                ui.label(f"• {m.name}: {m.weights_gb:.2f} GB pesos + {m.kv_gb:.2f} GB contexto (KV) + "
                         f"{m.overhead_gb:.1f} GB margen{off} = {m.total_gb:.2f} GB en GPU{note}") \
                    .classes("text-xs text-gray-500")
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
                        .classes("w-36").tooltip("Cuánto texto puede «ver» el modelo a la vez. Más contexto = más VRAM.")
                    ui.select({k: KV_LABELS[k] for k in KV_TYPES}, label="KV cache", value=m.kv_type,
                              on_change=lambda e, m=m: (setattr(m, "kv_type", e.value), changed())).classes("w-72")
                    ui.number("Capas en GPU (-1 = auto)", value=m.gpu_layers, min=-1, max=999, step=1,
                              format="%d",
                              on_change=lambda e, m=m: (setattr(m, "gpu_layers", int(e.value if e.value is not None else -1)),
                                                        changed())) \
                        .classes("w-44").tooltip("99 = todas en la GPU. -1 = automático: llama.cpp reparte "
                                                 "entre GPU y RAM para que quepa (ideal para modelos MoE).")
                    ui.input("Args extra de llama-server", value=m.extra_args,
                             on_change=lambda e, m=m: setattr(m, "extra_args", e.value or "")) \
                        .classes("grow").tooltip("Ej.: --n-cpu-moe 20 para modelos MoE que no caben")

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

    def save() -> None:
        warnings = state.save_config()
        ui.notify("Configuración guardada. Si el servidor está en marcha, reinícialo para aplicar "
                  "cambios de modelos o contexto.", type="positive", multi_line=True)
        for w in warnings:
            ui.notify(w, type="warning")

    async def recalculate() -> None:
        result = await run.io_bound(autotune.recalculate, cfg, state.catalog)
        if result.error:
            ui.notify(result.error, type="warning", multi_line=True)
            return
        if not result.changes:
            ui.notify(f"Tu configuración ya es la adecuada para {result.hardware}.", type="positive",
                      multi_line=True)
            return
        with ui.dialog() as dialog, ui.card().classes("w-full max-w-4xl"):
            ui.label("Parámetros recomendados para tu equipo").classes("text-lg font-semibold")
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
        ui.button("Recalcular", icon="auto_fix_high", on_click=recalculate).props("outline") \
            .tooltip("Calcula contexto, KV cache, capas en GPU y demás parámetros según tu GPU, "
                     "tu RAM y los modelos asignados. Te muestra la propuesta antes de aplicarla.")
        ui.button("Ir a Servidor", icon="arrow_forward", on_click=lambda: state.navigate("server")).props("flat")
    calculator()
    ui.label("Modelos y roles").classes("text-lg font-semibold mt-2")
    models_view()
    agent_section()
    server_section()
    ui.timer(0.3, load_gpu, once=True)
    state.on_config_saved(lambda: (models_view.refresh(), calculator.refresh()))
