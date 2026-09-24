"""Bucle agéntico: modelo -> herramientas -> modelo ... hasta terminar.

No conoce la interfaz: emite eventos (events.py) y pide confirmaciones con un callback.
Ningún fallo sale de `run`: todo se convierte en un evento AgentError + Done.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
import traceback
import uuid
from pathlib import Path
from typing import Awaitable, Callable

from ..config import AppConfig
from . import sessions
from .checkpoints import Checkpoint
from .context import ContextManager, TokenCounter, repair_history
from .errors import AgentFailure, ContextExhausted, ContextOverflowFromServer, ServerUnavailable
from .events import (AgentError, ContextUsage, Done, Event, FilesChanged, Notice, ReasoningDelta,
                     TextDelta, TextRewrite, ToolRequest, ToolResult)
from .llm import LLMClient, ToolCall, race
from .mcp_tools import MCPManager
from .prompts import SUMMARIZER_PROMPT, build_system
from .tools import TOOL_SCHEMAS, Toolbox, clip, dangerous_reason

log = logging.getLogger("agent.loop")
LOOP_WARN = 3  # misma llamada con el mismo resultado N veces seguidas: aviso al modelo
LOOP_STOP = 4  # ... y a la siguiente se para la tarea

Emit = Callable[[Event], Awaitable[None] | None]
# Decisión sobre una herramienta: "yes" (esta vez), "no", "session" (esta herramienta hasta cerrar
# la app) o "always" (esta herramienta para siempre; se guarda en cfg.always_allow).
Approve = Callable[[ToolRequest], Awaitable[str]]

TOOL_NAMES = {s["function"]["name"] for s in TOOL_SCHEMAS}
# Envoltorios con los que algunos modelos (p. ej. Qwen2.5-Coder) escriben llamadas como texto.
CALL_OPENER = re.compile(r"(<tool_call>|<tools>|```(?:json)?)\s*$")
_JSON = json.JSONDecoder()


def extract_text_tool_calls(content: str, names: set[str] = TOOL_NAMES) -> tuple[str, list[ToolCall]]:
    """Rescata una llamada escrita como texto en lugar del formato nativo: un objeto JSON con
    `name` de una herramienta y `arguments`, suelto o dentro de `<tool_call>`, `<tools>` o ```json.
    Solo se ejecuta la PRIMERA: lo que el modelo escribió después se basa en resultados que aún
    no ha visto (suele inventarlos), así que se descarta."""
    i = content.find("{")
    while i != -1:
        try:
            data, _ = _JSON.raw_decode(content, i)
        except ValueError:
            data = None
        if (isinstance(data, dict) and data.get("name") in names
                and ("arguments" in data or "parameters" in data)):
            args = data.get("arguments", data.get("parameters")) or {}
            call = ToolCall(f"call_{uuid.uuid4().hex[:8]}", data["name"],
                            args if isinstance(args, str) else json.dumps(args, ensure_ascii=False))
            return CALL_OPENER.sub("", content[:i].rstrip()).strip(), [call]
        i = content.find("{", i + 1)
    return content, []


def _norm_args(arguments: str) -> str:
    """Argumentos normalizados (orden de claves, espacios) para comparar llamadas."""
    try:
        return json.dumps(json.loads(arguments or "{}"), sort_keys=True, ensure_ascii=False)
    except ValueError:
        return arguments


class Agent:
    def __init__(self, cfg: AppConfig, llm: LLMClient, emit: Emit, approve: Approve,
                 hint: Callable[[str, int], str | None] | None = None, mcp: MCPManager | None = None,
                 persist: Callable[[], None] | None = None, sessions_dir: Path | None = None,
                 checkpoints_dir: Path | None = None, safe_mode: bool = False):
        self.cfg = cfg
        self.llm = llm
        self.mcp = mcp  # herramientas de servidores MCP (opcional)
        self.persist = persist  # guarda la config tras un «aprobar siempre»
        self.session_allowed: set[str] = set()  # aprobadas «en esta sesión»
        self._emit = emit
        self.approve = approve
        self.hint = hint  # (modelo, n_ctx) -> sugerencia de VRAM para ampliar contexto
        self.toolbox = Toolbox(cfg.workspace, cfg.max_tool_output, cfg.command_timeout)
        self.history: list[dict] = []
        self.summary: str | None = None
        # Modo seguro: ignora el modo automático y los permisos «siempre» guardados.
        self.safe_mode = safe_mode
        self.auto_approve = cfg.confirm == "auto" and not safe_mode
        self.sessions_dir = sessions_dir  # None: no se guarda la conversación
        self.checkpoints_dir = checkpoints_dir  # None: deshacer solo en memoria
        self.session_id = sessions.new_id()
        self.last_checkpoint: Checkpoint | None = None
        self.retry_delay = 3.0  # espera antes de reintentar una conexión fallida
        self._pending_note: str | None = None
        self._cancel: asyncio.Event | None = None  # «Detener» del turno en curso
        self._counters: dict[str, TokenCounter] = {}
        self._n_ctx: dict[str, int] = {}
        self.running = False

    # --- estado ----------------------------------------------------------
    def is_allowed(self, tool: str) -> bool:
        """Aprobada sin preguntar: modo automático, o permiso de sesión o permanente."""
        return (self.auto_approve or tool in self.session_allowed
                or (tool in self.cfg.always_allow and not self.safe_mode))

    def reset(self) -> None:
        self.history, self.summary = [], None
        self.session_id = sessions.new_id()

    def set_workspace(self, path: str) -> None:
        self.toolbox = Toolbox(path, self.cfg.max_tool_output, self.cfg.command_timeout)
        self.reset()

    def restore(self, session: sessions.Session) -> None:
        """Retoma una conversación guardada."""
        self.history, _ = repair_history(list(session.history))
        self.summary = session.summary
        self.session_id = session.id

    def save_session(self) -> None:
        if self.sessions_dir is None or not self.history:
            return
        try:
            sessions.save(self.sessions_dir, sessions.Session(
                self.session_id, str(self.toolbox.workspace), self.history, self.summary))
        except OSError:
            log.exception("No se pudo guardar la conversación")

    def undo(self, checkpoint_id: str) -> str:
        """Deshace los archivos escritos en un turno. Devuelve un resumen legible."""
        cp = self.last_checkpoint if self.last_checkpoint and self.last_checkpoint.id == checkpoint_id \
            else (Checkpoint.load(self.checkpoints_dir, checkpoint_id) if self.checkpoints_dir else None)
        if cp is None:
            return "No encontré ese punto de restauración (quizá ya se deshizo)."
        report = cp.undo()
        log.info("Deshacer %s: %s", checkpoint_id, report.text())
        # Que el modelo sepa que el disco ya no está como lo dejó (va delante del próximo mensaje:
        # dos mensajes `user` seguidos rompen algunas plantillas de chat).
        self._pending_note = (f"[Nota del sistema] El usuario deshizo los cambios de archivos de un "
                              f"turno anterior ({report.text()}). Vuelve a leer los archivos antes "
                              "de editarlos.")
        return report.text()

    async def _close_checkpoint(self) -> None:
        """Cierra el checkpoint del turno y avisa de los archivos cambiados (idempotente)."""
        cp, self.toolbox.checkpoint = self.toolbox.checkpoint, None
        if cp is None:
            return
        try:
            cp.finalize()
        except OSError:
            log.exception("No se pudo guardar el punto de restauración")
        if cp.files:
            self.last_checkpoint = cp
            await self.emit(FilesChanged(cp.id, cp.files))

    async def emit(self, event: Event) -> None:
        if isinstance(event, Done):  # FilesChanged va justo antes: Done es siempre el último
            await self._close_checkpoint()
        r = self._emit(event)
        if inspect.isawaitable(r):
            await r

    def system_prompt(self, summary: str | None) -> str:
        return build_system(str(self.toolbox.workspace), summary)

    async def context_limit(self, model: str) -> int:
        if model not in self._n_ctx:
            try:
                n = await self.llm.n_ctx(model)
            except Exception:  # noqa: BLE001
                n = None
            if n:
                self._n_ctx[model] = n  # solo cacheamos el valor real del servidor
            return n or self.cfg.ctx_for(model)
        return self._n_ctx[model]

    def _counter(self, model: str) -> TokenCounter:
        if model not in self._counters:
            self._counters[model] = TokenCounter(lambda t, m=model: self.llm.tokenize(m, t))
        return self._counters[model]

    async def _summarize(self, messages: list[dict], previous: str | None) -> str | None:
        model = self.cfg.roles.get("fast") or self.cfg.roles.get("main")
        if not model:
            return None
        budget_chars = max(4000, int((self.cfg.ctx_for(model) - 1200) * 2.5))
        lines = []
        for m in messages:
            body = m.get("content") or ""
            for tc in m.get("tool_calls") or []:
                body += f"\n-> {tc['function']['name']}({tc['function'].get('arguments', '')[:300]})"
            lines.append(f"[{m['role']}] {clip(body, 1500)}")
        transcript = clip("\n".join(lines), budget_chars)
        user = (f"Previous summary:\n{previous}\n\n" if previous else "") + f"Conversation:\n{transcript}"
        try:
            # Resumir puede tardar (y cargar el modelo «fast»): «Detener» no debe esperar a que acabe.
            cancelled, text = await race(
                self.llm.complete(model, [{"role": "system", "content": SUMMARIZER_PROMPT},
                                          {"role": "user", "content": user}], max_tokens=700),
                self._cancel, None)
            return None if cancelled else text
        except Exception:  # noqa: BLE001 - el ContextManager sabe seguir sin resumen
            return None

    # --- ejecución -------------------------------------------------------
    async def run(self, user_text: str, model: str | None = None,
                  cancel: asyncio.Event | None = None) -> None:
        cancel = cancel or asyncio.Event()
        self._cancel = cancel
        self.running = True
        self.toolbox.checkpoint = Checkpoint(self.toolbox.workspace, self.checkpoints_dir)
        try:
            try:
                model_name = self.cfg.resolve(model or "main")
            except KeyError as e:
                raise AgentFailure("Falta un modelo", str(e),
                                   ["Asigna un modelo al rol en la página «Modelos»."],
                                   action="settings") from e
            self.history, fixes = repair_history(self.history)
            if fixes:
                log.warning("Historial reparado antes del turno (%d arreglos)", fixes)
            if self._pending_note:
                user_text = f"{self._pending_note}\n\n{user_text}"
                self._pending_note = None
            self.history.append({"role": "user", "content": user_text})
            log.info("Turno con %s: %s", model_name, user_text[:200])
            await self._run(model_name, cancel)
        except AgentFailure as e:
            log.warning("Fallo explicable: %s: %s", e.title, e.cause)
            await self.emit(AgentError(e.title, e.cause, e.detail, e.suggestions, e.action))
            await self.emit(Done("context" if isinstance(e, ContextExhausted) else "error"))
        except asyncio.CancelledError:
            await self.emit(Done("cancelled"))
            raise
        except Exception as e:  # noqa: BLE001 - nunca romper la interfaz
            log.exception("Error inesperado en el turno")
            await self.emit(AgentError("Error inesperado", f"{type(e).__name__}: {e}",
                                       traceback.format_exc()))
            await self.emit(Done("error"))
        finally:
            try:
                await self._close_checkpoint()  # por si no llegó a emitirse Done
            except Exception:  # noqa: BLE001 - la interfaz pudo cerrarse
                log.exception("No se pudo cerrar el punto de restauración")
            # Pase lo que pase, el historial queda válido para el siguiente turno y en disco.
            self.history, _ = repair_history(self.history)
            self.save_session()
            self.running = False

    async def _run(self, model: str, cancel: asyncio.Event) -> None:
        n_ctx = await self.context_limit(model)
        reserve = max(256, min(self.cfg.max_output_tokens, n_ctx // 4))
        counter = self._counter(model)
        counter.exact = True  # reintentar /tokenize: el servidor pudo arrancar desde la última vez
        ctxm = ContextManager(counter, n_ctx, reserve, self.system_prompt, self._summarize,
                              hint=(lambda n: self.hint(model, n)) if self.hint else None)
        tools = TOOL_SCHEMAS + (self.mcp.schemas() if self.mcp else [])
        names = TOOL_NAMES | (self.mcp.tool_names() if self.mcp else set())
        aggressive = False
        continued = False
        retried = False
        last_sig, repeats = None, 0

        async def on_text(t: str) -> None:
            await self.emit(TextDelta(t))

        async def on_reasoning(t: str) -> None:
            await self.emit(ReasoningDelta(t))

        for _ in range(self.cfg.max_steps):
            if cancel.is_set():
                await self.emit(Done("cancelled"))
                return

            # 1. Encajar el contexto (compacta o lanza ContextExhausted con explicación).
            await counter.prime(ctxm.texts_for_priming(self.history, self.summary, tools))
            fit = await ctxm.fit(self.history, self.summary, tools, aggressive=aggressive)
            self.history, self.summary = fit.history, fit.summary
            for text in fit.notices:
                await self.emit(Notice(text, "warning"))
            await self.emit(ContextUsage(fit.used, n_ctx, ctxm.budget, fit.breakdown))

            # 2. Llamar al modelo.
            messages = [{"role": "system", "content": self.system_prompt(self.summary)}, *self.history]
            try:
                res = await self.llm.chat(model, messages, tools, reserve, on_text, on_reasoning, cancel)
            except ContextOverflowFromServer as e:
                if aggressive:
                    exc = ContextExhausted(fit.breakdown, n_ctx, ctxm.budget,
                                           ctxm.hint(n_ctx) if ctxm.hint else None)
                    exc.detail = str(e)
                    raise exc from e
                counter.overflowed()
                aggressive = True
                await self.emit(Notice("El servidor avisó de que el contexto no alcanza (mis cuentas "
                                       "se quedaron cortas). Compacto más y reintento.", "warning"))
                continue
            except ServerUnavailable:
                # Suele ser momentáneo (llama-swap cambiando de modelo): un reintento antes de rendirse.
                if retried:
                    raise
                retried = True
                log.warning("Servidor no disponible; reintento en %.0f s", self.retry_delay)
                await self.emit(Notice("No pude conectar con el servidor de modelos; reintento en "
                                       f"{self.retry_delay:.0f} s…", "warning"))
                try:
                    await asyncio.wait_for(cancel.wait(), self.retry_delay)
                except asyncio.TimeoutError:
                    pass
                continue
            aggressive = False
            if res.prompt_tokens and not counter.exact:
                counter.calibrate(fit.used, res.prompt_tokens)

            if res.cancelled:
                if res.content:
                    self.history.append({"role": "assistant",
                                         "content": res.content + "\n[interrumpido por el usuario]"})
                await self.emit(Done("cancelled"))
                return

            content, calls = res.content, res.tool_calls
            if not calls:
                content, calls = extract_text_tool_calls(content, names)
                if calls:  # la interfaz ya mostró el texto completo: se sustituye por el limpio
                    await self.emit(TextRewrite(content))
            truncated = res.finish_reason == "length"

            # 3a. Llamadas a herramientas.
            if calls:
                parsed: list[tuple[ToolCall, dict | None, str | None]] = []
                for call in calls:
                    try:
                        parsed.append((call, call.parse_args(), None))
                    except ValueError as e:
                        parsed.append((call, None, str(e)))
                if truncated and any(err for _, _, err in parsed):
                    await self._handle_truncated_call(content, parsed, reserve)
                    continue
                msg = {
                    "role": "assistant", "content": content or "",
                    "tool_calls": [{"id": c.id, "type": "function",
                                    "function": {"name": c.name, "arguments": c.arguments}}
                                   for c, _, _ in parsed],
                }
                if res.reasoning:
                    # Los modelos que piensan entre llamadas (Qwen3.x, gpt-oss) rinden mejor si
                    # reciben su razonamiento previo; el ContextManager lo quita de pasos antiguos.
                    msg["reasoning_content"] = res.reasoning
                self.history.append(msg)
                outputs = []
                for call, args, err in parsed:
                    ok, output = await self._run_tool(call, args, err, cancel)
                    await self.emit(ToolResult(call.id, call.name, ok, output))
                    self.history.append({"role": "tool", "tool_call_id": call.id, "content": output})
                    outputs.append(output)
                if cancel.is_set():
                    await self.emit(Done("cancelled"))
                    return
                # Modelos pequeños a veces repiten la misma llamada sin fin: se detecta y se corta.
                sig = tuple((c.name, _norm_args(c.arguments)) for c, _, _ in parsed) + tuple(outputs)
                repeats = repeats + 1 if sig == last_sig else 1
                last_sig = sig
                if repeats >= LOOP_STOP:
                    log.warning("Bucle detectado: %s", [c.name for c, _, _ in parsed])
                    await self.emit(Notice("El modelo repite la misma acción sin avanzar; detengo la "
                                           "tarea. Prueba a reformular la petición o a usar otro "
                                           "modelo.", "warning"))
                    await self.emit(Done("loop"))
                    return
                if repeats == LOOP_WARN:
                    self.history.append({"role": "user", "content": (
                        f"[Nota del sistema] Has repetido la misma llamada {repeats} veces con el mismo "
                        "resultado. No la repitas: cambia de enfoque o explica qué te bloquea.")})
                    await self.emit(Notice("El modelo repite la misma acción; le pedí que cambie de "
                                           "enfoque.", "warning"))
                continue

            # 3b. Respuesta de texto: fin del turno (o continuación si se cortó).
            self.history.append({"role": "assistant", "content": content})
            if truncated:
                if not continued:
                    continued = True
                    await self.emit(Notice("La respuesta llegó al límite de tokens de salida; le pido "
                                           "que continúe.", "warning"))
                    self.history.append({"role": "user",
                                         "content": "Continúa exactamente donde lo dejaste, sin repetir."})
                    continue
                await self.emit(Notice("La respuesta se volvió a cortar por el límite de salida. Sube "
                                       "«tokens de salida» en Configuración o escribe «continúa».",
                                       "warning"))
            await self.emit(Done("ok"))
            return

        await self.emit(Notice(f"Alcancé el máximo de {self.cfg.max_steps} pasos sin terminar. "
                               "Escribe «continúa» para seguir o sube el límite en Configuración.",
                               "warning"))
        await self.emit(Done("max_steps"))

    async def _handle_truncated_call(self, content: str, parsed: list, reserve: int) -> None:
        names = ", ".join(c.name for c, _, _ in parsed)
        if content:
            self.history.append({"role": "assistant", "content": content})
        self.history.append({"role": "user", "content": (
            f"[Nota del sistema] Tu llamada a {names} se cortó al llegar al límite de {reserve} "
            "tokens de salida y NO se ejecutó. Repítela en partes más pequeñas: por ejemplo varias "
            "llamadas edit_file cortas, o un write_file con la primera parte y luego edit_file para "
            "añadir el resto.")})
        await self.emit(Notice(f"La llamada a {names} se cortó por el límite de salida: no se "
                               "ejecutó y le pedí al modelo que la divida.", "warning"))

    async def _run_tool(self, call: ToolCall, args: dict | None, err: str | None,
                        cancel: asyncio.Event) -> tuple[bool, str]:
        if cancel.is_set():
            return False, "Cancelado por el usuario antes de ejecutarse."
        if err is not None or args is None:
            await self.emit(ToolRequest(call.id, call.name, {}, call.arguments[:500], False))
            return False, (f"Error: los argumentos no son JSON válido ({err}). Vuelve a llamar a "
                           f"{call.name} con un objeto JSON correcto.")
        is_mcp = self.mcp is not None and self.mcp.is_mcp(call.name)
        if is_mcp:
            needs = self.mcp.needs_approval(call.name)
            preview = json.dumps(args, ensure_ascii=False, indent=2)
        else:
            needs = self.toolbox.needs_approval(call.name)
            preview = await asyncio.to_thread(self.toolbox.preview, call.name, args)
        danger = dangerous_reason(str(args.get("command", ""))) if call.name == "run_command" else None
        # Lo destructivo pregunta siempre: ni el modo automático ni «aprobar siempre» lo saltan.
        needs = (needs and not self.is_allowed(call.name)) or danger is not None
        if danger:
            preview = f"# ⚠ Potencialmente destructivo: {danger}\n{preview}"
            log.warning("Comando destructivo pendiente de aprobación: %s", args.get("command"))
        request = ToolRequest(call.id, call.name, args, preview, needs)
        await self.emit(request)
        if needs:
            decision = await self.approve(request)
            log.info("Aprobación de %s: %s", call.name, decision)
            if decision == "session":
                self.session_allowed.add(call.name)
            elif decision == "always":
                if call.name not in self.cfg.always_allow:
                    self.cfg.always_allow.append(call.name)
                if self.persist:
                    self.persist()
            elif decision != "yes":
                return False, ("El usuario rechazó esta acción. No la repitas igual: pregúntale cómo "
                               "prefiere continuar o propón una alternativa.")
        if is_mcp:
            cancelled, result = await race(self.mcp.call(call.name, args, self.cfg.max_tool_output),
                                           cancel, None)
            return (False, "Cancelado por el usuario mientras se ejecutaba.") if cancelled else result
        return await asyncio.to_thread(self.toolbox.execute, call.name, args, cancel.is_set)
