"""Bucle agéntico: modelo -> herramientas -> modelo ... hasta terminar.

No conoce la interfaz: emite eventos (events.py) y pide confirmaciones con un callback.
Ningún fallo sale de `run`: todo se convierte en un evento AgentError + Done.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
import traceback
import uuid
from typing import Awaitable, Callable

from ..config import AppConfig
from .context import ContextManager, TokenCounter
from .errors import AgentFailure, ContextExhausted, ContextOverflowFromServer
from .events import (AgentError, ContextUsage, Done, Event, Notice, ReasoningDelta, TextDelta,
                     TextRewrite, ToolRequest, ToolResult)
from .llm import LLMClient, ToolCall
from .prompts import SUMMARIZER_PROMPT, build_system
from .tools import TOOL_SCHEMAS, Toolbox, clip

Emit = Callable[[Event], Awaitable[None] | None]
Approve = Callable[[ToolRequest], Awaitable[str]]  # devuelve "yes" | "no" | "always"

TOOL_NAMES = {s["function"]["name"] for s in TOOL_SCHEMAS}
# Envoltorios con los que algunos modelos (p. ej. Qwen2.5-Coder) escriben llamadas como texto.
CALL_OPENER = re.compile(r"(<tool_call>|<tools>|```(?:json)?)\s*$")
_JSON = json.JSONDecoder()


def extract_text_tool_calls(content: str) -> tuple[str, list[ToolCall]]:
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
        if (isinstance(data, dict) and data.get("name") in TOOL_NAMES
                and ("arguments" in data or "parameters" in data)):
            args = data.get("arguments", data.get("parameters")) or {}
            call = ToolCall(f"call_{uuid.uuid4().hex[:8]}", data["name"],
                            args if isinstance(args, str) else json.dumps(args, ensure_ascii=False))
            return CALL_OPENER.sub("", content[:i].rstrip()).strip(), [call]
        i = content.find("{", i + 1)
    return content, []


class Agent:
    def __init__(self, cfg: AppConfig, llm: LLMClient, emit: Emit, approve: Approve,
                 hint: Callable[[str, int], str | None] | None = None):
        self.cfg = cfg
        self.llm = llm
        self._emit = emit
        self.approve = approve
        self.hint = hint  # (modelo, n_ctx) -> sugerencia de VRAM para ampliar contexto
        self.toolbox = Toolbox(cfg.workspace, cfg.max_tool_output, cfg.command_timeout)
        self.history: list[dict] = []
        self.summary: str | None = None
        self.auto_approve = cfg.confirm == "auto"
        self._counters: dict[str, TokenCounter] = {}
        self._n_ctx: dict[str, int] = {}
        self.running = False

    # --- estado ----------------------------------------------------------
    def reset(self) -> None:
        self.history, self.summary = [], None

    def set_workspace(self, path: str) -> None:
        self.toolbox = Toolbox(path, self.cfg.max_tool_output, self.cfg.command_timeout)
        self.reset()

    async def emit(self, event: Event) -> None:
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
            return await self.llm.complete(model, [{"role": "system", "content": SUMMARIZER_PROMPT},
                                                   {"role": "user", "content": user}], max_tokens=700)
        except Exception:  # noqa: BLE001 - el ContextManager sabe seguir sin resumen
            return None

    # --- ejecución -------------------------------------------------------
    async def run(self, user_text: str, model: str | None = None,
                  cancel: asyncio.Event | None = None) -> None:
        cancel = cancel or asyncio.Event()
        self.running = True
        try:
            try:
                model_name = self.cfg.resolve(model or "main")
            except KeyError as e:
                raise AgentFailure("Falta un modelo", str(e),
                                   ["Asigna un modelo al rol en la página «Modelos»."],
                                   action="settings") from e
            self.history.append({"role": "user", "content": user_text})
            await self._run(model_name, cancel)
        except AgentFailure as e:
            await self.emit(AgentError(e.title, e.cause, e.detail, e.suggestions, e.action))
            await self.emit(Done("context" if isinstance(e, ContextExhausted) else "error"))
        except asyncio.CancelledError:
            await self.emit(Done("cancelled"))
            raise
        except Exception as e:  # noqa: BLE001 - nunca romper la interfaz
            await self.emit(AgentError("Error inesperado", f"{type(e).__name__}: {e}",
                                       traceback.format_exc()))
            await self.emit(Done("error"))
        finally:
            self.running = False

    async def _run(self, model: str, cancel: asyncio.Event) -> None:
        n_ctx = await self.context_limit(model)
        reserve = max(256, min(self.cfg.max_output_tokens, n_ctx // 4))
        counter = self._counter(model)
        counter.exact = True  # reintentar /tokenize: el servidor pudo arrancar desde la última vez
        ctxm = ContextManager(counter, n_ctx, reserve, self.system_prompt, self._summarize,
                              hint=(lambda n: self.hint(model, n)) if self.hint else None)
        tools = TOOL_SCHEMAS
        aggressive = False
        continued = False

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
                content, calls = extract_text_tool_calls(content)
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
                for call, args, err in parsed:
                    ok, output = await self._run_tool(call, args, err, cancel)
                    await self.emit(ToolResult(call.id, call.name, ok, output))
                    self.history.append({"role": "tool", "tool_call_id": call.id, "content": output})
                if cancel.is_set():
                    await self.emit(Done("cancelled"))
                    return
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
        needs = self.toolbox.needs_approval(call.name) and not self.auto_approve
        preview = await asyncio.to_thread(self.toolbox.preview, call.name, args)
        request = ToolRequest(call.id, call.name, args, preview, needs)
        await self.emit(request)
        if needs:
            decision = await self.approve(request)
            if decision == "always":
                self.auto_approve = True
            elif decision != "yes":
                return False, ("El usuario rechazó esta acción. No la repitas igual: pregúntale cómo "
                               "prefiere continuar o propón una alternativa.")
        return await asyncio.to_thread(self.toolbox.execute, call.name, args)
