"""Cliente único para cualquier servidor compatible con OpenAI (llama-swap, llama-server, Ollama...).

Traduce los fallos del servidor a errores explicables (ver errors.py).
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from typing import Awaitable, Callable

import httpx
import openai

from .errors import AgentFailure, ContextOverflowFromServer, ModelLoadFailed, ServerUnavailable

CONTEXT_ERROR_MARKERS = (
    "exceed_context_size", "exceeds the available context", "context size", "context length",
    "context window", "n_ctx", "too many tokens", "prompt is too long", "maximum context",
    "input is too long",
)
OOM_MARKERS = ("out of memory", "cudamalloc failed", "failed to allocate", "unable to allocate",
               "not enough memory", "error allocating")


def is_context_error(message: str) -> bool:
    m = message.lower()
    return any(k in m for k in CONTEXT_ERROR_MARKERS)


def is_oom(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in OOM_MARKERS)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # JSON sin parsear, tal como lo generó el modelo

    def parse_args(self) -> dict:
        """Lanza ValueError si el JSON no es válido o no es un objeto."""
        raw = self.arguments.strip() or "{}"
        args = json.loads(raw)
        if not isinstance(args, dict):
            raise ValueError("los argumentos deben ser un objeto JSON")
        return args


@dataclass
class ChatResult:
    content: str = ""
    reasoning: str = ""  # "pensamiento" de modelos como Qwen3.x o gpt-oss (reasoning_content)
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cancelled: bool = False


STREAM_IDLE_TIMEOUT = 180.0  # s sin recibir nada a mitad de una respuesta -> servidor colgado


class _Stalled(Exception):
    pass


async def race(aw: Awaitable, cancel: asyncio.Event | None, timeout: float | None):
    """Espera `aw`, pero vuelve antes si se pulsa «Detener» o pasan `timeout` segundos.
    Devuelve (cancelado, resultado); lanza _Stalled si venció el tiempo."""
    task = asyncio.ensure_future(aw)
    waiter = asyncio.ensure_future(cancel.wait()) if cancel is not None else None
    try:
        done, _ = await asyncio.wait({task, waiter} - {None}, timeout=timeout,
                                     return_when=asyncio.FIRST_COMPLETED)
    except BaseException:
        task.cancel()
        raise
    finally:
        if waiter is not None:
            waiter.cancel()
    if task in done:
        return False, task.result()
    task.cancel()
    try:
        await task
    except BaseException:  # noqa: BLE001 - solo queremos que termine
        pass
    if waiter is not None and waiter in done:
        return True, None
    raise _Stalled()


class LLMClient:
    def __init__(self, endpoint: str, log_provider: Callable[[], str] | None = None,
                 transport: httpx.AsyncBaseTransport | None = None):
        self.endpoint = endpoint.rstrip("/")
        self.root = self.endpoint[:-3] if self.endpoint.endswith("/v1") else self.endpoint
        self.log_provider = log_provider
        self.idle_timeout = STREAM_IDLE_TIMEOUT
        # `transport` permite a los tests simular un servidor que falla de mil maneras.
        self.client = openai.AsyncOpenAI(
            base_url=self.endpoint, api_key="local", max_retries=0,
            timeout=900.0,  # cargar un modelo grande tarda
            http_client=httpx.AsyncClient(transport=transport, timeout=900.0) if transport else None,
        )
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=3.0), transport=transport)
        self.active = 0  # peticiones de chat en curso (ver close_when_idle)
        self._idle = asyncio.Event()
        self._idle.set()

    # --- utilidades del servidor ----------------------------------------
    async def list_models(self) -> list[str]:
        try:
            page = await self.client.models.list()
            return [m.id for m in page.data]
        except openai.APIConnectionError as e:
            raise ServerUnavailable(self.endpoint, str(e)) from e

    async def _get_upstream(self, model: str, path: str) -> httpx.Response | None:
        for url in (f"{self.root}/upstream/{model}/{path}", f"{self.root}/{path}"):
            try:
                r = await self._http.get(url)
                if r.status_code == 200:
                    return r
            except httpx.ConnectError:
                return None  # servidor caído: no insistir (en Windows cada intento tarda ~2 s)
            except httpx.HTTPError:
                continue
        return None

    async def n_ctx(self, model: str) -> int | None:
        """Contexto real con el que arrancó llama-server (por slot)."""
        r = await self._get_upstream(model, "props")
        if r is None:
            return None
        try:
            data = r.json()
        except ValueError:
            return None
        gen = data.get("default_generation_settings") or {}
        value = gen.get("n_ctx") or data.get("n_ctx")
        return int(value) if value else None

    async def tokenize(self, model: str, text: str) -> int | None:
        for url in (f"{self.root}/upstream/{model}/tokenize", f"{self.root}/tokenize"):
            try:
                r = await self._http.post(url, json={"content": text})
                if r.status_code == 200:
                    return len(r.json().get("tokens", []))
            except httpx.ConnectError:
                return None
            except (httpx.HTTPError, ValueError):
                continue
        return None

    # --- chat ------------------------------------------------------------
    def _translate(self, model: str, e: Exception) -> Exception:
        if isinstance(e, openai.APITimeoutError):
            return AgentFailure(
                "El modelo tardó demasiado", "No hubo respuesta a tiempo del servidor.",
                ["Si el modelo se estaba cargando por primera vez, vuelve a intentarlo.",
                 "Revisa en «Servidor» si llama-server sigue vivo."], str(e), action="server")
        if isinstance(e, openai.APIConnectionError):
            return ServerUnavailable(self.endpoint, str(e))
        if isinstance(e, openai.APIStatusError):
            body = e.message or ""
            try:
                body = json.dumps(e.body, ensure_ascii=False) if e.body else body
            except (TypeError, ValueError):
                pass
            if is_context_error(body):
                return ContextOverflowFromServer(body)
            if e.status_code == 404:
                return AgentFailure(
                    "Modelo no configurado", f"El servidor no conoce el modelo «{model}».",
                    ["Guarda la configuración y reinicia el servidor para regenerar llama-swap.yaml."],
                    body, action="settings")
            if e.status_code in (500, 502, 503):
                logs = self.log_provider() if self.log_provider else ""
                return ModelLoadFailed(model, (body + "\n\n" + logs[-3000:]).strip(),
                                       out_of_memory=is_oom(body) or is_oom(logs))
            return AgentFailure("El servidor rechazó la petición",
                                f"HTTP {e.status_code}: {e.message}", [], body)
        if isinstance(e, openai.APIError) and is_context_error(str(e)):
            return ContextOverflowFromServer(str(e))
        return e

    async def chat(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
        on_text: Callable[[str], Awaitable[None]] | None = None,
        on_reasoning: Callable[[str], Awaitable[None]] | None = None,
        cancel: asyncio.Event | None = None,
    ) -> ChatResult:
        result = ChatResult()
        slots: dict[int, dict] = {}
        kwargs: dict = {"model": model, "messages": messages, "max_tokens": max_tokens,
                        "stream": True, "stream_options": {"include_usage": True}}
        if tools:
            kwargs["tools"] = tools
        stream = None
        first = True  # aún no llegó ningún trozo
        self.active += 1
        self._idle.clear()
        try:
            # Mientras el modelo se carga no llega nada: «Detener» debe funcionar igualmente.
            cancelled, stream = await race(self.client.chat.completions.create(**kwargs), cancel, None)
            if cancelled:
                result.cancelled = True
                return result
            chunks = stream.__aiter__()
            while True:
                try:
                    cancelled, chunk = await race(chunks.__anext__(), cancel,
                                                   None if first else self.idle_timeout)
                except StopAsyncIteration:
                    break
                first = False
                if cancelled or (cancel is not None and cancel.is_set()):
                    result.cancelled = True
                    break
                if getattr(chunk, "usage", None):
                    result.prompt_tokens = chunk.usage.prompt_tokens
                    result.completion_tokens = chunk.usage.completion_tokens
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                delta = choice.delta
                if choice.finish_reason:
                    result.finish_reason = choice.finish_reason
                if delta is None:
                    continue
                reasoning = getattr(delta, "reasoning_content", None)
                if reasoning:
                    result.reasoning += reasoning
                    if on_reasoning:
                        await on_reasoning(reasoning)
                if delta.content:
                    result.content += delta.content
                    if on_text:
                        await on_text(delta.content)
                for tc in delta.tool_calls or []:
                    slot = slots.setdefault(tc.index or 0, {"id": None, "name": "", "arguments": ""})
                    if tc.id:
                        slot["id"] = tc.id
                    if tc.function is not None:
                        if tc.function.name and not slot["name"]:
                            slot["name"] = tc.function.name
                        if tc.function.arguments:
                            slot["arguments"] += tc.function.arguments
        except openai.APIConnectionError as e:
            # Si ya llegaron trozos no es que no haya servidor: se cortó a mitad. Distinguirlo
            # evita el mensaje «no hay nada escuchando» y un reintento que duplicaría el texto.
            if not first:
                raise self._cut(model, e) from e
            raise self._translate(model, e) from e
        except openai.APIError as e:
            raise self._translate(model, e) from e
        except _Stalled as e:
            raise AgentFailure(
                "El modelo dejó de responder",
                f"No llegó nada del servidor en {self.idle_timeout:.0f} s a mitad de la respuesta.",
                ["Revisa en «Servidor» si llama-server sigue vivo o se quedó sin memoria.",
                 "Vuelve a enviar el mensaje o escribe «continúa»."], action="server") from e
        except httpx.HTTPError as e:  # la conexión se cortó a mitad del stream
            raise self._cut(model, e) from e
        finally:
            self.active -= 1
            if self.active == 0:
                self._idle.set()
            if stream is not None:
                try:
                    await stream.close()
                except Exception:  # noqa: BLE001
                    pass
        for idx in sorted(slots):
            s = slots[idx]
            if s["name"]:
                result.tool_calls.append(
                    ToolCall(s["id"] or f"call_{idx}_{uuid.uuid4().hex[:8]}", s["name"], s["arguments"]))
        return result

    def _cut(self, model: str, e: Exception) -> AgentFailure:
        """La conexión se cayó a mitad de la respuesta (llama-server murió, p. ej. sin memoria)."""
        logs = self.log_provider() if self.log_provider else ""
        if is_oom(logs):
            return ModelLoadFailed(model, (str(e) + "\n\n" + logs[-3000:]).strip(), out_of_memory=True)
        return AgentFailure(
            "Se cortó la conexión con el modelo",
            "El servidor cerró la conexión a mitad de la respuesta (¿se cayó llama-server?).",
            ["Revisa el registro en la página «Servidor».",
             "Vuelve a enviar el mensaje: lo ya hecho en disco se conserva."],
            f"{type(e).__name__}: {e}\n\n{logs[-3000:]}".strip(), action="server")

    async def complete(self, model: str, messages: list[dict], max_tokens: int = 800) -> str:
        """Petición corta sin herramientas (para resúmenes)."""
        res = await self.chat(model, messages, None, max_tokens)
        return res.content.strip()

    async def aclose(self) -> None:
        await self._http.aclose()
        await self.client.close()

    async def close_when_idle(self) -> None:
        """Cierra el cliente cuando no quede ninguna petición en curso. Cerrarlo antes cortaba la
        respuesta que se estaba recibiendo (p. ej. al guardar la configuración a mitad de un turno)."""
        await self._idle.wait()
        await self.aclose()
