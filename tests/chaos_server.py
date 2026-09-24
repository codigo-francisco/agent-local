"""Servidor «caótico» compatible con OpenAI para tests: un transporte httpx que responde a cada
petición de chat según un guion de modos de fallo, sin red ni procesos.

    transport = ChaosTransport(["hang_first", "ok"])
    llm = LLMClient("http://chaos/v1", transport=transport)
"""

from __future__ import annotations

import asyncio
import json

import httpx


def _chunk(delta: dict, finish: str | None = None) -> bytes:
    data = {"id": "x", "object": "chat.completion.chunk", "created": 0, "model": "m",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    return f"data: {json.dumps(data)}\n\n".encode()


DONE = b"data: [DONE]\n\n"


class ChaosTransport(httpx.AsyncBaseTransport):
    """Modos (uno por petición de chat, en orden; el último se repite):
    ok, slow (0,3 s entre trozos), stall (se cuelga tras el primer trozo), cut (corta la conexión
    a mitad), oom (500 sin
    memoria), context (400 por contexto), refused (no hay servidor), hang_first (tarda mucho en
    empezar), bad_tool_json (llamada con JSON roto)."""

    def __init__(self, script: list[str]):
        self.script = list(script)
        self.requests = 0

    def _next(self) -> str:
        return self.script.pop(0) if len(self.script) > 1 else self.script[0]

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if not request.url.path.endswith("/chat/completions"):
            return httpx.Response(404)  # /props, /tokenize: como un servidor sin esos endpoints
        self.requests += 1
        mode = self._next()
        if mode == "refused":
            raise httpx.ConnectError("connection refused", request=request)
        if mode == "oom":
            return httpx.Response(500, json={"error": {"message": "cudaMalloc failed: out of memory"}})
        if mode == "context":
            return httpx.Response(400, json={"error": {"message": "exceed_context_size_error: "
                                                                  "the request exceeds the available context size"}})
        if mode == "hang_first":
            await asyncio.sleep(30)

        async def body():
            if mode == "bad_tool_json":
                yield _chunk({"tool_calls": [{"index": 0, "id": "t1", "type": "function",
                                              "function": {"name": "read_file", "arguments": '{"path": x'}}]})
                yield _chunk({}, "tool_calls")
                yield DONE
                return
            yield _chunk({"role": "assistant", "content": "Hola"})
            if mode == "slow":  # respuesta larga: da tiempo a que pase algo a mitad
                await asyncio.sleep(0.3)
            if mode == "stall":
                await asyncio.sleep(30)
            if mode == "cut":
                raise httpx.RemoteProtocolError("peer closed connection without sending complete message body")
            yield _chunk({"content": ", listo."}, "stop")
            yield DONE

        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body())
