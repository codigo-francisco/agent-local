"""El agente con el cliente real contra un servidor que falla de todas las formas posibles:
siempre termina con Done, con un error explicable, y el historial queda válido."""

import asyncio
import time

from agent.config import AppConfig, ModelEntry
from agent.core.context import repair_history
from agent.core.events import AgentError, Done, ToolResult
from agent.core.llm import LLMClient
from agent.core.loop import Agent
from chaos_server import ChaosTransport


def run(tmp_path, script, cancel_after=None, idle_timeout=0.2):
    cfg = AppConfig(workspace=str(tmp_path), roles={"main": "m", "fast": "", "draft": ""},
                    models=[ModelEntry("m", "m.gguf", ctx=8192)], max_output_tokens=512)
    transport = ChaosTransport(script)
    events = []

    async def approve(req):
        return "yes"

    async def main():
        llm = LLMClient("http://chaos/v1", transport=transport)
        llm.idle_timeout = idle_timeout
        agent = Agent(cfg, llm, events.append, approve)
        agent.retry_delay = 0
        cancel = asyncio.Event()
        if cancel_after is not None:
            asyncio.get_running_loop().call_later(cancel_after, cancel.set)
        await agent.run("hola", cancel=cancel)
        await llm.aclose()
        return agent

    t0 = time.monotonic()
    agent = asyncio.run(main())
    elapsed = time.monotonic() - t0
    assert isinstance(events[-1], Done), events[-1]
    assert repair_history(agent.history)[1] == 0, "historial inconsistente"
    return agent, events, transport, elapsed


def error(events):
    return next((e for e in events if isinstance(e, AgentError)), None)


def test_ok(tmp_path):
    agent, events, _, _ = run(tmp_path, ["ok"])
    assert events[-1].reason == "ok" and agent.history[-1]["content"] == "Hola, listo."


def test_stall_mid_stream_is_detected(tmp_path):
    _, events, _, elapsed = run(tmp_path, ["stall"])
    assert events[-1].reason == "error" and error(events).title == "El modelo dejó de responder"
    assert elapsed < 10


def test_connection_cut_mid_stream(tmp_path):
    _, events, _, _ = run(tmp_path, ["cut"])
    assert events[-1].reason == "error" and "cortó la conexión" in error(events).title


def test_out_of_memory_is_explained(tmp_path):
    _, events, _, _ = run(tmp_path, ["oom"])
    err = error(events)
    assert events[-1].reason == "error" and "memoria" in err.cause and err.action == "server"


def test_context_overflow_retries_then_explains(tmp_path):
    _, events, transport, _ = run(tmp_path, ["context"])
    assert events[-1].reason == "context" and transport.requests == 2


def test_refused_retries_once_then_recovers(tmp_path):
    _, events, transport, _ = run(tmp_path, ["refused", "ok"])
    assert events[-1].reason == "ok" and transport.requests == 2


def test_refused_twice_is_explained(tmp_path):
    _, events, _, _ = run(tmp_path, ["refused"])
    assert events[-1].reason == "error" and error(events).action == "server"


def test_cancel_while_model_loads_is_immediate(tmp_path):
    _, events, _, elapsed = run(tmp_path, ["hang_first"], cancel_after=0.1)
    assert events[-1].reason == "cancelled" and elapsed < 5


def test_cancel_during_stall(tmp_path):
    _, events, _, elapsed = run(tmp_path, ["stall"], cancel_after=0.1, idle_timeout=60)
    assert events[-1].reason == "cancelled" and elapsed < 5


def test_broken_tool_json_goes_back_to_model(tmp_path):
    _, events, _, _ = run(tmp_path, ["bad_tool_json", "ok"])
    res = next(e for e in events if isinstance(e, ToolResult))
    assert not res.ok and "JSON válido" in res.output and events[-1].reason == "ok"


def test_replacing_client_mid_stream_does_not_cut_the_answer():
    """Guardar la configuración a mitad de un turno (p. ej. al terminar una descarga) reemplazaba
    el cliente y cerraba el viejo: la respuesta en curso se cortaba con «Se cortó la conexión»."""
    async def main():
        llm = LLMClient("http://chaos/v1", transport=ChaosTransport(["slow"]))
        chat = asyncio.create_task(llm.chat("m", [{"role": "user", "content": "hola"}]))
        await asyncio.sleep(0.1)  # ya llegó el primer trozo
        assert llm.active == 1
        closer = asyncio.create_task(llm.close_when_idle())
        await asyncio.sleep(0.05)
        assert not closer.done()  # espera a que termine la respuesta
        result = await chat
        await asyncio.wait_for(closer, 2)
        return result, llm

    result, llm = asyncio.run(main())
    assert result.content == "Hola, listo." and llm.active == 0
    assert llm.client.is_closed()
