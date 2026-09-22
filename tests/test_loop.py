"""Pruebas del bucle con un LLM falso: flujo de herramientas y todos los caminos de fallo."""

import asyncio

from agent.config import AppConfig, ModelEntry
from agent.core.errors import ContextOverflowFromServer, ServerUnavailable
from agent.core.events import AgentError, Done, Notice, ToolRequest, ToolResult
from agent.core.llm import ChatResult, ToolCall
from agent.core.loop import Agent, extract_text_tool_calls


class FakeLLM:
    def __init__(self, script, n_ctx=8192):
        self.script = list(script)
        self.calls = []
        self._n_ctx = n_ctx

    async def n_ctx(self, model):
        return self._n_ctx

    async def tokenize(self, model, text):
        return None  # fuerza la estimación

    async def complete(self, model, messages, max_tokens=800):
        return "resumen"

    async def chat(self, model, messages, tools=None, max_tokens=4096, on_text=None,
                   on_reasoning=None, cancel=None):
        self.calls.append(messages)
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        if on_text and step.content:
            await on_text(step.content)
        return step


def make_agent(tmp_path, script, approve_answer="yes", n_ctx=8192, confirm="ask"):
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    cfg = AppConfig(workspace=str(tmp_path), roles={"main": "m", "fast": "", "draft": ""},
                    models=[ModelEntry("m", "m.gguf", ctx=n_ctx)], confirm=confirm,
                    max_output_tokens=512)
    events = []

    async def approve(req):
        return approve_answer

    llm = FakeLLM(script, n_ctx)
    agent = Agent(cfg, llm, events.append, approve)
    return agent, llm, events


def text(content, finish="stop"):
    return ChatResult(content=content, finish_reason=finish)


def call(name, args_json, finish="tool_calls", cid="c1"):
    return ChatResult(tool_calls=[ToolCall(cid, name, args_json)], finish_reason=finish)


def kinds(events):
    return [type(e).__name__ for e in events]


def test_tool_flow_edit_approved(tmp_path):
    agent, llm, events = make_agent(tmp_path, [
        call("edit_file", '{"path": "calc.py", "old": "a - b", "new": "a + b"}'),
        text("Arreglado."),
    ])
    asyncio.run(agent.run("arregla add"))
    assert "a + b" in (tmp_path / "calc.py").read_text(encoding="utf-8")
    req = next(e for e in events if isinstance(e, ToolRequest))
    assert req.needs_approval and "+    return a + b" in req.preview
    res = next(e for e in events if isinstance(e, ToolResult))
    assert res.ok
    assert isinstance(events[-1], Done) and events[-1].reason == "ok"
    roles = [m["role"] for m in agent.history]
    assert roles == ["user", "assistant", "tool", "assistant"]


def test_tool_rejected_is_not_executed(tmp_path):
    agent, llm, events = make_agent(tmp_path, [
        call("run_command", '{"command": "echo peligro"}'),
        text("Vale, no lo ejecuto."),
    ], approve_answer="no")
    asyncio.run(agent.run("ejecuta algo"))
    res = next(e for e in events if isinstance(e, ToolResult))
    assert not res.ok and "rechazó" in res.output


def test_invalid_json_arguments_go_back_to_model(tmp_path):
    agent, llm, events = make_agent(tmp_path, [
        call("read_file", '{"path": calc.py}'),
        text("ok"),
    ])
    asyncio.run(agent.run("lee"))
    res = next(e for e in events if isinstance(e, ToolResult))
    assert not res.ok and "JSON válido" in res.output
    assert events[-1].reason == "ok"


def test_server_overflow_compacts_and_retries_once(tmp_path):
    agent, llm, events = make_agent(tmp_path, [
        ContextOverflowFromServer("exceed_context_size_error"),
        text("hecho"),
    ])
    asyncio.run(agent.run("hola"))
    assert any(isinstance(e, Notice) and "reintento" in e.text for e in events)
    assert events[-1].reason == "ok"


def test_server_overflow_twice_explains(tmp_path):
    agent, llm, events = make_agent(tmp_path, [
        ContextOverflowFromServer("exceed_context_size_error"),
        ContextOverflowFromServer("exceed_context_size_error"),
    ])
    asyncio.run(agent.run("hola"))
    err = next(e for e in events if isinstance(e, AgentError))
    assert err.title == "Sin espacio en el contexto" and err.suggestions
    assert events[-1].reason == "context"


def test_context_exhausted_before_calling_model(tmp_path):
    agent, llm, events = make_agent(tmp_path, [text("nunca")], n_ctx=1024)
    asyncio.run(agent.run("hola"))
    # Con 1K de contexto ni siquiera caben el sistema y las herramientas: se explica sin llamar.
    assert not llm.calls
    err = next(e for e in events if isinstance(e, AgentError))
    assert "herramientas" in err.cause and events[-1].reason == "context"


def test_truncated_tool_call_is_not_executed(tmp_path):
    agent, llm, events = make_agent(tmp_path, [
        call("write_file", '{"path": "big.py", "content": "print(1)\\nprint(2', finish="length"),
        text("Lo divido."),
    ])
    asyncio.run(agent.run("crea un archivo enorme"))
    assert not (tmp_path / "big.py").exists()
    assert any(isinstance(e, Notice) and "se cortó" in e.text for e in events)
    assert any(m["role"] == "user" and "NO se ejecutó" in m["content"] for m in agent.history)
    assert events[-1].reason == "ok"


def test_truncated_text_continues_once(tmp_path):
    agent, llm, events = make_agent(tmp_path, [
        text("primera parte", finish="length"),
        text(" y final", finish="stop"),
    ])
    asyncio.run(agent.run("explica"))
    assert len(llm.calls) == 2
    assert agent.history[-1]["content"] == " y final"
    assert events[-1].reason == "ok"


def test_server_unavailable_is_explained(tmp_path):
    agent, llm, events = make_agent(tmp_path, [ServerUnavailable("http://127.0.0.1:8080/v1")])
    asyncio.run(agent.run("hola"))
    err = next(e for e in events if isinstance(e, AgentError))
    assert err.action == "server" and events[-1].reason == "error"


def test_unexpected_exception_never_escapes(tmp_path):
    agent, llm, events = make_agent(tmp_path, [ZeroDivisionError("boom")])
    asyncio.run(agent.run("hola"))
    err = next(e for e in events if isinstance(e, AgentError))
    assert "ZeroDivisionError" in err.cause and events[-1].reason == "error"


def test_max_steps(tmp_path):
    script = [call("list_files", "{}", cid=f"c{i}") for i in range(3)]
    agent, llm, events = make_agent(tmp_path, script)
    agent.cfg.max_steps = 3
    asyncio.run(agent.run("bucle"))
    assert events[-1].reason == "max_steps"


def test_text_tool_call_rescue():
    content = 'Voy a leerlo.\n<tool_call>\n{"name": "read_file", "arguments": {"path": "a.py"}}\n</tool_call>'
    rest, calls = extract_text_tool_calls(content)
    assert rest == "Voy a leerlo." and calls[0].name == "read_file"
    assert calls[0].parse_args() == {"path": "a.py"}


def test_text_tool_call_rescue_qwen_coder_formats():
    # <tools> (lo que devuelve Qwen2.5-Coder con llama.cpp)
    rest, calls = extract_text_tool_calls('<tools>\n{"name": "read_file", "arguments": {"path": "calc.py"}}\n</tools>')
    assert rest == "" and calls[0].parse_args() == {"path": "calc.py"}
    # Varios ```json seguidos con resultados inventados: solo la primera, y se corta lo demás
    content = ("Paso 1:\n```json\n{\"name\": \"list_files\", \"arguments\": {\"pattern\": \"*.py\"}}\n```\n"
               "Paso 2:\n```json\n{\"name\": \"write_file\", \"arguments\": {\"path\": \"x\", \"content\": \"y\"}}\n```\n"
               "Listo, creé los archivos y los tests pasan.")
    rest, calls = extract_text_tool_calls(content)
    assert rest == "Paso 1:" and len(calls) == 1 and calls[0].name == "list_files"
    # JSON suelto, sin envoltorio
    rest, calls = extract_text_tool_calls('Ahora lo leo.\n\n{"name": "read_file", "arguments": {"path": "calc.py"}}\n')
    assert rest == "Ahora lo leo." and calls[0].name == "read_file"
    # Un bloque JSON que no es una llamada no se toca
    plain = 'Ejemplo:\n```json\n{"name": "Ana", "edad": 3}\n```'
    assert extract_text_tool_calls(plain) == (plain, [])


def test_rescued_text_call_is_executed(tmp_path):
    agent, llm, events = make_agent(tmp_path, [
        text('<tools>\n{"name": "read_file", "arguments": {"path": "calc.py"}}\n</tools>'),
        text("Hay un bug: resta en vez de sumar."),
    ])
    asyncio.run(agent.run("revisa calc.py"))
    res = next(e for e in events if isinstance(e, ToolResult))
    assert res.ok and "return a - b" in res.output
    assert agent.history[1]["tool_calls"][0]["function"]["name"] == "read_file"
    assert events[-1].reason == "ok"
