"""Pruebas del bucle con un LLM falso: flujo de herramientas y todos los caminos de fallo."""

import asyncio

from agent.config import AppConfig, ModelEntry
from agent.core.errors import ContextOverflowFromServer, ServerUnavailable
from agent.core.events import AgentError, Done, FilesChanged, Notice, ToolRequest, ToolResult
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
        self.tools_seen = tools or []
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
    # Aquí se prueba el bucle, no la shell: abrir PowerShell cuesta ~1 s por comando.
    agent.toolbox.tool_run_command = lambda command, timeout=None: f"exit code: 0\n--- stdout ---\n{command}"
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


def _asked_agent(tmp_path, script, answer):
    """Agente que registra por qué herramientas se le preguntó al usuario."""
    agent, llm, events = make_agent(tmp_path, script)
    asked = []

    async def approve(req):
        asked.append(req.name)
        return answer

    agent.approve = approve
    return agent, asked


def test_approve_session_is_per_tool_and_not_saved(tmp_path):
    agent, asked = _asked_agent(tmp_path, [
        call("run_command", '{"command": "echo 1"}', cid="a"),
        call("run_command", '{"command": "echo 2"}', cid="b"),
        call("write_file", '{"path": "x.txt", "content": "hola"}', cid="c"),
        text("listo"),
    ], "session")
    asyncio.run(agent.run("haz cosas"))
    assert asked == ["run_command", "write_file"]  # el 2º run_command ya no pregunta
    assert agent.session_allowed == {"run_command", "write_file"}
    assert agent.cfg.always_allow == []


def test_approve_always_is_saved_and_revocable(tmp_path):
    agent, asked = _asked_agent(tmp_path, [
        call("run_command", '{"command": "echo 1"}', cid="a"),
        text("listo"),
    ], "always")
    saved = []
    agent.persist = lambda: saved.append(list(agent.cfg.always_allow))
    asyncio.run(agent.run("haz cosas"))
    assert agent.cfg.always_allow == ["run_command"] and saved == [["run_command"]]
    # un agente nuevo con la misma config (p. ej. tras reiniciar) no vuelve a preguntar
    agent2, asked2 = _asked_agent(tmp_path, [call("run_command", '{"command": "echo 2"}'), text("ok")], "no")
    agent2.cfg = agent.cfg
    asyncio.run(agent2.run("otra vez"))
    assert asked2 == []
    agent.cfg.always_allow.remove("run_command")  # quitar el permiso: vuelve a preguntar
    assert not agent2.is_allowed("run_command")


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
    agent, llm, events = make_agent(tmp_path, [ServerUnavailable("http://127.0.0.1:8080/v1"),
                                               ServerUnavailable("http://127.0.0.1:8080/v1")])
    agent.retry_delay = 0
    asyncio.run(agent.run("hola"))
    err = next(e for e in events if isinstance(e, AgentError))
    assert err.action == "server" and events[-1].reason == "error"
    assert len(llm.calls) == 2  # un reintento y se rinde


def test_server_unavailable_retries_once(tmp_path):
    agent, llm, events = make_agent(tmp_path, [ServerUnavailable("http://127.0.0.1:8080/v1"),
                                               text("ya estoy")])
    agent.retry_delay = 0
    asyncio.run(agent.run("hola"))
    assert any(isinstance(e, Notice) and "reintento" in e.text for e in events)
    assert events[-1].reason == "ok"


def test_repeated_identical_call_is_stopped(tmp_path):
    script = [call("read_file", '{"path": "calc.py"}', cid=f"c{i}") for i in range(6)]
    agent, llm, events = make_agent(tmp_path, script)
    asyncio.run(agent.run("lee"))
    assert any(isinstance(e, Notice) and "cambie de enfoque" in e.text for e in events)
    assert events[-1].reason == "loop" and len(llm.calls) == 4


def test_undo_restores_files_of_the_turn(tmp_path):
    agent, llm, events = make_agent(tmp_path, [
        call("edit_file", '{"path": "calc.py", "old": "a - b", "new": "a + b"}', cid="a"),
        call("write_file", '{"path": "nuevo.txt", "content": "hola"}', cid="b"),
        text("hecho"),
    ])
    agent.checkpoints_dir = tmp_path / ".cp"
    original = (tmp_path / "calc.py").read_bytes()
    asyncio.run(agent.run("cambia cosas"))
    changed = next(e for e in events if isinstance(e, FilesChanged))
    assert sorted(changed.files) == ["calc.py", "nuevo.txt"]
    # Otro agente (como tras reiniciar la app) puede deshacerlo desde disco.
    other = tmp_path / "otro"
    other.mkdir()
    agent2, _, _ = make_agent(other, [])
    agent2.checkpoints_dir = tmp_path / ".cp"
    report = agent2.undo(changed.checkpoint_id)
    assert "restaurados: calc.py" in report and "nuevo.txt" in report
    assert (tmp_path / "calc.py").read_bytes() == original and not (tmp_path / "nuevo.txt").exists()


def test_undo_skips_files_changed_afterwards(tmp_path):
    agent, llm, events = make_agent(tmp_path, [
        call("edit_file", '{"path": "calc.py", "old": "a - b", "new": "a + b"}'), text("hecho")])
    asyncio.run(agent.run("arregla"))
    (tmp_path / "calc.py").write_text("lo edité yo\n", encoding="utf-8")
    cid = next(e for e in events if isinstance(e, FilesChanged)).checkpoint_id
    assert "cambiaron después: calc.py" in agent.undo(cid)
    assert (tmp_path / "calc.py").read_text(encoding="utf-8") == "lo edité yo\n"


def test_exception_mid_tools_leaves_valid_history(tmp_path):
    agent, llm, events = make_agent(tmp_path, [
        ChatResult(tool_calls=[ToolCall("a", "run_command", '{"command": "echo 1"}'),
                               ToolCall("b", "run_command", '{"command": "echo 2"}')],
                   finish_reason="tool_calls")])

    async def broken_approve(req):
        raise RuntimeError("la ventana se cerró")

    agent.approve = broken_approve
    asyncio.run(agent.run("haz algo"))
    assert events[-1].reason == "error"
    ids = [m["tool_call_id"] for m in agent.history if m["role"] == "tool"]
    assert ids == ["a", "b"]  # cada llamada tiene su resultado aunque se cortara


def test_session_is_saved_and_restored(tmp_path):
    from agent.core import sessions
    agent, llm, events = make_agent(tmp_path, [text("hola, ¿qué tal?")])
    agent.sessions_dir = tmp_path / ".sessions"
    asyncio.run(agent.run("saluda"))
    saved = sessions.latest(tmp_path / ".sessions", tmp_path)
    assert saved and saved.title == "saluda" and saved.history[-1]["content"] == "hola, ¿qué tal?"
    agent2, _, _ = make_agent(tmp_path, [])
    agent2.restore(saved)
    assert agent2.history == agent.history and agent2.session_id == agent.session_id


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
