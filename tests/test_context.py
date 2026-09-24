import asyncio

import pytest

from agent.core.context import ContextManager, TokenCounter, truncate_middle
from agent.core.errors import ContextExhausted


def system_builder(summary):
    return "SYS" + (f"\nRESUMEN: {summary}" if summary else "")


def make_cm(n_ctx=2000, reserve=200, summarizer=None, hint=None):
    counter = TokenCounter(chars_per_token=1.0)  # 1 carácter = 1 token: cuentas fáciles
    return ContextManager(counter, n_ctx, reserve, system_builder, summarizer, margin=0.0, hint=hint)


def tool_step(i, size):
    call_id = f"c{i}"
    return [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": "read_file",
                                                             "arguments": f'{{"path": "f{i}.py"}}'}}]},
        {"role": "tool", "tool_call_id": call_id, "content": "x" * size},
    ]


def run(coro):
    return asyncio.run(coro)


def test_fits_without_changes():
    cm = make_cm()
    history = [{"role": "user", "content": "hola"}]
    res = run(cm.fit(history, None, None))
    assert res.history == history and not res.notices and res.used < cm.budget


def test_stage1_elides_old_tool_results_but_keeps_last_step():
    cm = make_cm(n_ctx=2500)
    history = [{"role": "user", "content": "tarea"}] + tool_step(1, 900) + tool_step(2, 900) + tool_step(3, 900)
    res = run(cm.fit(history, None, None))
    contents = [m["content"] for m in res.history if m["role"] == "tool"]
    assert contents[0].startswith("[resultado omitido")
    assert contents[-1] == "x" * 900  # el último resultado se conserva intacto
    assert any("Omití" in n for n in res.notices)
    assert res.used <= cm.budget


def test_stage2_summarizes_old_turns_and_keeps_current_request():
    calls = []

    async def summarizer(messages, previous):
        calls.append(messages)
        return "el usuario pidió A; se editó a.py"

    cm = make_cm(n_ctx=1500, summarizer=summarizer)
    old = []
    for i in range(6):
        old += [{"role": "user", "content": f"petición antigua {i} " + "y" * 150},
                {"role": "assistant", "content": "respuesta " + "z" * 150}]
    history = old + [{"role": "user", "content": "petición actual"}]
    res = run(cm.fit(history, None, None))
    assert calls, "debió llamar al resumidor"
    assert res.summary == "el usuario pidió A; se editó a.py"
    assert res.history[-1]["content"] == "petición actual"
    assert res.history[0]["role"] == "user"
    assert any("Resumí" in n for n in res.notices)
    assert res.used <= cm.budget


def test_stage2_without_summarizer_drops_with_note():
    async def broken(messages, previous):
        raise RuntimeError("fast caído")

    cm = make_cm(n_ctx=1200, summarizer=broken)
    history = []
    for _ in range(6):
        history += [{"role": "user", "content": "u" * 200}, {"role": "assistant", "content": "a" * 200}]
    history.append({"role": "user", "content": "actual"})
    res = run(cm.fit(history, None, None))
    assert "Se descartaron" in res.summary
    assert any("sin resumen" in n for n in res.notices)


def test_stage3_truncates_huge_last_message():
    cm = make_cm(n_ctx=3000)
    history = [{"role": "user", "content": "lee esto"}] + tool_step(1, 20000)
    res = run(cm.fit(history, None, None))
    last = res.history[-1]["content"]
    assert len(last) < 3000 and "se omitieron las líneas" in last
    assert any("Recorté" in n for n in res.notices)
    assert res.used <= cm.budget


def test_exhausted_raises_with_breakdown_and_hint():
    cm = make_cm(n_ctx=300, reserve=100, hint=lambda n: "sube el contexto")
    tools = [{"type": "function", "function": {"name": "t", "description": "d" * 500}}]
    with pytest.raises(ContextExhausted) as exc:
        run(cm.fit([{"role": "user", "content": "hola"}], None, tools))
    e = exc.value
    assert e.breakdown["herramientas"] > 300
    assert e.suggestions[0] == "sube el contexto"
    assert "tokens" in e.cause


def test_aggressive_mode_uses_smaller_budget():
    cm = make_cm(n_ctx=4000)
    history = [{"role": "user", "content": "x" * 3500}]
    normal = run(cm.fit(history, None, None))
    assert not normal.notices  # cabe en modo normal
    res = run(cm.fit(history, None, None, aggressive=True))
    assert res.notices and res.used <= int(cm.budget * 0.8)


def test_counter_calibration_and_overflow():
    c = TokenCounter(chars_per_token=4.0)
    base = c.estimate("a" * 400)
    c.calibrate(base, base * 2)
    assert c.estimate("a" * 400) > base
    before = c.factor
    c.overflowed()
    assert c.factor > before


def test_truncate_middle_note():
    text = "\n".join(f"línea {i}" for i in range(1, 501))
    out = truncate_middle(text, 800)
    assert out.startswith("línea 1") and out.rstrip().endswith("línea 500")
    assert "read_file" in out
