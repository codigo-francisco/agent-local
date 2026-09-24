"""MCP de extremo a extremo con un servidor real (stdio) y el agente con un LLM simulado."""

import asyncio
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from agent.core.events import ToolRequest, ToolResult  # noqa: E402
from agent.core.mcp_tools import MCPManager  # noqa: E402
from test_loop import call, make_agent, text  # noqa: E402

SERVER = str(Path(__file__).with_name("mcp_demo_server.py"))


def write_config(tmp_path, servers: dict) -> Path:
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps({"mcpServers": servers}), encoding="utf-8")
    return path


@pytest.mark.slow
def test_manager_connects_lists_and_calls(tmp_path):
    async def run():
        mgr = MCPManager(write_config(tmp_path, {
            "demo": {"command": sys.executable, "args": [SERVER]},
            "roto": {"command": "no-existe-este-programa"},
            "apagado": {"command": "x", "disabled": True},
        }))
        await mgr.start()
        try:
            status = {s["name"]: s for s in mgr.summary()}
            assert status["demo"]["status"] == "conectado"
            assert set(status["demo"]["tools"]) == {"sumar", "fallar"}
            assert status["roto"]["status"] == "error" and status["roto"]["error"]
            assert status["apagado"]["status"] == "desactivado"
            names = {s["function"]["name"] for s in mgr.schemas()}
            assert names == {"mcp__demo__sumar", "mcp__demo__fallar"}
            assert mgr.needs_approval("mcp__demo__sumar")
            assert await mgr.call("mcp__demo__sumar", {"a": 2, "b": 3}) == (True, "5")
            ok, out = await mgr.call("mcp__demo__fallar", {})
            assert not ok and out.startswith("Error") and "fallar" in out  # el SDK oculta el detalle
        finally:
            await mgr.stop()
    asyncio.run(run())


def test_known_server_errors_get_a_plain_explanation():
    from agent.core.mcp_tools import _known_hint
    busy = ("codebase-memory-mcp: CBM daemon endpoint is held by pid 22596 but that process answered "
            "no rendezvous within 30000 ms")
    assert "compartido" in _known_hint(busy)
    assert "perfil de usuario" in _known_hint("owner is another account, a trusted identity required")
    assert _known_hint("algo desconocido") is None


@pytest.mark.slow
def test_agent_uses_mcp_tool_with_approval(tmp_path):
    async def run():
        mgr = MCPManager(write_config(tmp_path, {"demo": {"command": sys.executable, "args": [SERVER]}}))
        await mgr.start()
        try:
            agent, llm, events = make_agent(tmp_path, [
                call("mcp__demo__sumar", '{"a": 40, "b": 2}'),
                text("La suma es 42."),
            ])
            agent.mcp = mgr
            await agent.run("suma 40 y 2 con la herramienta")
            req = next(e for e in events if isinstance(e, ToolRequest))
            assert req.needs_approval and '"a": 40' in req.preview
            res = next(e for e in events if isinstance(e, ToolResult))
            assert res.ok and res.output == "42"
            sent_tools = {t["function"]["name"] for t in llm.tools_seen}
            assert "mcp__demo__sumar" in sent_tools and "read_file" in sent_tools
        finally:
            await mgr.stop()
    asyncio.run(run())
