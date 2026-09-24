"""CLI mínimo: el mismo agente que la GUI, en la terminal. Requiere el servidor en marcha."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Callable

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax

from . import config, log
from .config import CONFIG_FILE, GENERATED_DIR, AppConfig, load_config
from .core.events import (AgentError, ContextUsage, Done, Event, FilesChanged, Notice,
                          ReasoningDelta, TextDelta, ToolRequest, ToolResult)
from .core.llm import LLMClient
from .core.loop import Agent
from .core.mcp_tools import MCPManager

console = Console()
last_change: list[str] = []  # id del último checkpoint con cambios (para /deshacer)


def make_persist(cfg: AppConfig, path: Path = CONFIG_FILE) -> Callable[[], None]:
    """Guarda solo los permisos «siempre». `cfg` lleva aplicados --workspace y --auto, que son
    de esta ejecución: guardarlo entero dejaría la GUI en modo automático para siempre."""
    def persist() -> None:
        disk = load_config(path)
        for tool in cfg.always_allow:
            if tool not in disk.always_allow:
                disk.always_allow.append(tool)
        disk.save(path)
    return persist


def on_event(ev: Event) -> None:
    if isinstance(ev, TextDelta):
        console.print(ev.text, end="", markup=False, highlight=False)
    elif isinstance(ev, ReasoningDelta):
        pass
    elif isinstance(ev, ToolRequest):
        console.print()
        console.print(f"[bold cyan]🔧 {ev.name}[/] {', '.join(f'{k}={str(v)[:60]}' for k, v in ev.args.items() if k not in ('content', 'old', 'new'))}")
        if ev.needs_approval:
            console.print(Syntax(ev.preview, "diff" if ev.name != "run_command" else "powershell",
                                 word_wrap=True))
    elif isinstance(ev, ToolResult):
        color = "green" if ev.ok else "red"
        first = ev.output.splitlines()[0] if ev.output else ""
        console.print(f"[{color}]  ↳ {first[:150]}[/]")
    elif isinstance(ev, ContextUsage):
        console.print(f"[dim]contexto {ev.used:,}/{ev.budget:,} tokens[/]")
    elif isinstance(ev, Notice):
        console.print(f"\n[yellow]⚠ {ev.text}[/]")
    elif isinstance(ev, AgentError):
        body = ev.cause + ("\n\n" + "\n".join(f"• {s}" for s in ev.suggestions) if ev.suggestions else "")
        console.print(Panel(body, title=f"[red]{ev.title}[/]", border_style="red"))
    elif isinstance(ev, FilesChanged):
        last_change[:] = [ev.checkpoint_id]
        console.print(f"\n[dim]Archivos cambiados: {', '.join(ev.files)} (/deshacer para revertir)[/]")
    elif isinstance(ev, Done):
        console.print()


async def approve(ev: ToolRequest) -> str:
    answer = await asyncio.to_thread(
        console.input, f"[bold]¿Aprobar {ev.name}? [s]í / [n]o / [e]sta sesión / [p]ara siempre: [/]")
    return {"s": "yes", "si": "yes", "sí": "yes", "y": "yes", "e": "session",
            "p": "always"}.get(answer.strip().lower(), "no")


async def main_async(args: argparse.Namespace) -> None:
    cfg = load_config()
    if config.load_warning:
        console.print(f"[yellow]⚠ {config.load_warning}[/]")
    if args.workspace:
        cfg.workspace = args.workspace
    if args.auto:
        cfg.confirm = "auto"
    llm = LLMClient(cfg.endpoint)
    mcp = MCPManager()
    await mcp.start()
    for s in mcp.summary():
        detail = f"{len(s['tools'])} herramientas" if s["status"] == "conectado" else (s["error"] or s["status"])
        console.print(f"[dim]MCP {s['name']}: {detail}[/]")
    agent = Agent(cfg, llm, on_event, approve, mcp=mcp, persist=make_persist(cfg),
                  checkpoints_dir=GENERATED_DIR / "checkpoints")
    console.print(Markdown(f"**Agente local** · workspace `{agent.toolbox.workspace}` · modelo `{args.model}`\n\n"
                           "Escribe tu petición. `/nuevo` reinicia la conversación, `/deshacer` "
                           "revierte los archivos del último turno, `/salir` termina."))
    try:
        while True:
            text = (await asyncio.to_thread(console.input, "\n[bold green]> [/]")).strip()
            if not text:
                continue
            if text in ("/salir", "/exit", "/quit"):
                break
            if text in ("/nuevo", "/new"):
                agent.reset()
                console.print("[dim]Conversación nueva.[/]")
                continue
            if text in ("/deshacer", "/undo"):
                if not last_change:
                    console.print("[dim]No hay cambios que deshacer.[/]")
                else:
                    console.print(f"[dim]{agent.undo(last_change.pop())}[/]")
                continue
            await agent.run(text, args.model)
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        await mcp.stop()
        await llm.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(description="Agente de programación local (CLI)")
    parser.add_argument("--workspace", "-w", help="carpeta del proyecto (por defecto la de la config)")
    parser.add_argument("--model", "-m", default="main", help="rol o nombre de modelo (main, fast, ...)")
    parser.add_argument("--auto", action="store_true", help="no pedir confirmación")
    log.setup()
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
