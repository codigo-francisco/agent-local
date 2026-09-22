"""CLI mínimo: el mismo agente que la GUI, en la terminal. Requiere el servidor en marcha."""

from __future__ import annotations

import argparse
import asyncio

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax

from .config import load_config
from .core.events import (AgentError, ContextUsage, Done, Event, Notice, ReasoningDelta, TextDelta,
                          ToolRequest, ToolResult)
from .core.llm import LLMClient
from .core.loop import Agent

console = Console()


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
    elif isinstance(ev, Done):
        console.print()


async def approve(ev: ToolRequest) -> str:
    answer = await asyncio.to_thread(console.input, "[bold]¿Aprobar? [s]í / [n]o / [t]odo: [/]")
    return {"s": "yes", "si": "yes", "sí": "yes", "y": "yes", "t": "always"}.get(answer.strip().lower(), "no")


async def main_async(args: argparse.Namespace) -> None:
    cfg = load_config()
    if args.workspace:
        cfg.workspace = args.workspace
    if args.auto:
        cfg.confirm = "auto"
    llm = LLMClient(cfg.endpoint)
    agent = Agent(cfg, llm, on_event, approve)
    console.print(Markdown(f"**Agente local** · workspace `{agent.toolbox.workspace}` · modelo `{args.model}`\n\n"
                           "Escribe tu petición. `/nuevo` reinicia la conversación, `/salir` termina."))
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
            await agent.run(text, args.model)
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        await llm.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(description="Agente de programación local (CLI)")
    parser.add_argument("--workspace", "-w", help="carpeta del proyecto (por defecto la de la config)")
    parser.add_argument("--model", "-m", default="main", help="rol o nombre de modelo (main, fast, ...)")
    parser.add_argument("--auto", action="store_true", help="no pedir confirmación")
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
