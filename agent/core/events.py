"""Eventos que emite el bucle del agente. La GUI y el CLI solo consumen esto."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TextDelta:
    text: str


@dataclass
class TextRewrite:
    """Sustituye el texto del mensaje en curso (tras rescatar una llamada escrita como texto)."""
    text: str


@dataclass
class ReasoningDelta:
    text: str


@dataclass
class ToolRequest:
    call_id: str
    name: str
    args: dict
    preview: str  # diff, comando o resumen de argumentos
    needs_approval: bool


@dataclass
class ToolResult:
    call_id: str
    name: str
    ok: bool
    output: str


@dataclass
class ContextUsage:
    used: int
    limit: int
    budget: int
    breakdown: dict[str, int] = field(default_factory=dict)


@dataclass
class Notice:
    text: str
    level: str = "info"  # info | warning


@dataclass
class AgentError:
    title: str
    cause: str
    detail: str = ""
    suggestions: list[str] = field(default_factory=list)
    action: str | None = None  # "server" | "settings" | "new_chat"


@dataclass
class Done:
    reason: str  # ok | cancelled | error | context | max_steps


Event = TextDelta | TextRewrite | ReasoningDelta | ToolRequest | ToolResult | ContextUsage | Notice | AgentError | Done
