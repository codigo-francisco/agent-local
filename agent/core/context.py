"""Gestión de la ventana de contexto: medir, compactar de forma escalonada y, si no hay forma,
explicar por qué (ContextExhausted) en lugar de dejar que el servidor falle.

Escalones de `fit`, en orden y solo mientras no quepa:
  1. Omitir resultados de herramientas antiguos (y argumentos enormes de llamadas antiguas).
  2. Resumir los turnos más antiguos (con el modelo `fast`) dentro del prompt de sistema.
  3. Recortar por el medio los mensajes más grandes que queden, con una nota para el modelo.
  4. Rendirse con un desglose: ContextExhausted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from .errors import ContextExhausted

MSG_OVERHEAD = 6  # tokens de la plantilla de chat por mensaje
SUMMARY_ESTIMATE = 400  # tokens que suponemos que ocupará un resumen nuevo
SUMMARY_MAX_CHARS = 3000
MIN_TRUNCATE_CHARS = 1500  # no recortamos mensajes más pequeños que esto
ELIDED_PREFIX = "[resultado omitido"


class TokenCounter:
    """Cuenta tokens con el endpoint /tokenize del servidor si está disponible, si no estima.

    `prime` es async (consulta al servidor) y rellena una caché; `count` es síncrono y usa la
    caché o la estimación calibrada con el `usage` real de las respuestas.
    """

    def __init__(self, tokenize: Callable[[str], Awaitable[int | None]] | None = None,
                 chars_per_token: float = 3.2):
        self._tokenize = tokenize
        self._cache: dict[int, int] = {}
        self.chars_per_token = chars_per_token
        self.factor = 1.0
        self.exact = tokenize is not None

    def estimate(self, text: str) -> int:
        return int(len(text) / self.chars_per_token * self.factor) + 1 if text else 0

    def count(self, text: str) -> int:
        if not text:
            return 0
        cached = self._cache.get(hash(text))
        return cached if cached is not None else self.estimate(text)

    async def prime(self, texts: list[str]) -> None:
        if not self.exact or self._tokenize is None:
            return
        for t in texts:
            if not t or hash(t) in self._cache:
                continue
            try:
                n = await self._tokenize(t)
            except Exception:  # noqa: BLE001 - sin tokenizer seguimos estimando
                n = None
            if n is None:
                self.exact = False
                return
            self._cache[hash(t)] = n

    def calibrate(self, estimated: int, actual: int) -> None:
        """Ajusta la estimación con el número real de tokens del prompt que dio el servidor."""
        if estimated <= 0 or actual <= 0:
            return
        ratio = actual / estimated
        if abs(ratio - 1) > 0.05:
            self.factor = min(3.0, max(0.5, self.factor * ratio ** 0.8))

    def overflowed(self) -> None:
        """El servidor dijo que no cabía: nuestras cuentas se quedaron cortas."""
        self.factor = min(3.0, self.factor * 1.25)
        self._cache.clear()


def message_text(msg: dict) -> str:
    text = msg.get("content") or ""
    if not isinstance(text, str):
        text = json.dumps(text, ensure_ascii=False)
    text += msg.get("reasoning_content") or ""
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", {})
        text += fn.get("name", "") + (fn.get("arguments") or "")
    return text


@dataclass
class FitResult:
    history: list[dict]
    summary: str | None
    notices: list[str] = field(default_factory=list)
    breakdown: dict[str, int] = field(default_factory=dict)

    @property
    def used(self) -> int:
        return sum(self.breakdown.values())


Summarizer = Callable[[list[dict], str | None], Awaitable[str | None]]


class ContextManager:
    def __init__(
        self,
        counter: TokenCounter,
        n_ctx: int,
        reserve_output: int,
        system_builder: Callable[[str | None], str],
        summarizer: Summarizer | None = None,
        margin: float = 0.05,
        hint: Callable[[int], str | None] | None = None,
    ):
        self.counter = counter
        self.n_ctx = n_ctx
        self.reserve_output = reserve_output
        self.system_builder = system_builder
        self.summarizer = summarizer
        self.margin = margin
        self.hint = hint

    @property
    def budget(self) -> int:
        return int(self.n_ctx * (1 - self.margin)) - self.reserve_output

    # --- medición --------------------------------------------------------
    def msg_tokens(self, msg: dict) -> int:
        return self.counter.count(message_text(msg)) + MSG_OVERHEAD

    def breakdown(self, history: list[dict], summary: str | None, tools: list[dict] | None) -> dict[str, int]:
        system = self.system_builder(summary)
        return {
            "sistema": self.counter.count(system) + MSG_OVERHEAD,
            "herramientas": self.counter.count(json.dumps(tools, ensure_ascii=False)) if tools else 0,
            "historial": sum(self.msg_tokens(m) for m in history[:-1]),
            "último mensaje": self.msg_tokens(history[-1]) if history else 0,
        }

    def total(self, history: list[dict], summary: str | None, tools: list[dict] | None) -> int:
        return sum(self.breakdown(history, summary, tools).values())

    def texts_for_priming(self, history: list[dict], summary: str | None, tools: list[dict] | None) -> list[str]:
        texts = [self.system_builder(summary), *(message_text(m) for m in history)]
        if tools:
            texts.append(json.dumps(tools, ensure_ascii=False))
        return texts

    # --- compactación ----------------------------------------------------
    async def fit(self, history: list[dict], summary: str | None, tools: list[dict] | None,
                  aggressive: bool = False) -> FitResult:
        budget = int(self.budget * (0.8 if aggressive else 1.0))
        history = list(history)
        notices: list[str] = []

        def fits() -> bool:
            return self.total(history, summary, tools) <= budget

        if not fits():
            self._elide_old(history, fits, notices, aggressive)
        if not fits():
            history, summary = await self._summarize_old(history, summary, tools, budget, notices)
        if not fits():
            self._truncate_largest(history, summary, tools, budget, notices)
        bd = self.breakdown(history, summary, tools)
        if sum(bd.values()) > budget:
            raise ContextExhausted(bd, self.n_ctx, budget, self.hint(self.n_ctx) if self.hint else None)
        return FitResult(history, summary, notices, bd)

    def _last_step_start(self, history: list[dict]) -> int:
        """Índice del último mensaje user/assistant: ese paso nunca se omite ni resume."""
        for i in range(len(history) - 1, -1, -1):
            if history[i]["role"] in ("user", "assistant"):
                return i
        return 0

    def _elide_old(self, history: list[dict], fits: Callable[[], bool], notices: list[str],
                   aggressive: bool) -> None:
        # Normalmente el último paso se respeta; en modo agresivo solo el mensaje final.
        protect_from = len(history) - 1 if aggressive else self._last_step_start(history)
        calls = {tc["id"]: tc["function"] for m in history for tc in (m.get("tool_calls") or [])}
        count, freed = 0, 0
        for i in range(protect_from):
            if fits():
                break
            m = history[i]
            if m.get("reasoning_content"):  # el razonamiento de pasos viejos es lo más prescindible
                before = self.msg_tokens(m)
                m = history[i] = {k: v for k, v in m.items() if k != "reasoning_content"}
                freed += before - self.msg_tokens(m)
                count += 1
                if fits():
                    break
            if m["role"] == "tool" and not (m.get("content") or "").startswith(ELIDED_PREFIX):
                before = self.msg_tokens(m)
                if before < 150:
                    continue
                fn = calls.get(m.get("tool_call_id"), {})
                label = f"{fn.get('name', 'herramienta')} {_short_args(fn.get('arguments', ''))}"
                history[i] = {**m, "content": f"{ELIDED_PREFIX} para ahorrar contexto: {label}, "
                                             f"~{before} tokens. Vuelve a ejecutarla si lo necesitas.]"}
                freed += before - self.msg_tokens(history[i])
                count += 1
            elif m["role"] == "assistant" and m.get("tool_calls"):
                new_calls, changed = [], False
                for tc in m["tool_calls"]:
                    args = tc["function"].get("arguments") or ""
                    if len(args) > 2000:
                        tc = {**tc, "function": {**tc["function"], "arguments": _shrink_args(args)}}
                        changed = True
                    new_calls.append(tc)
                if changed:
                    before = self.msg_tokens(m)
                    history[i] = {**m, "tool_calls": new_calls}
                    freed += before - self.msg_tokens(history[i])
                    count += 1
        if count:
            notices.append(f"Omití {count} resultados, argumentos o razonamientos antiguos "
                           f"(≈{freed:,} tokens liberados).".replace(",", "."))

    async def _summarize_old(self, history: list[dict], summary: str | None, tools: list[dict] | None,
                             budget: int, notices: list[str]) -> tuple[list[dict], str | None]:
        last_user = max((i for i, m in enumerate(history) if m["role"] == "user"), default=-1)
        max_b = self._last_step_start(history)
        candidates = [b for b in range(1, max_b + 1)
                      if (b <= last_user and history[b]["role"] == "user")
                      or (b > last_user and history[b]["role"] == "assistant")]
        if not candidates:
            return history, summary

        def rebuild(b: int) -> list[dict]:
            keep_user = [history[last_user]] if 0 <= last_user < b else []
            return keep_user + history[b:]

        chosen = candidates[-1]
        for b in candidates:
            if self.total(rebuild(b), summary, tools) + SUMMARY_ESTIMATE <= budget:
                chosen = b
                break
        removed = [m for i, m in enumerate(history[:chosen]) if i != last_user]
        if not removed:
            return history, summary
        new_history = rebuild(chosen)
        freed = self.total(history, summary, tools) - self.total(new_history, summary, tools)

        new_summary = None
        if self.summarizer is not None:
            try:
                new_summary = await self.summarizer(removed, summary)
            except Exception:  # noqa: BLE001 - un resumen fallido no debe tumbar la tarea
                new_summary = None
        if new_summary:
            summary = new_summary.strip()[:SUMMARY_MAX_CHARS]
            notices.append(f"Resumí {len(removed)} mensajes antiguos para liberar ≈{freed:,} tokens."
                           .replace(",", "."))
        else:
            note = f"(Se descartaron {len(removed)} mensajes anteriores por falta de contexto.)"
            summary = f"{summary}\n{note}" if summary else note
            notices.append(f"Descarté {len(removed)} mensajes antiguos sin resumen (el modelo de "
                           f"resúmenes no respondió); liberé ≈{freed:,} tokens.".replace(",", "."))
        return new_history, summary

    def _truncate_largest(self, history: list[dict], summary: str | None, tools: list[dict] | None,
                          budget: int, notices: list[str]) -> None:
        stuck: set[int] = set()  # mensajes que ya no se pueden recortar más
        for _ in range(len(history) * 3):
            excess = self.total(history, summary, tools) - budget
            if excess <= 0:
                return
            sizes = [(self.msg_tokens(m), i) for i, m in enumerate(history)
                     if i not in stuck and isinstance(m.get("content"), str)
                     and len(m["content"]) > MIN_TRUNCATE_CHARS]
            if not sizes:
                return
            _, i = max(sizes)
            content = history[i]["content"]
            chars_per_tok = len(content) / max(1, self.counter.count(content))
            target = max(MIN_TRUNCATE_CHARS, len(content) - int(excess * chars_per_tok * 1.1) - 400)
            new_content = truncate_middle(content, target) if target < len(content) else content
            if len(new_content) >= len(content):
                stuck.add(i)
                continue
            history[i] = {**history[i], "content": new_content}
            who = {"tool": "un resultado de herramienta", "user": "tu mensaje",
                   "assistant": "una respuesta anterior"}.get(history[i]["role"], "un mensaje")
            notices.append(f"Recorté {who} de {len(content):,} a {target:,} caracteres para que quepa "
                           f"en el contexto.".replace(",", "."))


MISSING_RESULT = "[sin resultado: la ejecución se interrumpió antes de terminar]"


def repair_history(history: list[dict]) -> tuple[list[dict], int]:
    """Garantiza el invariante que exigen las plantillas de chat: cada `tool_call` de un mensaje
    assistant va seguida de su mensaje `tool`, y no hay mensajes `tool` sueltos. Si una tarea se
    cortó a mitad (excepción, cierre), el historial podía quedar roto y el servidor respondería con
    un 400 críptico en el siguiente turno. Devuelve (historial reparado, nº de arreglos)."""
    out: list[dict] = []
    fixes = 0
    pending: list[str] = []  # ids de llamadas del último assistant aún sin resultado

    def close_pending() -> None:
        nonlocal fixes
        for cid in pending:
            out.append({"role": "tool", "tool_call_id": cid, "content": MISSING_RESULT})
            fixes += 1
        pending.clear()

    for m in history:
        role = m.get("role")
        if role == "tool":
            cid = m.get("tool_call_id")
            if cid in pending:
                pending.remove(cid)
                out.append(m)
            else:
                fixes += 1  # huérfano o duplicado: fuera
            continue
        close_pending()
        out.append(m)
        if role == "assistant":
            pending = [tc.get("id") for tc in m.get("tool_calls") or []]
    close_pending()
    return out, fixes


def truncate_middle(text: str, target: int) -> str:
    head = int(target * 0.6)
    tail = max(0, target - head)
    first_omitted = text[:head].count("\n") + 2
    last_omitted = max(first_omitted, text.count("\n") - text[len(text) - tail:].count("\n"))
    note = (f"\n[... se omitieron las líneas {first_omitted}-{last_omitted} de este texto "
            f"(~{len(text) - head - tail:,} caracteres) por falta de contexto. Si necesitas esa "
            f"parte, léela por rangos con read_file(start, end) ...]\n").replace(",", ".")
    return text[:head] + note + (text[-tail:] if tail else "")


def _short_args(arguments: str) -> str:
    try:
        args = json.loads(arguments or "{}")
    except ValueError:
        return ""
    if not isinstance(args, dict):
        return ""
    return ", ".join(f"{k}={str(v)[:40]}" for k, v in args.items() if len(str(v)) < 200)[:120]


def _shrink_args(arguments: str) -> str:
    try:
        args = json.loads(arguments)
    except ValueError:
        return json.dumps({"_omitido": f"argumentos de {len(arguments)} caracteres"})
    if not isinstance(args, dict):
        return json.dumps({"_omitido": f"argumentos de {len(arguments)} caracteres"})
    return json.dumps({k: (v if len(str(v)) < 300 else f"[omitido: {len(str(v))} caracteres]")
                       for k, v in args.items()}, ensure_ascii=False)
