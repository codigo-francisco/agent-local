"""Registro a archivo: en la ventana nativa no hay consola, así que sin esto las trazas se pierden.

`generated/logs/agent.log`, rotando a 1 MB × 3. Además captura excepciones no controladas del
hilo principal, de otros hilos y del bucle asyncio.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .config import GENERATED_DIR

LOG_DIR = GENERATED_DIR / "logs"
LOG_FILE = LOG_DIR / "agent.log"

log = logging.getLogger("agent")
_configured = False


def setup(path: Path = LOG_FILE, level: int = logging.INFO) -> None:
    global _configured
    if _configured:
        return
    _configured = True
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler: logging.Handler = RotatingFileHandler(path, maxBytes=1_000_000, backupCount=3,
                                                       encoding="utf-8", delay=True)
    except OSError:  # sin permisos de escritura: al menos a stderr
        handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    log.addHandler(handler)
    log.setLevel(level)
    logging.captureWarnings(True)

    previous = sys.excepthook

    def excepthook(tp, value, tb):
        log.critical("Excepción no controlada", exc_info=(tp, value, tb))
        previous(tp, value, tb)

    sys.excepthook = excepthook
    threading.excepthook = lambda a: log.error(
        "Excepción en el hilo %s", a.thread.name if a.thread else "?",
        exc_info=(a.exc_type, a.exc_value, a.exc_traceback))


def asyncio_handler(loop: asyncio.AbstractEventLoop, context: dict) -> None:
    """Para `loop.set_exception_handler`: tareas en segundo plano que fallan sin que nadie mire."""
    exc = context.get("exception")
    log.error("asyncio: %s", context.get("message", ""),
              exc_info=(type(exc), exc, exc.__traceback__) if exc else None)
    loop.default_exception_handler(context)
