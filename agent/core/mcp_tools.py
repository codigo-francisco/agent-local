"""Servidores MCP: conecta con los de `config/mcp.json` y expone sus herramientas al agente.

El formato es el mismo que usan Claude Desktop o Cursor, para poder copiar configuraciones:

    {"mcpServers": {
        "fs":  {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "D:/docs"]},
        "web": {"url": "https://ejemplo.com/mcp", "headers": {"Authorization": "Bearer ..."}},
        "x":   {"command": "...", "disabled": true, "autoApprove": true}
    }}

Cada servidor vive en su propia tarea: los transportes del SDK (anyio) deben abrirse y cerrarse
en la misma tarea. Las herramientas se ofrecen al modelo como `mcp__<servidor>__<herramienta>`.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..config import CONFIG_DIR, ROOT
from .tools import clip

MCP_FILE = CONFIG_DIR / "mcp.json"
LOG_DIR = ROOT / "generated" / "mcp-logs"  # stderr de cada servidor stdio (para diagnosticar)
CONNECT_TIMEOUT = 45.0  # algunos servidores (codebase-memory) esperan 30 s antes de rendirse
CONNECT_RETRIES = 2
CALL_TIMEOUT = 120.0
PREFIX = "mcp__"
log = logging.getLogger("agent.mcp")


def _safe(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


@dataclass
class ServerState:
    name: str
    spec: dict
    status: str = "desconectado"  # conectando | conectado | error | desactivado | desconectado
    error: str = ""
    tools: list = field(default_factory=list)  # mcp.types.Tool
    session: object = None
    stop: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task | None = None

    @property
    def auto_approve(self) -> bool:
        return bool(self.spec.get("autoApprove"))


def load_spec(path: Path = MCP_FILE) -> dict:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8") or "{}")
    return data.get("mcpServers", data) if isinstance(data, dict) else {}


class MCPManager:
    def __init__(self, path: Path = MCP_FILE):
        self.path = path
        self.servers: dict[str, ServerState] = {}
        self._by_tool: dict[str, tuple[ServerState, str]] = {}
        self.load_error = ""

    # --- ciclo de vida ---------------------------------------------------
    async def start(self) -> None:
        """(Re)conecta todos los servidores del archivo. Nunca lanza: los fallos quedan en status."""
        await self.stop()
        try:
            spec = load_spec(self.path)
            self.load_error = ""
        except (OSError, ValueError) as e:
            self.load_error = f"No se pudo leer {self.path.name}: {e}"
            spec = {}
        for name, s in spec.items():
            st = ServerState(_safe(name), s if isinstance(s, dict) else {})
            self.servers[st.name] = st
            if st.spec.get("disabled"):
                st.status = "desactivado"
                continue
            for attempt in range(1 + CONNECT_RETRIES):
                if attempt:  # reintento: los fallos de arranque suelen ser momentáneos
                    if not st.task.done():
                        st.task.cancel()  # que el intento anterior no compita con el nuevo
                    await asyncio.wait([st.task], timeout=10)
                    st.stop, st.error = asyncio.Event(), ""
                ready = asyncio.Event()
                st.task = asyncio.create_task(self._run(st, ready))
                try:
                    await asyncio.wait_for(ready.wait(), CONNECT_TIMEOUT)
                except asyncio.TimeoutError:
                    st.status, st.error = "error", f"No respondió en {CONNECT_TIMEOUT:.0f} s."
                    tail = _log_tail(st.name)
                    if tail:
                        st.error += f"\nSalida del servidor:\n{tail}"
                    st.stop.set()
                if st.status != "error" or "no se encontró el programa" in st.error:
                    break  # conectado, o un error que no se arregla reintentando
            if st.status == "error":
                hint = _known_hint(st.error)
                if hint:
                    st.error = f"{hint}\n\n{st.error}"
        self._index()

    async def stop(self) -> None:
        for st in self.servers.values():
            st.stop.set()
        tasks = [st.task for st in self.servers.values() if st.task]
        if tasks:
            await asyncio.wait(tasks, timeout=10)
        self.servers.clear()
        self._by_tool.clear()

    async def _run(self, st: ServerState, ready: asyncio.Event) -> None:
        from mcp import ClientSession  # import tardío: la app funciona aunque falte el paquete

        st.status = "conectando"
        try:
            # El stack cierra también el archivo de log y el cliente HTTP: antes quedaban abiertos
            # en cada «Guardar y reconectar».
            async with contextlib.AsyncExitStack() as stack:
                read, write = await stack.enter_async_context(self._transport(st.spec, st.name, stack))
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                st.tools = list((await session.list_tools()).tools)
                st.session, st.status = session, "conectado"
                log.info("MCP %s conectado (%d herramientas)", st.name, len(st.tools))
                ready.set()
                await st.stop.wait()
        except BaseException as e:  # noqa: BLE001 - incluye ExceptionGroup de anyio
            if st.status != "conectado" or not st.stop.is_set():
                st.status, st.error = "error", _describe(e)
                log.warning("MCP %s: %s", st.name, st.error)
                tail = _log_tail(st.name)
                if tail:  # lo que el servidor escribió en stderr suele explicar el fallo
                    st.error += f"\nSalida del servidor:\n{tail}"
        finally:
            st.session = None
            if st.status == "conectado":
                st.status = "desconectado"
            ready.set()

    @staticmethod
    def _transport(spec: dict, name: str, stack: contextlib.AsyncExitStack):
        """Contexto del transporte; los recursos auxiliares se registran en `stack` para cerrarse
        al terminar (después del transporte, por el orden LIFO)."""
        if spec.get("url"):
            import httpx2
            from mcp.client.streamable_http import streamable_http_client
            client = httpx2.AsyncClient(headers=spec.get("headers") or {}, timeout=CALL_TIMEOUT)
            stack.push_async_callback(client.aclose)
            return streamable_http_client(spec["url"], http_client=client)
        if not spec.get("command"):
            raise ValueError("falta «command» o «url»")
        from mcp import StdioServerParameters
        from mcp.client.stdio import stdio_client
        env = {**os.environ, **{k: str(v) for k, v in (spec.get("env") or {}).items()}}
        command = spec["command"]
        local = ROOT / command  # rutas relativas (p. ej. "bin/…") se buscan en la carpeta del proyecto
        if not Path(command).is_absolute() and local.exists():
            command = str(local)
        params = StdioServerParameters(command=command, args=[str(a) for a in spec.get("args", [])],
                                       env=env, cwd=spec.get("cwd"))
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        errlog = open(LOG_DIR / f"{name}.log", "w", encoding="utf-8", errors="replace")  # noqa: SIM115
        stack.callback(errlog.close)
        return stdio_client(params, errlog=errlog)

    # --- herramientas ----------------------------------------------------
    def _index(self) -> None:
        self._by_tool.clear()
        for st in self.servers.values():
            if st.status != "conectado":
                continue
            for tool in st.tools:
                full = f"{PREFIX}{st.name}__{_safe(tool.name)}"[:64]
                self._by_tool[full] = (st, tool.name)

    def schemas(self) -> list[dict]:
        out = []
        for full, (st, name) in self._by_tool.items():
            tool = next(t for t in st.tools if t.name == name)
            schema = getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None) or {}
            out.append({"type": "function", "function": {
                "name": full,
                "description": f"[MCP {st.name}] {tool.description or tool.name}"[:1024],
                "parameters": schema or {"type": "object", "properties": {}},
            }})
        return out

    def tool_names(self) -> set[str]:
        return set(self._by_tool)

    def is_mcp(self, name: str) -> bool:
        return name in self._by_tool

    def needs_approval(self, name: str) -> bool:
        st, _ = self._by_tool[name]
        return not st.auto_approve

    async def call(self, name: str, args: dict, max_output: int = 12000) -> tuple[bool, str]:
        """Ejecuta una herramienta MCP. Nunca lanza: los errores vuelven como texto al modelo."""
        st, tool = self._by_tool[name]
        if st.session is None:
            return False, f"Error: el servidor MCP «{st.name}» no está conectado ({st.error or st.status})."
        try:
            res = await asyncio.wait_for(st.session.call_tool(tool, args), CALL_TIMEOUT)
        except asyncio.TimeoutError:
            return False, f"Error: la herramienta MCP «{tool}» no respondió en {CALL_TIMEOUT:.0f} s."
        except Exception as e:  # noqa: BLE001
            return False, f"Error del servidor MCP «{st.name}»: {_describe(e)}"
        parts = []
        for block in getattr(res, "content", None) or []:
            kind = getattr(block, "type", "")
            if kind == "text":
                parts.append(block.text)
            elif kind == "resource":
                r = block.resource
                parts.append(getattr(r, "text", None) or f"[recurso {getattr(r, 'uri', '')}]")
            else:
                parts.append(f"[contenido {kind} omitido]")
        structured = getattr(res, "structured_content", None)
        if not parts and structured is not None:
            parts.append(json.dumps(structured, ensure_ascii=False, indent=2))
        is_error = bool(getattr(res, "is_error", False) or getattr(res, "isError", False))
        text = "\n".join(parts) or "(sin salida)"
        return not is_error, clip(("Error: " if is_error else "") + text, max_output)

    def summary(self) -> list[dict]:
        return [{"name": st.name, "status": st.status, "error": st.error,
                 "tools": [t.name for t in st.tools], "auto": st.auto_approve}
                for st in self.servers.values()]


# Fallos conocidos de servidores concretos -> explicación en lenguaje llano.
KNOWN_HINTS = [
    (re.compile(r"daemon endpoint is held by pid (\d+)|active account daemon uses a different cache"),
     "codebase-memory usa un único proceso compartido por usuario, y ahora lo tiene abierto otra "
     "aplicación (por ejemplo Claude Code). Se puede compartir, pero si esa app lo lanzó desde un "
     "entorno aislado (como Claude de la Microsoft Store), conectarse puede tardar más de los 30 s "
     "que espera codebase-memory. Pulsa «Guardar y reconectar» para reintentar, o cierra la otra "
     "app para que la conexión sea inmediata."),
    (re.compile(r"owner is another account, a trusted identity required"),
     "La carpeta de caché (CBM_CACHE_DIR) debe estar dentro de tu perfil de usuario, por ejemplo "
     "en %LOCALAPPDATA%. Quita CBM_CACHE_DIR o cámbiala."),
]


def _known_hint(error: str) -> str | None:
    return next((hint for rx, hint in KNOWN_HINTS if rx.search(error)), None)


def _log_tail(name: str, lines: int = 8) -> str:
    try:
        text = (LOG_DIR / f"{name}.log").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return "\n".join(l for l in text.strip().splitlines()[-lines:])


def _describe(e: BaseException) -> str:
    """Mensaje legible, desenvolviendo los ExceptionGroup de anyio."""
    while isinstance(e, BaseExceptionGroup) and e.exceptions:
        e = e.exceptions[0]
    if isinstance(e, FileNotFoundError):
        return f"no se encontró el programa ({e.filename or e}). ¿Está instalado y en el PATH?"
    return f"{type(e).__name__}: {e}"
