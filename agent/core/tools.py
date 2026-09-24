"""Herramientas del agente y sandbox de rutas dentro del workspace."""

from __future__ import annotations

import codecs
import difflib
import fnmatch
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable

from ..config import atomic_write

IGNORED_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", ".mypy_cache",
                ".pytest_cache", ".idea", ".vs", "dist", "build", ".next", "target"}
READ_MAX_LINES = 400
SEARCH_MAX_FILE_BYTES = 2_000_000
APPROVAL_TOOLS = {"edit_file", "write_file", "run_command"}
WRITE_TOOLS = {"edit_file", "write_file"}
MAX_COMMAND_TIMEOUT = 600  # tope para el `timeout` que pida el modelo (segundos)


class ToolError(Exception):
    pass


# Comandos que siempre piden confirmación, aunque esté el modo automático o «aprobar siempre».
_SEP = r"[^|;&\n]*"  # dentro del mismo comando (sin cruzar | ; &)
DANGEROUS: list[tuple[re.Pattern, str]] = [(re.compile(rx, re.I), why) for rx, why in [
    (r"\brm\s+" + _SEP + r"-[a-z]*r", "borra carpetas recursivamente"),
    (r"\b(remove-item|ri|rm|del|erase|rmdir|rd)\b" + _SEP + r"\s-r(ecurse)?\b", "borra carpetas recursivamente"),
    (r"\b(del|erase|rd|rmdir)\b" + _SEP + r"\s/s\b", "borra carpetas recursivamente"),
    (r"\b(format(-volume)?\s+[a-z]:|diskpart|mkfs|dd\s+if=)", "formatea o sobrescribe un disco"),
    (r"\bgit\s+push\b" + _SEP + r"(--force|\s-f\b)", "reescribe el historial remoto (push --force)"),
    (r"\bgit\s+reset\s+--hard\b", "descarta cambios sin commitear (reset --hard)"),
    (r"\bgit\s+clean\b" + _SEP + r"\s-[a-z]*f", "borra archivos no versionados (git clean)"),
    (r"\bgit\s+(checkout|restore)\s+(--\s+)?\.(\s|$)", "descarta cambios sin commitear"),
    (r"\b(shutdown|stop-computer|restart-computer)\b", "apaga o reinicia el equipo"),
    (r"\breg(\.exe)?\s+delete\b|\bremove-itemproperty\b", "borra claves del registro de Windows"),
    (r"\bset-executionpolicy\b", "cambia la política de seguridad de PowerShell"),
    (r"\b(iwr|irm|invoke-webrequest|invoke-restmethod|curl|wget)\b" + r"[^;\n]*\|\s*(iex|invoke-expression|sh|bash|python)\b",
     "descarga y ejecuta código de internet"),
]]


def dangerous_reason(command: str) -> str | None:
    """Por qué un comando es potencialmente destructivo, o None."""
    return next((why for rx, why in DANGEROUS if rx.search(command or "")), None)


def decode_text(data: bytes) -> tuple[str, str]:
    """(texto, codificación) sin perder bytes: UTF-8 (con o sin BOM) y, si no lo es, cp1252 o
    latin-1 (que acepta cualquier byte). Así editar un archivo antiguo de Windows no convierte sus
    «ñ» y tildes en �."""
    first = "utf-8-sig" if data.startswith(codecs.BOM_UTF8) else "utf-8"
    for enc in (first, "cp1252", "latin-1"):
        try:
            return data.decode(enc), enc
        except UnicodeDecodeError:
            continue
    raise AssertionError("latin-1 decodifica cualquier secuencia de bytes")


def encode_text(text: str, encoding: str) -> tuple[bytes, str]:
    """Codifica en la codificación original del archivo; si el texto nuevo trae caracteres que no
    caben en ella, pasa a UTF-8 (sin pérdida) y lo indica en la codificación devuelta."""
    try:
        return text.encode(encoding), encoding
    except UnicodeEncodeError:
        return text.encode("utf-8"), "utf-8"


class _Capture:
    """Lee un pipe en un hilo guardando solo el principio y el final: un comando que escupe
    gigas no llena la memoria."""

    def __init__(self, stream, limit: int):
        self.limit = limit
        self.head = bytearray()
        self.tail = bytearray()
        self.total = 0
        self.thread = threading.Thread(target=self._read, args=(stream,), daemon=True)
        self.thread.start()

    def _read(self, stream) -> None:
        try:
            while True:
                chunk = stream.read1(65536)
                if not chunk:
                    break
                self.total += len(chunk)
                room = self.limit - len(self.head)
                if room > 0:
                    self.head += chunk[:room]
                    chunk = chunk[room:]
                if chunk:
                    self.tail += chunk
                    if len(self.tail) > 2 * self.limit:
                        del self.tail[:-self.limit]
        except (OSError, ValueError):
            pass
        finally:
            try:
                stream.close()
            except OSError:
                pass

    def text(self) -> str:
        head = self.head.decode("utf-8", errors="replace")
        tail = bytes(self.tail[-self.limit:])
        omitted = self.total - len(self.head) - len(tail)
        if omitted > 0:
            head += f"\n[... se omitieron {omitted:,} bytes de salida ...]\n".replace(",", ".")
        return (head + tail.decode("utf-8", errors="replace")).strip()


def kill_tree(pid: int) -> None:
    """Mata un proceso y sus hijos (en Windows, terminate() deja huérfanos a los hijos)."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        try:
            if os.getpgid(pid) == pid:  # líder de su grupo (start_new_session): matar el grupo
                os.killpg(pid, 9)
            else:
                os.kill(pid, 9)
        except OSError:
            pass


def clip(text: str, limit: int, hint: str = "") -> str:
    """Recorta conservando el principio y el final, con una nota en medio."""
    if len(text) <= limit:
        return text
    head = int(limit * 0.6)
    tail = limit - head
    omitted = len(text) - head - tail
    note = f"\n\n[... se omitieron {omitted:,} caracteres ...{(' ' + hint) if hint else ''}]\n\n"
    return text[:head] + note.replace(",", ".") + text[-tail:]


def _fn(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


TOOL_SCHEMAS: list[dict] = [
    _fn("list_files",
        "List files in the workspace recursively (ignores .git, node_modules, venvs). "
        "Use a glob pattern like '*.py' to filter.",
        {"path": {"type": "string", "description": "Directory relative to the workspace. Default '.'"},
         "pattern": {"type": "string", "description": "Glob on file name, e.g. '*.py'. Default '*'"}},
        []),
    _fn("read_file",
        f"Read a text file with line numbers. Files longer than {READ_MAX_LINES} lines are returned "
        "partially: pass start/end (1-based, inclusive) to read other ranges.",
        {"path": {"type": "string"},
         "start": {"type": "integer", "description": "First line (1-based)"},
         "end": {"type": "integer", "description": "Last line (inclusive)"}},
        ["path"]),
    _fn("search",
        "Search a regular expression in workspace files. Returns 'path:line: text' matches.",
        {"pattern": {"type": "string", "description": "Python regular expression"},
         "path": {"type": "string", "description": "Directory to search. Default '.'"},
         "glob": {"type": "string", "description": "Only files matching this glob, e.g. '*.py'"}},
        ["pattern"]),
    _fn("edit_file",
        "Replace an exact snippet of a file. `old` must appear exactly once in the file: copy it "
        "exactly (indentation included) and add surrounding lines if it is not unique. "
        "Prefer several small edits over rewriting whole files.",
        {"path": {"type": "string"},
         "old": {"type": "string", "description": "Exact current text"},
         "new": {"type": "string", "description": "Replacement text"}},
        ["path", "old", "new"]),
    _fn("write_file",
        "Create a new file or fully overwrite an existing one. For changes to existing files "
        "use edit_file instead.",
        {"path": {"type": "string"}, "content": {"type": "string"}},
        ["path", "content"]),
    _fn("run_command",
        "Run a shell command in the workspace root (PowerShell on Windows) and return exit code, "
        "stdout and stderr. Use it for tests, linters, builds, git. Not for interactive programs.",
        {"command": {"type": "string"},
         "timeout": {"type": "integer", "description": "Seconds (default from settings)"}},
        ["command"]),
]


class Toolbox:
    def __init__(self, workspace: str | Path, max_output: int = 12000, command_timeout: int = 120):
        self.workspace = Path(workspace).resolve()
        self.max_output = max_output
        self.command_timeout = command_timeout
        self.checkpoint = None  # Checkpoint del turno: guarda originales antes de escribir
        self._should_stop: Callable[[], bool] | None = None

    # --- sandbox ---------------------------------------------------------
    def resolve(self, path: str | None) -> Path:
        raw = (path or ".").strip() or "."
        p = Path(raw)
        p = (p if p.is_absolute() else self.workspace / p).resolve()
        if p != self.workspace and not p.is_relative_to(self.workspace):
            raise ToolError(f"Ruta fuera del workspace ({self.workspace}): {raw}. "
                            "Solo puedo acceder a archivos dentro del proyecto.")
        return p

    def rel(self, p: Path) -> str:
        try:
            return p.relative_to(self.workspace).as_posix() or "."
        except ValueError:
            return str(p)

    # --- aprobación y vista previa --------------------------------------
    @staticmethod
    def needs_approval(name: str) -> bool:
        return name in APPROVAL_TOOLS

    def preview(self, name: str, args: dict) -> str:
        try:
            if name == "run_command":
                return f"> {args.get('command', '')}"
            if name == "edit_file":
                p = self.resolve(args.get("path"))
                before = decode_text(p.read_bytes())[0] if p.is_file() else ""
                after = self._apply_edit(before, args.get("old", ""), args.get("new", ""))
                return self._diff(before, after, self.rel(p))
            if name == "write_file":
                p = self.resolve(args.get("path"))
                before = decode_text(p.read_bytes())[0] if p.is_file() else ""
                return self._diff(before, args.get("content", ""), self.rel(p))
        except (ToolError, OSError) as e:
            return f"(no se puede previsualizar: {e})"
        return ", ".join(f"{k}={str(v)[:80]}" for k, v in args.items())

    @staticmethod
    def _diff(before: str, after: str, label: str) -> str:
        diff = difflib.unified_diff(before.splitlines(), after.splitlines(),
                                    f"a/{label}", f"b/{label}", lineterm="")
        text = "\n".join(diff)
        return clip(text or "(sin cambios)", 20000)

    # --- ejecución -------------------------------------------------------
    def execute(self, name: str, args: dict,
                should_stop: Callable[[], bool] | None = None) -> tuple[bool, str]:
        """Ejecuta una herramienta. Nunca lanza: los errores vuelven como texto al modelo.
        `should_stop` permite interrumpir run_command cuando el usuario pulsa «Detener»."""
        fn = getattr(self, f"tool_{name}", None)
        if fn is None:
            return False, f"Error: la herramienta '{name}' no existe. Disponibles: " + \
                ", ".join(s["function"]["name"] for s in TOOL_SCHEMAS)
        self._should_stop = should_stop
        try:
            return True, clip(fn(**args), self.max_output,
                              "Usa rangos o filtros más concretos para ver el resto.")
        except TypeError as e:
            return False, f"Error: argumentos inválidos para {name}: {e}"
        except ToolError as e:
            return False, f"Error: {e}"
        except OSError as e:
            return False, f"Error de sistema de archivos: {e}"
        except Exception as e:  # noqa: BLE001 - el modelo debe ver cualquier fallo
            return False, f"Error inesperado en {name}: {type(e).__name__}: {e}"
        finally:
            self._should_stop = None

    def _walk(self, root: Path):
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in IGNORED_DIRS)
            for f in sorted(filenames):
                yield Path(dirpath) / f

    def tool_list_files(self, path: str = ".", pattern: str = "*") -> str:
        root = self.resolve(path)
        if not root.is_dir():
            raise ToolError(f"No es un directorio: {path}")
        lines, total = [], 0
        for f in self._walk(root):
            if not fnmatch.fnmatch(f.name, pattern or "*"):
                continue
            total += 1
            if len(lines) < 500:
                try:
                    size = f.stat().st_size
                except OSError:
                    size = 0
                lines.append(f"{self.rel(f)}  ({size:,} B)".replace(",", "."))
        if not lines:
            return "No hay archivos que coincidan."
        more = f"\n... y {total - len(lines)} más (usa un patrón o subcarpeta)" if total > len(lines) else ""
        return "\n".join(lines) + more

    def tool_read_file(self, path: str, start: int | None = None, end: int | None = None) -> str:
        p = self.resolve(path)
        if not p.is_file():
            raise ToolError(f"No existe el archivo: {path}")
        data = p.read_bytes()
        if b"\x00" in data[:8192]:
            raise ToolError(f"{path} parece binario; no lo puedo leer como texto.")
        lines = decode_text(data)[0].splitlines()
        n = len(lines)
        if n == 0:
            return f"{path} está vacío."
        s = max(1, int(start)) if start else 1
        e = min(n, int(end)) if end else n
        partial_note = ""
        if not end and e - s + 1 > READ_MAX_LINES:
            e = s + READ_MAX_LINES - 1
            partial_note = (f"\n[Archivo de {n} líneas: mostradas {s}-{e}. "
                            f"Usa start/end para leer otros rangos.]")
        if s > n:
            raise ToolError(f"{path} solo tiene {n} líneas.")
        width = len(str(e))
        body = "\n".join(f"{i:>{width}}| {lines[i - 1]}" for i in range(s, e + 1))
        return f"{self.rel(p)} (líneas {s}-{e} de {n})\n{body}{partial_note}"

    def tool_search(self, pattern: str, path: str = ".", glob: str = "*") -> str:
        try:
            rx = re.compile(pattern)
        except re.error as e:
            raise ToolError(f"Expresión regular inválida: {e}") from e
        root = self.resolve(path)
        files = [root] if root.is_file() else self._walk(root)
        out: list[str] = []
        for f in files:
            if not fnmatch.fnmatch(f.name, glob or "*"):
                continue
            try:
                if f.stat().st_size > SEARCH_MAX_FILE_BYTES:
                    continue
                data = f.read_bytes()
            except OSError:
                continue
            if b"\x00" in data[:8192]:
                continue
            for i, line in enumerate(decode_text(data)[0].splitlines(), 1):
                if rx.search(line):
                    out.append(f"{self.rel(f)}:{i}: {line.strip()[:200]}")
                    if len(out) >= 200:
                        return "\n".join(out) + "\n[... hay más coincidencias; afina el patrón]"
        return "\n".join(out) if out else "Sin coincidencias."

    @staticmethod
    def _apply_edit(content: str, old: str, new: str) -> str:
        if not old:
            raise ToolError("`old` no puede estar vacío. Para crear archivos usa write_file.")
        crlf = "\r\n" in content
        text = content.replace("\r\n", "\n") if crlf else content
        old_n, new_n = old.replace("\r\n", "\n"), new.replace("\r\n", "\n")
        count = text.count(old_n)
        if count == 0:
            raise ToolError("No encontré `old` en el archivo. Vuelve a leerlo con read_file y copia "
                            "el fragmento exacto (espacios e indentación incluidos).")
        if count > 1:
            raise ToolError(f"`old` aparece {count} veces. Incluye más líneas de contexto para que "
                            "sea único.")
        text = text.replace(old_n, new_n, 1)
        return text.replace("\n", "\r\n") if crlf else text

    def _write(self, p: Path, text: str, encoding: str = "utf-8") -> str:
        """Escritura atómica que conserva la codificación; antes guarda el original para poder
        deshacer. Devuelve una nota si hubo que cambiar la codificación."""
        data, used = encode_text(text, encoding)
        if self.checkpoint is not None:
            self.checkpoint.record(p)
        atomic_write(p, data)
        if used != encoding and encoding not in ("utf-8", "utf-8-sig"):
            return f" Nota: pasó de {encoding} a UTF-8 porque el texto nuevo tiene caracteres que no caben en {encoding}."
        return ""

    def tool_edit_file(self, path: str, old: str, new: str) -> str:
        p = self.resolve(path)
        if not p.is_file():
            raise ToolError(f"No existe el archivo: {path}. Para crearlo usa write_file.")
        content, encoding = decode_text(p.read_bytes())
        updated = self._apply_edit(content, old, new)
        note = self._write(p, updated, encoding)
        delta = updated.count("\n") - content.count("\n")
        return f"Editado {self.rel(p)} ({delta:+d} líneas).{note}"

    def tool_write_file(self, path: str, content: str) -> str:
        p = self.resolve(path)
        if p.is_dir():
            raise ToolError(f"{path} es una carpeta, no un archivo.")
        existed = p.exists()
        # Un archivo existente conserva su codificación (p. ej. un .bat en cp1252).
        encoding = decode_text(p.read_bytes())[1] if existed else "utf-8"
        note = self._write(p, content, encoding)
        verb = "Sobrescrito" if existed else "Creado"
        return f"{verb} {self.rel(p)} ({content.count(chr(10)) + 1} líneas).{note}"

    def tool_run_command(self, command: str, timeout: int | None = None) -> str:
        try:
            timeout = int(timeout or self.command_timeout)
        except (TypeError, ValueError) as e:
            raise ToolError(f"`timeout` debe ser un número de segundos: {timeout!r}") from e
        timeout = max(1, min(timeout, max(self.command_timeout, MAX_COMMAND_TIMEOUT)))
        if os.name == "nt":
            argv = ["powershell", "-NoProfile", "-NonInteractive", "-Command",
                    "[Console]::OutputEncoding=[Text.Encoding]::UTF8; " + command]
            flags = subprocess.CREATE_NO_WINDOW  # sin ventanas de consola sueltas en la GUI
        else:
            argv, flags = ["bash", "-lc", command], 0
        proc = subprocess.Popen(argv, cwd=self.workspace, stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=flags,
                                start_new_session=os.name != "nt")
        limit = max(4096, self.max_output)
        out, err = _Capture(proc.stdout, limit), _Capture(proc.stderr, limit)
        should_stop = self._should_stop
        deadline = time.monotonic() + timeout
        note, code = "", None
        while True:
            try:
                code = proc.wait(timeout=0.2)
                break
            except subprocess.TimeoutExpired:
                pass
            if should_stop is not None and should_stop():
                note = "\n[Detenido por el usuario.]"
            elif time.monotonic() > deadline:
                note = f"\n[El comando superó el límite de {timeout}s y se detuvo.]"
            else:
                continue
            kill_tree(proc.pid)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            break
        # Un nieto que heredó el pipe puede mantenerlo abierto: no esperamos para siempre.
        out.thread.join(timeout=5)
        err.thread.join(timeout=5)
        half = self.max_output // 2
        status = code if code is not None else ("cancelado" if "usuario" in note else "timeout")
        parts = [f"exit code: {status}"]
        if out.total:
            parts.append("--- stdout ---\n" + clip(out.text(), half))
        if err.total:
            parts.append("--- stderr ---\n" + clip(err.text(), half))
        return "\n".join(parts) + note
