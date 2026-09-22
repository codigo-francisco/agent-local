"""Herramientas del agente y sandbox de rutas dentro del workspace."""

from __future__ import annotations

import difflib
import fnmatch
import os
import re
import subprocess
from pathlib import Path

IGNORED_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", ".mypy_cache",
                ".pytest_cache", ".idea", ".vs", "dist", "build", ".next", "target"}
READ_MAX_LINES = 400
SEARCH_MAX_FILE_BYTES = 2_000_000
APPROVAL_TOOLS = {"edit_file", "write_file", "run_command"}


class ToolError(Exception):
    pass


def kill_tree(pid: int) -> None:
    """Mata un proceso y sus hijos (en Windows, terminate() deja huérfanos a los hijos)."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        try:
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
                before = p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""
                after = self._apply_edit(before, args.get("old", ""), args.get("new", ""))
                return self._diff(before, after, self.rel(p))
            if name == "write_file":
                p = self.resolve(args.get("path"))
                before = p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""
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
    def execute(self, name: str, args: dict) -> tuple[bool, str]:
        """Ejecuta una herramienta. Nunca lanza: los errores vuelven como texto al modelo."""
        fn = getattr(self, f"tool_{name}", None)
        if fn is None:
            return False, f"Error: la herramienta '{name}' no existe. Disponibles: " + \
                ", ".join(s["function"]["name"] for s in TOOL_SCHEMAS)
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
        lines = data.decode("utf-8", errors="replace").splitlines()
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
            for i, line in enumerate(data.decode("utf-8", errors="replace").splitlines(), 1):
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

    def tool_edit_file(self, path: str, old: str, new: str) -> str:
        p = self.resolve(path)
        if not p.is_file():
            raise ToolError(f"No existe el archivo: {path}. Para crearlo usa write_file.")
        content = p.read_bytes().decode("utf-8", errors="replace")
        updated = self._apply_edit(content, old, new)
        p.write_bytes(updated.encode("utf-8"))
        delta = updated.count("\n") - content.count("\n")
        return f"Editado {self.rel(p)} ({delta:+d} líneas)."

    def tool_write_file(self, path: str, content: str) -> str:
        p = self.resolve(path)
        existed = p.exists()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content.encode("utf-8"))
        verb = "Sobrescrito" if existed else "Creado"
        return f"{verb} {self.rel(p)} ({content.count(chr(10)) + 1} líneas)."

    def tool_run_command(self, command: str, timeout: int | None = None) -> str:
        timeout = int(timeout or self.command_timeout)
        if os.name == "nt":
            argv = ["powershell", "-NoProfile", "-NonInteractive", "-Command",
                    "[Console]::OutputEncoding=[Text.Encoding]::UTF8; " + command]
        else:
            argv = ["bash", "-lc", command]
        proc = subprocess.Popen(argv, cwd=self.workspace, stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            out, err = proc.communicate(timeout=timeout)
            code, note = proc.returncode, ""
        except subprocess.TimeoutExpired:
            kill_tree(proc.pid)
            try:
                out, err = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                out, err = b"", b""
            code = None
            note = f"\n[El comando superó el límite de {timeout}s y se detuvo.]"
        dec = lambda b: b.decode("utf-8", errors="replace").strip()  # noqa: E731
        half = self.max_output // 2
        parts = [f"exit code: {code if code is not None else 'timeout'}"]
        if out:
            parts.append("--- stdout ---\n" + clip(dec(out), half))
        if err:
            parts.append("--- stderr ---\n" + clip(dec(err), half))
        return "\n".join(parts) + note
