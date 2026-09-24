"""Conversaciones guardadas en disco: cerrar la app o un fallo no pierden el historial.

Cada conversación es un .json en generated/sessions. La interfaz las lista (multichat), permite
renombrarlas y borrarlas, y exportarlas a un archivo para retomarlas después (o en otra PC).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..config import atomic_write

KEEP = 1000  # tope de seguridad: las conversaciones se borran a mano desde el chat
EXPORT_FORMAT = "agent-local-chat"


@dataclass
class Session:
    id: str
    workspace: str
    history: list[dict] = field(default_factory=list)
    summary: str | None = None
    updated: float = 0.0
    name: str = ""  # título puesto por el usuario («Renombrar»)

    @property
    def title(self) -> str:
        if self.name.strip():
            return self.name.strip()
        first = next((m.get("content") or "" for m in self.history if m.get("role") == "user"), "")
        first = " ".join(str(first).split())
        return (first[:60] + "…") if len(first) > 60 else (first or "(vacía)")

    def to_dict(self) -> dict:
        return {"id": self.id, "workspace": self.workspace, "updated": self.updated, "name": self.name,
                "summary": self.summary, "history": self.history}


def new_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"


def _path(root: Path, session_id: str) -> Path:
    if not session_id or any(c in session_id for c in "/\\:") or session_id.startswith("."):
        raise ValueError(f"identificador de conversación no válido: {session_id!r}")
    return root / f"{session_id}.json"


def save(root: Path, session: Session, touch: bool = True) -> None:
    path = _path(root, session.id)
    if not session.name and path.exists():  # el agente no conoce el nombre: conservar el guardado
        old = load(path)
        if old:
            session.name = old.name
    if touch:
        session.updated = time.time()
    atomic_write(path, json.dumps(session.to_dict(), ensure_ascii=False).encode("utf-8"))
    _prune(root)


def _from_dict(data: dict) -> Session | None:
    history = data.get("history")
    if not isinstance(history, list) or not all(isinstance(m, dict) and "role" in m for m in history):
        return None
    try:
        return Session(str(data["id"]), str(data["workspace"]), history, data.get("summary"),
                       float(data.get("updated") or 0), str(data.get("name") or ""))
    except (KeyError, TypeError, ValueError):
        return None


def load(path: Path) -> Session | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None  # archivo dañado: se ignora
    return _from_dict(data) if isinstance(data, dict) else None


def get(root: Path, session_id: str) -> Session | None:
    try:
        return load(_path(root, session_id))
    except ValueError:
        return None


def _same_folder(a: str | Path, b: str | Path) -> bool:
    try:
        return str(Path(a).resolve()).lower() == str(Path(b).resolve()).lower()
    except OSError:
        return False


def list_sessions(root: Path, workspace: str | Path | None = None) -> list[Session]:
    """Conversaciones con al menos un mensaje (de esa carpeta, si se indica), la más reciente primero."""
    try:
        files = list(root.glob("*.json"))
    except OSError:
        return []
    out = []
    for f in files:
        s = load(f)
        if s and s.history and (workspace is None or _same_folder(s.workspace, workspace)):
            out.append(s)
    return sorted(out, key=lambda s: (s.updated, s.id), reverse=True)


def latest(root: Path, workspace: str | Path) -> Session | None:
    """La conversación más reciente de esa carpeta de proyecto (con al menos un mensaje)."""
    found = list_sessions(root, workspace)
    return found[0] if found else None


def delete(root: Path, session_id: str) -> bool:
    try:
        _path(root, session_id).unlink()
        return True
    except (OSError, ValueError):
        return False


def rename(root: Path, session_id: str, name: str) -> Session | None:
    s = get(root, session_id)
    if s is None:
        return None
    s.name = " ".join((name or "").split())[:120]
    atomic_write(_path(root, s.id), json.dumps(s.to_dict(), ensure_ascii=False).encode("utf-8"))
    return s


# --- exportar / importar ------------------------------------------------------------

def export_bytes(session: Session) -> bytes:
    data = {"format": EXPORT_FORMAT, "version": 1, "exported": time.time(), **session.to_dict()}
    return json.dumps(data, ensure_ascii=False, indent=1).encode("utf-8")


def export_filename(session: Session) -> str:
    safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in session.title)[:50].strip() or "chat"
    return f"{safe} ({session.id}).chat.json"


class SessionImportError(ValueError):
    pass


def import_bytes(root: Path, raw: bytes, workspace: str | Path) -> Session:
    """Guarda una conversación exportada como una nueva de esta carpeta de proyecto (con otro id
    si ya existe, para no pisar la original)."""
    try:
        data = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, ValueError) as e:
        raise SessionImportError(f"No es un archivo de conversación válido (JSON ilegible: {e}).") from e
    if not isinstance(data, dict) or data.get("format", EXPORT_FORMAT) != EXPORT_FORMAT:
        raise SessionImportError("No es un archivo de conversación exportado por agent-local.")
    data.setdefault("id", new_id())
    data["workspace"] = str(workspace)
    s = _from_dict(data)
    if s is None or not s.history:
        raise SessionImportError("El archivo no contiene mensajes de conversación válidos.")
    try:
        if _path(root, s.id).exists():
            s.id = new_id()
    except ValueError:
        s.id = new_id()
    save(root, s)
    return s


def _prune(root: Path) -> None:
    try:
        files = sorted(root.glob("*.json"), key=lambda p: p.stat().st_mtime)
    except OSError:
        return
    for f in files[:-KEEP]:
        try:
            f.unlink()
        except OSError:
            pass
