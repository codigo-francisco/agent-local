"""Checkpoints por turno: el contenido original de cada archivo que el agente escribe, para poder
deshacer el turno entero. Se guardan en disco (sobreviven a cerrar la app).

Al deshacer no se pisa nada que haya cambiado después: si un archivo ya no es como lo dejó el
agente (lo editaste tú o un turno posterior), se omite y se avisa.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from ..config import atomic_write

KEEP = 20  # checkpoints que se conservan en disco


def _hash(data: bytes | None) -> str | None:
    return hashlib.sha256(data).hexdigest() if data is not None else None


def _read(p: Path) -> bytes | None:
    try:
        return p.read_bytes() if p.is_file() else None
    except OSError:
        return None


@dataclass
class UndoReport:
    restored: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)  # cambiaron después: no se tocan

    def text(self) -> str:
        parts = []
        if self.restored:
            parts.append("restaurados: " + ", ".join(self.restored))
        if self.deleted:
            parts.append("eliminados (eran nuevos): " + ", ".join(self.deleted))
        if self.skipped:
            parts.append("sin tocar porque cambiaron después: " + ", ".join(self.skipped))
        return "; ".join(parts) or "no había nada que deshacer"


class Checkpoint:
    def __init__(self, workspace: Path, root: Path | None = None, cid: str | None = None):
        self.workspace = Path(workspace)
        self.root = root  # None: solo en memoria (tests, CLI)
        self.id = cid or f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        self.originals: dict[str, bytes | None] = {}  # ruta relativa -> bytes (None = no existía)
        self.after: dict[str, str | None] = {}  # hash tras el turno

    @property
    def files(self) -> list[str]:
        return list(self.originals)

    def _rel(self, p: Path) -> str:
        return p.resolve().relative_to(self.workspace.resolve()).as_posix()

    def record(self, p: Path) -> None:
        """Guarda el original la primera vez que el turno va a escribir `p`."""
        rel = self._rel(p)
        if rel not in self.originals:
            self.originals[rel] = _read(p)

    def finalize(self) -> None:
        """Anota cómo quedaron los archivos y lo persiste (si hay carpeta)."""
        self.after = {rel: _hash(_read(self.workspace / rel)) for rel in self.originals}
        if self.root is None or not self.originals:
            return
        d = self.root / self.id
        manifest = {"workspace": str(self.workspace), "created": time.time(), "files": []}
        for i, (rel, data) in enumerate(self.originals.items()):
            blob = None
            if data is not None:
                blob = f"{i}.bin"
                atomic_write(d / blob, data)
            manifest["files"].append({"path": rel, "blob": blob, "after": self.after[rel]})
        atomic_write(d / "manifest.json", json.dumps(manifest, ensure_ascii=False).encode("utf-8"))
        _prune(self.root)

    def undo(self) -> UndoReport:
        report = UndoReport()
        for rel, original in self.originals.items():
            p = self.workspace / rel
            if _hash(_read(p)) != self.after.get(rel):
                report.skipped.append(rel)
                continue
            try:
                if original is None:
                    if p.exists():
                        p.unlink()
                    report.deleted.append(rel)
                else:
                    atomic_write(p, original)
                    report.restored.append(rel)
            except OSError:
                report.skipped.append(rel)
        # Tras deshacer, el estado «después» es el original: un segundo deshacer no hace nada.
        self.originals = {}
        if self.root is not None:
            shutil.rmtree(self.root / self.id, ignore_errors=True)
        return report

    @classmethod
    def load(cls, root: Path, cid: str) -> "Checkpoint | None":
        d = root / cid
        try:
            manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
            cp = cls(Path(manifest["workspace"]), root, cid)
            for f in manifest["files"]:
                cp.originals[f["path"]] = (d / f["blob"]).read_bytes() if f["blob"] else None
                cp.after[f["path"]] = f["after"]
            return cp
        except (OSError, ValueError, KeyError, TypeError):
            return None


def _prune(root: Path) -> None:
    try:
        dirs = sorted((d for d in root.iterdir() if d.is_dir()), key=lambda d: d.name)
    except OSError:
        return
    for d in dirs[:-KEEP]:
        shutil.rmtree(d, ignore_errors=True)
