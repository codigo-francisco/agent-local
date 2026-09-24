"""Descarga de archivos GGUF desde Hugging Face, con progreso, reanudación y cancelación."""

from __future__ import annotations

import asyncio
import hashlib
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import httpx

from ..config import MODELS_DIR


def hf_url(repo: str, filename: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/main/{filename}"


def hf_page(repo: str) -> str:
    return f"https://huggingface.co/{repo}"


@dataclass
class DownloadState:
    repo: str
    filename: str
    total: int = 0
    done: int = 0
    speed: float = 0.0  # bytes/s
    status: str = "pendiente"  # pendiente | descargando | completado | error | cancelado
    error: str = ""
    cancel: asyncio.Event = field(default_factory=asyncio.Event)

    @property
    def fraction(self) -> float:
        return self.done / self.total if self.total else 0.0


class DownloadError(Exception):
    pass


SPACE_MARGIN = 1 << 30  # 1 GB libre de sobra tras la descarga


def expected_sha256(head: httpx.Response) -> str | None:
    """SHA-256 que publica Hugging Face para archivos LFS (cabecera x-linked-etag de la
    respuesta de redirección)."""
    for r in (*head.history, head):
        etag = (r.headers.get("x-linked-etag") or "").strip('"').lower()
        if len(etag) == 64 and all(c in "0123456789abcdef" for c in etag):
            return etag
    return None


def check_space(folder: Path, needed: int) -> None:
    try:
        free = shutil.disk_usage(folder).free
    except OSError:
        return
    if needed > 0 and free < needed + SPACE_MARGIN:
        raise DownloadError(f"No hay espacio: faltan {needed / 1e9:.1f} GB y solo quedan "
                            f"{free / 1e9:.1f} GB libres en {folder}. Libera espacio y reintenta.")


def _hash_file(path: Path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            digest.update(block)
    return digest


async def download(state: DownloadState, dest_dir: Path = MODELS_DIR,
                   on_progress: Callable[[DownloadState], None] | None = None,
                   transport: httpx.AsyncBaseTransport | None = None) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / state.filename
    part = dest.with_suffix(dest.suffix + ".part")
    url = hf_url(state.repo, state.filename)
    state.status = "descargando"
    timeout = httpx.Timeout(60.0, connect=15.0)
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=timeout, transport=transport) as client:
            head = await client.head(url)
            if head.status_code in (401, 403):
                raise DownloadError("El repositorio requiere aceptar una licencia o iniciar sesión en "
                                    "Hugging Face. Descárgalo desde el navegador y ponlo en models/.")
            if head.status_code == 404:
                raise DownloadError(f"No existe {state.filename} en {state.repo}. El nombre del "
                                    "archivo puede haber cambiado: abre la página del repositorio.")
            head.raise_for_status()
            state.total = int(head.headers.get("content-length", 0))
            expected = expected_sha256(head)
            if dest.exists() and state.total and dest.stat().st_size == state.total:
                state.done, state.status = state.total, "completado"
                return dest
            offset = part.stat().st_size if part.exists() else 0
            check_space(dest_dir, state.total - offset)
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            async with client.stream("GET", url, headers=headers) as r:
                if r.status_code == 200:
                    offset = 0  # el servidor no soporta reanudar: empezamos de cero
                elif r.status_code != 206:
                    r.raise_for_status()
                # El hash se calcula mientras se descarga (al reanudar, primero lo ya bajado).
                digest = await asyncio.to_thread(_hash_file, part) if offset else hashlib.sha256()
                state.done = offset
                t0, b0 = time.monotonic(), offset
                with open(part, "ab" if offset else "wb") as f:
                    async for chunk in r.aiter_bytes(1 << 20):
                        if state.cancel.is_set():
                            state.status = "cancelado"
                            return part
                        f.write(chunk)
                        digest.update(chunk)
                        state.done += len(chunk)
                        dt = time.monotonic() - t0
                        if dt > 0.5:
                            state.speed = (state.done - b0) / dt
                        if on_progress:
                            on_progress(state)
        if state.total and part.stat().st_size != state.total:
            raise DownloadError("La descarga quedó incompleta; vuelve a pulsar Descargar para reanudarla.")
        if expected and digest.hexdigest() != expected:
            part.unlink(missing_ok=True)  # corrupto: reanudarlo no lo arreglaría
            raise DownloadError("El archivo descargado está dañado (su huella SHA-256 no coincide con "
                                "la de Hugging Face). Lo borré: pulsa Descargar para bajarlo de nuevo.")
        part.replace(dest)
        state.status = "completado"
        return dest
    except DownloadError as e:
        state.status, state.error = "error", str(e)
        raise
    except httpx.HTTPError as e:
        state.status = "error"
        state.error = f"Error de red: {e}. Pulsa Descargar de nuevo para reanudar."
        raise DownloadError(state.error) from e
