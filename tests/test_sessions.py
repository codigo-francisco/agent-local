"""Multichat: listar, renombrar, borrar, exportar e importar conversaciones."""

import json

import pytest

from agent.core import sessions
from agent.core.sessions import Session


def chat(sid: str, ws, text: str) -> Session:
    return Session(sid, str(ws), [{"role": "user", "content": text},
                                  {"role": "assistant", "content": "ok"}])


def test_list_rename_delete(tmp_path):
    root, ws, other = tmp_path / "s", tmp_path / "proyecto", tmp_path / "otro"
    ws.mkdir()
    other.mkdir()
    sessions.save(root, chat("a", ws, "primera tarea"))
    sessions.save(root, chat("b", ws, "segunda tarea"))
    sessions.save(root, chat("c", other, "de otra carpeta"))
    sessions.save(root, Session("vacia", str(ws)))  # sin mensajes: no se lista
    assert [s.id for s in sessions.list_sessions(root, ws)] == ["b", "a"]  # la más reciente primero
    assert len(sessions.list_sessions(root)) == 3
    assert sessions.rename(root, "a", "  Arreglar   tests ").title == "Arreglar tests"
    # el agente vuelve a guardar sin conocer el nombre: se conserva
    sessions.save(root, chat("a", ws, "primera tarea"))
    assert sessions.get(root, "a").title == "Arreglar tests"
    assert sessions.delete(root, "a") and sessions.get(root, "a") is None
    assert not sessions.delete(root, "a")
    assert sessions.get(root, "../fuera") is None  # nada de rutas fuera de la carpeta


def test_old_limit_no_longer_prunes_history(tmp_path):
    for i in range(40):  # antes solo se guardaban 30
        sessions.save(tmp_path, chat(f"id{i:02d}", tmp_path, f"tarea {i}"))
    assert len(sessions.list_sessions(tmp_path)) == 40


def test_export_import_roundtrip(tmp_path):
    root, ws = tmp_path / "s", tmp_path / "ws"
    original = chat("20260101-120000-001", "C:/otra/pc/proyecto", "migra la base de datos")
    original.name = "Migración"
    original.summary = "resumen previo"
    data = sessions.export_bytes(original)
    assert json.loads(data)["format"] == sessions.EXPORT_FORMAT
    assert sessions.export_filename(original).endswith(".chat.json")
    imported = sessions.import_bytes(root, data, ws)
    assert imported.workspace == str(ws) and imported.title == "Migración"
    assert imported.history == original.history and imported.summary == "resumen previo"
    again = sessions.import_bytes(root, data, ws)  # mismo archivo otra vez: no pisa el primero
    assert again.id != imported.id and len(sessions.list_sessions(root, ws)) == 2


@pytest.mark.parametrize("raw", [b"no es json", b'{"format": "otra-app", "history": []}',
                                 b'{"history": "x"}', b'{"history": []}'])
def test_import_rejects_invalid(tmp_path, raw):
    with pytest.raises(sessions.SessionImportError):
        sessions.import_bytes(tmp_path, raw, tmp_path)
