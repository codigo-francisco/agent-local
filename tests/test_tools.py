import pytest

from agent.core.tools import Toolbox, ToolError, clip


@pytest.fixture
def tb(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    return Toolbox(tmp_path, max_output=5000, command_timeout=20)


def test_sandbox_rejects_outside_paths(tb, tmp_path):
    with pytest.raises(ToolError):
        tb.resolve("../fuera.txt")
    with pytest.raises(ToolError):
        tb.resolve(str(tmp_path.parent / "otro"))
    ok, out = tb.execute("read_file", {"path": "../../etc/passwd"})
    assert not ok and "fuera del workspace" in out


def test_read_file_numbers_and_ranges(tb):
    ok, out = tb.execute("read_file", {"path": "src/calc.py"})
    assert ok and "1| def add" in out and "2|     return a - b" in out
    ok, out = tb.execute("read_file", {"path": "src/calc.py", "start": 2, "end": 2})
    assert ok and "def add" not in out and "return a - b" in out


def test_read_file_large_is_partial(tb, tmp_path):
    (tmp_path / "big.txt").write_text("\n".join(f"line {i}" for i in range(1000)), encoding="utf-8")
    ok, out = tb.execute("read_file", {"path": "big.txt"})
    assert ok and "Archivo de 1000 líneas" in out and "line 999" not in out


def test_edit_file_unique_and_errors(tb, tmp_path):
    ok, out = tb.execute("edit_file", {"path": "src/calc.py", "old": "a - b", "new": "a + b"})
    assert ok, out
    assert "a + b" in (tmp_path / "src" / "calc.py").read_text(encoding="utf-8")
    ok, out = tb.execute("edit_file", {"path": "src/calc.py", "old": "no existe", "new": "x"})
    assert not ok and "No encontré" in out
    (tmp_path / "dup.py").write_text("x = 1\nx = 1\n", encoding="utf-8")
    ok, out = tb.execute("edit_file", {"path": "dup.py", "old": "x = 1", "new": "x = 2"})
    assert not ok and "2 veces" in out


def test_edit_preserves_crlf(tb, tmp_path):
    p = tmp_path / "win.txt"
    p.write_bytes(b"uno\r\ndos\r\ntres\r\n")
    ok, out = tb.execute("edit_file", {"path": "win.txt", "old": "uno\ndos", "new": "UNO\nDOS"})
    assert ok, out
    assert p.read_bytes() == b"UNO\r\nDOS\r\ntres\r\n"


def test_write_and_list_and_search(tb):
    ok, _ = tb.execute("write_file", {"path": "pkg/nuevo.py", "content": "print('hola')\n"})
    assert ok
    ok, out = tb.execute("list_files", {"pattern": "*.py"})
    assert "pkg/nuevo.py" in out and "src/calc.py" in out
    ok, out = tb.execute("search", {"pattern": r"print\("})
    assert ok and "pkg/nuevo.py:1" in out
    ok, out = tb.execute("search", {"pattern": "("})
    assert not ok and "inválida" in out


def test_unknown_tool_and_bad_args(tb):
    ok, out = tb.execute("borrar_todo", {})
    assert not ok and "no existe" in out
    ok, out = tb.execute("read_file", {"ruta": "x"})
    assert not ok and "argumentos inválidos" in out


def test_run_command(tb):
    ok, out = tb.execute("run_command", {"command": "echo hola"})
    assert ok and "exit code: 0" in out and "hola" in out


def test_preview_diff(tb):
    diff = tb.preview("edit_file", {"path": "src/calc.py", "old": "a - b", "new": "a + b"})
    assert "-    return a - b" in diff and "+    return a + b" in diff


def test_clip_keeps_head_and_tail():
    text = "A" * 1000 + "B" * 1000
    out = clip(text, 200)
    assert out.startswith("A") and out.endswith("B") and "se omitieron" in out
