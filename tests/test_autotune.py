"""«Recalcular» con hardware y tamaños de modelo simulados (sin GPU ni archivos grandes)."""

from pathlib import Path

import pytest

from agent.config import AppConfig, ModelEntry, load_catalog
from agent.server import autotune
from agent.server.vram import GPUInfo

GPU16 = GPUInfo("RTX 4070 Ti SUPER", 16.0, 1.0, 15.0, "x")
GPU8 = GPUInfo("RTX 4060", 8.0, 0.5, 7.5, "x")
FILES = {
    "Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf": 8.37,
    "Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf": 1.80,
    "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf": 20.61,
}


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    # Archivos diminutos; el tamaño "real" lo da file_gb simulado. (En Windows, truncar a varios
    # GB escribe los ceros de verdad: no crear archivos grandes en los tests.)
    for name in FILES:
        (tmp_path / name).write_bytes(b"x")
    monkeypatch.setattr("agent.config.MODELS_DIR", tmp_path)
    monkeypatch.setattr("agent.server.vram.file_gb", lambda p: FILES[Path(p).name])
    return tmp_path


def cfg(main: str, main_file: str, **kw) -> AppConfig:
    return AppConfig(
        roles={"main": main, "fast": "qwen2.5-coder-3b", "draft": ""},
        models=[ModelEntry(main, main_file, ctx=4096, kv_type="f16", gpu_layers=10),
                ModelEntry("qwen2.5-coder-3b", "Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf", ctx=65536)],
        **kw)


def changed(result, model, field):
    return next((c for c in result.changes if c.model == model and c.field == field), None)


def test_dense_model_fits_whole_gpu(models_dir):
    c = cfg("qwen2.5-coder-14b", "Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf", keep_loaded=False)
    r = autotune.recalculate(c, load_catalog(), GPU16, 32)
    assert r.error is None
    main = r.config.model("qwen2.5-coder-14b")
    assert main.gpu_layers == 99 and main.kv_type == "q8_0" and main.ctx >= 16384
    assert r.config.keep_loaded is True
    assert r.config.model("qwen2.5-coder-3b").ctx == 16384
    assert changed(r, "qwen2.5-coder-14b", "ctx").reason
    # la propuesta no toca la configuración viva hasta aplicarla
    assert c.model("qwen2.5-coder-14b").ctx == 4096
    autotune.apply(c, r.config)
    assert c.model("qwen2.5-coder-14b").ctx == main.ctx and c.keep_loaded is True


def test_moe_model_goes_auto_with_big_context(models_dir):
    c = cfg("qwen3.6-35b-a3b", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
    r = autotune.recalculate(c, load_catalog(), GPU16, 32)
    main = r.config.model("qwen3.6-35b-a3b")
    assert main.gpu_layers == -1 and main.kv_type == "q8_0" and main.ctx >= 65536
    assert r.config.keep_loaded is True
    assert any("RAM" in n for n in r.notes)


def test_dense_model_too_big_warns(models_dir):
    c = cfg("qwen2.5-coder-14b", "Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf")
    r = autotune.recalculate(c, load_catalog(), GPU8, 16)
    main = r.config.model("qwen2.5-coder-14b")
    assert main.gpu_layers == -1 and r.config.keep_loaded is False
    assert any("MUCHO más lento" in n for n in r.notes)


def test_already_optimal_has_no_changes(models_dir):
    c = cfg("qwen2.5-coder-14b", "Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf")
    first = autotune.recalculate(c, load_catalog(), GPU16, 32)
    autotune.apply(c, first.config)
    assert autotune.recalculate(c, load_catalog(), GPU16, 32).changes == []


def test_errors_are_explained(models_dir, monkeypatch):
    monkeypatch.setattr("agent.server.vram.gpu_info", lambda: None)
    c = cfg("qwen2.5-coder-14b", "Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf")
    assert "GPU" in autotune.recalculate(c, load_catalog()).error
    c.roles["main"] = ""
    assert "main" in autotune.recalculate(c, load_catalog(), GPU16, 32).error
