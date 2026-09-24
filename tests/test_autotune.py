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
    r = autotune.recalculate(c, load_catalog(), GPU16, 32, choose_roles=False)
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
    r = autotune.recalculate(c, load_catalog(), GPU16, 32, choose_roles=False)
    main = r.config.model("qwen3.6-35b-a3b")
    assert main.gpu_layers == -1 and main.kv_type == "q8_0" and main.ctx >= 65536
    assert r.config.keep_loaded is True
    assert any("RAM" in n for n in r.notes)


def test_dense_model_too_big_warns(models_dir):
    c = cfg("qwen2.5-coder-14b", "Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf")
    r = autotune.recalculate(c, load_catalog(), GPU8, 16, choose_roles=False)
    main = r.config.model("qwen2.5-coder-14b")
    assert main.gpu_layers == -1 and r.config.keep_loaded is False
    assert any("MUCHO más lento" in n for n in r.notes)


def test_already_optimal_has_no_changes(models_dir):
    c = cfg("qwen2.5-coder-14b", "Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf")
    first = autotune.recalculate(c, load_catalog(), GPU16, 32, choose_roles=False)
    autotune.apply(c, first.config)
    assert autotune.recalculate(c, load_catalog(), GPU16, 32, choose_roles=False).changes == []


def test_errors_are_explained(models_dir, monkeypatch):
    monkeypatch.setattr("agent.server.vram.gpu_info", lambda: None)
    c = cfg("qwen2.5-coder-14b", "Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf")
    assert "GPU" in autotune.recalculate(c, load_catalog(), choose_roles=False).error
    c.roles["main"] = ""
    assert "main" in autotune.recalculate(c, load_catalog(), GPU16, 32, choose_roles=False).error


# --- modo completo: roles + ubicación ---------------------------------------------

def test_chooses_roles_from_downloaded_models(models_dir):
    # Solo el 14B y el 3B están configurados; el 35B está en models/ sin configurar.
    c = cfg("qwen2.5-coder-14b", "Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf")
    steps = []
    r = autotune.recalculate(c, load_catalog(), GPU16, 32, progress=steps.append)
    assert r.error is None
    assert r.config.roles["main"] == "qwen3.6-35b-a3b"  # el más capaz que corre bien (MoE + RAM)
    # Un 3B sin herramientas aporta poco y quitaría VRAM a los expertos de main: resume main.
    assert r.config.roles["fast"] == ""
    assert "main hará también los resúmenes" in changed(r, None, "fast").reason
    assert changed(r, None, "main").reason and changed(r, "qwen3.6-35b-a3b", "added")
    assert any("Analizando modelos" in s for s in steps) and any("reparto" in s for s in steps)
    # aplicar añade el modelo descubierto y el segundo cálculo ya no propone nada
    autotune.apply(c, r.config)
    assert c.model("qwen3.6-35b-a3b") and c.roles["main"] == "qwen3.6-35b-a3b"
    assert autotune.recalculate(c, load_catalog(), GPU16, 32).changes == []


def test_small_gpu_prefers_model_that_fits(models_dir):
    c = cfg("qwen2.5-coder-14b", "Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf")
    r = autotune.recalculate(c, load_catalog(), GPU8, 8)  # 8 GB de VRAM y poca RAM: el 35B no entra
    assert r.config.roles["main"] != "qwen3.6-35b-a3b"


def test_params_from_name():
    assert autotune._params_from_name("Qwen3.6-35B-A3B-UD-Q4_K_M.gguf") == 35
    assert autotune._params_from_name("gpt-oss-20b-F16.gguf") == 20
    assert autotune._params_from_name("Qwen2.5-Coder-0.5B-Instruct-Q8_0.gguf") == 0.5
    assert autotune._params_from_name("modelo.gguf") is None


def test_discover_models_adds_downloaded_files(models_dir):
    c = cfg("qwen2.5-coder-14b", "Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf")
    for aux in ("mmproj-F16.gguf", "mtp-Algo-Q8_0.gguf", "Grande-Q4_K_M-00001-of-00002.gguf",
                "Nuevo-Coder-8B-Q4_K_M.gguf.part"):
        (models_dir / aux).write_bytes(b"x")
    (models_dir / "Nuevo-Coder-8B-Q4_K_M.gguf").write_bytes(b"x")
    c.ignored_files = ["Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"]  # el usuario lo quitó: no vuelve solo
    added = autotune.discover_models(c, load_catalog())
    assert [e.name for e in added] == ["nuevo-coder-8b-q4_k_m"]
    assert c.model("nuevo-coder-8b-q4_k_m").placement == "local"
    assert autotune.discover_models(c, load_catalog()) == []  # idempotente
    c.ignored_files = []
    added = autotune.discover_models(c, load_catalog())
    # del catálogo: nombre del catálogo y sus valores por defecto
    assert [e.name for e in added] == ["qwen3.6-35b-a3b"] and added[0].gpu_layers == -1


def test_ignored_files_roundtrip():
    c = AppConfig(ignored_files=["a.gguf", 3, ""])
    back = AppConfig.from_dict(c.to_dict())
    back.sanitize()
    assert back.ignored_files == ["a.gguf"]
    assert AppConfig.from_dict({}).ignored_files == []
