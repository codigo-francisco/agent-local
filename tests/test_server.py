"""VRAM y generación de llama-swap.yaml (sin GPU ni binarios reales)."""

from agent.config import AppConfig, ModelEntry, load_catalog
from agent.server import swapconfig, vram
from agent.server.vram import ArchInfo, GPUInfo, kv_cache_bytes

GB = 1024 ** 3


def test_kv_cache_formula_qwen14b():
    arch = ArchInfo(layers=48, kv_heads=8, key_dim=128, value_dim=128)
    assert abs(kv_cache_bytes(arch, 32768, "f16") / GB - 6.0) < 0.01
    assert abs(kv_cache_bytes(arch, 32768, "q8_0") / GB - 3.19) < 0.01


def test_kv_cache_hybrid_attention_qwen36():
    # 40 capas, solo 1 de cada 4 con KV, 2 cabezas KV de 256: ~20 KB/token en f16
    arch = ArchInfo(layers=40, kv_heads=2, key_dim=256, value_dim=256, attn_interval=4)
    assert arch.kv_layers == 10
    assert kv_cache_bytes(arch, 1, "f16") == 20480
    assert abs(kv_cache_bytes(arch, 131072, "f16") / GB - 2.5) < 0.01


def test_plan_auto_offloads_moe_to_ram():
    gpu = GPUInfo("RTX 4070 Ti SUPER", 16.0, 1.0, 15.0, "x")
    cfg = AppConfig(roles={"main": "qwen3.6-35b-a3b", "fast": "", "draft": ""},
                    models=[ModelEntry("qwen3.6-35b-a3b", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf", ctx=65536,
                                       gpu_layers=-1)])
    plan = vram.plan_usage(cfg, load_catalog(), gpu)
    assert plan.fits and plan.ram_offload_gb > 4
    assert "expertos MoE en RAM" in plan.models[0].note
    cfg.models[0].gpu_layers = 99  # sin modo automático: no cabe
    assert vram.plan_usage(cfg, load_catalog(), gpu).fits is False


def test_catalog_is_valid():
    catalog = load_catalog()
    assert catalog, "config/catalog.yaml vacío o ausente"
    for item in catalog:
        for key in ("id", "name", "repo", "file", "size_gb", "role", "summary"):
            assert key in item, f"{item.get('id')}: falta {key}"
        assert item["file"].endswith(".gguf")
        assert vram.arch_from_catalog(item) is not None


def cfg_two_models(keep_loaded=True, kv="q8_0", main_ctx=24576):
    return AppConfig(
        roles={"main": "qwen2.5-coder-14b", "fast": "qwen2.5-coder-3b", "draft": ""},
        models=[ModelEntry("qwen2.5-coder-14b", "Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf", ctx=main_ctx, kv_type=kv),
                ModelEntry("qwen2.5-coder-3b", "Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf", ctx=16384, kv_type=kv)],
        keep_loaded=keep_loaded,
    )


def test_plan_fits_on_16gb():
    gpu = GPUInfo("RTX 4070 Ti SUPER", 16.0, 1.0, 15.0, "x")
    plan = vram.plan_usage(cfg_two_models(), load_catalog(), gpu)
    assert plan.fits and 12 < plan.total_gb < 15.2


def test_plan_too_big_gives_suggestions():
    gpu = GPUInfo("RTX 4070 Ti SUPER", 16.0, 1.0, 15.0, "x")
    plan = vram.plan_usage(cfg_two_models(kv="f16", main_ctx=131072), load_catalog(), gpu)
    assert plan.fits is False
    text = " ".join(plan.suggestions)
    assert "contexto máximo" in text and "q8_0" in text


def test_ctx_hint_mentions_vram():
    hint = vram.ctx_hint(cfg_two_models(), load_catalog(), "qwen2.5-coder-14b", 24576)
    assert hint and "48K" in hint and "GB" in hint


def test_swap_config_commands(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    for name in ("Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf", "Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf"):
        (models_dir / name).write_bytes(b"GGUF")
    monkeypatch.setattr("agent.config.MODELS_DIR", models_dir)
    cfg = cfg_two_models()
    data, warnings = swapconfig.build(cfg)
    cmd = data["models"]["qwen2.5-coder-14b"]["cmd"]
    for flag in ("--jinja", "-c 24576", "-np 1", "-ctk q8_0", "-ctv q8_0", "${PORT}", "-fa on"):
        assert flag in cmd, flag
    assert data["groups"]["agent"]["members"] == ["qwen2.5-coder-14b", "qwen2.5-coder-3b"]
    cfg.models[0].gpu_layers = -1  # automático: sin -ngl, con margen para el modelo fast
    cmd = swapconfig.build(cfg)[0]["models"]["qwen2.5-coder-14b"]["cmd"]
    assert "-ngl" not in cmd and "--fit-target" in cmd
    cfg.keep_loaded = False
    data, _ = swapconfig.build(cfg)
    assert "groups" not in data


def test_swap_config_skips_missing_files(tmp_path, monkeypatch):
    monkeypatch.setattr("agent.config.MODELS_DIR", tmp_path)
    data, warnings = swapconfig.build(cfg_two_models())
    assert data["models"] == {}
    assert any("falta el archivo" in w for w in warnings)
