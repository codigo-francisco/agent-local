"""«Recargar recomendaciones» contra una API de Hugging Face simulada (sin red)."""

import json

import httpx

from agent.config import AppConfig, RpcWorker
from agent.server import recommend
from agent.server.vram import GPUInfo

GPU16 = GPUInfo("RTX 4070 Ti SUPER", 16.0, 1.0, 15.0, "x")
TOOLS = "{% if tools %}{{ tools }}{% endif %}<think>"

REPOS = [
    {"id": "unsloth/Coder-30B-A3B-GGUF", "pipeline_tag": "text-generation", "downloads": 900000,
     "createdAt": "2026-08-01T00:00:00Z", "tags": ["base_model:quantized:org/Coder-30B-A3B"],
     "gguf": {"architecture": "qwen3moe", "total": 30.5e9, "chat_template": TOOLS}},
    {"id": "bartowski/org_Coder-30B-A3B-GGUF", "pipeline_tag": "text-generation", "downloads": 5,
     "tags": ["base_model:quantized:org/Coder-30B-A3B"],  # mismo modelo base: se queda el de unsloth
     "gguf": {"architecture": "qwen3moe", "total": 30.5e9, "chat_template": TOOLS}},
    {"id": "unsloth/Dense-14B-GGUF", "pipeline_tag": "text-generation", "downloads": 1000,
     "gguf": {"architecture": "qwen3", "total": 14e9, "chat_template": TOOLS}},
    {"id": "unsloth/NoTools-8B-GGUF", "pipeline_tag": "text-generation",
     "gguf": {"architecture": "llama", "total": 8e9, "chat_template": "{{ messages }}"}},
    {"id": "unsloth/Image-GGUF", "pipeline_tag": "text-to-image",
     "gguf": {"architecture": "flux", "total": 9e9, "chat_template": TOOLS}},
    {"id": "unsloth/Huge-400B-GGUF", "pipeline_tag": "text-generation",
     "gguf": {"architecture": "qwen3moe", "total": 400e9, "chat_template": TOOLS}},
]
TREES = {
    "unsloth/Coder-30B-A3B-GGUF": [("Coder-30B-A3B-Q8_0.gguf", 32.5e9), ("Coder-30B-A3B-UD-Q4_K_XL.gguf", 17.7e9),
                                   ("Coder-30B-A3B-Q3_K_M.gguf", 14.7e9), ("mmproj-F16.gguf", 1e9),
                                   ("mtp-Coder-Q8_0.gguf", 2e9)],
    "unsloth/Dense-14B-GGUF": [("Dense-14B-Q8_0.gguf", 15.7e9), ("Dense-14B-Q4_K_M.gguf", 9.0e9),
                               ("BF16/Dense-14B-BF16-00001-of-00002.gguf", 15e9)],
}


def handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/api/models":
        return httpx.Response(200, json=REPOS if request.url.params.get("author") == "unsloth" else
                              [r for r in REPOS if r["id"].startswith(request.url.params.get("author", "") + "/")])
    if path.endswith("/tree/main"):
        repo = path.removeprefix("/api/models/").removesuffix("/tree/main")
        return httpx.Response(200, json=[{"type": "file", "path": p, "size": int(s)}
                                         for p, s in TREES.get(repo, [])])
    return httpx.Response(404)


def test_quant_of():
    assert recommend.quant_of("M-UD-Q4_K_XL.gguf") == ("UD-Q4_K_XL", 0.82)
    assert recommend.quant_of("M-Q6_K.gguf")[0] == "Q6_K"
    assert recommend.quant_of("M-Q2_K.gguf") is None  # demasiado degradado: no se recomienda


def test_refresh_filters_and_picks_quant(tmp_path, monkeypatch):
    monkeypatch.setattr(recommend, "CACHE_FILE", tmp_path / "rec.json")
    monkeypatch.setattr("agent.server.recommend.MODELS_DIR", tmp_path)
    monkeypatch.setattr(recommend, "arch_supported", lambda arch: arch != "flux")
    steps = []
    rep = recommend.refresh(AppConfig(), GPU16, 32, steps.append, httpx.MockTransport(handler))
    assert rep.error == ""
    repos = [r.repo for r in rep.items]
    assert repos == ["unsloth/Coder-30B-A3B-GGUF", "unsloth/Dense-14B-GGUF"]
    coder, dense = rep.items
    assert coder.moe and coder.quant == "UD-Q4_K_XL" and coder.ram_gb > 0 and "RAM" in coder.placement
    # El Q8 (15,7 GB + contexto) no cabe entero en 16 GB: mejor Q4_K_M entero en la GPU.
    assert dense.quant == "Q4_K_M" and dense.ram_gb == 0 and dense.speed == 1.0
    assert "tarjeta gráfica" in dense.placement and "remota" not in dense.placement
    assert coder.coding and coder.thinking
    assert any("Hugging Face" in s for s in steps)
    # queda en caché para la próxima vez que se abra la página
    cached = recommend.load(tmp_path / "rec.json")
    assert [r.repo for r in cached.items] == repos and cached.hardware == rep.hardware


def test_remote_pc_changes_the_fit(tmp_path, monkeypatch):
    monkeypatch.setattr(recommend, "CACHE_FILE", tmp_path / "rec.json")
    monkeypatch.setattr("agent.server.recommend.MODELS_DIR", tmp_path)
    monkeypatch.setattr(recommend, "arch_supported", lambda arch: True)
    cfg = AppConfig(rpc_workers=[RpcWorker("10.0.0.2", vram_gb=12.0)])
    gpu8 = GPUInfo("RTX 4060", 8.0, 0.5, 7.5, "x")
    rep = recommend.refresh(cfg, gpu8, 4, None, httpx.MockTransport(handler))  # casi sin RAM
    dense = next(r for r in rep.items if r.repo == "unsloth/Dense-14B-GGUF")
    # Denso que no cabe en 8 GB: repartido con la PC remota, que ya permite el Q8.
    assert dense.quant == "Q8_0" and "PC remota" in dense.placement and "PCs remotas" in rep.hardware


def test_offline_error(tmp_path, monkeypatch):
    monkeypatch.setattr(recommend, "CACHE_FILE", tmp_path / "rec.json")

    def down(request):
        raise httpx.ConnectError("sin red")

    rep = recommend.refresh(AppConfig(), GPU16, 32, None, httpx.MockTransport(down))
    assert "Hugging Face" in rep.error
    assert not (tmp_path / "rec.json").exists()
    json.dumps(rep.error)


def test_remote_vram_counts_as_one_pool(tmp_path, monkeypatch):
    """GPU de 16 GB + PC remota de 12 GB = ~25,8 GB de VRAM utilizable para el modelo."""
    monkeypatch.setattr(recommend, "CACHE_FILE", tmp_path / "rec.json")
    monkeypatch.setattr("agent.server.recommend.MODELS_DIR", tmp_path)
    monkeypatch.setattr(recommend, "arch_supported", lambda arch: True)
    alone = recommend.refresh(AppConfig(), GPU16, 32, None, httpx.MockTransport(handler))
    cfg = AppConfig(rpc_workers=[RpcWorker("10.0.0.2", vram_gb=12.0)])
    pooled = recommend.refresh(cfg, GPU16, 32, None, httpx.MockTransport(handler))
    coder_alone = next(r for r in alone.items if r.repo == "unsloth/Coder-30B-A3B-GGUF")
    coder = next(r for r in pooled.items if r.repo == "unsloth/Coder-30B-A3B-GGUF")
    assert coder_alone.ram_gb > 0 and coder_alone.speed < 1
    # Con la VRAM sumada el Q4 (17,7 GB) cabe entero en VRAM: sin RAM y a velocidad plena.
    assert coder.ram_gb == 0 and "PC remota" in coder.placement and coder.speed == 1.0
    dense = next(r for r in pooled.items if r.repo == "unsloth/Dense-14B-GGUF")
    assert dense.quant == "Q8_0" and "PC remota" in dense.placement  # antes no cabía: Q4 solo
    assert "de VRAM en total" in pooled.hardware


def test_recompute_when_remote_list_changes(tmp_path, monkeypatch):
    """Añadir o quitar una PC remota rehace las recomendaciones sin volver a consultar la red."""
    monkeypatch.setattr(recommend, "CACHE_FILE", tmp_path / "rec.json")
    monkeypatch.setattr("agent.server.recommend.MODELS_DIR", tmp_path)
    monkeypatch.setattr(recommend, "arch_supported", lambda arch: True)
    cfg = AppConfig()
    rep = recommend.refresh(cfg, GPU16, 32, None, httpx.MockTransport(handler))
    coder = next(r for r in rep.items if r.repo == "unsloth/Coder-30B-A3B-GGUF")
    assert coder.ram_gb > 0
    cfg.rpc_workers.append(RpcWorker("10.0.0.2", vram_gb=12.0))  # se añade a la lista

    with_remote = recommend.recompute(recommend.load(), cfg, GPU16, 32)
    coder = next(r for r in with_remote.items if r.repo == "unsloth/Coder-30B-A3B-GGUF")
    assert coder.ram_gb == 0 and "PC remota" in coder.placement
    assert "de VRAM en total" in with_remote.hardware
    cfg.rpc_workers.clear()  # se quita de la lista
    without = recommend.recompute(with_remote, cfg, GPU16, 32)
    assert next(r for r in without.items if r.repo == "unsloth/Coder-30B-A3B-GGUF").ram_gb > 0
    assert "PCs remotas" not in without.hardware


def test_recommend_respects_tune_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(recommend, "CACHE_FILE", tmp_path / "rec.json")
    monkeypatch.setattr("agent.server.recommend.MODELS_DIR", tmp_path)
    monkeypatch.setattr(recommend, "arch_supported", lambda arch: True)
    cfg = AppConfig(rpc_workers=[RpcWorker("10.0.0.2", vram_gb=12.0)], tune_mode="local")
    rep = recommend.refresh(cfg, GPU16, 32, None, httpx.MockTransport(handler))
    coder = next(r for r in rep.items if r.repo == "unsloth/Coder-30B-A3B-GGUF")
    assert coder.ram_gb > 0 and "remota" not in coder.placement  # local primero: RAM de esta PC
    assert "preferir lo local" in rep.hardware
    cfg.tune_mode = "vram"
    rep = recommend.recompute(rep, cfg, GPU16, 32)
    coder = next(r for r in rep.items if r.repo == "unsloth/Coder-30B-A3B-GGUF")
    assert coder.ram_gb == 0 and "PC remota" in coder.placement
