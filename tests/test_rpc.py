"""PCs remotas (llama.cpp RPC): configuración, comando de llama-server, VRAM, Recalcular y paquete."""

import zipfile
from pathlib import Path

import pytest

from agent.config import AppConfig, ModelEntry, RpcWorker, load_catalog
from agent.server import autotune, rpc, swapconfig, vram
from agent.server.vram import GPUInfo

GPU8 = GPUInfo("RTX 4060", 8.0, 0.5, 7.5, "x")
GPU16 = GPUInfo("RTX 4070 Ti SUPER", 16.0, 1.0, 15.0, "x")
FILES = {
    "Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf": 8.37,
    "Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf": 1.80,
    "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf": 20.61,
}
LIST_LOCAL = """Available devices:
  CUDA0: NVIDIA GeForce RTX 4070 Ti SUPER (16375 MiB, 15111 MiB free)
"""
LIST_REMOTE = LIST_LOCAL + "  RPC0: 192.168.1.50:50052 (12282 MiB, 11000 MiB free)\n"


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    for name in FILES:
        (tmp_path / name).write_bytes(b"x")
    monkeypatch.setattr("agent.config.MODELS_DIR", tmp_path)
    monkeypatch.setattr("agent.server.vram.file_gb", lambda p: FILES[Path(p).name])
    return tmp_path


def cfg(main="qwen2.5-coder-14b", main_file="Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf", workers=(), **kw):
    return AppConfig(
        roles={"main": main, "fast": "qwen2.5-coder-3b", "draft": ""},
        models=[ModelEntry(main, main_file, ctx=16384),
                ModelEntry("qwen2.5-coder-3b", "Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf", ctx=16384)],
        rpc_workers=list(workers), **kw)


def test_config_roundtrip_and_sanitize():
    c = cfg(workers=[RpcWorker("192.168.1.50", vram_gb=12.0), RpcWorker(" 192.168.1.50 "),
                     RpcWorker(""), RpcWorker("pc2", port="abc", enabled=0)])
    c.models[0].placement = "split"
    back = AppConfig.from_dict(c.to_dict())
    fixes = back.sanitize()
    assert [w.endpoint for w in back.rpc_workers] == ["192.168.1.50:50052", "pc2:50052"]
    assert back.rpc_workers[0].vram_gb == 12.0 and back.rpc_workers[1].enabled is False
    assert back.models[0].rpc is True and back.models[1].rpc is False
    assert any("repetida" in f for f in fixes)
    assert back.rpc_endpoints() == ["192.168.1.50:50052"]
    # config antigua sin los campos nuevos
    old = AppConfig.from_dict({"models": [{"name": "a", "file": "a.gguf"}]})
    assert old.rpc_workers == [] and old.models[0].rpc is False
    legacy = AppConfig.from_dict({"models": [{"name": "a", "file": "a.gguf", "rpc": True}]})
    assert legacy.models[0].placement == "split"


def test_swap_command_adds_rpc_only_where_enabled(models_dir):
    c = cfg(workers=[RpcWorker("10.0.0.2"), RpcWorker("10.0.0.3", 50053), RpcWorker("10.0.0.4", enabled=False)])
    c.models[0].placement = "split"
    data, warnings = swapconfig.build(c)
    assert "--rpc 10.0.0.2:50052,10.0.0.3:50053" in data["models"]["qwen2.5-coder-14b"]["cmd"]
    assert "--rpc" not in data["models"]["qwen2.5-coder-3b"]["cmd"]
    assert data["healthCheckTimeout"] == 900
    assert "--device" not in data["models"]["qwen2.5-coder-14b"]["cmd"]
    c.models[1].placement = "remote"  # fast entero en las remotas: solo sus GPUs, sin reservar la local
    data, _ = swapconfig.build(c)
    assert "--device RPC0,RPC1" in data["models"]["qwen2.5-coder-3b"]["cmd"]
    assert "--fit-target" not in data["models"]["qwen2.5-coder-14b"]["cmd"]
    c.models[1].placement = "local"
    c.models[0].extra_args = "--rpc 1.2.3.4:5"  # el usuario ya lo puso a mano: no se duplica
    cmd = swapconfig.build(c)[0]["models"]["qwen2.5-coder-14b"]["cmd"]
    assert cmd.count("--rpc") == 1 and "1.2.3.4:5" in cmd
    c.models[0].extra_args = ""
    c.rpc_workers = []
    data, warnings = swapconfig.build(c)
    assert "--rpc" not in data["models"]["qwen2.5-coder-14b"]["cmd"]
    assert data["healthCheckTimeout"] == 600
    assert any("PC remota" in w for w in warnings)


def test_parse_devices():
    devs = rpc.parse_devices(LIST_REMOTE)
    assert [d.name for d in devs] == ["CUDA0", "RPC0"]
    assert devs[1].description == "192.168.1.50:50052"
    assert abs(devs[1].total_gb - 12.0) < 0.01


def test_probe(monkeypatch):
    rpc._local_device_names.cache_clear()
    monkeypatch.setattr(rpc, "reachable", lambda *a, **k: True)
    monkeypatch.setattr(swapconfig, "find_binary", lambda name: Path("llama-server.exe"))
    outputs = {None: (LIST_LOCAL, 0), "192.168.1.50:50052": (LIST_REMOTE, 0)}
    monkeypatch.setattr(rpc, "_list_devices", lambda exe, where, timeout: outputs[where])
    res = rpc.probe(RpcWorker("192.168.1.50"))
    assert res.ok and abs(res.total_gb - 12.0) < 0.01 and "12.0 GB" in res.summary()
    outputs["192.168.1.50:50052"] = ("ggml-rpc.cpp:547: Failed to connect to 192.168.1.50:50052", 9)
    res = rpc.probe(RpcWorker("192.168.1.50"))
    assert not res.ok and "firewall" in res.error
    monkeypatch.setattr(rpc, "reachable", lambda *a, **k: False)
    assert "start-worker.bat" in rpc.probe(RpcWorker("192.168.1.50")).error
    rpc._local_device_names.cache_clear()


def test_plan_counts_remote_vram(models_dir):
    c = cfg(main="qwen3.6-35b-a3b", main_file="Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
            workers=[RpcWorker("10.0.0.2", vram_gb=12.0)])
    plan = vram.plan_usage(c, load_catalog(), GPU16)
    assert plan.fits is False and plan.remote_gb == 0
    assert any("PC remota" in s for s in plan.suggestions)
    c.models[0].placement = "split"
    plan = vram.plan_usage(c, load_catalog(), GPU16)
    assert plan.fits and abs(plan.remote_gb - (12.0 - vram.DESKTOP_RESERVE_GB - vram.OVERHEAD_GB)) < 0.01


def test_autotune_uses_remote_for_dense_model(models_dir):
    c = cfg(workers=[RpcWorker("10.0.0.2", vram_gb=12.0)])
    r = autotune.recalculate(c, load_catalog(), GPU8, 32, choose_roles=False)
    main = r.config.model("qwen2.5-coder-14b")
    assert main.placement == "split" and main.gpu_layers == 99 and r.config.keep_loaded is True
    ch = next(x for x in r.changes if x.model == main.name and x.field == "placement")
    assert ch.reason and autotune.describe(ch.new, "placement") == "repartido con PC remota"
    # en una GPU donde cabe, se quita la PC remota
    autotune.apply(c, r.config)
    r = autotune.recalculate(c, load_catalog(), GPU16, 32, choose_roles=False)
    assert r.config.model("qwen2.5-coder-14b").placement == "local"


def test_autotune_uses_total_vram_before_ram(models_dir):
    """La VRAM remota cuenta como VRAM total aunque sea más lenta: antes que mandar expertos a la
    RAM, el modelo se reparte con la PC remota."""
    c = cfg(main="qwen3.6-35b-a3b", main_file="Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
            workers=[RpcWorker("10.0.0.2", vram_gb=12.0)])
    r = autotune.recalculate(c, load_catalog(), GPU16, 32, choose_roles=False)
    main = r.config.model("qwen3.6-35b-a3b")
    assert main.placement == "split" and main.gpu_layers == 99  # 15,2 + 10,6 GB: cabe entero en VRAM
    # Sin la PC remota (quitada de la lista), vuelve a esta PC con expertos en RAM.
    c.rpc_workers = []
    r = autotune.recalculate(c, load_catalog(), GPU16, 32, choose_roles=False)
    main = r.config.model("qwen3.6-35b-a3b")
    assert main.placement == "local" and main.gpu_layers == -1


def test_autotune_keeps_rpc_when_remote_not_measured(models_dir):
    c = cfg(workers=[RpcWorker("10.0.0.2")])
    c.models[0].placement = "split"
    r = autotune.recalculate(c, load_catalog(), GPU16, 32, choose_roles=False)
    assert r.config.model("qwen2.5-coder-14b").placement == "split"
    assert any("Probar" in n for n in r.notes)


def test_worker_package(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("ggml-rpc-server.exe", "ggml.dll", "ggml-base.dll", "cublas64_12.dll",
                 "cublas64_13.dll", "cublasLt64_13.dll", "cudart64_13.dll", "llama.dll", "LICENSE.md"):
        (bin_dir / name).write_bytes(b"x")
    (bin_dir / "ggml-cuda.dll").write_bytes(b"....cublas64_13.dll....")
    monkeypatch.setattr(swapconfig, "BIN_DIR", bin_dir)
    dest = rpc.build_worker_package("192.168.1.10", tmp_path / "out" / "rpc-worker.zip")
    with zipfile.ZipFile(dest) as z:
        names = {Path(n).name for n in z.namelist()}
        fw = z.read("agent-rpc/permitir-firewall.bat").decode()
        start = z.read("agent-rpc/start-worker.bat").decode()
    assert {"ggml-rpc-server.exe", "ggml-cuda.dll", "cublas64_13.dll", "cublasLt64_13.dll",
            "cudart64_13.dll", "start-worker.bat", "LEEME.txt"} <= names
    assert "cublas64_12.dll" not in names and "llama.dll" not in names
    assert "remoteip=192.168.1.10" in fw and "profile=private" in fw
    assert "-H 0.0.0.0 -p 50052 -c" in start
    assert not list(dest.parent.glob("*.tmp"))


@pytest.mark.parametrize("mode, moe_place, dense_place", [
    ("vram", "split", "split"),    # VRAM total antes que la RAM
    ("local", "local", "local"),   # esta PC (GPU + RAM) primero; la remota solo si no cabe
    ("speed", "local", "split"),   # lo más rápido: MoE con RAM gana a la red; denso con RAM no
])
def test_tune_mode_preference(models_dir, mode, moe_place, dense_place):
    moe = cfg(main="qwen3.6-35b-a3b", main_file="Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
              workers=[RpcWorker("10.0.0.2", vram_gb=12.0)], tune_mode=mode)
    r = autotune.recalculate(moe, load_catalog(), GPU16, 32, choose_roles=False)
    assert r.config.model("qwen3.6-35b-a3b").placement == moe_place
    assert mode in r.hardware or autotune.MODE_LABELS[mode] in r.hardware
    dense = cfg(workers=[RpcWorker("10.0.0.2", vram_gb=12.0)], tune_mode=mode)
    r = autotune.recalculate(dense, load_catalog(), GPU8, 32, choose_roles=False)
    assert r.config.model("qwen2.5-coder-14b").placement == dense_place


def test_local_mode_still_uses_remote_when_nothing_else_fits(models_dir):
    c = cfg(workers=[RpcWorker("10.0.0.2", vram_gb=12.0)], tune_mode="local")
    r = autotune.recalculate(c, load_catalog(), GPU8, 1, choose_roles=False)  # sin RAM para modelos
    assert r.config.model("qwen2.5-coder-14b").placement == "split"


def test_tune_mode_sanitized():
    c = AppConfig(tune_mode="rapidisimo")
    assert any("tune_mode" in f for f in c.sanitize()) and c.tune_mode == "vram"
