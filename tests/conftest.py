import pytest

from agent import config
from agent.server import vram


@pytest.fixture(autouse=True)
def isolated_models(tmp_path_factory, monkeypatch):
    """Los tests no leen los .gguf reales de models/ (GGUFReader tarda ~5 s por modelo y el
    resultado dependería de lo que haya descargado cada uno): se usa el catálogo, igual que en CI."""
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path_factory.mktemp("models"))
    monkeypatch.setattr(vram, "GGUF_CACHE_FILE", tmp_path_factory.mktemp("cache") / "gguf-cache.json")
    monkeypatch.setattr(vram, "_disk_cache", None)
