"""Config: escritura atómica, arranque con archivo roto y el CLI que no guarda sus opciones."""

import yaml

from agent import config
from agent.config import AppConfig, atomic_write, load_config


def test_atomic_write_replaces_and_cleans(tmp_path):
    p = tmp_path / "x.yaml"
    atomic_write(p, b"uno")
    atomic_write(p, b"dos")
    assert p.read_bytes() == b"dos" and [f.name for f in tmp_path.iterdir()] == ["x.yaml"]


def test_broken_yaml_is_backed_up_and_defaults_load(tmp_path):
    p = tmp_path / "models.yaml"
    p.write_text("port: [8080\nroles: {", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.port == 8080 and cfg.roles["main"]
    assert "no se pudo leer" in config.load_warning
    backups = [f for f in tmp_path.iterdir() if ".bak-" in f.name]
    assert len(backups) == 1 and "port: [8080" in backups[0].read_text(encoding="utf-8")


def test_unknown_keys_are_ignored(tmp_path):
    p = tmp_path / "models.yaml"
    p.write_text(yaml.safe_dump({"port": 9000, "clave_vieja": 1,
                                 "models": [{"name": "m", "file": "m.gguf", "obsoleto": True}]}),
                 encoding="utf-8")
    cfg = load_config(p)
    assert cfg.port == 9000 and cfg.models[0].name == "m" and config.load_warning == ""


def test_cli_persist_only_saves_permissions(tmp_path):
    from agent.cli import make_persist
    p = tmp_path / "models.yaml"
    AppConfig(workspace="C:/proyecto", confirm="ask").save(p)
    cfg = load_config(p)
    cfg.workspace, cfg.confirm = "D:/temporal", "auto"  # lo que hacen --workspace y --auto
    cfg.always_allow.append("run_command")
    make_persist(cfg, p)()
    disk = load_config(p)
    assert disk.always_allow == ["run_command"]
    assert disk.confirm == "ask" and disk.workspace == "C:/proyecto"
