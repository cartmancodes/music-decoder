from pathlib import Path

import pytest

from music_decoder.config.runtime import RuntimeConfig, load_runtime_config


def test_load_runtime_config_reads_yaml(tmp_path: Path):
    cfg_path = tmp_path / "runtime.yaml"
    cfg_path.write_text(
        """
        db_path: /tmp/db.sqlite3
        artifact_dir: /tmp/artifacts
        log_level: DEBUG
        ffmpeg_path: /usr/local/bin/ffmpeg
        model_cache_dir: /tmp/models
        fixture_dir: /tmp/fixtures
        """
    )
    cfg = load_runtime_config(cfg_path)
    assert isinstance(cfg, RuntimeConfig)
    assert cfg.db_path == Path("/tmp/db.sqlite3")
    assert cfg.log_level == "DEBUG"
    assert cfg.ffmpeg_path == Path("/usr/local/bin/ffmpeg")


def test_env_overrides_yaml(tmp_path: Path, monkeypatch):
    cfg_path = tmp_path / "runtime.yaml"
    cfg_path.write_text("db_path: /tmp/a.sqlite3\nartifact_dir: /tmp/a\nlog_level: INFO\n")
    monkeypatch.setenv("MUSIC_DECODER_DB_PATH", "/tmp/override.sqlite3")
    cfg = load_runtime_config(cfg_path)
    assert cfg.db_path == Path("/tmp/override.sqlite3")


def test_missing_required_key_raises(tmp_path: Path):
    cfg_path = tmp_path / "runtime.yaml"
    cfg_path.write_text("artifact_dir: /tmp/a\n")
    with pytest.raises(ValueError, match="db_path"):
        load_runtime_config(cfg_path)
