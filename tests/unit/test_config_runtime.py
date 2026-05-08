from pathlib import Path

import pytest

from music_decoder.config.runtime import RuntimeConfig, load_runtime_config


def test_load_runtime_config_reads_yaml(tmp_path: Path):
    cfg_path = tmp_path / "runtime.yaml"
    cfg_path.write_text(
        """
        youtube_cache_dir: /tmp/yt_cache
        composition_out_dir: /tmp/compositions
        log_level: DEBUG
        ffmpeg_path: /usr/local/bin/ffmpeg
        model_cache_dir: /tmp/models
        fixture_dir: /tmp/fixtures
        """
    )
    cfg = load_runtime_config(cfg_path)
    assert isinstance(cfg, RuntimeConfig)
    assert cfg.youtube_cache_dir == Path("/tmp/yt_cache")
    assert cfg.composition_out_dir == Path("/tmp/compositions")
    assert cfg.log_level == "DEBUG"
    assert cfg.ffmpeg_path == Path("/usr/local/bin/ffmpeg")


def test_env_overrides_yaml(tmp_path: Path, monkeypatch):
    cfg_path = tmp_path / "runtime.yaml"
    cfg_path.write_text(
        "youtube_cache_dir: /tmp/yt_cache\n"
        "composition_out_dir: /tmp/compositions\n"
        "log_level: INFO\n"
    )
    monkeypatch.setenv("MUSIC_DECODER_YOUTUBE_CACHE_DIR", "/tmp/override_yt")
    cfg = load_runtime_config(cfg_path)
    assert cfg.youtube_cache_dir == Path("/tmp/override_yt")


def test_missing_required_key_raises(tmp_path: Path):
    cfg_path = tmp_path / "runtime.yaml"
    cfg_path.write_text("composition_out_dir: /tmp/c\n")
    with pytest.raises(ValueError, match="youtube_cache_dir"):
        load_runtime_config(cfg_path)


def test_runtime_config_has_youtube_cache_dir(tmp_path, monkeypatch):
    yaml_path = tmp_path / "runtime.yaml"
    yaml_path.write_text(
        "data_dir: ~/.music-decoder\n"
        "log_level: INFO\n"
        "youtube_cache_dir: ~/.music-decoder/yt_cache\n"
        "composition_out_dir: ~/.music-decoder/compositions\n"
    )
    from music_decoder.config.runtime import load_runtime_config

    cfg = load_runtime_config(yaml_path)
    assert str(cfg.youtube_cache_dir).endswith("yt_cache")
    assert str(cfg.composition_out_dir).endswith("compositions")
