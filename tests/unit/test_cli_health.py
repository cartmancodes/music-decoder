from pathlib import Path

import pytest

from music_decoder.cli.health import (
    FfmpegMissing,
    check_ffmpeg,
    ensure_data_dirs,
)


def test_ensure_data_dirs_creates_paths(tmp_path: Path):
    db_path = tmp_path / "appdata" / "db.sqlite3"
    artifact_dir = tmp_path / "appdata" / "artifacts"
    model_cache = tmp_path / "cache" / "models"
    ensure_data_dirs(db_path=db_path, artifact_dir=artifact_dir, model_cache=model_cache)
    assert db_path.parent.exists()
    assert artifact_dir.exists()
    assert model_cache.exists()


def test_check_ffmpeg_passes_when_present(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/ffmpeg")
    check_ffmpeg()


def test_check_ffmpeg_raises_when_missing(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: None)
    with pytest.raises(FfmpegMissing):
        check_ffmpeg()
