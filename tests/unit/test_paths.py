# tests/unit/test_paths.py
from pathlib import Path

from music_decoder import paths


def test_project_root_is_repo_root():
    root = paths.project_root()
    assert (root / "pyproject.toml").is_file()
    assert (root / "src" / "music_decoder").is_dir()


def test_bundled_config_points_into_config_dir():
    p = paths.bundled_config("runtime.yaml")
    assert p == paths.project_root() / "config" / "runtime.yaml"
    assert p.is_file()


def test_project_root_matches_legacy_parents3():
    # Byte-identical to the idiom being replaced in config/runtime.py
    legacy = (
        Path(__import__("music_decoder.config.runtime", fromlist=["x"]).__file__)
        .resolve()
        .parents[3]
    )
    assert paths.project_root() == legacy
