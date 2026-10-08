"""scripts/download_soundfont.py delegates to music_decoder.assets.fetch_soundfont."""

import importlib.util
from pathlib import Path
from types import ModuleType
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "download_soundfont", ROOT / "scripts" / "download_soundfont.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_script_delegates_to_assets(tmp_path: Path) -> None:
    script = _load()
    dest = tmp_path / "x.sf2"
    with mock.patch("music_decoder.assets.fetch_soundfont", return_value=dest) as fetch:
        rc = script.main(["--dest", str(dest), "--url", "https://example.org/x.sf2", "--force"])
    assert rc == 0
    fetch.assert_called_once_with(dest, "https://example.org/x.sf2", force=True)


def test_script_reports_failure(tmp_path: Path) -> None:
    script = _load()
    with mock.patch("music_decoder.assets.fetch_soundfont", side_effect=RuntimeError("bad")):
        assert script.main(["--dest", str(tmp_path / "x.sf2")]) == 1
