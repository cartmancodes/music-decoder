"""Model / asset provisioning: bundled soundfont, soundfont download, Demucs cache."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from music_decoder import assets

_SF2_BYTES = b"RIFF" + (1024).to_bytes(4, "little") + b"sfbk" + b"\0" * 1024


def test_bundled_soundfont_is_a_real_soundfont() -> None:
    sf = assets.bundled_soundfont()
    assert sf is not None
    assert sf.name == "TimGM6mb.sf2"
    assert assets.is_soundfont(sf)


def test_is_soundfont_rejects_html_error_pages(tmp_path: Path) -> None:
    good = tmp_path / "good.sf2"
    good.write_bytes(_SF2_BYTES)
    bad = tmp_path / "bad.sf2"
    bad.write_bytes(b"<!DOCTYPE html><html>404</html>")
    assert assets.is_soundfont(good)
    assert not assets.is_soundfont(bad)
    assert not assets.is_soundfont(tmp_path / "missing.sf2")


def _fake_urlopen(payload: bytes) -> Any:
    def opener(url: str, *args: Any, **kwargs: Any) -> Any:
        return io.BytesIO(payload)

    return opener


def test_fetch_soundfont_downloads_and_validates(tmp_path: Path) -> None:
    dest = tmp_path / "sf" / "GeneralUser-GS.sf2"
    with mock.patch("urllib.request.urlopen", _fake_urlopen(_SF2_BYTES)):
        out = assets.fetch_soundfont(dest)
    assert out == dest
    assert assets.is_soundfont(dest)
    assert not list(dest.parent.glob("*.part"))


def test_fetch_soundfont_rejects_invalid_download_and_leaves_nothing(tmp_path: Path) -> None:
    dest = tmp_path / "GeneralUser-GS.sf2"
    with (
        mock.patch("urllib.request.urlopen", _fake_urlopen(b"<html>not found</html>")),
        pytest.raises(RuntimeError, match="not a SoundFont"),
    ):
        assets.fetch_soundfont(dest)
    assert list(tmp_path.iterdir()) == []


def test_fetch_soundfont_is_idempotent(tmp_path: Path) -> None:
    dest = tmp_path / "GeneralUser-GS.sf2"
    dest.write_bytes(_SF2_BYTES)
    with mock.patch("urllib.request.urlopen", side_effect=AssertionError("no download")):
        assert assets.fetch_soundfont(dest) == dest


def test_soundfont_target_is_the_configured_runtime_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MUSIC_DECODER_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MUSIC_DECODER_FLUIDSYNTH_SOUNDFONT", "GeneralUser-GS.sf2")
    assert assets.soundfont_target() == tmp_path / "GeneralUser-GS.sf2"


def test_demucs_signatures_for_htdemucs_6s() -> None:
    assert assets.demucs_signatures("htdemucs_6s") == ["5c90dfd2"]


def test_demucs_weights_cached_checks_torch_hub_dir(tmp_path: Path) -> None:
    with mock.patch("torch.hub.get_dir", return_value=str(tmp_path)):
        assert not assets.demucs_weights_cached("htdemucs_6s")
        (tmp_path / "checkpoints").mkdir()
        (tmp_path / "checkpoints" / "5c90dfd2-34c22ccb.th").write_bytes(b"x")
        assert assets.demucs_weights_cached("htdemucs_6s")


def test_synth_falls_back_to_bundled_soundfont(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Nothing installed at the configured path → compose still gets a real
    # soundfont instead of degrading to sine waves.
    from music_decoder.synth import _resolve_soundfont_path

    monkeypatch.setenv("MUSIC_DECODER_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MUSIC_DECODER_FLUIDSYNTH_SOUNDFONT", "not-installed.sf2")
    assert _resolve_soundfont_path() == assets.bundled_soundfont()


def test_synth_prefers_installed_soundfont_over_bundled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from music_decoder.synth import _resolve_soundfont_path

    (tmp_path / "GeneralUser-GS.sf2").write_bytes(_SF2_BYTES)
    monkeypatch.setenv("MUSIC_DECODER_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MUSIC_DECODER_FLUIDSYNTH_SOUNDFONT", "GeneralUser-GS.sf2")
    assert _resolve_soundfont_path() == tmp_path / "GeneralUser-GS.sf2"
