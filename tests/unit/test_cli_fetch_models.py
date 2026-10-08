"""`music-decoder fetch-models` and the model checks in `doctor`."""

from pathlib import Path
from unittest import mock

from click.testing import CliRunner

from music_decoder.cli.main import main


def test_fetch_models_fetches_demucs_and_soundfont(tmp_path: Path) -> None:
    sf = tmp_path / "GeneralUser-GS.sf2"
    with (
        mock.patch("music_decoder.assets.fetch_demucs") as demucs,
        mock.patch("music_decoder.assets.fetch_soundfont", return_value=sf) as soundfont,
    ):
        r = CliRunner().invoke(main, ["fetch-models"])
    assert r.exit_code == 0, r.output
    demucs.assert_called_once_with()
    soundfont.assert_called_once_with()
    assert "Demucs" in r.output
    assert str(sf) in r.output


def test_fetch_models_flags_skip_steps() -> None:
    with (
        mock.patch("music_decoder.assets.fetch_demucs") as demucs,
        mock.patch("music_decoder.assets.fetch_soundfont") as soundfont,
    ):
        r = CliRunner().invoke(main, ["fetch-models", "--no-demucs", "--no-soundfont"])
    assert r.exit_code == 0, r.output
    demucs.assert_not_called()
    soundfont.assert_not_called()


def test_fetch_models_continues_after_a_failure_and_exits_nonzero(tmp_path: Path) -> None:
    with (
        mock.patch("music_decoder.assets.fetch_demucs", side_effect=OSError("offline")),
        mock.patch(
            "music_decoder.assets.fetch_soundfont", return_value=tmp_path / "x.sf2"
        ) as soundfont,
    ):
        r = CliRunner().invoke(main, ["fetch-models"])
    assert r.exit_code == 1
    soundfont.assert_called_once()
    assert "offline" in r.output


def test_doctor_reports_demucs_cache_and_soundfont_without_failing_on_missing_demucs() -> None:
    with (
        mock.patch("music_decoder.assets.demucs_weights_cached", return_value=False),
        mock.patch("shutil.which", return_value="/usr/bin/ffmpeg"),
    ):
        r = CliRunner().invoke(main, ["doctor"])
    assert "demucs weights" in r.output
    assert "fetch-models" in r.output
    assert "soundfont" in r.output
    assert "FAIL" not in r.output.split("demucs weights")[1].splitlines()[0]
