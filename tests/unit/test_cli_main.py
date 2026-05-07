# tests/unit/test_cli_main.py
from unittest import mock

from click.testing import CliRunner

from music_decoder.cli.main import main


def test_help_runs():
    r = CliRunner().invoke(main, ["--help"])
    assert r.exit_code == 0
    assert "analyze" in r.output and "compose" in r.output


def test_analyze_subcommand_calls_api(tmp_path):
    with mock.patch("music_decoder.cli.main.analyze") as a:
        a.return_value = mock.Mock(
            source=str(tmp_path / "x.wav"), duration_s=10.0,
            sample_rate_hz=22050,
            key=mock.Mock(tonic="C", mode="major"),
            chord_progression=(), tab=(), tempo_bpm=120.0,
            beat_times_s=(), metadata={},
        )
        r = CliRunner().invoke(main, ["analyze", str(tmp_path / "x.wav"),
                                      "--format", "json"])
    assert r.exit_code == 0


def test_compose_subcommand_calls_api(tmp_path):
    with mock.patch("music_decoder.cli.main.compose") as c:
        c.return_value = mock.Mock(
            midi_path=tmp_path / "a.mid", wav_path=tmp_path / "a.wav",
            ascii_tab="e|---", melody_notes=(), chord_voicings=(),
            metadata={"seed": 42},
        )
        r = CliRunner().invoke(main, [
            "compose",
            "--scale", "C:major",
            "--progression", "Cmaj7 Am7 Dm7 G7",
            "--out", str(tmp_path),
        ])
    assert r.exit_code == 0
    c.assert_called_once()


def test_compose_invalid_scale():
    r = CliRunner().invoke(main, [
        "compose", "--scale", "C:dorian", "--progression", "C", "--out", "/tmp",
    ])
    assert r.exit_code != 0


def test_doctor_runs():
    with mock.patch("music_decoder.cli.main._doctor_checks", return_value=0):
        r = CliRunner().invoke(main, ["doctor"])
    assert r.exit_code == 0
