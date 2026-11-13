import json
from pathlib import Path
from unittest import mock

from click.testing import CliRunner

from music_decoder.cli.main import main
from music_decoder.productionize import Production, ProductionError
from music_decoder.productionize.config import PercussionLayer
from music_decoder.productionize.master import MasterReport
from music_decoder.productionize.qc import Check, QcReport
from music_decoder.productionize.song import Song


def _production(tmp_path: Path, *, passed: bool = True) -> Production:
    return Production(
        wav_path=tmp_path / "production.wav",
        mp3_path=tmp_path / "production.mp3",
        raw_mix_path=tmp_path / "pro_raw.wav",
        stem_paths={"drums": tmp_path / "stem_drums.wav"},
        master=MasterReport(-10.5, -1.93, 4.8, 2.0, 0.9, 2),
        qc=QcReport((Check("loudness (WAV)", passed, "-10.5 LUFS", "-10.5 +-0.3 LU"),)),
        metadata={"credits": ["Piano: Salamander Grand Piano V3 by Alexander Holm (CC BY 3.0)"]},
    )


def test_help_lists_the_new_commands():
    r = CliRunner().invoke(main, ["--help"])
    assert "productionize" in r.output and "fetch-samples" in r.output


def test_productionize_passes_its_options_to_the_api(tmp_path):
    with mock.patch("music_decoder.cli.main.productionize") as p:
        p.return_value = _production(tmp_path)
        r = CliRunner().invoke(
            main,
            [
                "productionize", str(tmp_path / "take"),
                "--out", str(tmp_path / "out"),
                "--name", "night",
                "--title", "Night Drive",
                "--key", "Am",
                "--progression", "Am F C G",
                "--percussion", "shaker",
                "--percussion", "tambourine@1,6",
            ],
        )  # fmt: skip
    assert r.exit_code == 0, r.output
    args, kwargs = p.call_args
    assert args == (str(tmp_path / "take"),)
    assert kwargs["out_dir"] == tmp_path / "out"
    assert (kwargs["name"], kwargs["title"], kwargs["key"]) == ("night", "Night Drive", "Am")
    assert [c.to_label() for c in kwargs["progression"]] == ["Am", "F", "C", "G"]
    assert kwargs["config"].percussion.layers == (
        PercussionLayer("shaker"),
        PercussionLayer("tambourine", (1, 6)),
    )
    assert kwargs["check"] is True
    assert "WAV:" in r.output and "PASS" in r.output and "Salamander" in r.output


def test_productionize_json_output(tmp_path):
    with mock.patch("music_decoder.cli.main.productionize") as p:
        p.return_value = _production(tmp_path)
        r = CliRunner().invoke(
            main,
            ["productionize", "take", "--out", str(tmp_path), "--no-qc", "--format", "json"],
        )
    assert r.exit_code == 0, r.output
    payload = json.loads(r.stdout)
    assert payload["wav_path"].endswith("production.wav")
    assert payload["qc"]["passed"] is True
    assert payload["master"]["iterations"] == 2
    assert p.call_args.kwargs["check"] is False


def test_failed_checks_exit_non_zero_after_printing_the_report(tmp_path):
    with mock.patch("music_decoder.cli.main.productionize") as p:
        p.return_value = _production(tmp_path, passed=False)
        r = CliRunner().invoke(main, ["productionize", "take", "--out", str(tmp_path)])
    assert r.exit_code == 1
    assert "FAIL" in r.output and "1 acceptance check(s) failed: loudness (WAV)" in r.output


def test_hisaab_flag_renders_the_reference_song(tmp_path):
    with mock.patch("music_decoder.cli.main.productionize") as p:
        p.return_value = _production(tmp_path)
        r = CliRunner().invoke(main, ["productionize", "--hisaab", "--out", str(tmp_path)])
    assert r.exit_code == 0, r.output
    (song,) = p.call_args.args
    assert isinstance(song, Song) and song.title == "Hisaab (Instrumental)"
    assert len(song.piano) == 263


def test_a_source_is_required(tmp_path):
    r = CliRunner().invoke(main, ["productionize", "--out", str(tmp_path)])
    assert r.exit_code != 0 and "composition folder" in r.output


def test_unknown_percussion_is_rejected(tmp_path):
    r = CliRunner().invoke(
        main, ["productionize", "take", "--out", str(tmp_path), "--percussion", "cowbell"]
    )
    assert r.exit_code != 0 and "cowbell" in r.output
    r = CliRunner().invoke(
        main, ["productionize", "take", "--out", str(tmp_path), "--percussion", "shaker@a"]
    )
    assert r.exit_code != 0 and "must be numbers" in r.output


def test_errors_are_shown_without_a_traceback(tmp_path):
    with mock.patch("music_decoder.cli.main.productionize") as p:
        p.side_effect = ProductionError("no composer MIDI file at take/composition.mid")
        r = CliRunner().invoke(main, ["productionize", "take", "--out", str(tmp_path)])
    assert r.exit_code == 1 and "no composer MIDI file" in r.output


def test_fetch_samples_list_prints_url_and_path():
    r = CliRunner().invoke(main, ["fetch-samples", "--list"])
    assert r.exit_code == 0
    lines = r.output.strip().splitlines()
    assert len(lines) == 98 + 48  # drums + percussion
    url, path = lines[0].split("\t")
    assert url.startswith("https://raw.githubusercontent.com/sfzinstruments/virtuosity_drums/")
    assert path.startswith("drums/")
    with_piano = CliRunner().invoke(main, ["fetch-samples", "--list", "--piano"])
    assert len(with_piano.output.strip().splitlines()) == 98 + 48 + 390


def test_fetch_samples_downloads_into_the_given_folder(tmp_path):
    with mock.patch("music_decoder.cli.main.SampleStore") as store:
        store.return_value.ensure.return_value = 146
        store.return_value.root = tmp_path
        r = CliRunner().invoke(main, ["fetch-samples", "--samples-dir", str(tmp_path)])
    assert r.exit_code == 0
    store.assert_called_once_with(tmp_path)
    assert "146 samples" in r.output and "146 downloaded" in r.output


def test_verse_style_is_passed_in_the_config(tmp_path):
    with mock.patch("music_decoder.cli.main.productionize") as p:
        p.return_value = _production(tmp_path)
        r = CliRunner().invoke(
            main, ["productionize", "take", "--out", str(tmp_path), "--verse", "build"]
        )
    assert r.exit_code == 0, r.output
    assert p.call_args.kwargs["config"].arrange.verse_style == "build"
    r = CliRunner().invoke(main, ["productionize", "take", "--out", str(tmp_path), "--verse", "x"])
    assert r.exit_code != 0
