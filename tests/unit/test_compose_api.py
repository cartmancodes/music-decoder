from pathlib import Path
from unittest import mock

import pytest

from music_decoder.compose.api import arrange, compose
from music_decoder.errors import InvalidProgressionError, InvalidScaleError
from music_decoder.types import ChordSymbol, Scale


def test_compose_returns_paths_and_data(tmp_path):
    with mock.patch("music_decoder.compose.api.render_wav") as render:
        render.side_effect = lambda midi_path, out_path: out_path.write_bytes(b"RIFFWAV")
        result = compose(
            scale=Scale("C", "major"),
            progression=[ChordSymbol.parse(c) for c in ("Cmaj7", "Am7", "Dm7", "G7")],
            tempo_bpm=120.0,
            bars_per_chord=1,
            style="strum",
            seed=7,
            out_dir=tmp_path,
        )
    assert result.midi_path.exists()
    assert result.wav_path.exists()
    assert result.ascii_tab.strip() != ""
    assert len(result.melody_notes) > 0
    assert len(result.chord_voicings) == 4
    assert result.metadata["seed"] == 7


def test_compose_invalid_progression_raises():
    with pytest.raises(InvalidProgressionError):
        compose(
            scale=Scale("C", "major"),
            progression=[],  # empty
            out_dir=Path("/tmp"),
        )


def test_compose_invalid_scale_raises():
    # non-Literal mode is only constructible by going around the parser:
    bad = Scale(tonic="C", mode="dorian")  # type: ignore[arg-type]
    with pytest.raises(InvalidScaleError):
        compose(scale=bad, progression=[ChordSymbol.parse("C")], out_dir=Path("/tmp"))


# ---- arrange(): the filesystem-free core of compose() ----------------------

_PROG = [ChordSymbol.parse(c) for c in ("Cmaj7", "Am7", "Dm7", "G7")]


def test_arrange_is_deterministic_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    a = arrange(Scale("C", "major"), _PROG, tempo_bpm=120.0, style="strum", seed=3)
    b = arrange(Scale("C", "major"), _PROG, tempo_bpm=120.0, style="strum", seed=3)
    assert a == b
    assert list(tmp_path.iterdir()) == []
    assert len(a.voicings) == 4 and a.accompaniment
    assert [tn.note for tn in a.melody_tab] == list(a.melody)


def test_compose_returns_the_arrangement(tmp_path):
    with mock.patch("music_decoder.compose.api.render_wav"):
        result = compose(scale=Scale("C", "major"), progression=_PROG, seed=1, out_dir=tmp_path)
    arr = arrange(Scale("C", "major"), _PROG, seed=1)
    assert result.melody_notes == arr.melody
    assert result.chord_voicings == arr.voicings
