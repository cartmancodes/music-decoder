import re
from pathlib import Path
from unittest import mock

import pytest

from music_decoder.compose import metrics
from music_decoder.compose.api import (
    COMPOSE_TAB_WEIGHTS,
    COMPOSER_VERSION,
    _melody_floor,
    _melody_tab,
    arrange,
    compose,
)
from music_decoder.compose.melody import MAX_INTERVAL, generate_melody
from music_decoder.compose.tabtext import render_melody_tab
from music_decoder.compose.voicings import shape_faults, voicings_for
from music_decoder.errors import InvalidProgressionError, InvalidScaleError
from music_decoder.tabs import _DEFAULT_WEIGHTS
from music_decoder.tabs.tuning import PRESETS, STANDARD_EADGBE, get_preset
from music_decoder.types import VALID_TONICS, ChordSymbol, Note, Scale


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


def _chords(*labels):
    return [ChordSymbol.parse(c) for c in labels]


def _decode(block):
    """(string, fret) per melody note from a rendered melody tab block."""
    out = []
    for system in block.split("\n\n"):
        rows = system.splitlines()[::-1]
        hits = []
        for s, row in enumerate(rows):
            body = row.split("|", 1)[1]
            hits.extend((m.start(), s, int(m.group())) for m in re.finditer(r"\d+", body))
        out.extend((s, f) for _, s, f in sorted(hits))
    return out


def test_arrange_is_deterministic_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    a = arrange(Scale("C", "major"), _PROG, tempo_bpm=120.0, style="strum", seed=3)
    b = arrange(Scale("C", "major"), _PROG, tempo_bpm=120.0, style="strum", seed=3)
    assert a == b
    assert list(tmp_path.iterdir()) == []
    assert len(a.voicings) == 4 and a.accompaniment


def test_arrange_reads_no_configuration():
    with (
        mock.patch("music_decoder.config.hyperparameters.load_hyperparameters") as load,
        mock.patch("music_decoder.tabs.assign_tabs") as configurable,
    ):
        arrange(Scale("C", "major"), _PROG, seed=3)
    load.assert_not_called()
    configurable.assert_not_called()


def test_arrange_ignores_the_analysis_hyperparameters_file(tmp_path, monkeypatch):
    before = arrange(Scale("G", "major"), _PROG, seed=5)
    other = tmp_path / "hp.yaml"
    other.write_text("id: test\ntab_assignment:\n  weights: {w_move: 9.0}\n  max_fret: 5\n")
    monkeypatch.setenv("MUSIC_DECODER_HYPERPARAMETERS", str(other))
    assert arrange(Scale("G", "major"), _PROG, seed=5) == before


def test_pinned_tab_weights_match_the_analysis_defaults():
    assert COMPOSE_TAB_WEIGHTS == _DEFAULT_WEIGHTS


def test_compose_returns_the_arrangement_and_version(tmp_path):
    with mock.patch("music_decoder.compose.api.render_wav"):
        result = compose(scale=Scale("C", "major"), progression=_PROG, seed=1, out_dir=tmp_path)
    arr = arrange(Scale("C", "major"), _PROG, seed=1)
    assert result.melody_notes == arr.melody
    assert result.chord_voicings == arr.voicings
    assert result.metadata["composer_version"] == COMPOSER_VERSION == 2


def test_arrange_tabs_every_melody_note_in_a_playable_register():
    arr = arrange(Scale("G", "major"), _PROG, bars_per_chord=2, seed=9)
    assert [tn.note for tn in arr.melody_tab] == list(arr.melody)
    for tn in arr.melody_tab:
        assert STANDARD_EADGBE.open_pitches[tn.position.string] + tn.position.fret == tn.note.pitch
    assert max(tn.position.fret for tn in arr.melody_tab) <= 15


def test_melody_tab_falls_back_when_astar_drops_a_note():
    melody = [
        Note(start_s=0.0, end_s=0.4, pitch=64, velocity=80, confidence=1.0),
        Note(start_s=0.5, end_s=0.9, pitch=67, velocity=80, confidence=1.0),
    ]
    empty = mock.Mock(tabbed_notes=[])
    with mock.patch("music_decoder.compose.api.assign_tab", return_value=empty):
        tab = _melody_tab(melody, STANDARD_EADGBE)
    assert [(t.position.string, t.position.fret) for t in tab] == [(5, 0), (5, 3)]


@pytest.mark.parametrize("bars", [0, -1])
def test_bars_per_chord_below_one_raises(bars):
    with pytest.raises(InvalidProgressionError):
        arrange(Scale("C", "major"), _PROG, bars_per_chord=bars)


@pytest.mark.parametrize("tuning_name", ["Drop_D", "DADGAD", "D_standard"])
def test_arrange_in_other_tunings_tabs_every_note(tuning_name):
    tuning = PRESETS[tuning_name]
    arr = arrange(Scale("D", "major"), _chords("D", "G", "A", "D"), tuning=tuning, seed=2)
    assert len(arr.melody_tab) == len(arr.melody)
    for tn in arr.melody_tab:
        assert tuning.open_pitches[tn.position.string] + tn.position.fret == tn.note.pitch


def test_melody_floor_sits_a_minor_third_above_the_defining_tones():
    d = voicings_for(ChordSymbol.parse("D"), STANDARD_EADGBE)[0]  # D3 A3 D4 F#4
    assert _melody_floor(d, STANDARD_EADGBE) == 66 + 3
    g7 = voicings_for(ChordSymbol.parse("G7"), STANDARD_EADGBE)[0]  # G2 B2 D3 G3 B3 F4
    assert _melody_floor(g7, STANDARD_EADGBE) == 65 + 3  # the seventh F4 is the highest


def test_d_major_cadence_has_its_third_under_the_melody():
    # review reproduction: D G A D, seed 9
    arr = arrange(Scale("D", "major"), _chords("D", "G", "A", "D"), tempo_bpm=120.0, seed=9)
    final_melody = [n for n in arr.melody if n.start_s >= 6.0]
    final_acc = [n for n in arr.accompaniment if n.start_s >= 6.0 - 1e-9]
    assert 6 in {n.pitch % 12 for n in final_acc}  # F# is heard
    assert max(n.pitch for n in final_acc) < min(n.pitch for n in final_melody)
    assert metrics.final_resolution(arr.melody, Scale("D", "major"), _chords("D", "G", "A", "D"))


@pytest.mark.parametrize(
    ("key", "labels", "bars", "seed"),
    [
        ("C", ("C", "G", "Am", "F"), 1, 0),  # review reproduction
        ("A#", ("A#", "Gm", "D#", "F", "Cm", "F", "A#", "A#"), 2, 1004),  # 16 bars
    ],
)
def test_rendered_tab_shows_every_melody_position(tmp_path, key, labels, bars, seed):
    with mock.patch("music_decoder.compose.api.render_wav"):
        result = compose(
            scale=Scale(key, "major"),
            progression=_chords(*labels),
            bars_per_chord=bars,
            tempo_bpm=120.0,
            seed=seed,
            out_dir=tmp_path,
        )
    arr = arrange(
        Scale(key, "major"), _chords(*labels), bars_per_chord=bars, tempo_bpm=120.0, seed=seed
    )
    melody_block = result.ascii_tab.split("\n\nChords\n")[0].removeprefix("Melody\n")
    expected = [(tn.position.string, tn.position.fret) for tn in arr.melody_tab]
    assert _decode(melody_block) == expected
    assert melody_block == render_melody_tab(
        arr.melody_tab, n_bars=len(labels) * bars, tempo_bpm=120.0, tuning=STANDARD_EADGBE
    )


@pytest.mark.parametrize(
    ("key", "mode", "labels", "seed"),
    [
        ("C", "major", ("C", "Am", "F", "G7"), 16),  # review r2: 65 -> 79 jump
        ("C", "major", ("C", "G", "Am", "G7"), 16),
        ("A", "major", ("A", "Dsus4"), 0),  # review r2: no cadence target
        ("A", "minor", ("Am", "Dsus4"), 0),
        ("A", "major", ("Dsus4",), 0),
    ],
)
def test_review_reproductions_compose_within_the_invariants(key, mode, labels, seed):
    progression = _chords(*labels)
    arr = arrange(Scale(key, mode), progression, tempo_bpm=120.0, seed=seed)
    assert metrics.max_interval(arr.melody) <= MAX_INTERVAL
    assert metrics.final_resolution(arr.melody, Scale(key, mode), progression)
    assert all(
        metrics.essential_tone_coverage(
            arr.melody, arr.accompaniment, arr.voicings, 1, 120.0, STANDARD_EADGBE
        )
    )


@pytest.mark.parametrize(
    "labels",
    [("C", "Am", "F", "G7"), ("C", "G", "Am", "G7"), ("C", "F", "Csus4"), ("C", "F", "Dm7")],
)
def test_floor_coupled_sweep_keeps_the_octave_limit(labels):
    # the generator with the floors arrange() would pass, without the A* cost
    progression = _chords(*labels)
    voicings = [voicings_for(c, STANDARD_EADGBE)[0] for c in progression]
    floors = [_melody_floor(v, STANDARD_EADGBE) for v in voicings]
    for seed in range(150):
        notes = generate_melody(
            scale=Scale("C", "major"), progression=progression, seed=seed, chord_floors=floors
        )
        assert metrics.max_interval(notes) <= MAX_INTERVAL, seed
        assert metrics.final_resolution(notes, Scale("C", "major"), progression), seed


_QUALITY_SUFFIXES = ("", "m", "7", "maj7", "m7", "dim", "sus4", "aug")


@pytest.mark.parametrize(("key", "mode", "home"), [("C", "major", "C"), ("A", "minor", "Am")])
def test_every_chord_quality_on_every_root_composes(key, mode, home):
    scale = Scale(key, mode)
    for root in VALID_TONICS:
        for suffix in _QUALITY_SUFFIXES:
            progression = _chords(home, root + suffix)
            arr = arrange(scale, progression, tempo_bpm=120.0, seed=7)
            label = root + suffix
            assert metrics.max_interval(arr.melody) <= MAX_INTERVAL, label
            assert metrics.final_resolution(arr.melody, scale, progression), label
            coverage = metrics.essential_tone_coverage(
                arr.melody, arr.accompaniment, arr.voicings, 1, 120.0, STANDARD_EADGBE
            )
            assert all(coverage), label


@pytest.mark.parametrize("style", ["arpeggio", "strum", "fingerstyle"])
@pytest.mark.parametrize(
    ("key", "mode", "labels", "seed"),
    [
        ("C", "major", ("C", "Am", "F", "G7"), 5),  # review r3: G7 ending lost its F
        ("A", "major", ("A", "Dsus4"), 1),  # review r3: Dsus4 ending lost its G
        ("A", "minor", ("Am", "Dsus4"), 1),
        ("C", "major", ("C", "Fm", "C", "G7"), 1006),  # from the edge matrix
        ("A", "major", ("A", "Dsus4"), 1002),
    ],
)
def test_sustained_cadence_keeps_the_chord_identity(key, mode, labels, seed, style):
    progression = _chords(*labels)
    arr = arrange(Scale(key, mode), progression, tempo_bpm=120.0, style=style, seed=seed)
    assert metrics.cadence_identity(arr.melody, arr.accompaniment, arr.voicings, STANDARD_EADGBE)
    assert metrics.final_chord_rings(arr.accompaniment, len(labels), 120.0)
    assert metrics.max_interval(arr.melody) <= MAX_INTERVAL


@pytest.mark.parametrize("style", ["arpeggio", "strum", "fingerstyle"])
@pytest.mark.parametrize(
    ("tuning_name", "labels"),
    [
        ("Drop_C", ("C", "G", "Am", "G7", "Gmaj7", "C")),
        ("Drop_D", ("C", "G7", "C")),
        ("DADGAD", ("D", "A#", "D#dim", "A#sus4", "Bmaj7", "D")),
        ("Eb", ("C", "C7", "F", "B7", "C")),
    ],
)
def test_searched_voicings_compose_with_their_identity_heard(tuning_name, labels, style):
    tuning = get_preset(tuning_name)
    arr = arrange(Scale(labels[0], "major"), _chords(*labels), style=style, tuning=tuning, seed=3)
    assert all(
        shape_faults([p.fret for p in v.positions], tuning, v.chord) == () for v in arr.voicings
    )
    assert all(
        metrics.essential_tone_coverage(
            arr.melody, arr.accompaniment, arr.voicings, 1, 100.0, tuning
        )
    )
    assert metrics.cadence_identity(arr.melody, arr.accompaniment, arr.voicings, tuning)
