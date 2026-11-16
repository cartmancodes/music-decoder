"""v6 task 1: melody features on the eighth grid and the band distance."""

from __future__ import annotations

import pytest

from music_decoder.compose.tune_metrics import (
    Band,
    GridNote,
    band_distance,
    grid_features,
    tune_features,
)
from music_decoder.types import ChordSymbol, Note, Scale

C_MAJOR = [frozenset({0, 4, 7})] * 64  # eight bars of C


def _notes(pairs):
    return [GridNote(slot=s, pitch=p) for s, p in pairs]


def _features(notes, *, n_bars=8, chord_tones=C_MAJOR):
    return grid_features(
        notes,
        n_bars=n_bars,
        tonic_pc=0,
        mode="major",
        chord_tones=chord_tones,
        seconds_per_bar=2.0,
    )


def test_interval_shares():
    f = _features(_notes([(0, 60), (2, 60), (4, 62), (6, 65), (8, 72)]))
    assert (f["repeat"], f["step"], f["third"], f["leap"]) == (0.25, 0.25, 0.25, 0.25)
    assert f["notes_per_second"] == 5 / 16.0 and f["range"] == 12.0


def test_rhythm_features():
    bars = {0: (0, 2, 4), 1: (0, 3), 2: (0, 2, 4), 3: (1, 5), 4: (0, 2, 4)}
    f = _features(_notes([(b * 8 + s, 60) for b, slots in bars.items() for s in slots]))
    assert f["rhythm_variety"] == 3 / 5
    assert f["rhythm_lag2"] == 2 / 3 and f["rhythm_lag4"] == 1.0
    assert f["late_entry"] == 4 / 8  # bar 3 and the three empty bars
    assert f["offbeat"] == 3 / 13


def test_figure_recurrence_counts_repeated_three_move_figures():
    up = [60, 62, 64, 65]
    f = _features(_notes(list(enumerate(up + up + [60, 67, 59]))))
    # (2, 2, 1) and (2, 1, -5) each occur twice among eight figures
    assert f["figure_recurrence"] == 4 / 8


def test_figure_recurrence_is_zero_with_fewer_than_four_notes():
    assert _features(_notes([(0, 60), (2, 62), (4, 64)]))["figure_recurrence"] == 0.0


def test_chord_tone_strong_and_pentatonic():
    f = _features(_notes([(0, 60), (4, 62), (8, 65), (9, 71)]))
    assert f["chord_tone_strong"] == 1 / 3 and f["pentatonic"] == 0.5


def test_chord_tone_strong_is_omitted_without_strong_notes():
    assert "chord_tone_strong" not in _features(_notes([(1, 60), (3, 62)]))


def test_minor_pentatonic_set():
    f = grid_features(
        _notes([(0, 57), (2, 60), (4, 62), (6, 64)]),
        n_bars=8,
        tonic_pc=9,
        mode="minor",
        chord_tones=C_MAJOR,
        seconds_per_bar=2.0,
    )
    assert f["pentatonic"] == 1.0  # A, C, D, E sit at 0, 3, 5, 7 above the tonic


def test_sixteen_bars_average_two_windows():
    f = _features(
        _notes([(0, 60), (2, 60), (64, 60), (66, 62)]), n_bars=16, chord_tones=C_MAJOR * 2
    )
    assert f["repeat"] == 0.5  # 1.0 in the first window, 0.0 in the second


def test_short_piece_is_one_window_and_trailing_bars_are_ignored():
    short = _features(_notes([(0, 60), (2, 62)]), n_bars=4, chord_tones=C_MAJOR[:32])
    assert short["notes_per_second"] == 2 / 8.0 and short["late_entry"] == 3 / 4
    # 10 bars: only bars 0-7 count, so the note in bar 9 is ignored
    long = _features(
        _notes([(0, 60), (2, 62), (72, 90)]), n_bars=10, chord_tones=C_MAJOR + C_MAJOR[:16]
    )
    assert long["range"] == 2.0


def test_tune_features_matches_grid_features():
    scale, prog = Scale.parse("C:major"), [ChordSymbol.parse("C"), ChordSymbol.parse("G")]
    e = 60.0 / 120 / 2
    melody = [
        Note(start_s=k * e, end_s=k * e + 0.2, pitch=p, velocity=80, confidence=1.0)
        for k, p in [(0, 60), (3, 62), (32, 67)]
    ]
    f = tune_features(melody, scale, prog, bars_per_chord=4, tempo_bpm=120.0)
    assert f["notes_per_second"] == 3 / 16.0 and f["chord_tone_strong"] == 1.0  # C over C, G over G


def test_band_distance():
    bands = {
        "repeat": Band(0.1, 0.2, 0.3),
        "range": Band(12, 14, 16),
        "offbeat": Band(0.4, 0.4, 0.4),
    }
    assert band_distance({"repeat": 0.2, "range": 14.0}, bands) == 0.0
    f = {"repeat": 0.5, "range": 10.0, "offbeat": 0.5, "leap": 0.9}
    assert band_distance(f, bands) == pytest.approx(1.0 + 0.5 + 2.0)
