"""Phrase model: tokens and conditioning streams."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from music_decoder.compose import phrase_tokens as pt
from music_decoder.compose.tune_corpus import read_song
from music_decoder.types import ChordSymbol, Note, Scale


@pytest.fixture
def mini() -> Path:
    return Path(__file__).parents[1] / "fixtures" / "compose" / "pop909_mini"


def _note(start: float, pitch: int) -> Note:
    return Note(start_s=start, end_s=start + 0.2, pitch=pitch, velocity=80, confidence=1.0)


def _c_major(n_chords: int = 4):
    return Scale.parse("C:major"), [ChordSymbol.parse(c) for c in ["C", "Am", "F", "G"][:n_chords]]


def test_anchor_is_the_tonic_at_or_below_the_middle():
    assert pt.anchor_pitch(0, 64) == 60
    assert pt.anchor_pitch(0, 60) == 60
    assert pt.anchor_pitch(9, 60) == 57


def test_piece_conditioning_marks_restatements():
    scale, prog = _c_major()
    cond = pt.piece_conditioning(scale=scale, progression=prog, bars_per_chord=2, tempo_bpm=100.0)
    assert set(cond) == set(pt.CONDITION_KEYS)
    assert all(len(v) == 8 * pt.SLOTS_PER_BAR for v in cond.values())
    per_bar = lambda key: cond[key][:: pt.SLOTS_PER_BAR].tolist()  # noqa: E731
    assert per_bar("restate") == [0, 0, 0, 0, 1, 1, 1, 1]
    assert per_bar("since") == [0, 1, 2, 3, 0, 1, 2, 3]
    assert per_bar("until") == [3, 2, 1, 0, 3, 2, 1, 0]
    assert cond["slot"][:32].tolist() == list(range(16)) * 2
    assert per_bar("root") == [0, 0, 9, 9, 5, 5, 7, 7]
    assert per_bar("next_root")[:2] == [0, 9]
    assert per_bar("next_quality")[-1] == pt.QUALITIES.index("none")
    assert set(cond["tempo"].tolist()) == {2}
    assert set(cond["mode"].tolist()) == {0}
    for key, size in pt.CONDITION_SIZES.items():
        assert 0 <= cond[key].min() and cond[key].max() < size


def test_chord_quality_maps_by_pitch_class_set():
    assert pt.chord_code(frozenset({9, 0, 4, 7}), 0) == (9, pt.QUALITIES.index("min7"))
    assert pt.chord_code(frozenset({0, 4, 8}), 0)[1] == pt.QUALITIES.index("other")
    assert pt.chord_code(frozenset(), 0)[1] == pt.QUALITIES.index("none")
    assert pt.chord_code(frozenset({2, 6, 9}), 2) == (0, pt.QUALITIES.index("maj"))


def test_encode_piece_round_trips_onsets():
    scale, prog = _c_major(1)
    enc = pt.encode_piece(
        [_note(0.0, 64), _note(0.25, 67)],
        scale=scale,
        progression=prog,
        bars_per_chord=1,
        tempo_bpm=120.0,
        middle=64,
        pitch_lo=-12,
        pitch_hi=17,
    )
    assert len(enc.tokens) == pt.SLOTS_PER_BAR
    assert enc.tokens[0] == 4 - (-12) + 1 and enc.tokens[2] == 7 - (-12) + 1
    assert np.count_nonzero(enc.tokens) == 2
    assert enc.out_of_range_bars == frozenset()


def test_out_of_range_note_is_reported():
    scale, prog = _c_major(2)
    enc = pt.encode_piece(
        [_note(0.0, 64), _note(2.0, 90)],
        scale=scale,
        progression=prog,
        bars_per_chord=1,
        tempo_bpm=120.0,
        middle=64,
        pitch_lo=-12,
        pitch_hi=17,
    )
    assert enc.out_of_range_bars == frozenset({1})
    assert np.count_nonzero(enc.tokens) == 1


def test_encode_song_reads_the_mini_corpus(mini):
    song = read_song(mini / "001")
    assert pt.phrase_starts(song) == [0, 4]
    enc = pt.encode_song(song, -12, 17)
    assert len(enc.tokens) % pt.SLOTS_PER_BAR == 0
    assert 0 < np.count_nonzero(enc.tokens) <= len(song.notes)
    assert set(enc.cond) == set(pt.CONDITION_KEYS)
    rel = pt.song_relative_pitches(song)
    assert len(rel) == len(song.notes)
    assert pt.eighth_only(enc, 0, 1) == bool((enc.tokens[: pt.SLOTS_PER_BAR][1::2] == 0).all())


def test_eighth_only_rejects_an_odd_slot_onset():
    scale, prog = _c_major(1)
    enc = pt.encode_piece(
        [_note(15 / 120.0, 64)],
        scale=scale,
        progression=prog,
        bars_per_chord=1,
        tempo_bpm=120.0,
        middle=64,
        pitch_lo=-12,
        pitch_hi=17,
    )
    assert enc.tokens[1] != 0
    assert not pt.eighth_only(enc, 0, 1)
    assert not pt.eighth_only(enc, 0, 2)  # runs past the end


def test_figure_hashes_ignore_transposition():
    ons, ps = [0, 2, 4, 8, 12], [60, 62, 64, 62, 60]
    base = pt.figure_hashes(ons, ps, 4)
    assert base.dtype == np.uint32 and len(base) == 2
    assert (pt.figure_hashes(ons, [p + 5 for p in ps], 4) == base).all()
    assert pt.figure_hashes([0, 2, 6, 8, 12], ps, 4)[0] != base[0]
    assert len(pt.figure_hashes(ons, ps, 6)) == 0


def test_next_chord_is_read_in_the_current_bar_key():
    from music_decoder.compose.tune_corpus import Chord, Song, SongNote

    song = Song(
        name="x",
        keys=((0, "major"), (7, "major")),
        tempo=100,
        phrases=(("A", 0, 2),),
        notes=(SongNote(0, 4, 64), SongNote(16, 4, 67)),
        chords=(Chord(0, 16, frozenset({0, 4, 7})), Chord(16, 32, frozenset({7, 11, 2}))),
        bar_keys=((0, "major"), (7, "major")),
    )
    enc = pt.encode_song(song, -12, 17)
    assert enc.cond["next_root"][0] == 7  # G seen from C
    assert enc.cond["root"][16] == 0  # G in G
