import statistics
import time

import pytest

from music_decoder.compose import metrics
from music_decoder.compose.api import arrange
from music_decoder.compose.fingerstyle import arrange_fingerstyle, final_bar_pitches, hand_shape
from music_decoder.compose.harmony import essential_pcs
from music_decoder.compose.melody import cadence_pc, generate_melody
from music_decoder.compose.shapes import (
    MAX_CANDIDATES,
    bar_cost,
    choose_bar_shapes,
    move_cost,
    repair_identity,
    shape_candidates,
)
from music_decoder.compose.voicings import shape_faults, voicings_for
from music_decoder.tabs.tuning import PRESETS, STANDARD_EADGBE
from music_decoder.types import VALID_TONICS, ChordSymbol, Scale, Tuning


def _frets(voicing):
    return tuple(p.fret for p in voicing.positions)


def test_candidates_start_with_the_v4_shape_and_are_faithful():
    for label in ("C", "Dm7", "G7", "F#m", "A#maj7", "Bdim", "Caug", "Esus4"):
        chord = ChordSymbol.parse(label)
        cands = shape_candidates(chord, STANDARD_EADGBE)
        assert cands[0] == voicings_for(chord, STANDARD_EADGBE)[0]
        assert 1 <= len(cands) <= MAX_CANDIDATES
        assert len({_frets(v) for v in cands}) == len(cands)
        assert all(hand_shape(v, STANDARD_EADGBE).allowed for v in cands)
        assert all(not shape_faults(_frets(v), STANDARD_EADGBE, chord) for v in cands)


@pytest.mark.parametrize("tuning", list(PRESETS.values()), ids=list(PRESETS))
def test_every_quality_and_root_gets_a_shape_in_every_tuning(tuning):
    for root in VALID_TONICS:
        for suffix in ("", "m", "7", "maj7", "m7", "dim", "sus4", "aug"):
            prog = [ChordSymbol.parse(root + suffix)]
            assert len(choose_bar_shapes(prog, bars_per_chord=2, tuning=tuning)) == 2


def test_a_held_chord_keeps_its_shape():
    shapes = choose_bar_shapes([ChordSymbol.parse("Dm7")], bars_per_chord=4, tuning=STANDARD_EADGBE)
    assert len({_frets(v) for v in shapes}) == 1


def test_seventh_chords_prefer_shapes_that_keep_their_seventh_off_the_top_string():
    prog = [ChordSymbol.parse(c) for c in ("Cmaj7", "Fmaj7", "Dm7", "G7")]
    chosen = choose_bar_shapes(prog, bars_per_chord=1, tuning=STANDARD_EADGBE)
    v4 = [voicings_for(c, STANDARD_EADGBE)[0] for c in prog]

    def risk(vs):
        return sum(
            bar_cost(hand_shape(v, STANDARD_EADGBE), essential_pcs(v.chord), 0)
            + 0.15 * len(hand_shape(v, STANDARD_EADGBE).allowed)
            for v in vs
        )

    assert risk(chosen) < risk(v4)  # identity-risk part of the cost drops


def test_choose_is_deterministic():
    prog = [ChordSymbol.parse(c) for c in ("G", "D7", "Em7", "C")]
    assert choose_bar_shapes(prog, bars_per_chord=1, tuning=STANDARD_EADGBE) == choose_bar_shapes(
        prog, bars_per_chord=1, tuning=STANDARD_EADGBE
    )


def test_move_cost_is_zero_for_the_same_shape_and_grows_with_the_window_shift():
    c = hand_shape(voicings_for(ChordSymbol.parse("C"), STANDARD_EADGBE)[0], STANDARD_EADGBE)
    f = hand_shape(voicings_for(ChordSymbol.parse("F"), STANDARD_EADGBE)[0], STANDARD_EADGBE)
    assert move_cost(c, c) == 0
    assert move_cost(c, f) == pytest.approx(0.25 * abs(c.window[0] - f.window[0]))


def test_a_tuning_where_the_v4_shape_reaches_no_melody_still_gets_shapes():
    tuning = Tuning("cluster", (37, 40, 42, 44, 46, 48))
    chord = ChordSymbol.parse("C#")
    assert not hand_shape(voicings_for(chord, tuning)[0], tuning).allowed
    assert len(shape_candidates(chord, tuning)) >= 1
    assert len(choose_bar_shapes([chord], bars_per_chord=2, tuning=tuning)) == 2


# ---- Task 2: per-bar shapes in the fingerstyle arrangement ----------------

REPAIR_CASES = (
    ("C", "major", ("Cmaj7", "Fmaj7", "Dm7", "G7")),
    ("G", "major", ("G", "D7", "G")),
    ("D", "minor", ("Dm", "A7", "Dm")),
)


def test_repair_never_breaks_playability():
    for key, mode, labels in REPAIR_CASES:
        for seed in range(10):
            prog = [ChordSymbol.parse(c) for c in labels]
            a = arrange(Scale(key, mode), prog, tempo_bpm=120.0, style="fingerstyle", seed=seed)
            part = [*a.melody_tab, *a.accompaniment_tab]
            assert metrics.playable_moment_ratio(part) == 1.0
            assert metrics.melody_onset_on_top_ratio(a.melody, a.accompaniment) == 1.0
            assert len(a.bar_voicings) == len(prog)


def test_sevenths_keep_their_seventh_more_often_than_v4():
    prog = [ChordSymbol.parse(c) for c in ("Cmaj7", "Fmaj7", "Dm7", "G7")]
    hits = total = 0
    for seed in range(20):
        a = arrange(
            Scale("C", "major"),
            prog,
            bars_per_chord=2,
            tempo_bpm=70.0,
            style="fingerstyle",
            seed=seed,
        )
        ident = metrics.bar_identity(a.accompaniment, prog, 2, 70.0)
        hits += sum(ident)
        total += len(ident)
    assert hits / total >= 0.95  # v4 sample 13 met 3 of 8 bars


def test_a_long_piece_arranges_quickly():
    prog = [ChordSymbol.parse(c) for c in ("C", "Am7", "Dm7", "G7") * 4]
    t = time.perf_counter()
    arrange(Scale("C", "major"), prog, tempo_bpm=100.0, style="fingerstyle", seed=1)
    assert time.perf_counter() - t < 1.0


def test_voicings_stay_one_per_chord_and_are_the_first_bars_shape():
    prog = [ChordSymbol.parse(c) for c in ("Cmaj7", "Fmaj7", "Dm7", "G7")]
    a = arrange(Scale("C", "major"), prog, bars_per_chord=2, tempo_bpm=70.0, seed=3)
    assert len(a.bar_voicings) == 8
    assert a.voicings == a.bar_voicings[::2]
    assert [v.chord for v in a.bar_voicings] == [c for c in prog for _ in range(2)]
    strum = arrange(Scale("C", "major"), prog, style="strum", seed=3)
    assert strum.bar_voicings == ()


def _unrepaired(scale, prog, bars, tempo, seed):
    """Steps 1-5 of the fingerstyle arrangement: the part before repair."""
    shapes = choose_bar_shapes(prog, bars_per_chord=bars, tuning=STANDARD_EADGBE)
    hands = [hand_shape(v, STANDARD_EADGBE) for v in shapes]
    allowed = [h.allowed for h in hands]
    allowed[-1] = final_bar_pitches(hands[-1], cadence_pc(scale, prog[-1]), essential_pcs(prog[-1]))
    center = statistics.median_low([p for pitches in allowed for p in sorted(pitches)])
    melody = generate_melody(
        scale=scale,
        progression=prog,
        bars_per_chord=bars,
        tempo_bpm=tempo,
        seed=seed,
        tessitura_center=center,
        allowed_pitches=allowed,
        final_whole_note=True,
    )
    part = arrange_fingerstyle(shapes, melody, hands, bars_per_chord=1, tempo_bpm=tempo)
    return part, shapes


def test_repair_keeps_the_melody_and_only_clears_misses():
    prog = [ChordSymbol.parse(c) for c in ("Cmaj7", "Am7", "Dm7", "G7")]
    repaired = 0
    for bars in (1, 2):
        for seed in range(1000, 1005):
            part, shapes = _unrepaired(Scale("C", "major"), prog, bars, 120.0, seed)
            fixed, fixed_shapes = repair_identity(
                part, shapes, prog, bars_per_chord=bars, tempo_bpm=120.0, tuning=STANDARD_EADGBE
            )
            notes = [(t.note.start_s, t.note.end_s, t.note.pitch) for t in fixed.melody_tab]
            assert notes == [(t.note.start_s, t.note.end_s, t.note.pitch) for t in part.melody_tab]
            assert set(fixed.identity_misses) <= set(part.identity_misses)
            heard = metrics.bar_identity(
                [t.note for t in fixed.accompaniment_tab], prog, bars, 120.0
            )
            assert fixed.identity_misses == tuple(b for b, ok in enumerate(heard) if not ok)
            assert [v.chord for v in fixed_shapes] == [v.chord for v in shapes]
            for bar, (old, new) in enumerate(zip(shapes, fixed_shapes, strict=True)):
                assert old == new or (
                    bar in part.identity_misses and bar not in fixed.identity_misses
                )
            whole = [*fixed.melody_tab, *fixed.accompaniment_tab]
            assert metrics.playable_moment_ratio(whole) == 1.0
            assert (
                metrics.melody_onset_on_top_ratio(
                    [t.note for t in fixed.melody_tab], [t.note for t in fixed.accompaniment_tab]
                )
                == 1.0
            )
            repaired += len(part.identity_misses) - len(fixed.identity_misses)
    assert repaired >= 3  # seed 1001 alone repairs bars 0 and 1 (one bar per chord)


def test_shape_moves_counts_window_changes_between_bars():
    c, g = (voicings_for(ChordSymbol.parse(x), STANDARD_EADGBE)[0] for x in ("C", "G"))
    barre_f = shape_candidates(ChordSymbol.parse("F"), STANDARD_EADGBE)
    assert metrics.shape_moves([c], STANDARD_EADGBE) is None
    assert metrics.shape_moves([c, c, c], STANDARD_EADGBE) == 0.0
    windows = [hand_shape(v, STANDARD_EADGBE).window[0] for v in (c, *barre_f)]
    far = next(v for v, w in zip((c, *barre_f), windows, strict=True) if w != windows[0])
    assert metrics.shape_moves([c, far, far], STANDARD_EADGBE) == 0.5
    assert metrics.shape_moves([c, g], STANDARD_EADGBE) == float(
        hand_shape(c, STANDARD_EADGBE).window[0] != hand_shape(g, STANDARD_EADGBE).window[0]
    )
