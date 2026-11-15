import pytest

from music_decoder.compose.fingerstyle import hand_shape
from music_decoder.compose.harmony import essential_pcs
from music_decoder.compose.shapes import (
    MAX_CANDIDATES,
    bar_cost,
    choose_bar_shapes,
    move_cost,
    shape_candidates,
)
from music_decoder.compose.voicings import shape_faults, voicings_for
from music_decoder.tabs.tuning import PRESETS, STANDARD_EADGBE
from music_decoder.types import VALID_TONICS, ChordSymbol, Tuning


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
