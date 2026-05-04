# tests/unit/test_tab_candidates.py

from music_decoder.tab_assignment.candidates import (
    chord_combinations,
    note_candidates,
)
from music_decoder.tab_assignment.tuning import get_preset


def test_e4_in_eadgbe_has_two_candidates_below_fret_22():
    cands = note_candidates(pitch=64, tuning=get_preset("EADGBE"), max_fret=22)
    pairs = sorted([(c.string, c.fret) for c in cands])
    assert (5, 0) in pairs and (4, 5) in pairs and (3, 9) in pairs


def test_below_range_returns_empty():
    cands = note_candidates(pitch=30, tuning=get_preset("EADGBE"), max_fret=22)
    assert cands == []


def test_chord_combinations_filters_string_collisions():
    pitches = [40, 40]   # two E2s; cannot both play on string 0
    combos = chord_combinations(pitches, tuning=get_preset("EADGBE"), max_fret=22)
    for combo in combos:
        strings = [p.string for p in combo]
        assert len(strings) == len(set(strings))   # no duplicates


def test_chord_combinations_respects_span_5_frets():
    pitches = [60, 67]  # C4 and G4
    combos = chord_combinations(pitches, tuning=get_preset("EADGBE"), max_fret=22)
    for combo in combos:
        frets = [p.fret for p in combo]
        assert max(frets) - min(frets) <= 5
