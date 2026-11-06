import pytest

from music_decoder.chords.templates import QUALITIES
from music_decoder.compose import voicings as compose_voicings
from music_decoder.compose.harmony import (
    _DEGREE_STEPS,
    CHORD_INTERVALS,
    ESSENTIAL_INTERVALS,
    chord_pcs,
    chord_scale_pcs,
    essential_pcs,
    leading_tone_pc,
    scale_pcs,
    tonic_pc,
)
from music_decoder.types import VALID_TONICS, ChordSymbol, Scale

C = Scale("C", "major")
A_MINOR = Scale("A", "minor")


def ch(label: str) -> ChordSymbol:
    return ChordSymbol.parse(label)


def test_scale_and_chord_pitch_classes():
    assert scale_pcs(C) == (0, 2, 4, 5, 7, 9, 11)
    assert scale_pcs(A_MINOR) == (9, 11, 0, 2, 4, 5, 7)
    assert chord_pcs(ch("G7")) == (7, 11, 2, 5)
    assert tonic_pc(A_MINOR) == 9
    assert leading_tone_pc(A_MINOR) == 8  # G#


@pytest.mark.parametrize(
    ("label", "expected"),
    [("C", {4}), ("Am", {0}), ("G7", {11, 5}), ("Bdim", {2, 5}), ("Dsus4", {7}), ("Caug", {4, 8})],
)
def test_essential_pcs(label, expected):
    assert essential_pcs(ch(label)) == frozenset(expected)


def test_chord_scale_is_the_scale_over_diatonic_chords():
    for label in ("C", "Dm7", "Em", "F", "G7", "Am", "Bdim"):
        assert chord_scale_pcs(C, ch(label)) == frozenset(scale_pcs(C))


def test_e7_in_a_minor_gives_harmonic_minor_and_keeps_the_tonic():
    assert chord_scale_pcs(A_MINOR, ch("E7")) == frozenset({9, 11, 0, 2, 4, 5, 8})


def test_secondary_dominant_keeps_its_target():
    pool = chord_scale_pcs(C, ch("D7"))  # D F# A C
    assert 6 in pool and 5 not in pool  # F -> F#
    assert 7 in pool  # G (where D7 resolves) is kept


@pytest.mark.parametrize(
    ("label", "added", "removed"),
    [
        ("C7", 10, 11),  # Bb replaces B
        ("Fm", 8, 9),  # Ab replaces A
        ("A", 1, 0),  # C# replaces C over A major
        ("A#", 10, 11),  # bVII: foreign root Bb replaces B
    ],
)
def test_borrowed_and_secondary_chords_in_c(label, added, removed):
    pool = chord_scale_pcs(C, ch(label))
    assert added in pool and removed not in pool
    assert len(pool) == 7


def test_csharp_dim_in_c_alters_the_tonic_degree():
    pool = chord_scale_pcs(C, ch("C#dim"))  # vii°/ii: C# E G
    assert 1 in pool and 0 not in pool


def test_bii_maj7_keeps_its_own_chord_tones():
    pool = chord_scale_pcs(C, ch("C#maj7"))  # C# F G# C
    assert {1, 5, 8, 0} <= pool
    assert 2 not in pool and 9 not in pool  # D and A are the lowered degrees


def test_b7_in_c_major_raises_d_and_f_and_keeps_g():
    pool = chord_scale_pcs(C, ch("B7"))  # B D# F# A, resolving to Em
    assert {3, 6} <= pool  # D# and F#
    assert 2 not in pool and 5 not in pool  # D and F replaced
    assert {7, 0} <= pool  # G (third of Em) and the tonic C kept


def test_f_sharp_7_in_c_major_raises_f_a_and_c_and_keeps_g():
    pool = chord_scale_pcs(C, ch("F#7"))  # F# A# C# E
    assert {6, 10, 1} <= pool
    assert not {5, 9, 0} & pool
    assert {7, 2, 4, 11} <= pool


@pytest.mark.parametrize("key", VALID_TONICS)
def test_secondary_dominants_alter_exactly_the_right_degrees_in_every_key(key):
    t = VALID_TONICS.index(key)
    scale = Scale(key, "major")
    cases = {  # root offset -> {raised/lowered pc offset: replaced pc offset}
        2: {6: 5},  # V7/V  (D7 in C): F -> F#
        9: {1: 0},  # V7/ii (A7 in C): C -> C#
        4: {8: 7},  # V7/vi (E7 in C): G -> G#
        11: {3: 2, 6: 5},  # V7/iii (B7 in C): D -> D#, F -> F#
        0: {10: 11},  # V7/IV (C7 in C): B -> Bb
    }
    for root_offset, swaps in cases.items():
        pool = chord_scale_pcs(scale, ch(VALID_TONICS[(t + root_offset) % 12] + "7"))
        expected = set(scale_pcs(scale))
        for raised, replaced in swaps.items():
            expected.discard((t + replaced) % 12)
            expected.add((t + raised) % 12)
        assert pool == frozenset(expected), (key, root_offset)


def test_ambiguous_chromatic_chord_uses_the_flat_side_spelling():
    # F# major in C: F#-A#-C# and Gb-Bb-Db alter 3 semitones each (a tie);
    # the documented tie-break spells it Gb-Bb-Db, replacing G, B and D.
    assert chord_scale_pcs(C, ch("F#")) == frozenset({0, 1, 4, 5, 6, 9, 10})


@pytest.mark.parametrize(
    "table",
    [CHORD_INTERVALS, ESSENTIAL_INTERVALS, _DEGREE_STEPS, compose_voicings._CHORD_PCS],
    ids=["CHORD_INTERVALS", "ESSENTIAL_INTERVALS", "_DEGREE_STEPS", "voicings._CHORD_PCS"],
)
def test_every_chord_quality_has_a_row_in_each_compose_table(table):
    # docs/architecture-technical.md "Add a new chord quality" lists these tables
    assert set(table) == set(QUALITIES)
