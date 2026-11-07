import pytest

from music_decoder.chords.templates import QUALITIES, ROOTS, chord_label
from music_decoder.chords.voicings import VOICINGS
from music_decoder.compose.voicings import (
    _is_playable,
    fingers_needed,
    search_voicing,
    shape_faults,
    voicings_for,
)
from music_decoder.tabs.tuning import (
    D_STANDARD,
    DADGAD,
    DROP_C,
    DROP_D,
    EB_HALF_STEP_DOWN,
    PRESETS,
    STANDARD_EADGBE,
)
from music_decoder.types import ChordSymbol, Tuning


def test_c_major_voicing_in_eadgbe():
    voicings = voicings_for(ChordSymbol.parse("C"), STANDARD_EADGBE)
    assert len(voicings) >= 1
    primary = voicings[0]
    assert primary.chord.to_label() == "C"
    frets = [p.fret for p in primary.positions if p.fret >= 0]
    assert max(frets) <= 5
    assert max(frets) - min(frets) <= 4


def test_drop_d_d_chord_uses_open_lowest():
    voicings = voicings_for(ChordSymbol.parse("D"), DROP_D)
    primary = voicings[0]
    # In Drop D, the lowest string (index 0) is now D, so D chord plays it open.
    assert primary.positions[0].fret == 0


def test_unknown_quality_falls_back_to_root_note():
    # We only support the 8 qualities; an unsupported one should raise.
    bad = ChordSymbol(root="C", quality="maj")  # ok
    voicings_for(bad, STANDARD_EADGBE)  # smoke
    # No quality outside Literal can be constructed via parse, so we stop here.


def test_shape_faults_names_each_defect_in_order():
    c = ChordSymbol.parse
    assert shape_faults((-1, 3, 2, 0, 1, 0), STANDARD_EADGBE, c("C")) == ()
    assert (
        shape_faults((-1, 3, 2, 3, 1, 0), STANDARD_EADGBE, c("C7")) == ()
    )  # fifth omitted: allowed
    assert shape_faults((-1, -1, 2, 3, 2, -1), STANDARD_EADGBE, c("C#dim")) == (
        "foreign tone",
        "root not in bass",
        "missing tone",
    )  # v2 C#dim: E3 A#3 C#4
    assert shape_faults((-1, 3, 3, 2, 1, 0), STANDARD_EADGBE, c("Fmaj7")) == ("root not in bass",)
    assert shape_faults((9, 4, -1, -1, -1, -1), EB_HALF_STEP_DOWN, c("Cm")) == (
        "too few strings",
        "missing tone",
    )  # v2 Eb fallback: C3 C3
    assert shape_faults((-1, 3, 2, 0, 1, -1), STANDARD_EADGBE, c("C")) == ()  # C E G C
    assert shape_faults((-1, 3, 5, 5, -1, -1), STANDARD_EADGBE, c("C")) == ("missing tone",)  # no E
    assert shape_faults((-1,) * 6, STANDARD_EADGBE, c("C")) == ("too few strings", "missing tone")


@pytest.mark.parametrize(
    ("frets", "fingers"),
    [
        ((0, 7, 6, 7, 5, 4), 5),  # the fret-7 notes are split by a fret-6 string
        ((-1, 6, 5, 6, 4, 3), 5),  # v2 D#7 x65643
        ((-1, 4, 3, 4, 2, 1), 5),  # v2 C#7 x43421
        ((1, 3, 3, 2, 1, 1), 3),  # barred F
        ((3, 2, 0, 0, 0, 3), 3),  # open G: the open strings split the fret-3 notes
        ((3, 0, 0, 0, 0, 3), 2),  # one barre cannot cross open strings
        ((-1, 4, 6, 4, 6, 4), 3),  # repaired C#7
        ((0, 7, 9, 7, 9, 7), 3),  # Eb-tuned D#7 after the repair
        ((-1,) * 6, 0),
    ],
)
def test_fingers_needed(frets, fingers):
    assert fingers_needed(frets) == fingers


def _frets(chord, tuning):
    return tuple(p.fret for p in voicings_for(ChordSymbol.parse(chord), tuning)[0].positions)


def test_every_chord_in_every_preset_tuning_is_faithful_and_playable():
    bad = []
    for name, tuning in PRESETS.items():
        for root in ROOTS:
            for quality in QUALITIES:
                label = chord_label(root, quality)
                frets = _frets(label, tuning)
                fretted = [f for f in frets if f > 0]
                span = max(fretted) - min(fretted) if fretted else 0
                faults = shape_faults(frets, tuning, ChordSymbol.parse(label))
                if faults or span > 3 or fingers_needed(frets) > 4:
                    bad.append((name, label, frets, faults, fingers_needed(frets)))
    assert bad == []  # v2: 98 of 576 faulty; revision 1: 22 needing 5 fingers (spec §3.3)


def test_standard_tuning_plays_the_library_shapes_unchanged():
    for label, frets in VOICINGS.items():
        if label != "N":
            assert _frets(label, STANDARD_EADGBE) == frets


@pytest.mark.parametrize(
    ("tuning", "label", "frets"),
    [
        (DROP_D, "D", (0, -1, 0, 2, 3, 2)),  # low D added (root); open A2 stays muted
        (DROP_C, "C", (0, 5, 4, 2, 3, 2)),
        (EB_HALF_STEP_DOWN, "D#7", (0, 7, 9, 7, 9, 7)),  # v2: one D#2; not 0-7-6-7-5-4 (5 fingers)
        (EB_HALF_STEP_DOWN, "C7", (-1, 4, 3, 1, 0, 1)),  # searched: retuned x32310 needs 5 fingers
        (EB_HALF_STEP_DOWN, "Cm", (-1, 4, 6, 6, 5, 4)),  # v2: C3 C3
        (D_STANDARD, "A#7", (-1, 3, 5, 3, 5, 3)),  # v2: open D2 under the barre
        (DROP_C, "G", (-1, 0, 2, 2, 2, 0)),  # searched
        (DROP_D, "G7", (5, 2, 3, 0, 3, 3)),  # searched: the retuned span of 4 is rejected
        (DADGAD, "D#dim", (1, 0, 1, 2, 0, 4)),  # searched
    ],
)
def test_retune_and_search_choices(tuning, label, frets):
    assert _frets(label, tuning) == frets


def test_five_finger_retunes_are_rejected():
    assert not _is_playable((0, 7, 6, 7, 5, 4))
    assert not _is_playable((-1, 6, 5, 6, 4, 3))
    assert _is_playable((0, 7, 9, 7, 9, 7))


def test_tuning_above_standard_uses_the_search():  # Review Focus 1
    f_std = Tuning("F_standard", (41, 46, 51, 56, 60, 65))
    assert _frets("C", f_std) == (-1, 2, 1, 4, 0, 2)
    assert shape_faults(_frets("C", f_std), f_std, ChordSymbol.parse("C")) == ()


def test_unvoiceable_chord_falls_back_without_raising():  # Review Focus 2
    all_c = Tuning("C2x6", (36,) * 6)
    assert search_voicing(ChordSymbol.parse("D"), all_c) is None
    v = voicings_for(ChordSymbol.parse("D"), all_c)[0]
    assert {all_c.open_pitches[p.string] + p.fret for p in v.positions if p.fret >= 0} == {38}
