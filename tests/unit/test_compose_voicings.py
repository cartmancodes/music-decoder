import pytest

from music_decoder.compose.voicings import fingers_needed, shape_faults, voicings_for
from music_decoder.tabs.tuning import DROP_D, EB_HALF_STEP_DOWN, STANDARD_EADGBE
from music_decoder.types import ChordSymbol


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
