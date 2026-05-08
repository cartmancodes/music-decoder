# tests/unit/test_chord_detection_templates.py
import numpy as np
import pytest

from music_decoder.chords.templates import (
    NO_CHORD,
    QUALITIES,
    ROOTS,
    all_templates,
    chord_label,
    label_index,
    label_to_root_quality,
    template_for,
)


def test_roots_are_twelve_chromatic():
    assert ROOTS == ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def test_qualities_match_spec():
    # Phase B-3 expanded the vocabulary from 4 to 8 qualities.
    assert QUALITIES == (
        "maj",
        "min",
        "7",
        "maj7",
        "min7",
        "dim",
        "sus4",
        "aug",
    )


def test_template_for_c_major_is_root_third_fifth():
    t = template_for("C", "maj")
    assert t.shape == (12,)
    assert t[0] == 1.0  # C
    assert t[4] == 1.0  # E
    assert t[7] == 1.0  # G
    assert t.sum() == 3.0


def test_template_for_a_minor_has_minor_third():
    t = template_for("A", "min")
    assert t[9] == 1.0  # A
    assert t[0] == 1.0  # C (minor third)
    assert t[4] == 1.0  # E (perfect fifth)
    assert t.sum() == 3.0


def test_template_for_g7_has_minor_seventh():
    t = template_for("G", "7")
    assert t[7] == 1.0  # G
    assert t[11] == 1.0  # B (major third)
    assert t[2] == 1.0  # D (perfect fifth)
    assert t[5] == 1.0  # F (minor seventh)
    assert t.sum() == 4.0


def test_template_for_cmaj7_has_major_seventh():
    t = template_for("C", "maj7")
    assert t[0] == 1.0  # C
    assert t[4] == 1.0  # E
    assert t[7] == 1.0  # G
    assert t[11] == 1.0  # B (major seventh)


# Phase B-3 templates
def test_template_for_am7_has_minor_third_minor_seventh():
    t = template_for("A", "min7")
    assert t[9] == 1.0  # A
    assert t[0] == 1.0  # C (minor third)
    assert t[4] == 1.0  # E (perfect fifth)
    assert t[7] == 1.0  # G (minor seventh)
    assert t.sum() == 4.0


def test_template_for_bdim_has_minor_third_diminished_fifth():
    t = template_for("B", "dim")
    assert t[11] == 1.0  # B
    assert t[2] == 1.0  # D (minor third)
    assert t[5] == 1.0  # F (diminished fifth)
    assert t.sum() == 3.0


def test_template_for_csus4_has_perfect_fourth_fifth():
    t = template_for("C", "sus4")
    assert t[0] == 1.0  # C
    assert t[5] == 1.0  # F (perfect fourth)
    assert t[7] == 1.0  # G (perfect fifth)
    assert t.sum() == 3.0


def test_template_for_caug_has_major_third_augmented_fifth():
    t = template_for("C", "aug")
    assert t[0] == 1.0  # C
    assert t[4] == 1.0  # E (major third)
    assert t[8] == 1.0  # G# (augmented fifth)
    assert t.sum() == 3.0


def test_no_chord_template_is_uniform():
    t = template_for(NO_CHORD, "")
    assert t.shape == (12,)
    assert np.allclose(t, np.full(12, 1.0 / 12.0))


def test_all_templates_returns_97_rows():
    """12 roots * 8 qualities + N = 97 rows after Phase B-3."""
    arr = all_templates()
    assert arr.shape == (97, 12)


def test_chord_label_round_trips():
    assert chord_label("C", "maj") == "C"
    assert chord_label("A", "min") == "Am"
    assert chord_label("G", "7") == "G7"
    assert chord_label("D", "maj7") == "Dmaj7"
    assert chord_label("A", "min7") == "Am7"
    assert chord_label("B", "dim") == "Bdim"
    assert chord_label("C", "sus4") == "Csus4"
    assert chord_label("C", "aug") == "Caug"
    assert chord_label(NO_CHORD, "") == "N"


def test_label_to_root_quality_round_trip():
    labels = (
        "C",
        "Am",
        "G7",
        "Dmaj7",
        "C#m",
        "F#7",
        "G#maj7",
        "N",
        # Phase B-3 additions
        "Am7",
        "Dm7",
        "Bdim",
        "F#dim",
        "Csus4",
        "Gsus4",
        "Caug",
        "Faug",
    )
    for label in labels:
        root, quality = label_to_root_quality(label)
        assert chord_label(root, quality) == label


def test_label_index_is_stable():
    # Index 0 is C maj; the no-chord index lives at 12*8 = 96.
    assert label_index("C", "maj") == 0
    assert label_index(NO_CHORD, "") == 96
    # First minor7 row: C is root index 0, min7 is quality index 4.
    assert label_index("C", "min7") == 4


def test_unknown_quality_raises():
    with pytest.raises(KeyError):
        template_for("C", "made_up_quality")
