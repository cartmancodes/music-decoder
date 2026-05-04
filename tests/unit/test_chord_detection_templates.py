# tests/unit/test_chord_detection_templates.py
import numpy as np
import pytest

from music_decoder.chord_detection.templates import (
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
    assert ROOTS == ("C", "C#", "D", "D#", "E", "F",
                     "F#", "G", "G#", "A", "A#", "B")


def test_qualities_match_spec():
    assert QUALITIES == ("maj", "min", "7", "maj7")


def test_template_for_c_major_is_root_third_fifth():
    t = template_for("C", "maj")
    assert t.shape == (12,)
    assert t[0] == 1.0   # C
    assert t[4] == 1.0   # E
    assert t[7] == 1.0   # G
    assert t.sum() == 3.0


def test_template_for_a_minor_has_minor_third():
    t = template_for("A", "min")
    assert t[9] == 1.0    # A
    assert t[0] == 1.0    # C (minor third)
    assert t[4] == 1.0    # E (perfect fifth)
    assert t.sum() == 3.0


def test_template_for_g7_has_minor_seventh():
    t = template_for("G", "7")
    assert t[7] == 1.0    # G
    assert t[11] == 1.0   # B (major third)
    assert t[2] == 1.0    # D (perfect fifth)
    assert t[5] == 1.0    # F (minor seventh)
    assert t.sum() == 4.0


def test_template_for_cmaj7_has_major_seventh():
    t = template_for("C", "maj7")
    assert t[0] == 1.0    # C
    assert t[4] == 1.0    # E
    assert t[7] == 1.0    # G
    assert t[11] == 1.0   # B (major seventh)


def test_no_chord_template_is_uniform():
    t = template_for(NO_CHORD, "")
    assert t.shape == (12,)
    assert np.allclose(t, np.full(12, 1.0 / 12.0))


def test_all_templates_returns_49_rows():
    arr = all_templates()
    assert arr.shape == (49, 12)


def test_chord_label_round_trips():
    assert chord_label("C", "maj") == "C"
    assert chord_label("A", "min") == "Am"
    assert chord_label("G", "7") == "G7"
    assert chord_label("D", "maj7") == "Dmaj7"
    assert chord_label(NO_CHORD, "") == "N"


def test_label_to_root_quality_round_trip():
    for label in ("C", "Am", "G7", "Dmaj7", "C#m", "F#7", "G#maj7", "N"):
        root, quality = label_to_root_quality(label)
        assert chord_label(root, quality) == label


def test_label_index_is_stable():
    # Index 0 must be C major; index 48 must be N.
    assert label_index("C", "maj") == 0
    assert label_index(NO_CHORD, "") == 48


def test_unknown_quality_raises():
    with pytest.raises(KeyError):
        template_for("C", "sus4")
