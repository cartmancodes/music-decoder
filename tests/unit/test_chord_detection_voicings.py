# tests/unit/test_chord_detection_voicings.py
import pytest

from music_decoder.chord_detection.voicings import VOICINGS, voicing_for


def test_c_major_open_position():
    # Standard C: x32010 (low-to-high: muted, 3, 2, 0, 1, 0)
    v = voicing_for("C", "maj")
    assert v == (-1, 3, 2, 0, 1, 0)


def test_a_minor_open_position():
    v = voicing_for("A", "min")
    assert v == (-1, 0, 2, 2, 1, 0)


def test_g_major_open_position():
    v = voicing_for("G", "maj")
    assert v == (3, 2, 0, 0, 0, 3)


def test_g7_open_position():
    # G7: 320001
    v = voicing_for("G", "7")
    assert v == (3, 2, 0, 0, 0, 1)


def test_cmaj7_open_position():
    # Cmaj7: x32000
    v = voicing_for("C", "maj7")
    assert v == (-1, 3, 2, 0, 0, 0)


def test_voicing_six_strings_each():
    for label, v in VOICINGS.items():
        assert len(v) == 6, f"{label} voicing has {len(v)} strings, expected 6"


def test_voicing_frets_in_range():
    for label, v in VOICINGS.items():
        for f in v:
            assert -1 <= f <= 22, f"{label}: fret {f} out of range"


def test_voicing_for_unknown_returns_fallback_barre():
    # Implementation choice: return None or raise. Spec says lookup table is
    # static; missing keys signal a gap in coverage.
    with pytest.raises(KeyError):
        voicing_for("C", "sus4")


def test_no_chord_voicing_is_all_muted():
    v = voicing_for("N", "")
    assert v == (-1, -1, -1, -1, -1, -1)


def test_all_48_chords_have_voicings():
    from music_decoder.chord_detection.templates import QUALITIES, ROOTS, chord_label
    missing = []
    for root in ROOTS:
        for quality in QUALITIES:
            label = chord_label(root, quality)
            if label not in VOICINGS:
                missing.append(label)
    assert missing == [], f"Voicings missing for: {missing}"
