# tests/unit/test_chord_detection_voicings.py
import pytest

from music_decoder.chords.voicings import VOICINGS, voicing_for
from music_decoder.compose.voicings import fingers_needed, shape_faults
from music_decoder.tabs.tuning import STANDARD_EADGBE
from music_decoder.types import ChordSymbol


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


def test_voicing_for_unknown_quality_raises():
    # Implementation choice: missing keys signal a gap in coverage.
    # Phase B-3 added sus4 to the vocabulary, so we test with a genuinely
    # unsupported quality.
    with pytest.raises(KeyError):
        voicing_for("C", "alt")  # not in our 8-quality vocabulary


def test_no_chord_voicing_is_all_muted():
    v = voicing_for("N", "")
    assert v == (-1, -1, -1, -1, -1, -1)


def test_all_chords_have_voicings():
    """Phase B-3 grew the vocabulary from 48 to 96 chords (12 roots x 8 qualities)."""
    from music_decoder.chords.templates import QUALITIES, ROOTS, chord_label

    missing = []
    for root in ROOTS:
        for quality in QUALITIES:
            label = chord_label(root, quality)
            if label not in VOICINGS:
                missing.append(label)
    assert missing == [], f"Voicings missing for: {missing}"


def test_every_library_shape_is_faithful_and_playable():
    bad = {}
    for label, v in VOICINGS.items():
        if label == "N":
            continue
        fretted = [f for f in v if f > 0]
        span = max(fretted) - min(fretted) if fretted else 0
        faults = shape_faults(v, STANDARD_EADGBE, ChordSymbol.parse(label))
        if faults or span > 3 or fingers_needed(v) > 4:
            bad[label] = (faults, span, fingers_needed(v))
    assert bad == {}


@pytest.mark.parametrize(
    ("label", "frets"),
    [
        ("C#dim", (-1, 4, 5, 6, 5, -1)),  # C#3 G3 C#4 E4
        ("Ddim", (-1, -1, 0, 1, 3, 1)),  # D3 G#3 D4 F4
        ("D#dim", (-1, -1, 1, 2, 4, 2)),  # D#3 A3 D#4 F#4
        ("Fmaj7", (-1, -1, 3, 2, 1, 0)),  # F3 A3 C4 E4
        ("G#aug", (-1, -1, 6, 5, 5, 4)),  # G#3 C4 E4 G#4
        ("C#7", (-1, 4, 6, 4, 6, 4)),  # C#3 G#3 B3 F4 G#4 (v2 x43421 needs 5 fingers)
        ("D#7", (-1, 6, 8, 6, 8, 6)),  # D#3 A#3 C#4 G4 A#4 (v2 x65643 needs 5 fingers)
    ],
)
def test_repaired_shapes(label, frets):
    assert VOICINGS[label] == frets
