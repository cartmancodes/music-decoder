"""Tests for the shared JAMS chord-label parser."""
from __future__ import annotations

from music_decoder.chords.labels import parse_jams_chord_label


def test_no_chord_label():
    assert parse_jams_chord_label("N") == ("N", "")


def test_basic_qualities():
    assert parse_jams_chord_label("C:maj") == ("C", "maj")
    assert parse_jams_chord_label("A:min") == ("A", "min")
    assert parse_jams_chord_label("G:7") == ("G", "7")
    assert parse_jams_chord_label("D:maj7") == ("D", "maj7")


def test_sharp_and_flat_roots():
    assert parse_jams_chord_label("C#:maj") == ("C#", "maj")
    assert parse_jams_chord_label("Bb:7") == ("Bb", "7")
    assert parse_jams_chord_label("F#:min") == ("F#", "min")


def test_min7_first_class_after_b3():
    # Phase B-3 promoted min7 from a downgrade rule to a first-class quality.
    assert parse_jams_chord_label("D:min7") == ("D", "min7")


def test_maj9_downgrades_to_maj7():
    assert parse_jams_chord_label("C:maj9") == ("C", "maj7")


def test_min9_downgrades_to_min7():
    assert parse_jams_chord_label("C:min9") == ("C", "min7")


def test_sus4_first_class_after_b3():
    assert parse_jams_chord_label("F:sus4") == ("F", "sus4")


def test_sus2_downgrades_to_sus4():
    # Both are suspensions; sus4 is the first-class shape.
    assert parse_jams_chord_label("C:sus2") == ("C", "sus4")


def test_diminished_first_class_after_b3():
    assert parse_jams_chord_label("B:dim") == ("B", "dim")
    # dim7 + half-dim downgrade to dim (same triad, different 7th).
    assert parse_jams_chord_label("B:dim7") == ("B", "dim")
    assert parse_jams_chord_label("B:hdim7") == ("B", "dim")


def test_aug_first_class_after_b3():
    assert parse_jams_chord_label("C:aug") == ("C", "aug")
    assert parse_jams_chord_label("C:+") == ("C", "aug")


def test_x_label_returns_none():
    assert parse_jams_chord_label("X") is None


def test_extension_specifier_stripped():
    # JAMS allows "C:maj(9,11)" and similar — base quality should still parse.
    assert parse_jams_chord_label("C:maj(9)") == ("C", "maj")


def test_slash_chord_bass_ignored():
    # G major over B should parse as G major; the bass is ignored for our v1.
    assert parse_jams_chord_label("G:maj/B") == ("G", "maj")


def test_bare_root_treated_as_major():
    assert parse_jams_chord_label("C") == ("C", "maj")


def test_invalid_root_returns_none():
    assert parse_jams_chord_label("H:maj") is None  # H is not a valid root


def test_empty_label_returns_none():
    assert parse_jams_chord_label("") is None


def test_non_string_returns_none():
    assert parse_jams_chord_label(None) is None  # type: ignore[arg-type]
