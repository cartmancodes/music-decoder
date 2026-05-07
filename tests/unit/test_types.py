import pytest
from music_decoder.types import Scale, ChordSymbol


def test_scale_parse_major():
    s = Scale.parse("C:major")
    assert s.tonic == "C"
    assert s.mode == "major"


def test_scale_parse_minor():
    s = Scale.parse("A:minor")
    assert s.tonic == "A"
    assert s.mode == "minor"


def test_scale_parse_invalid_tonic_raises():
    with pytest.raises(ValueError):
        Scale.parse("H:major")


def test_scale_parse_invalid_mode_raises():
    with pytest.raises(ValueError):
        Scale.parse("C:dorian")


def test_chord_symbol_parse_major():
    c = ChordSymbol.parse("C")
    assert c.root == "C"
    assert c.quality == "maj"


def test_chord_symbol_parse_min7():
    c = ChordSymbol.parse("Am7")
    assert c.root == "A"
    assert c.quality == "min7"


def test_chord_symbol_parse_maj7():
    c = ChordSymbol.parse("Cmaj7")
    assert c.root == "C"
    assert c.quality == "maj7"


def test_chord_symbol_parse_seventh():
    c = ChordSymbol.parse("G7")
    assert c.root == "G"
    assert c.quality == "7"


def test_chord_symbol_to_label_roundtrip():
    for label in ["C", "Am", "G7", "Cmaj7", "Dm7", "F#dim", "Bsus4", "Eaug"]:
        assert ChordSymbol.parse(label).to_label() == label
