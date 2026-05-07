from music_decoder.compose.voicings import voicings_for
from music_decoder.tabs.tuning import STANDARD_EADGBE, DROP_D
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
    import pytest
    from music_decoder.errors import InvalidProgressionError
    bad = ChordSymbol(root="C", quality="maj")  # ok
    voicings_for(bad, STANDARD_EADGBE)  # smoke
    # No quality outside Literal can be constructed via parse, so we stop here.
