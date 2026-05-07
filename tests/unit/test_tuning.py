import pytest

from music_decoder.tabs.tuning import (
    PRESETS,
    get_preset,
)


def test_eadgbe_preset_pitches():
    t = get_preset("EADGBE")
    assert t.name == "EADGBE"
    assert t.open_pitches == (40, 45, 50, 55, 59, 64)


def test_drop_d_preset_pitches():
    t = get_preset("Drop_D")
    assert t.open_pitches == (38, 45, 50, 55, 59, 64)


def test_eb_preset_pitches():
    t = get_preset("Eb")
    assert t.open_pitches == (39, 44, 49, 54, 58, 63)


def test_d_standard_pitches():
    t = get_preset("D_standard")
    assert t.open_pitches == (38, 43, 48, 53, 57, 62)


def test_drop_c_pitches():
    t = get_preset("Drop_C")
    assert t.open_pitches == (36, 43, 48, 53, 57, 62)


def test_dadgad_pitches():
    t = get_preset("DADGAD")
    assert t.open_pitches == (38, 45, 50, 55, 57, 62)


def test_unknown_preset_raises():
    with pytest.raises(KeyError):
        get_preset("not_a_tuning")


def test_tuning_is_frozen():
    t = get_preset("EADGBE")
    with pytest.raises(AttributeError):
        t.name = "X"  # type: ignore[misc]


def test_presets_listed():
    assert set(PRESETS.keys()) == {
        "EADGBE", "Drop_D", "Eb", "D_standard", "Drop_C", "DADGAD"
    }
