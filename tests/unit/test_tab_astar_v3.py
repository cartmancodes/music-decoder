from music_decoder.tabs.astar import _representative
from music_decoder.types import TabPosition


def test_representative_ignores_open_strings() -> None:
    chord = (TabPosition(0, 0), TabPosition(1, 3), TabPosition(2, 2))
    assert _representative(chord) == TabPosition(2, 2)


def test_representative_all_open() -> None:
    chord = (TabPosition(0, 0), TabPosition(1, 0))
    assert _representative(chord).fret == 0
