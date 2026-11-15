from music_decoder.compose.fingerstyle import hand_shape
from music_decoder.tabs.tuning import STANDARD_EADGBE
from music_decoder.types import ChordSymbol, TabPosition, VoicedChord


def P(string: int, fret: int) -> TabPosition:
    return TabPosition(string=string, fret=fret)


def _shape(label: str, frets: tuple[int, ...]) -> VoicedChord:
    return VoicedChord(
        chord=ChordSymbol.parse(label),
        positions=tuple(TabPosition(string=s, fret=f) for s, f in enumerate(frets)),
    )


def test_open_c_reaches_g3_to_a4_on_the_top_three_strings():
    h = hand_shape(_shape("C", (-1, 3, 2, 0, 1, 0)), STANDARD_EADGBE)
    assert (h.root, h.alternate) == (P(1, 3), P(2, 2))
    assert h.melody_strings == (3, 4, 5) and h.window == (1, 5) and h.barre_fret is None
    assert h.fill_strings == (3, 4, 5)
    assert h.allowed == frozenset(range(55, 70))
    assert h.positions(64) == (P(5, 0), P(4, 5))
    assert h.positions(60) == (P(4, 1), P(3, 5))
    assert h.positions(70) == ()


def test_open_d_keeps_the_melody_a_minor_third_over_the_thumb():
    h = hand_shape(_shape("D", (-1, -1, 0, 2, 3, 2)), STANDARD_EADGBE)
    assert (h.root, h.alternate) == (P(2, 0), P(3, 2)) and h.melody_strings == (4, 5)
    assert h.allowed == frozenset(range(60, 70))  # B3 (59) is under A3 + 3


def test_barre_shape_has_no_open_melody_notes():
    h = hand_shape(_shape("F", (1, 3, 3, 2, 1, 1)), STANDARD_EADGBE)
    assert h.barre_fret == 1 and h.window == (1, 4)
    assert h.allowed == frozenset({*range(56, 64), *range(65, 69)})  # no E4, no G3


def test_open_thumb_widens_the_shape_to_a_four_fret_window():
    h = hand_shape(_shape("A", (-1, 0, 0, 2, 1, 0)), STANDARD_EADGBE)
    assert h.window == (1, 4) and h.allowed == frozenset(range(55, 69))
    assert h.positions(59) == (P(4, 0), P(3, 4))
    h0 = hand_shape(_shape("Em", (0, 0, 0, 0, 0, 0)), STANDARD_EADGBE)
    assert h0.window == (1, 4) and h0.allowed == frozenset(range(55, 69))


def test_hand_shapes_compare_and_hash_by_their_shape():
    a = hand_shape(_shape("C", (-1, 3, 2, 0, 1, 0)), STANDARD_EADGBE)
    b = hand_shape(_shape("C", (-1, 3, 2, 0, 1, 0)), STANDARD_EADGBE)
    assert a == b and hash(a) == hash(b)
