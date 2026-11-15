from collections.abc import Sequence

import pytest

from music_decoder.compose import fingerstyle, metrics
from music_decoder.compose.fingerstyle import (
    arrange_fingerstyle,
    body_bar,
    final_bar,
    final_bar_pitches,
    final_positions,
    hand_shape,
)
from music_decoder.tabs.tuning import STANDARD_EADGBE
from music_decoder.types import ChordSymbol, Note, TabPosition, VoicedChord


def P(string: int, fret: int) -> TabPosition:
    return TabPosition(string=string, fret=fret)


def _shape(label: str, frets: tuple[int, ...]) -> VoicedChord:
    return VoicedChord(
        chord=ChordSymbol.parse(label),
        positions=tuple(TabPosition(string=s, fret=f) for s, f in enumerate(frets)),
    )


def note(start_s: float, end_s: float, pitch: int) -> Note:
    return Note(start_s=start_s, end_s=end_s, pitch=pitch, velocity=80, confidence=1.0)


C_HAND = hand_shape(_shape("C", (-1, 3, 2, 0, 1, 0)), STANDARD_EADGBE)
D_HAND = hand_shape(_shape("D", (-1, -1, 0, 2, 3, 2)), STANDARD_EADGBE)


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


# ---- one body bar (120 BPM: beat_s = 0.5, the bar starts at 0) ------------


def _quarters(*pitches: int) -> Sequence[Note]:
    return [note(k * 0.5, k * 0.5 + 0.475, p) for k, p in enumerate(pitches)]


def test_thumb_alternates_on_every_beat():
    bar = body_bar(
        C_HAND, [note(0.0, 1.9, 64)], bar_start_s=0.0, beat_s=0.5, essential=frozenset({4})
    )
    thumb = [(t.note.start_s, t.position) for t in bar.accompaniment_tab if t.position.string < 3]
    assert thumb == [(0.0, P(1, 3)), (0.5, P(2, 2)), (1.0, P(1, 3)), (1.5, P(2, 2))]


def test_fills_fall_on_the_and_of_two_and_four_under_the_melody():
    bar = body_bar(
        C_HAND, _quarters(64, 65, 67, 65), bar_start_s=0.0, beat_s=0.5, essential=frozenset({4})
    )
    fills = [(t.note.start_s, t.position) for t in bar.accompaniment_tab if t.position.string >= 3]
    assert fills == [(0.75, P(4, 1)), (1.75, P(4, 1))]  # C4, nearest under F4
    assert [t.position for t in bar.melody_tab] == [P(5, 0), P(5, 1), P(5, 3), P(5, 1)]
    assert bar.identity_met


def test_missing_third_is_filled_over_a_held_b_string_note():
    melody = [note(0.0, 0.95, 62), note(1.0, 1.95, 61)]  # D4 and C#4, both on the B string
    bar = body_bar(D_HAND, melody, bar_start_s=0.0, beat_s=0.5, essential=frozenset({6}))
    fills = [(t.note.start_s, t.position) for t in bar.accompaniment_tab if t.position.string >= 4]
    assert fills == [(0.75, P(5, 2))] and bar.identity_met


def test_identity_moves_one_melody_note_to_free_the_top_string():
    melody = [note(0.0, 0.95, 64), note(1.0, 1.95, 64)]  # E4 held on the e string all bar
    bar = body_bar(D_HAND, melody, bar_start_s=0.0, beat_s=0.5, essential=frozenset({6}))
    assert [t.position for t in bar.melody_tab] == [P(4, 5), P(5, 0)]
    assert (0.75, P(5, 2)) in [(t.note.start_s, t.position) for t in bar.accompaniment_tab]
    assert bar.identity_met


def test_every_moment_of_a_bar_is_playable():
    bar = body_bar(
        C_HAND, _quarters(64, 65, 67, 69), bar_start_s=0.0, beat_s=0.5, essential=frozenset({4})
    )
    assert metrics.playable_moment_ratio([*bar.melody_tab, *bar.accompaniment_tab]) == 1.0
    assert (
        metrics.melody_onset_on_top_ratio(
            [t.note for t in bar.melody_tab], [t.note for t in bar.accompaniment_tab]
        )
        == 1.0
    )


def test_an_unreachable_melody_pitch_is_rejected():
    with pytest.raises(ValueError, match="not reachable"):
        body_bar(
            D_HAND, [note(0.0, 0.95, 59)], bar_start_s=0.0, beat_s=0.5, essential=frozenset({6})
        )


def test_with_no_melody_sounding_a_fill_takes_the_lowest_pitch():
    bar = body_bar(C_HAND, [], bar_start_s=0.0, beat_s=0.5, essential=frozenset({4}))
    fills = [(t.note.start_s, t.position) for t in bar.accompaniment_tab if t.position.string >= 3]
    assert fills == [(0.75, P(3, 0)), (1.75, P(3, 0))]  # G3


# ---- the final bar and the whole part --------------------------------------


def test_d_ending_is_revoiced_with_its_third_under_the_final_note():
    part, fallback = final_bar(
        D_HAND, note(6.0, 7.9, 62), bar_start_s=6.0, bar_s=2.0, essential=frozenset({6})
    )
    assert not fallback and part.melody_tab[0].position == P(4, 3)
    acc = part.accompaniment_tab
    assert [t.position for t in acc] == [P(1, 5), P(2, 4), P(3, 2)]  # x-5-4-2-(3)-x
    assert [t.note.start_s for t in acc] == pytest.approx([6.0, 6.01, 6.02])
    assert all(t.note.end_s == pytest.approx(8.0) for t in acc)


def test_final_bar_falls_back_and_says_so(monkeypatch):
    monkeypatch.setattr(fingerstyle, "ranked_shapes", lambda chord, tuning: ())
    part, fallback = final_bar(
        D_HAND, note(6.0, 7.9, 62), bar_start_s=6.0, bar_s=2.0, essential=frozenset({6})
    )
    assert fallback and [t.note.pitch for t in part.accompaniment_tab] == [50, 57]


def test_final_pitches_extend_to_an_unreachable_cadence():
    assert final_bar_pitches(D_HAND, 2) == D_HAND.allowed  # D is reachable
    assert final_bar_pitches(D_HAND, 11) == D_HAND.allowed | {71}  # B only at B4
    assert final_positions(D_HAND, 71) == (P(5, 7), P(4, 12))


def test_arrange_fingerstyle_joins_body_and_final_bars():
    voicings = [C_HAND.voicing, D_HAND.voicing]
    melody = [*_quarters(64, 65, 67, 65), note(2.0, 3.9, 62)]
    part = arrange_fingerstyle(
        voicings, melody, [C_HAND, D_HAND], bars_per_chord=1, tempo_bpm=120.0
    )
    assert [t.note for t in part.melody_tab] == melody
    assert part.identity_misses == () and not part.final_fallback
    assert metrics.playable_moment_ratio([*part.melody_tab, *part.accompaniment_tab]) == 1.0
