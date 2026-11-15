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
from music_decoder.compose.harmony import essential_pcs
from music_decoder.compose.melody import cadence_pc
from music_decoder.compose.voicings import voicings_for
from music_decoder.tabs.tuning import STANDARD_EADGBE
from music_decoder.types import ChordSymbol, Note, Scale, TabPosition, VoicedChord


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


# Identity, step 4: the alternate bass takes a missing tone (ruling R3).

A7_HAND = hand_shape(_shape("A7", (-1, 0, 2, 0, 2, 0)), STANDARD_EADGBE)


def _thumb(bar) -> list[tuple[float, TabPosition, int]]:
    return [
        (t.note.start_s, t.position, t.note.pitch)
        for t in bar.accompaniment_tab
        if round(t.note.start_s / 0.25) in (0, 2, 4, 6) and t.position.string <= 3
    ]


def test_d_over_a_held_top_string_puts_f_sharp_in_the_alternate_bass():
    # A4 sits on the e string all bar and is reachable nowhere else, so open
    # D's only F# (e string, fret 2) is never free: steps 1-3 all fail.
    bar = body_bar(
        D_HAND, [note(0.0, 1.9, 69)], bar_start_s=0.0, beat_s=0.5, essential=frozenset({6})
    )
    assert bar.identity_met
    assert _thumb(bar) == [
        (0.0, P(2, 0), 50),
        (0.5, P(2, 4), 54),
        (1.0, P(2, 0), 50),
        (1.5, P(2, 4), 54),
    ]
    subs = [t for t in bar.accompaniment_tab if t.position == P(2, 4)]
    assert [(t.note.end_s, t.note.velocity) for t in subs] == [
        (pytest.approx(0.975), 72),
        (pytest.approx(1.975), 72),
    ]  # the alternate bass's length and velocity


def test_a7_gets_c_sharp_on_the_a_string_when_the_b_string_is_held():
    # D4 is reachable only on the B string, where A7's C# sits.
    bar = body_bar(
        A7_HAND,
        [note(0.0, 1.9, 62)],
        bar_start_s=0.0,
        beat_s=0.5,
        essential=essential_pcs(ChordSymbol.parse("A7")),
    )
    assert bar.identity_met
    alternates = [(s, p, pitch) for s, p, pitch in _thumb(bar) if s in (0.5, 1.5)]
    assert alternates == [(0.5, P(1, 4), 49), (1.5, P(1, 4), 49)]  # C#3 for the fifth E3


@pytest.mark.parametrize(
    ("hand", "melody", "essential"),
    [
        (D_HAND, [note(0.0, 1.9, 69)], frozenset({6})),
        (A7_HAND, [note(0.0, 1.9, 62)], essential_pcs(ChordSymbol.parse("A7"))),
    ],
)
def test_the_substitute_keeps_the_root_on_one_and_three_and_stays_above_it(hand, melody, essential):
    bar = body_bar(hand, melody, bar_start_s=0.0, beat_s=0.5, essential=essential)
    root = STANDARD_EADGBE.open_pitches[hand.root.string] + hand.root.fret
    on_one_and_three = [
        (t.position, t.note.end_s) for t in bar.accompaniment_tab if t.note.start_s in (0.0, 1.0)
    ]
    assert on_one_and_three == [
        (hand.root, pytest.approx(0.475)),
        (hand.root, pytest.approx(1.475)),
    ]
    assert min(t.note.pitch for t in bar.accompaniment_tab) == root
    assert all(t.note.pitch > root for t in bar.accompaniment_tab if t.note.start_s in (0.5, 1.5))
    part = [*bar.melody_tab, *bar.accompaniment_tab]
    assert metrics.playable_moment_ratio(part) == 1.0


def test_the_substitute_never_drops_a_tone_the_alternate_bass_supplies():
    # In 3-2-x-x-x-1 the alternate bass B2 is G7's only B, and A4 holds the
    # e string (G7's only F). Swapping both alternates for F3 would lose the
    # B, so only beat 2 takes the F.
    g7 = hand_shape(_shape("G7", (3, 2, -1, -1, -1, 1)), STANDARD_EADGBE)
    bar = body_bar(
        g7, [note(0.0, 1.9, 69)], bar_start_s=0.0, beat_s=0.5, essential=frozenset({11, 5})
    )
    assert bar.identity_met
    assert _thumb(bar) == [
        (0.0, P(0, 3), 43),
        (0.5, P(1, 8), 53),
        (1.0, P(0, 3), 43),
        (1.5, P(1, 2), 47),
    ]


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
    d_essential = frozenset({6})
    assert final_bar_pitches(D_HAND, 2, d_essential) == D_HAND.allowed  # D is reachable
    assert final_bar_pitches(D_HAND, 11, d_essential) == D_HAND.allowed | {71}  # B only at B4
    assert final_positions(D_HAND, 71) == (P(5, 7), P(4, 12))


def test_final_pitches_prefer_a_cadence_pitch_that_can_be_voiced():
    chord = ChordSymbol.parse("A#maj7")
    hand = hand_shape(voicings_for(chord, STANDARD_EADGBE)[0], STANDARD_EADGBE)
    essential = essential_pcs(chord)
    pitches = final_bar_pitches(hand, 10, essential)
    assert {p for p in pitches if p % 12 == 10} == {70}  # A#3 (58) has no valid voicing
    assert pitches - {70} == {p for p in hand.allowed if p % 12 != 10}
    _, fallback = final_bar(
        hand, note(0.0, 1.9, 70), bar_start_s=0.0, bar_s=2.0, essential=essential
    )
    assert not fallback


def test_final_chord_sounds_the_final_notes_own_tone_under_it():
    # C4 over Am: C is Am's third. Open Am's only C is the held note's own
    # (B string, fret 1), so the chord is re-voiced to put a C under it.
    am = hand_shape(_shape("Am", (-1, 0, 2, 2, 1, 0)), STANDARD_EADGBE)
    part, fallback = final_bar(
        am, note(0.0, 1.9, 60), bar_start_s=0.0, bar_s=2.0, essential=frozenset({0})
    )
    assert not fallback and part.identity_met
    assert any(t.note.pitch % 12 == 0 and t.note.pitch < 60 for t in part.accompaniment_tab)


@pytest.mark.parametrize(("key", "mode", "label"), [("C", "major", "Am"), ("A", "minor", "F")])
def test_every_final_cadence_pitch_has_every_essential_tone_under_it(key, mode, label):
    chord = ChordSymbol.parse(label)
    hand = hand_shape(voicings_for(chord, STANDARD_EADGBE)[0], STANDARD_EADGBE)
    essential = essential_pcs(chord)
    cadence = cadence_pc(Scale(key, mode), chord)
    cadences = sorted(p for p in final_bar_pitches(hand, cadence, essential) if p % 12 == cadence)
    assert cadences
    for pitch in cadences:
        part, fallback = final_bar(
            hand, note(0.0, 1.9, pitch), bar_start_s=0.0, bar_s=2.0, essential=essential
        )
        under = {t.note.pitch % 12 for t in part.accompaniment_tab if t.note.pitch < pitch}
        assert not fallback and essential <= under, pitch


def test_final_positions_add_the_melody_strings_past_the_hand():
    assert D_HAND.positions(69) == (P(5, 5),)
    assert final_positions(D_HAND, 69) == (P(5, 5), P(4, 10))
    assert final_positions(D_HAND, 64) == (P(5, 0), P(4, 5))  # both reachable


def test_final_note_may_leave_the_hand_to_find_a_voicing():
    # Bmaj7 (x-2-4-3-4-2) reaches B3 only at G/4, where no chord fits under
    # it; on the open B string it sits over x-2-1-3 (B2, D#3, A#3).
    chord = ChordSymbol.parse("Bmaj7")
    hand = hand_shape(voicings_for(chord, STANDARD_EADGBE)[0], STANDARD_EADGBE)
    assert hand.positions(59) == (P(3, 4),)
    assert final_positions(hand, 59)[:2] == (P(3, 4), P(4, 0))
    essential = essential_pcs(chord)
    assert 59 in final_bar_pitches(hand, 11, essential)
    part, fallback = final_bar(
        hand, note(0.0, 1.9, 59), bar_start_s=0.0, bar_s=2.0, essential=essential
    )
    assert not fallback and part.melody_tab[0].position == P(4, 0)
    assert [t.position for t in part.accompaniment_tab] == [P(1, 2), P(2, 1), P(3, 3)]


def test_final_note_must_start_with_the_final_bar():
    voicings = [C_HAND.voicing, D_HAND.voicing]
    melody = [*_quarters(64, 65, 67, 65), note(2.5, 3.9, 62)]
    with pytest.raises(ValueError, match="start"):
        arrange_fingerstyle(voicings, melody, [C_HAND, D_HAND], bars_per_chord=1, tempo_bpm=120.0)


def test_arrange_fingerstyle_joins_body_and_final_bars():
    voicings = [C_HAND.voicing, D_HAND.voicing]
    melody = [*_quarters(64, 65, 67, 65), note(2.0, 3.9, 62)]
    part = arrange_fingerstyle(
        voicings, melody, [C_HAND, D_HAND], bars_per_chord=1, tempo_bpm=120.0
    )
    assert [t.note for t in part.melody_tab] == melody
    assert part.identity_misses == () and not part.final_fallback
    assert metrics.playable_moment_ratio([*part.melody_tab, *part.accompaniment_tab]) == 1.0
