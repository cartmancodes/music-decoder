import math

import pytest

from music_decoder.compose import metrics
from music_decoder.compose.api import arrange
from music_decoder.compose.fingerstyle import FingerstylePart, arrange_fingerstyle, hand_shape
from music_decoder.compose.performance import let_ring, shape_dynamics
from music_decoder.tabs.tuning import STANDARD_EADGBE
from music_decoder.types import ChordSymbol, Note, Scale, TabbedNote, TabPosition


def _fs(labels, key="C", mode="major", seed=3, bars=1, tempo=120.0, tuning=STANDARD_EADGBE):
    prog = [ChordSymbol.parse(c) for c in labels]
    return prog, arrange(
        Scale(key, mode),
        prog,
        bars_per_chord=bars,
        tempo_bpm=tempo,
        style="fingerstyle",
        tuning=tuning,
        seed=seed,
    )


def test_melody_is_at_least_8_louder_than_the_accompaniment_under_it():
    _, a = _fs(("C", "Am", "F", "G"))
    for m in a.melody:
        under = [n.velocity for n in a.accompaniment if n.start_s <= m.start_s + 1e-4 < n.end_s]
        assert not under or m.velocity >= max(under) + 8


def test_velocities_spread_and_stay_in_range():
    _, a = _fs(("C", "Am", "F", "G", "C", "F", "G", "C"))
    notes = [*a.melody, *a.accompaniment]
    assert metrics.velocity_spread(notes) >= 8
    assert all(30 <= n.velocity <= 118 for n in notes)


def test_accompaniment_rings_longer_than_v4():
    _, a = _fs(("C", "Am", "F", "G"))
    assert metrics.mean_length_beats(a.accompaniment, 120.0) >= 1.2  # v4 ≈ 0.8


def test_ringing_stops_at_the_bar_line():
    _, a = _fs(("C", "Am"), bars=2)
    bar = 2.0
    for n in a.accompaniment:
        assert (
            int((n.start_s + 1e-6) // bar) == int((n.end_s - 1e-6) // bar)
            or n.start_s >= 3 * bar - 1e-6
        )


@pytest.mark.parametrize("tempo", [40.0, 200.0, 240.0])
def test_let_ring_keeps_every_guarantee_at_extreme_tempos(tempo):
    for labels in (("C", "Am", "F", "G"), ("D", "G", "A7", "D"), ("F", "A#", "C", "F")):
        prog, a = _fs(labels, key=labels[0], tempo=tempo)
        part = [*a.melody_tab, *a.accompaniment_tab]
        assert metrics.playable_moment_ratio(part) == 1.0
        assert metrics.melody_onset_on_top_ratio(a.melody, a.accompaniment) == 1.0
        assert metrics.tab_consistent([*a.melody, *a.accompaniment], part, STANDARD_EADGBE)
        assert metrics.final_chord_rings(a.accompaniment, len(prog), tempo)


def test_onsets_stay_on_the_grid():
    _, a = _fs(("C", "Am", "F", "G"))
    heads = metrics.stroke_heads([n.start_s for n in a.accompaniment])
    assert metrics.onset_grid_ratio(heads, 120.0, 2) == 1.0
    assert metrics.onset_grid_ratio([n.start_s for n in a.melody], 120.0, 2) == 1.0


# ---- the pinned values, on the v4 part ---------------------------------------


def _v4_part(labels=("C", "Am", "F", "G"), tempo=120.0, seed=3):
    """The v4 part arrange() starts from, with arrange()'s own melody."""
    prog, a = _fs(labels, tempo=tempo, seed=seed)
    hands = [hand_shape(v, STANDARD_EADGBE) for v in a.voicings]
    melody = [
        Note(start_s=n.start_s, end_s=n.end_s, pitch=n.pitch, velocity=90, confidence=1.0)
        for n in a.melody
    ]
    part = arrange_fingerstyle(a.voicings, melody, hands, bars_per_chord=1, tempo_bpm=tempo)
    return prog, part


def test_accompaniment_roles_get_their_pinned_velocities():
    prog, part = _v4_part()
    shaped = shape_dynamics(part, n_bars=len(prog), tempo_bpm=120.0)
    final_start = (len(prog) - 1) * 2.0
    for t in shaped.accompaniment_tab:
        n = t.note
        if n.start_s >= final_start - 1e-6:
            assert n.velocity == 70
    hands = [hand_shape(v, STANDARD_EADGBE) for v in _fs(("C", "Am", "F", "G"))[1].voicings]
    body = [t for t in shaped.accompaniment_tab if t.note.start_s < final_start - 1e-6]
    for t in body:
        k = round(t.note.start_s / 0.25)
        hand = hands[k // 8]
        on_thumb = t.position.string in (hand.root.string, hand.alternate.string)
        if k % 8 in (0, 4) and on_thumb:
            assert t.note.velocity == 66
        elif k % 8 in (2, 6) and on_thumb:
            assert t.note.velocity == 58
        else:
            assert t.note.velocity == 50


def test_final_melody_note_is_84_and_strong_beats_are_louder():
    prog, part = _v4_part()
    shaped = shape_dynamics(part, n_bars=len(prog), tempo_bpm=120.0)
    assert shaped.melody_tab[-1].note.velocity == 84
    for t in shaped.melody_tab[:-1]:
        k = round(t.note.start_s / 0.25) % 8
        assert 78 + (6 if k in (0, 4) else 0) <= t.note.velocity <= 78 + 6 + 10

    # positions and timing are untouched
    def shape(tab):
        return [(t.position, t.note.start_s, t.note.end_s, t.note.pitch) for t in tab]

    assert shape(shaped.melody_tab) == shape(part.melody_tab)


def _tab(start, end, pitch, string=4):
    note = Note(start_s=start, end_s=end, pitch=pitch, velocity=90, confidence=1.0)
    return TabbedNote(note=note, position=TabPosition(string=string, fret=pitch - 59))


def test_phrase_swell_rises_to_the_first_highest_note_then_falls_to_the_last():
    # four bars at 120 bpm (one phrase); off-beat onsets, so no strong-beat boost
    melody = (
        _tab(0.25, 0.5, 60),
        _tab(0.75, 1.0, 64),
        _tab(1.25, 1.5, 67),  # the peak: the first of the tied highest notes
        _tab(1.75, 2.0, 67),
        _tab(2.25, 2.5, 62),
        _tab(6.0, 8.0, 60),  # the final note
    )
    part = FingerstylePart(melody, (), (), False)
    shaped = shape_dynamics(part, n_bars=4, tempo_bpm=120.0)
    # w = 0, 0.5, 1, 4.25/4.75, 3.75/4.75; the final note is pinned to 84
    assert [t.note.velocity for t in shaped.melody_tab] == [78, 83, 88, 87, 86, 84]


def test_strong_beats_add_6_and_a_lone_low_note_on_beat_2_is_the_thumb():
    melody = (_tab(0.0, 0.5, 64), _tab(0.5, 1.0, 64), _tab(6.0, 8.0, 64))
    low = Note(start_s=0.5, end_s=1.0, pitch=50, velocity=90, confidence=1.0)
    part = FingerstylePart(melody, (TabbedNote(note=low, position=TabPosition(0, 0)),), (), False)
    shaped = shape_dynamics(part, n_bars=4, tempo_bpm=120.0)
    assert shaped.melody_tab[0].note.velocity == 78 + 10 + 6  # the peak (w = 1), on beat 1
    assert shaped.accompaniment_tab[0].note.velocity == 58


def test_let_ring_never_shortens_and_keeps_the_final_bar_and_melody():
    prog, part = _v4_part()
    rung = let_ring(part, prog, bars_per_chord=1, tempo_bpm=120.0)
    assert rung.melody_tab == part.melody_tab
    assert len(rung.accompaniment_tab) == len(part.accompaniment_tab)
    final_start = (len(prog) - 1) * 2.0
    for old, new in zip(part.accompaniment_tab, rung.accompaniment_tab, strict=True):
        assert new.position == old.position
        assert new.note.start_s == old.note.start_s and new.note.pitch == old.note.pitch
        assert new.note.velocity == old.note.velocity
        assert new.note.end_s >= old.note.end_s - 1e-9
        if old.note.start_s >= final_start - 1e-6:
            assert new.note == old.note


def test_arrange_melody_matches_its_tab():
    _, a = _fs(("C", "Am", "F", "G"))
    assert a.melody == tuple(t.note for t in a.melody_tab)
    assert a.accompaniment == tuple(t.note for t in a.accompaniment_tab)


# ---- metrics -------------------------------------------------------------------


def _n(start, end, velocity=64):
    return Note(start_s=start, end_s=end, pitch=60, velocity=velocity, confidence=1.0)


def test_velocity_spread_is_the_population_standard_deviation():
    assert metrics.velocity_spread([]) is None
    assert metrics.velocity_spread([_n(0, 1, 60), _n(0, 1, 70)]) == pytest.approx(5.0)
    assert metrics.velocity_spread([_n(0, 1, 60)]) == 0.0


def test_mean_length_beats():
    assert metrics.mean_length_beats([], 120.0) is None
    assert metrics.mean_length_beats([_n(0, 0.5), _n(0, 1.5)], 120.0) == pytest.approx(2.0)
    assert math.isclose(metrics.mean_length_beats([_n(0, 1.0)], 60.0) or 0, 1.0)
