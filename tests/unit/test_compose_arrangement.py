import pretty_midi
import pytest

from music_decoder.compose import metrics
from music_decoder.compose.arrangement import _accompaniment_notes, _thin_below, build_midi
from music_decoder.compose.melody import generate_melody
from music_decoder.compose.voicings import voicings_for
from music_decoder.tabs.tuning import STANDARD_EADGBE
from music_decoder.types import ChordSymbol, Note, Scale


def test_build_midi_produces_two_tracks(tmp_path):
    progression = [ChordSymbol.parse(c) for c in ("Cmaj7", "Am7", "Dm7", "G7")]
    voicings = [voicings_for(c, STANDARD_EADGBE)[0] for c in progression]
    melody = generate_melody(
        scale=Scale("C", "major"),
        progression=progression,
        bars_per_chord=1,
        tempo_bpm=120.0,
        seed=42,
    )
    out = tmp_path / "x.mid"
    build_midi(
        melody=melody,
        voicings=voicings,
        bars_per_chord=1,
        tempo_bpm=120.0,
        style="strum",
        out_path=out,
    )
    assert out.exists()
    pm = pretty_midi.PrettyMIDI(str(out))
    assert len(pm.instruments) == 2
    assert sum(len(i.notes) for i in pm.instruments) > 0


def test_build_midi_styles_run(tmp_path):
    progression = [ChordSymbol.parse(c) for c in ("C", "G")]
    voicings = [voicings_for(c, STANDARD_EADGBE)[0] for c in progression]
    melody = generate_melody(
        scale=Scale("C", "major"),
        progression=progression,
        bars_per_chord=1,
        tempo_bpm=120.0,
        seed=42,
    )
    for style in ("strum", "arpeggio", "fingerstyle"):
        out = tmp_path / f"{style}.mid"
        build_midi(
            melody=melody,
            voicings=voicings,
            bars_per_chord=1,
            tempo_bpm=120.0,
            style=style,
            out_path=out,
        )
        assert out.exists()


# ---- melody-aware, on-grid accompaniment (compose improvements) ------------


def _note(start, end, pitch):
    return Note(start_s=start, end_s=end, pitch=pitch, velocity=80, confidence=1.0)


def _acc(labels, style, melody=(), bars=1):
    voicings = [voicings_for(ChordSymbol.parse(c), STANDARD_EADGBE)[0] for c in labels]
    notes = _accompaniment_notes(voicings, bars, 120.0, style, STANDARD_EADGBE, melody=melody)
    return [_note(n.start, n.end, n.pitch) for n in notes]


E = frozenset({4})  # the third of C major


def test_thin_below_keeps_accompaniment_a_minor_third_under_the_melody():
    pitches = [48, 52, 55, 60, 64]  # C3 E3 G3 C4 E4 (open C)
    melody = [_note(0.0, 0.5, 65)]  # F4
    assert _thin_below(pitches, E, melody, 0.0, 0.5) == [48, 52, 55, 60]
    assert _thin_below(pitches, E, [], 0.0, 0.5) == pitches
    assert _thin_below(pitches, E, [_note(1.0, 1.5, 50)], 0.0, 0.5) == pitches  # not sounding


def test_thin_below_always_keeps_the_bass():
    assert _thin_below([48, 52, 55], E, [_note(0.0, 0.5, 49)], 0.0, 0.5) == [48, 52]


def test_thin_below_restores_a_missing_third():
    # D chord (D3 A3 D4 F#4) under a D4 melody: F#4 would be dropped with D4
    d_shape = [50, 57, 62, 66]
    kept = _thin_below(d_shape, frozenset({6}), [_note(0.0, 2.0, 62)], 0.0, 2.0)
    assert kept == [50, 57, 66]  # third restored, the unison D4 removed


def test_thin_below_does_not_restore_what_the_melody_sounds():
    d_shape = [50, 57, 62, 66]
    kept = _thin_below(d_shape, frozenset({6}), [_note(0.0, 2.0, 66)], 0.0, 2.0)
    assert kept == [50, 57, 62]  # the melody's F#4 is the third


def test_thin_below_keeps_minor_thirds_and_sevenths():
    am = [45, 52, 57, 60, 64]  # A2 E3 A3 C4 E4 (open Am)
    assert 60 in _thin_below(am, frozenset({0}), [_note(0.0, 1.0, 62)], 0.0, 1.0)
    g7 = [43, 47, 50, 55, 59, 65]  # G2 B2 D3 G3 B3 F4 (open G7)
    kept = _thin_below(g7, frozenset({11, 5}), [_note(0.0, 1.0, 62)], 0.0, 1.0)
    assert {p % 12 for p in kept} >= {11, 5}  # B and F both still heard


@pytest.mark.parametrize("style", ["arpeggio", "strum", "fingerstyle"])
def test_stroke_heads_sit_on_the_sixteenth_grid(style):
    acc = _acc(["C", "Am", "F", "G7"], style)
    heads = metrics.stroke_heads([a.start_s for a in acc])
    assert metrics.onset_grid_ratio(heads, 120.0, 4) == 1.0


def test_arpeggio_plays_eight_eighths_per_bar():
    acc = _acc(["C", "G"], "arpeggio")  # bar 1 arpeggio, bar 2 final chord
    bar1 = sorted(a.start_s for a in acc if a.start_s < 2.0)
    assert bar1 == [k * 0.25 for k in range(8)]
    assert acc[0].pitch == 48  # C3 bass on the downbeat


def test_strum_pattern_and_up_strokes():
    acc = _acc(["C", "G"], "strum")
    bar1 = [a for a in acc if a.start_s < 2.0]
    heads = metrics.stroke_heads([a.start_s for a in bar1])
    assert [round(h / 0.25) for h in heads] == [0, 2, 3, 5, 6, 7]
    up = [a for a in bar1 if 0.75 - 1e-9 <= a.start_s < 0.75 + 0.06]
    assert len(up) == 3
    assert up[0].pitch > up[-1].pitch  # up stroke goes high -> low


def test_fingerstyle_alternates_the_bass():
    acc = _acc(["C", "G"], "fingerstyle")  # open C: C3 E3 G3 C4 E4
    beat1 = [a.pitch for a in acc if abs(a.start_s - 0.0) < 1e-9]
    beat3 = [a.pitch for a in acc if abs(a.start_s - 1.0) < 1e-9]
    assert beat1 == [48] and beat3 == [52]


@pytest.mark.parametrize("style", ["arpeggio", "strum", "fingerstyle"])
def test_final_bar_is_one_ringing_chord(style):
    acc = _acc(["C", "G", "C"], style)
    assert metrics.final_chord_rings(acc, 3, 120.0)


def test_melody_stays_on_top_of_the_strum():
    melody = [_note(0.0, 1.9, 67)]  # G4 held over the C chord
    acc = _acc(["C", "G"], "strum", melody=melody)
    sounding = [a.pitch for a in acc if a.start_s < 1.9 and a.end_s > 0.0]
    assert max(sounding) <= 64  # E4 is a minor third under G4


@pytest.mark.parametrize("style", ["arpeggio", "strum", "fingerstyle"])
def test_single_bar_accompaniment_is_the_final_chord(style):
    acc = _acc(["C"], style)
    assert acc and metrics.final_chord_rings(acc, 1, 120.0)


@pytest.mark.parametrize("style", ["arpeggio", "strum", "fingerstyle"])
def test_d_major_cadence_keeps_its_third(style):
    # review reproduction: D G A D, seed 9 used to end on bare D3 A3 D4
    progression = [ChordSymbol.parse(c) for c in ("D", "G", "A", "D")]
    voicings = [voicings_for(c, STANDARD_EADGBE)[0] for c in progression]
    melody = generate_melody(
        scale=Scale("D", "major"), progression=progression, tempo_bpm=120.0, seed=9
    )
    notes = _accompaniment_notes(voicings, 1, 120.0, style, STANDARD_EADGBE, melody=melody)
    acc = [_note(n.start, n.end, n.pitch) for n in notes]
    coverage = metrics.essential_tone_coverage(melody, acc, voicings, 1, 120.0, STANDARD_EADGBE)
    assert all(coverage)


G7_SHAPE = [43, 47, 50, 55, 59, 65]  # G2 B2 D3 G3 B3 F4
G7_ESSENTIAL = frozenset({11, 5})  # B and F


def test_a_passing_melody_tone_does_not_stand_in_for_the_held_harmony():
    # review r3: a brief F4 early in the final bar used to exempt the seventh
    melody = [_note(6.0, 6.475, 67), _note(6.5, 6.975, 65), _note(7.0, 7.95, 67)]
    assert 65 in _thin_below(G7_SHAPE, G7_ESSENTIAL, melody, 6.0, 8.0)


def test_the_held_final_melody_note_may_supply_an_essential_tone():
    melody = [_note(6.0, 6.475, 67), _note(6.5, 7.95, 65)]  # F4 held to the end
    assert 65 not in _thin_below(G7_SHAPE, G7_ESSENTIAL, melody, 6.0, 8.0)


@pytest.mark.parametrize("style", ["arpeggio", "strum", "fingerstyle"])
def test_final_stroke_keeps_the_seventh_under_the_cadence(style):
    voicings = [voicings_for(ChordSymbol.parse(c), STANDARD_EADGBE)[0] for c in ("C", "G7")]
    melody = [_note(2.0, 2.475, 67), _note(2.5, 2.975, 65), _note(3.0, 3.95, 67)]
    notes = _accompaniment_notes(voicings, 1, 120.0, style, STANDARD_EADGBE, melody=melody)
    acc = [_note(n.start, n.end, n.pitch) for n in notes]
    assert metrics.cadence_identity(melody, acc, voicings, STANDARD_EADGBE) is True
    assert metrics.final_chord_rings(acc, 2, 120.0)
