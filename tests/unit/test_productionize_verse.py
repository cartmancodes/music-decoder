from itertools import pairwise

import pytest

from music_decoder.compose.api import arrange
from music_decoder.compose.arrangement import build_midi
from music_decoder.productionize.config import (
    COMPOSER_SECTIONS,
    ComposerArrangeConfig,
    composer_sections,
)
from music_decoder.productionize.song import song_from_midi
from music_decoder.productionize.verse import (
    bar_harmonies,
    guide_lines,
    left_hand,
    pitch_name,
    skeleton_pitches,
    voice_led_voicings,
)
from music_decoder.types import ChordSymbol, Scale

CFG = ComposerArrangeConfig()
C_MAJOR = Scale("C", "major")
TURNAROUND = [ChordSymbol.parse(c) for c in ("Cmaj7", "Am7", "Dm7", "G7")]


def _harmonies(progression=TURNAROUND, scale=C_MAJOR):
    return bar_harmonies(
        [[] for _ in progression], [0] * len(progression),
        progression=progression, scale=scale, all_pitches=[],
    )  # fmt: skip


def _names(pitches):
    return " ".join(pitch_name(p) for p in pitches)


def test_harmony_roles_and_in_scale_ninths():
    cmaj7, _, _, g7 = _harmonies()
    assert [cmaj7.role(pc) for pc in (0, 4, 7, 11, 2)] == [
        "root",
        "third",
        "fifth",
        "seventh",
        "ninth",
    ]
    assert cmaj7.essential == {4, 11} and g7.essential == {11, 5}
    assert g7.line_tones == {7, 11, 2, 5, 9}  # G B D F + the ninth A
    (em7,) = _harmonies([ChordSymbol.parse("Em7")])
    assert 6 not in em7.line_tones  # F# is not in C major: no ninth over Em7
    (e7,) = _harmonies([ChordSymbol.parse("E7")], Scale("A", "minor"))
    assert 8 in e7.tones and 8 in e7.scale  # G# comes from the chord: A harmonic minor


def test_harmony_without_chord_symbols_reads_the_bar():
    (h,) = bar_harmonies([[48, 52, 55, 59]], [0], progression=None, scale=None, all_pitches=[62])
    assert h.tones == {0, 4, 7, 11} and h.essential == {4, 11} and 2 in h.line_tones


def test_voice_led_voicings_keep_third_and_seventh_and_move_little():
    harmonies = _harmonies()
    voicings = voice_led_voicings(harmonies, CFG)
    assert [_names(v) for v in voicings] == ["B3 E4 G4", "C4 E4 G4", "C4 F4 A4", "B3 F4 G4"]
    for h, v in zip(harmonies, voicings, strict=True):
        assert h.essential <= {p % 12 for p in v}
        assert all(CFG.rh_low <= p <= CFG.rh_high for p in v)
        assert all(b - a >= 2 for a, b in pairwise(v))
    loop = [*voicings, voicings[0]]
    assert [sum(abs(x - y) for x, y in zip(a, b, strict=True)) for a, b in pairwise(loop)] == [
        1,
        3,
        3,
        1,
    ]


def test_ceilings_keep_the_voicing_under_a_melody():
    voicings = voice_led_voicings(_harmonies(), CFG, [None, None, 66, None])
    assert max(voicings[2]) <= 66


def test_left_hand_is_root_and_fifth_in_the_low_register():
    harmonies = _harmonies()
    assert [_names(left_hand(h, CFG)) for h in harmonies] == ["C3 G3", "A2 E3", "D3 A3", "G2 D3"]
    (dim,) = _harmonies([ChordSymbol.parse("Bdim")])
    assert _names(left_hand(dim, CFG)) == "B2 F3"


def test_guide_line_moves_by_step_over_chord_tones():
    harmonies = _harmonies()
    floors = [max(v) + 3 for v in voice_led_voicings(harmonies, CFG)]
    lines = guide_lines(harmonies, floors, 79, [frozenset()] * 4, limit=5)
    assert lines[0].names() == "E5 G5 F5 D5"  # third, seventh, third, fifth
    assert [line.cost for line in lines] == sorted(line.cost for line in lines)
    assert len({line.pitches[0] for line in lines}) == len(lines)  # one per starting note
    for line in lines:
        assert all(f <= p <= 79 for f, p in zip(floors, line.pitches, strict=True))
        assert all(p % 12 in h.line_tones for h, p in zip(harmonies, line.pitches, strict=True))
    best = lines[0].pitches
    loop = [*best, *best[:2]]
    assert all(
        a != c for a, c in zip(loop, loop[2:], strict=False)
    )  # never rocks between two notes


def test_guide_line_for_longer_sections_and_single_bars():
    harmonies = _harmonies(TURNAROUND * 2)
    (line,) = guide_lines(harmonies, [70] * 8, 79, [frozenset()] * 8, limit=1)
    assert len(line.pitches) == 8
    (single,) = guide_lines(harmonies[:1], [70], 79, [frozenset()], limit=1)
    assert len(single.pitches) == 1


def test_skeleton_keeps_the_notes_on_beats_1_and_3():
    bar = 2.4
    step = bar / 16
    melody = [
        (0.0, 4 * step, 76), (4 * step, 8 * step, 74), (8 * step, 12 * step, 71),
        (12 * step, 16 * step, 76),
        (bar, bar + 16 * step, 72),  # one note held through bar 2
    ]  # fmt: skip
    assert skeleton_pitches(melody, bar, 2, (0.0, 8.0)) == [
        (0, 0.0, 76),
        (0, 8.0, 71),
        (1, 0.0, 72),
    ]


def test_verse_styles():
    assert composer_sections("held") == COMPOSER_SECTIONS
    build = [
        (s.chord_voicing, s.verse_melody) for s in composer_sections("build") if s.role == "verse"
    ]
    assert build == [("voice_led", "guide_line")] * 2 + [("voice_led", "skeleton")] * 2
    others = [s for s in composer_sections("guide-line") if s.role != "verse"]
    assert others == [s for s in COMPOSER_SECTIONS if s.role != "verse"]
    with pytest.raises(ValueError, match="verse style"):
        composer_sections("rap")


@pytest.fixture
def turnaround_midi(tmp_path):
    arr = arrange(C_MAJOR, TURNAROUND, tempo_bpm=100.0, style="fingerstyle", seed=42)
    return build_midi(
        melody=arr.melody, voicings=arr.voicings, bars_per_chord=1, tempo_bpm=100.0,
        style="fingerstyle", out_path=tmp_path / "composition.mid",
    )  # fmt: skip


def _verse_notes(song, section):
    g = song.grid
    start, end = g.time(section) - 0.05, g.time(section + 1) - 0.05
    return [n for n in song.piano if start <= n.start_s < end]


def test_guide_line_verse_is_one_melody_note_per_bar_over_voice_led_chords(turnaround_midi):
    song = song_from_midi(
        turnaround_midi, progression=TURNAROUND, scale=C_MAJOR,
        arrange=ComposerArrangeConfig(verse_style="guide-line"),
    )  # fmt: skip
    g = song.grid
    notes = _verse_notes(song, 2)
    line = [n for n in notes if abs((n.start_s - g.time(2)) % g.bar_s - 10 * g.step_s) < 1e-6]
    assert _names(n.pitch for n in line) == "E5 G5 F5 D5"
    chord_tops = [
        max(
            n.pitch
            for n in notes
            if n not in line and int((n.start_s - g.time(2) + 0.05) // g.bar_s) == b
        )
        for b in range(4)
    ]
    assert all(p.pitch >= top + 3 for p, top in zip(line, chord_tops, strict=True))
    assert len(notes) == 4 * (2 + 3 + 1)  # left hand, right hand, line note in each bar


def test_skeleton_verse_quotes_the_hook_at_half_density(turnaround_midi):
    song = song_from_midi(
        turnaround_midi, progression=TURNAROUND, scale=C_MAJOR,
        arrange=ComposerArrangeConfig(verse_style="skeleton"),
    )  # fmt: skip
    hook = song_from_midi(turnaround_midi, progression=TURNAROUND, scale=C_MAJOR)
    g = song.grid
    verse = _verse_notes(song, 2)
    melody = [
        n for n in verse if n.pitch >= 67 and round((n.start_s - g.time(2)) / g.step_s) % 8 == 0
    ]
    hook_melody = [n for n in _verse_notes(hook, 1) if n.velocity >= 70 and n.pitch >= 67]
    assert 0 < len(melody) <= len(hook_melody) / 2 + 1
    for n in melody:
        bar = int((n.start_s - g.time(2) + 0.05) // g.bar_s)
        below = [
            m.pitch
            for m in verse
            if m not in melody and int((m.start_s - g.time(2) + 0.05) // g.bar_s) == bar
        ]
        assert max(below) <= n.pitch - 3 or n.pitch in below  # the chord stays a minor third under


def test_default_verse_builds_and_held_keeps_the_original(turnaround_midi):
    default = song_from_midi(turnaround_midi, progression=TURNAROUND)
    assert default.sections == composer_sections("build")
    held = song_from_midi(
        turnaround_midi, progression=TURNAROUND, arrange=ComposerArrangeConfig(verse_style="held")
    )
    assert held.sections == COMPOSER_SECTIONS
    assert default.piano != held.piano


def test_tonic_harmony_uses_the_loops_chord_on_the_tonic():
    from music_decoder.productionize.verse import tonic_harmony

    harmonies = _harmonies()  # Cmaj7 Am7 Dm7 G7
    h = tonic_harmony(harmonies, 0, minor=False, scale_pcs=frozenset({0, 2, 4, 5, 7, 9, 11}))
    assert h is harmonies[0]


def test_tonic_harmony_builds_a_triad_when_the_loop_lacks_it():
    from music_decoder.productionize.verse import tonic_harmony

    loop = [ChordSymbol.parse(c) for c in ("F", "G", "Am", "F")]
    h = tonic_harmony(_harmonies(loop), 0, minor=False, scale_pcs=frozenset({0, 2, 4, 5, 7, 9, 11}))
    assert (h.root, h.tones, h.essential) == (0, frozenset({0, 4, 7}), frozenset({4}))
    minor = tonic_harmony(
        _harmonies(loop), 9, minor=True, scale_pcs=frozenset({9, 11, 0, 2, 4, 5, 7})
    )
    assert minor.tones == frozenset({9, 0, 4})


def test_resolve_voicing_moves_least_from_the_last_chord():
    from music_decoder.productionize.verse import resolve_voicing

    harmonies = _harmonies()
    g7 = voice_led_voicings(harmonies, CFG)[-1]
    out = resolve_voicing(harmonies[0], g7, CFG)
    assert len(out) == len(g7)
    assert {p % 12 for p in out} <= harmonies[0].tones
    assert sum(abs(a - b) for a, b in zip(sorted(out), sorted(g7), strict=True)) <= 4
