import re

from music_decoder.compose.tabtext import (
    render_chord_shapes,
    render_composition_tab,
    render_melody_tab,
    string_labels,
)
from music_decoder.compose.voicings import voicings_for
from music_decoder.tabs.tuning import DADGAD, EB_HALF_STEP_DOWN, STANDARD_EADGBE
from music_decoder.types import ChordSymbol, Note, TabbedNote, TabPosition


def tabbed(start_s: float, string: int, fret: int) -> TabbedNote:
    pitch = STANDARD_EADGBE.open_pitches[string] + fret
    note = Note(start_s=start_s, end_s=start_s + 0.4, pitch=pitch, velocity=80, confidence=1.0)
    return TabbedNote(note=note, position=TabPosition(string=string, fret=fret))


def decode(block: str) -> list[tuple[int, int]]:
    """(string, fret) per melody note, reading each system's columns left to right."""
    out: list[tuple[int, int]] = []
    for system in block.split("\n\n"):
        rows = system.splitlines()[::-1]  # low string first
        hits: list[tuple[int, int, int]] = []
        for s, row in enumerate(rows):
            body = row.split("|", 1)[1]
            hits.extend((m.start(), s, int(m.group())) for m in re.finditer(r"\d+", body))
        out.extend((s, f) for _, s, f in sorted(hits))
    return out


def test_string_labels():
    assert string_labels(STANDARD_EADGBE) == ["E", "A", "D", "G", "B", "e"]
    assert string_labels(DADGAD) == ["D", "A", "D", "G", "A", "d"]
    assert string_labels(EB_HALF_STEP_DOWN) == ["D#", "G#", "C#", "F#", "A#", "d#"]


def test_melody_tab_keeps_every_note_including_two_digit_frets():
    notes = [tabbed(0.0, 3, 9), tabbed(0.5, 5, 0), tabbed(1.0, 4, 12), tabbed(2.0, 5, 15)]
    block = render_melody_tab(notes, n_bars=2, tempo_bpm=120.0, tuning=STANDARD_EADGBE)
    assert decode(block) == [(3, 9), (5, 0), (4, 12), (5, 15)]
    lines = block.splitlines()
    assert len(lines) == 6 and len({len(line) for line in lines}) == 1
    assert lines[0].count("|") == 3  # label bar line + 2 bar ends


def test_melody_tab_wraps_every_four_bars_and_marks_empty_bars():
    notes = [tabbed(2.0 * b, 5, b) for b in range(6) if b != 4]
    block = render_melody_tab(notes, n_bars=6, tempo_bpm=120.0, tuning=STANDARD_EADGBE)
    systems = block.split("\n\n")
    assert len(systems) == 2
    assert decode(block) == [(5, b) for b in range(6) if b != 4]
    assert "---|" in systems[1].splitlines()[0]  # bar 5 has no notes


def test_chord_shapes_block():
    shapes = [voicings_for(ChordSymbol.parse(c), STANDARD_EADGBE)[0] for c in ("C", "Am7")]
    block = render_chord_shapes(shapes, STANDARD_EADGBE)
    header, *rows = block.splitlines()
    assert header.split() == ["C", "Am7"]
    assert rows[0].startswith("e|-0") and rows[-1].startswith("E|-x")  # C: x32010
    assert rows[1].startswith("B|-1")


def test_composition_tab_does_not_let_chords_overwrite_the_melody():
    # review reproduction: an E4 on the G string, fret 9, at the same time as a C chord
    melody = [tabbed(0.0, 3, 9)]
    shapes = [voicings_for(ChordSymbol.parse("C"), STANDARD_EADGBE)[0]]
    text = render_composition_tab(
        melody, shapes, bars_per_chord=1, tempo_bpm=120.0, tuning=STANDARD_EADGBE
    )
    melody_block = text.split("\n\nChords\n")[0].removeprefix("Melody\n")
    assert decode(melody_block) == [(3, 9)]
    assert text.startswith("Melody\n") and "\n\nChords\n" in text
