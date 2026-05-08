from music_decoder.types import (
    ChordSegment,
    TabbedNote,
    TabPosition,
    TranscribedNote,
)
from music_decoder.ui.components.chord_progression import (
    render_chord_diagram_svg,
    render_chord_labels_above_tab,
    render_chord_progression_text,
)


def _seg(start: float, end: float, root: str, quality: str, conf: float = 0.85) -> ChordSegment:
    return ChordSegment(start_s=start, end_s=end, root=root, quality=quality, confidence=conf)


def test_progression_text_pipe_format():
    segments = [
        _seg(0.0, 4.0, "A", "min"),
        _seg(4.0, 8.0, "F", "maj"),
        _seg(8.0, 12.0, "C", "maj"),
        _seg(12.0, 16.0, "G", "maj"),
    ]
    text = render_chord_progression_text(segments)
    assert "Am" in text
    assert "F" in text
    assert "|" in text


def test_progression_text_handles_empty():
    assert render_chord_progression_text([]) == ""


def test_diagram_svg_starts_with_svg_tag():
    svg = render_chord_diagram_svg("C", "maj")
    assert svg.startswith("<svg")
    assert "</svg>" in svg


def test_diagram_svg_includes_chord_label():
    svg = render_chord_diagram_svg("A", "min")
    assert "Am" in svg


def test_diagram_svg_for_no_chord_returns_empty_box():
    svg = render_chord_diagram_svg("N", "")
    assert svg.startswith("<svg")
    # No fret marks should be rendered for a fully muted N.
    assert svg.count("circle") == 0 or "muted" in svg.lower()


def test_labels_above_tab_aligns_chord_changes():
    notes = [
        TabbedNote(
            note=TranscribedNote(start_s=0.0, end_s=1.0, pitch=60, velocity=80, confidence=0.9),
            position=TabPosition(string=4, fret=1),
            cost_breakdown={},
        ),
        TabbedNote(
            note=TranscribedNote(start_s=4.0, end_s=5.0, pitch=65, velocity=80, confidence=0.9),
            position=TabPosition(string=5, fret=1),
            cost_breakdown={},
        ),
    ]
    segments = [
        _seg(0.0, 4.0, "C", "maj"),
        _seg(4.0, 8.0, "F", "maj"),
    ]
    out = render_chord_labels_above_tab(notes, segments, columns=64)
    rows = out.split("\n")
    # Top row contains the chord labels positioned above their respective columns.
    assert rows[0].startswith(" ")  # the chord row is offset to align with the "X|" prefix
    assert "C" in rows[0]
    assert "F" in rows[0]
    # Followed by 6 string rows (the existing ASCII-tab format).
    assert len(rows) >= 7
