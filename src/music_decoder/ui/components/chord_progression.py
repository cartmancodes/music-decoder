"""Plain-Python renderers for the chord-progression UI tab.

Pure functions; no Streamlit imports here. The Streamlit page in
ui/pages/03_Results.py composes these and feeds them to st.code / st.markdown.
"""
from __future__ import annotations

from collections.abc import Iterable

from music_decoder.chords.templates import chord_label
from music_decoder.chords.voicings import voicing_for
from music_decoder.types import ChordSegment, TabbedNote
from music_decoder.tabs.render import render_ascii_tab


def render_chord_progression_text(segments: list[ChordSegment]) -> str:
    if not segments:
        return ""
    parts = []
    for seg in segments:
        parts.append(f"{chord_label(seg.root, seg.quality)} ({seg.start_s:.1f}s)")
    return " | ".join(parts)


def render_chord_timeline(segments: Iterable[ChordSegment]) -> None:
    """Render a chord progression timeline into the active Streamlit context.

    Thin Streamlit-aware adapter over :func:`render_chord_progression_text`.
    Streamlit is imported lazily so this module still imports cleanly outside
    a running Streamlit server (e.g. during pytest collection).
    """
    import streamlit as st  # local import keeps module-level import side-effect-free

    seg_list = list(segments)
    text = render_chord_progression_text(seg_list)
    if not text:
        st.info("No chords detected.")
        return
    st.code(text)


def render_chord_diagram_svg(
    root: str, quality: str, *, n_strings: int = 6, max_fret: int = 5,
) -> str:
    label = chord_label(root, quality)
    voicing = voicing_for(root, quality)

    width, height = 140, 180
    margin_x, margin_y = 25, 35
    fret_w = (width - 2 * margin_x) / max_fret
    string_h = (height - 2 * margin_y) / (n_strings - 1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<text x="{width // 2}" y="20" font-size="14" font-weight="bold" '
        f'text-anchor="middle">{label}</text>',
    ]
    # Strings (horizontal lines)
    for i in range(n_strings):
        y = margin_y + i * string_h
        parts.append(
            f'<line x1="{margin_x}" y1="{y}" x2="{width - margin_x}" y2="{y}" '
            'stroke="#444" stroke-width="1.2"/>'
        )
    # Frets (vertical lines)
    for f in range(max_fret + 1):
        x = margin_x + f * fret_w
        sw = 2 if f == 0 else 1
        parts.append(
            f'<line x1="{x}" y1="{margin_y}" x2="{x}" y2="{height - margin_y}" '
            f'stroke="#888" stroke-width="{sw}"/>'
        )
    # Voicing markers (low-to-high).
    # String 0 in voicing = lowest = topmost row in standard chord diagrams.
    for s_idx, fret in enumerate(voicing):
        # Display string row: lowest string at the bottom.
        y = margin_y + (n_strings - 1 - s_idx) * string_h
        if fret == -1:
            parts.append(
                f'<text x="{margin_x - 12}" y="{y + 4}" font-size="10" '
                f'text-anchor="middle" fill="#888">x</text>'
            )
        elif fret == 0:
            parts.append(
                f'<circle cx="{margin_x - 8}" cy="{y}" r="4" fill="none" stroke="#222"/>'
            )
        elif fret <= max_fret:
            cx = margin_x + (fret - 0.5) * fret_w
            parts.append(
                f'<circle cx="{cx}" cy="{y}" r="6" fill="#222"/>'
            )
        else:
            # Fret out of displayed range (e.g., barre at fret 6 with max_fret=5):
            # render the marker at the rightmost fret with a "+" annotation.
            cx = margin_x + (max_fret - 0.5) * fret_w
            parts.append(
                f'<circle cx="{cx}" cy="{y}" r="6" fill="#222"/>'
                f'<text x="{cx + 12}" y="{y + 4}" font-size="9" fill="#222">+{fret}</text>'
            )
    parts.append("</svg>")
    return "".join(parts)


def render_chord_labels_above_tab(
    notes: Iterable[TabbedNote],
    segments: list[ChordSegment],
    *,
    n_strings: int = 6,
    columns: int = 64,
) -> str:
    """Build a chord-label header row aligned above the existing ASCII tab.

    The ASCII tab renderer prefixes each row with two characters (e.g. "e|").
    The chord-label row receives the same two-character offset so chord names
    line up with the column where each chord starts.
    """
    notes_list = list(notes)
    tab_rows = render_ascii_tab(notes_list, n_strings=n_strings, columns=columns)

    # Compute the timeline span the tab covers.
    if notes_list:
        end_time = max(t.note.end_s for t in notes_list)
    else:
        end_time = max((s.end_s for s in segments), default=0.0)
    if end_time <= 0:
        return tab_rows

    inner_columns = columns          # tab body width (between the bars)
    # Tab rows are formatted as "e|" + columns chars + "|" -> length columns + 3.
    # Match that width so the trailing pipe also has whitespace above it and
    # the monospace alignment doesn't drift on wide UIs.
    label_row = [" "] * (inner_columns + 3)
    for seg in segments:
        col = int((seg.start_s / end_time) * (inner_columns - 4)) + 2
        label = chord_label(seg.root, seg.quality)
        for i, ch in enumerate(label):
            if col + i < len(label_row):
                label_row[col + i] = ch
    return "".join(label_row) + "\n" + tab_rows
