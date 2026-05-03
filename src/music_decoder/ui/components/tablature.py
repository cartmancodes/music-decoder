from __future__ import annotations

from collections.abc import Iterable

from music_decoder.pipeline.contracts import TabbedNote

_STRING_LABELS = ("E", "A", "D", "G", "B", "e")   # standard EADGBE labels


def render_ascii_tab(
    notes: Iterable[TabbedNote], *, n_strings: int = 6, columns: int = 64,
) -> str:
    """Render a single-bar ASCII tab. Notes are placed in time-sorted order."""
    sorted_notes = sorted(notes, key=lambda t: t.note.start_s)
    if not sorted_notes:
        return "\n".join(f"{_STRING_LABELS[i]}|{'-' * columns}|" for i in range(5, -1, -1))
    end_time = max(t.note.end_s for t in sorted_notes)
    rows: list[list[str]] = [["-"] * columns for _ in range(n_strings)]
    for t in sorted_notes:
        col = int((t.note.start_s / end_time) * (columns - 4))
        fret = str(t.position.fret)
        for i, ch in enumerate(fret):
            if col + i < columns:
                rows[t.position.string][col + i] = ch
    out_rows = []
    for s_idx in range(n_strings - 1, -1, -1):
        label = _STRING_LABELS[s_idx] if s_idx < len(_STRING_LABELS) else "?"
        out_rows.append(f"{label}|{''.join(rows[s_idx])}|")
    return "\n".join(out_rows)


def render_svg_fretboard(
    notes: Iterable[TabbedNote], *, n_strings: int = 6, max_fret: int = 12,
) -> str:
    """Render a static fretboard SVG with marker dots at every (string, fret).

    Confidence determines marker fill color:
        >= 0.8: green, >= 0.5: gold, < 0.5: red.
    """
    width, height = 800, 200
    margin_x, margin_y = 40, 20
    fret_w = (width - 2 * margin_x) / max_fret
    string_h = (height - 2 * margin_y) / (n_strings - 1)

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">']
    # Strings (horizontal lines)
    for i in range(n_strings):
        y = margin_y + i * string_h
        parts.append(
            f'<line x1="{margin_x}" y1="{y}" x2="{width - margin_x}" y2="{y}" '
            'stroke="#444" stroke-width="1.5"/>'
        )
    # Frets (vertical lines)
    for f in range(max_fret + 1):
        x = margin_x + f * fret_w
        parts.append(
            f'<line x1="{x}" y1="{margin_y}" x2="{x}" y2="{height - margin_y}" '
            f'stroke="#888" stroke-width="{2 if f == 0 else 1}"/>'
        )
    # Note markers
    for t in notes:
        if t.position.fret > max_fret:
            continue
        cx = margin_x + (t.position.fret + 0.5) * fret_w if t.position.fret > 0 else margin_x - 12
        cy = margin_y + (n_strings - 1 - t.position.string) * string_h
        c = t.note.confidence
        fill = "green" if c >= 0.8 else ("gold" if c >= 0.5 else "red")
        parts.append(
            f'<circle cx="{cx}" cy="{cy}" r="8" fill="{fill}" stroke="#222"/>'
        )
        parts.append(
            f'<text x="{cx}" y="{cy + 4}" font-size="10" fill="#fff" '
            f'text-anchor="middle">{t.position.fret}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)
