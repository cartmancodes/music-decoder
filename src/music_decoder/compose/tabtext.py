"""Plain-text tablature for a composition: the melody tab and the chord shapes.

Each melody note gets its own column in playing order (bars separated by
``|``, a few bars per line), so no note can overwrite another; the chord
shapes are listed separately under their names. Rhythm is not drawn: the
MIDI file is the source of truth for timing.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from music_decoder.types import VALID_TONICS, TabbedNote, Tuning, VoicedChord

BARS_PER_LINE = 4
CHORDS_PER_LINE = 8


def string_labels(tuning: Tuning) -> list[str]:
    """One label per string (low to high), equal width; a top string that
    repeats the bottom string's name is lower-cased, as in ``E A D G B e``."""
    names = [VALID_TONICS[p % 12] for p in tuning.open_pitches]
    if len(names) > 1 and names[-1] == names[0]:
        names[-1] = names[-1].lower()
    width = max(len(n) for n in names)
    return [n.ljust(width) for n in names]


def render_melody_tab(
    melody_tab: Sequence[TabbedNote], *, n_bars: int, tempo_bpm: float, tuning: Tuning
) -> str:
    labels = string_labels(tuning)
    bar_s = 4.0 * 60.0 / tempo_bpm
    by_bar: list[list[TabbedNote]] = [[] for _ in range(max(n_bars, 1))]
    for tn in sorted(melody_tab, key=lambda t: t.note.start_s):
        bar = min(math.floor(tn.note.start_s / bar_s + 1e-6), len(by_bar) - 1)
        by_bar[bar].append(tn)
    systems: list[str] = []
    for first in range(0, len(by_bar), BARS_PER_LINE):
        rows = [f"{label}|" for label in labels]
        for bar_notes in by_bar[first : first + BARS_PER_LINE]:
            for tn in bar_notes:
                text = str(tn.position.fret)
                for s in range(len(rows)):
                    rows[s] += "-" + text if s == tn.position.string else "-" * (len(text) + 1)
            rows = [row + ("-|" if bar_notes else "---|") for row in rows]
        systems.append("\n".join(reversed(rows)))
    return "\n\n".join(systems)


def render_chord_shapes(voicings: Sequence[VoicedChord], tuning: Tuning) -> str:
    labels = string_labels(tuning)
    pad = " " * (len(labels[0]) + 1)
    systems: list[str] = []
    for first in range(0, len(voicings), CHORDS_PER_LINE):
        chunk = voicings[first : first + CHORDS_PER_LINE]
        header = pad
        rows = [f"{label}|" for label in labels]
        for vc in chunk:
            name = vc.chord.to_label()
            frets = {p.string: ("x" if p.fret < 0 else str(p.fret)) for p in vc.positions}
            width = max([len(name), *(len(f) for f in frets.values())]) + 2
            header += " " + name.ljust(width - 1)
            for s in range(len(rows)):
                text = frets.get(s, "x")
                rows[s] += "-" + text + "-" * (width - len(text) - 1)
        rows = [row + "|" for row in rows]
        systems.append("\n".join([header.rstrip(), *reversed(rows)]))
    return "\n\n".join(systems)


def render_composition_tab(
    melody_tab: Sequence[TabbedNote],
    voicings: Sequence[VoicedChord],
    *,
    bars_per_chord: int,
    tempo_bpm: float,
    tuning: Tuning,
) -> str:
    n_bars = len(voicings) * bars_per_chord
    melody = render_melody_tab(melody_tab, n_bars=n_bars, tempo_bpm=tempo_bpm, tuning=tuning)
    chords = render_chord_shapes(voicings, tuning)
    return f"Melody\n{melody}\n\nChords\n{chords}"


_COLUMNS_PER_BAR = 8
_COUNT_SYMBOLS = "1&2&3&4&"


def render_fingerstyle_tab(
    part: Sequence[TabbedNote],
    voicings: Sequence[VoicedChord],
    *,
    bars_per_chord: int,
    tempo_bpm: float,
    tuning: Tuning,
) -> str:
    """The fingerstyle part (melody and accompaniment together) with rhythm:
    eight eighth-note columns per bar, a chord-name row and a count row above
    each system. Held notes are drawn at their onset only."""
    labels = string_labels(tuning)
    n_strings = len(labels)
    n_bars = max(len(voicings) * bars_per_chord, 1)
    bar_s = 4.0 * 60.0 / tempo_bpm
    eighth_s = bar_s / _COLUMNS_PER_BAR
    # cells[bar][string][column] -> fret text
    cells: list[list[list[str]]] = [
        [[""] * _COLUMNS_PER_BAR for _ in range(n_strings)] for _ in range(n_bars)
    ]
    for tn in part:
        start = tn.note.start_s
        bar = min(math.floor(start / bar_s + 1e-6), n_bars - 1)
        column = min(max(round((start - bar * bar_s) / eighth_s), 0), _COLUMNS_PER_BAR - 1)
        slot = cells[bar][tn.position.string]
        if slot[column]:
            raise ValueError(
                f"two notes on string {tn.position.string} in bar {bar}, column {column}"
            )
        slot[column] = str(tn.position.fret)

    prefix = len(labels[0]) + 1
    systems: list[str] = []
    for first in range(0, n_bars, BARS_PER_LINE):
        rows = [f"{label}|" for label in labels]
        chord_row = [" "] * prefix
        count_row = [" "] * prefix
        for bar in range(first, min(first + BARS_PER_LINE, n_bars)):
            widths = [
                max(1, *(len(cells[bar][s][c]) for s in range(n_strings)))
                for c in range(_COLUMNS_PER_BAR)
            ]
            name = voicings[min(bar // bars_per_chord, len(voicings) - 1)].chord.to_label()
            offset = len(rows[0])
            starts: list[int] = []
            for width in widths:
                starts.append(offset + 1)
                offset += 1 + width
            for s in range(n_strings):
                rows[s] += (
                    "".join("-" + cells[bar][s][c].ljust(widths[c], "-") for c in range(8)) + "-|"
                )
            chord_row += [" "] * (starts[0] - len(chord_row))
            chord_row += list(name)
            for c, at in enumerate(starts):
                count_row += [" "] * (at - len(count_row))
                count_row.append(_COUNT_SYMBOLS[c])
            total = offset + 2
            chord_row += [" "] * (total - len(chord_row))
            count_row += [" "] * (total - len(count_row))
        systems.append(
            "\n".join(
                [
                    "".join(chord_row).rstrip(),
                    *reversed(rows),
                    "".join(count_row).rstrip(),
                ]
            )
        )
    chords = render_chord_shapes(voicings, tuning)
    return "Fingerstyle\n" + "\n\n".join(systems) + "\n\nChords\n" + chords
