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
