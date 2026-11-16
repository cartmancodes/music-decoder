"""Melody features on the eighth-note grid (composer v6).

The same feature functions measure real melodies and generated ones, so the
generator can be gated against the ranges real melodies occupy.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

from music_decoder.compose.harmony import chord_pcs, tonic_pc
from music_decoder.types import ChordSymbol, Note, Scale

Mode = Literal["major", "minor"]

FEATURES: tuple[str, ...] = (
    "repeat",
    "step",
    "third",
    "leap",
    "notes_per_second",
    "rhythm_variety",
    "rhythm_lag2",
    "rhythm_lag4",
    "figure_recurrence",
    "offbeat",
    "late_entry",
    "chord_tone_strong",
    "pentatonic",
    "range",
)

SLOTS_PER_BAR = 8
WINDOW_BARS = 8
_PENTATONIC: dict[str, frozenset[int]] = {
    "major": frozenset({0, 2, 4, 7, 9}),
    "minor": frozenset({0, 3, 5, 7, 10}),
}
_FLOORS: dict[str, float] = {"range": 1.0, "notes_per_second": 0.2}
_DEFAULT_FLOOR = 0.05


@dataclass(frozen=True)
class GridNote:
    slot: int  # eighths from the start of the excerpt
    pitch: int  # MIDI


@dataclass(frozen=True)
class Band:
    low: float
    mid: float
    high: float


def _share(count: int, total: int) -> float:
    return count / total if total else 0.0


def _window_features(
    notes: Sequence[GridNote],
    *,
    first_bar: int,
    bars: int,
    tonic_pc: int,
    mode: Mode,
    chord_tones: Sequence[frozenset[int]],
    seconds_per_bar: float,
) -> dict[str, float]:
    lo, hi = first_bar * SLOTS_PER_BAR, (first_bar + bars) * SLOTS_PER_BAR
    win = sorted((n for n in notes if lo <= n.slot < hi), key=lambda n: (n.slot, n.pitch))
    out: dict[str, float] = {}

    moves = [b.pitch - a.pitch for a, b in pairwise(win)]
    sizes = [abs(m) for m in moves]
    out["repeat"] = _share(sum(s == 0 for s in sizes), len(sizes))
    out["step"] = _share(sum(1 <= s <= 2 for s in sizes), len(sizes))
    out["third"] = _share(sum(3 <= s <= 4 for s in sizes), len(sizes))
    out["leap"] = _share(sum(s >= 5 for s in sizes), len(sizes))
    out["notes_per_second"] = len(win) / (bars * seconds_per_bar)

    patterns: list[tuple[int, ...]] = []
    for b in range(bars):
        base = lo + b * SLOTS_PER_BAR
        patterns.append(tuple(n.slot - base for n in win if base <= n.slot < base + SLOTS_PER_BAR))
    non_empty = [p for p in patterns if p]
    out["rhythm_variety"] = _share(len(set(non_empty)), len(non_empty))
    for lag in (2, 4):
        pairs = [(patterns[b], patterns[b - lag]) for b in range(lag, bars)]
        pairs = [(x, y) for x, y in pairs if x and y]
        out[f"rhythm_lag{lag}"] = _share(sum(x == y for x, y in pairs), len(pairs))

    figures = [tuple(moves[i : i + 3]) for i in range(len(moves) - 2)] if len(win) >= 4 else []
    counts = Counter(figures)
    out["figure_recurrence"] = _share(sum(counts[f] > 1 for f in figures), len(figures))

    out["offbeat"] = _share(sum(n.slot % 2 == 1 for n in win), len(win))
    bars_with_downbeat = sum(1 for p in patterns if 0 in p)
    out["late_entry"] = _share(bars - bars_with_downbeat, bars)

    strong = [n for n in win if n.slot % SLOTS_PER_BAR in (0, 4)]
    if strong:
        hits = sum(n.pitch % 12 in chord_tones[n.slot] for n in strong)
        out["chord_tone_strong"] = hits / len(strong)

    pent = _PENTATONIC[mode]
    out["pentatonic"] = _share(sum((n.pitch - tonic_pc) % 12 in pent for n in win), len(win))
    out["range"] = float(max(n.pitch for n in win) - min(n.pitch for n in win)) if win else 0.0
    return out


def grid_features(
    notes: Sequence[GridNote],
    *,
    n_bars: int,
    tonic_pc: int,
    mode: Mode,
    chord_tones: Sequence[frozenset[int]],
    seconds_per_bar: float,
) -> dict[str, float]:
    """Features of a melody on the eighth grid, averaged over 8-bar windows.

    A trailing window shorter than 8 bars is ignored, unless the whole piece is
    shorter than 8 bars, when it is the one window. ``chord_tone_strong`` is
    averaged over the windows that have a note on slot 0 or 4 and left out when
    none does.
    """
    if n_bars < WINDOW_BARS:
        starts, bars = [0], n_bars
    else:
        starts, bars = list(range(0, n_bars - n_bars % WINDOW_BARS, WINDOW_BARS)), WINDOW_BARS
    windows = [
        _window_features(
            notes,
            first_bar=start,
            bars=bars,
            tonic_pc=tonic_pc,
            mode=mode,
            chord_tones=chord_tones,
            seconds_per_bar=seconds_per_bar,
        )
        for start in starts
    ]
    out: dict[str, float] = {}
    for name in FEATURES:
        values = [w[name] for w in windows if name in w]
        if values:
            out[name] = sum(values) / len(values)
    return out


def tune_features(
    melody: Sequence[Note],
    scale: Scale,
    progression: Sequence[ChordSymbol],
    bars_per_chord: int,
    tempo_bpm: float,
) -> dict[str, float]:
    """Features of a melody over a chord progression at ``tempo_bpm`` in 4/4."""
    eighth_s = 60.0 / tempo_bpm / 2
    n_bars = len(progression) * bars_per_chord
    slots = n_bars * SLOTS_PER_BAR
    grid = [GridNote(slot=round(n.start_s / eighth_s), pitch=n.pitch) for n in melody]
    grid = [g for g in grid if 0 <= g.slot < slots]
    chord_tones = [
        frozenset(chord_pcs(progression[slot // SLOTS_PER_BAR // bars_per_chord]))
        for slot in range(slots)
    ]
    return grid_features(
        grid,
        n_bars=n_bars,
        tonic_pc=tonic_pc(scale),
        mode=scale.mode,
        chord_tones=chord_tones,
        seconds_per_bar=8 * eighth_s,
    )


def band_distance(features: Mapping[str, float], bands: Mapping[str, Band]) -> float:
    """Summed distance of each shared feature outside its band, in band widths."""
    total = 0.0
    for name, value in features.items():
        band = bands.get(name)
        if band is None:
            continue
        floor = _FLOORS.get(name, _DEFAULT_FLOOR)
        total += max(0.0, band.low - value, value - band.high) / max(band.high - band.low, floor)
    return total
