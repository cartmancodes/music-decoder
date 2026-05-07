# src/music_decoder/tab_assignment/heuristic.py
from __future__ import annotations

from collections.abc import Sequence

from music_decoder.types import TabPosition


def remaining_high_fret_penalty(
    groups: Sequence[Sequence[Sequence[TabPosition]]],
    *,
    weights: dict[str, float],
) -> float:
    """Admissible lower bound: for each remaining note/chord group, take
    the minimum fret across all candidate states and apply only the
    high-fret penalty term.
    """
    total = 0.0
    for group in groups:
        if not group:
            continue
        min_fret = min(min(p.fret for p in candidate) for candidate in group)
        total += weights["w_high"] * max(0, min_fret - 12) ** 1.2
    return float(total)
