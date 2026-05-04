# src/music_decoder/tab_assignment/cost.py
from __future__ import annotations

from collections.abc import Sequence

from music_decoder.pipeline.contracts import TabPosition


def chord_span_penalty(chord: Sequence[TabPosition]) -> float:
    fretted = [p.fret for p in chord if p.fret > 0]
    if len(fretted) < 2:
        return 0.0
    span = max(fretted) - min(fretted)
    if span <= 4:
        return 0.0
    return float((span - 4) ** 1.5)


def chord_collides(chord: Sequence[TabPosition]) -> bool:
    strings = [p.string for p in chord]
    return len(strings) != len(set(strings))


def transition_cost(
    *,
    prev: TabPosition,
    curr: TabPosition,
    weights: dict[str, float],
    hand_anchor: float,
) -> float:
    move = weights["w_move"] * max(0, curr.fret - prev.fret)
    string = weights["w_string"] * abs(curr.string - prev.string)
    span = weights["w_span"] * max(0.0, curr.fret - hand_anchor) ** 1.5
    high = weights["w_high"] * max(0, curr.fret - 12) ** 1.2
    open_bonus = weights["w_open"] * (-1.0 if curr.fret == 0 else 0.0)
    return float(move + string + span + high + open_bonus)
