# src/music_decoder/tab_assignment/candidates.py
from __future__ import annotations

from itertools import product

from music_decoder.tabs.tuning import Tuning
from music_decoder.types import TabPosition

_MAX_CHORD_SPAN = 5


def note_candidates(
    *,
    pitch: int,
    tuning: Tuning,
    max_fret: int,
) -> list[TabPosition]:
    out: list[TabPosition] = []
    for s, open_p in enumerate(tuning.open_pitches):
        f = pitch - open_p
        if 0 <= f <= max_fret:
            out.append(TabPosition(string=s, fret=f))
    return out


def chord_combinations(
    pitches: list[int],
    *,
    tuning: Tuning,
    max_fret: int,
) -> list[tuple[TabPosition, ...]]:
    per_pitch = [note_candidates(pitch=p, tuning=tuning, max_fret=max_fret) for p in pitches]
    if any(len(c) == 0 for c in per_pitch):
        return []
    combos: list[tuple[TabPosition, ...]] = []
    for combo in product(*per_pitch):
        strings = [p.string for p in combo]
        if len(set(strings)) != len(strings):
            continue
        frets = [p.fret for p in combo if p.fret > 0]
        if frets and (max(frets) - min(frets) > _MAX_CHORD_SPAN):
            continue
        combos.append(combo)
    return combos
