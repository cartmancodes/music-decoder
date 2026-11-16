"""The v6 tune engine: a melody built from the statistics of real tunes.

Two halves, both driven by ``tune_stats.TuneStats`` and a passed-in
``numpy.random.Generator``:

* Rhythm plan (:func:`plan_rhythm`): each bar draws a real one-bar onset
  pattern for its role (open, body, cadence), weighted by how often that
  pattern follows the previous bar's and by closeness to a target density
  taken from the tempo. A' phrases often restate the matching bar of phrase A.
* Pitch (added in a later task): notes on those onsets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from music_decoder.compose.rhythm import form_labels, split_phrases
from music_decoder.compose.tune_stats import Pattern, Role, TuneStats

P_RESTATE_RHYTHM = 0.5  # chance an A' bar copies the matching bar of phrase A
DENSITY_SIGMA = 1.5  # width of the onsets-per-bar preference, in onsets
DENSITY_MIN, DENSITY_MAX = 3.0, 6.5  # clamp on the target onsets per bar
PAIR_MIN_COUNT = 20  # following bars a previous pattern needs for its own row to count


# --- Rhythm plan -----------------------------------------------------------------------


@dataclass(frozen=True)
class BarRhythm:
    pattern: Pattern
    role: Role
    restated_from: int | None  # the bar whose pattern this bar copied


def bar_roles(n_bars: int) -> list[Role]:
    """Role of each bar: a phrase opens on its first bar and cadences on its last
    (a one-bar phrase is a cadence); the rest are body."""
    roles: list[Role] = []
    for length in split_phrases(n_bars):
        for k in range(length):
            if k == length - 1:
                roles.append("cadence")
            elif k == 0:
                roles.append("open")
            else:
                roles.append("body")
    return roles


def _target_onsets(tempo_bpm: float, stats: TuneStats) -> float:
    target = stats.notes_per_second.mid * 240.0 / tempo_bpm
    return min(max(target, DENSITY_MIN), DENSITY_MAX)


def _draw_pattern(
    rng: np.random.Generator,
    role: Role,
    previous: Pattern | None,
    target: float,
    stats: TuneStats,
    *,
    need_slot_zero: bool,
) -> Pattern:
    counts = stats.rhythm[role]
    patterns = sorted(p for p in counts if not need_slot_zero or 0 in p)
    if not patterns:
        return (0,)
    total = sum(counts.values())  # shares are over the whole role, even when restricted
    row = stats.rhythm_pairs[role].get(previous) if previous is not None else None
    row_total = sum(row.values()) if row else 0
    use_row = row is not None and row_total >= PAIR_MIN_COUNT
    weights = np.empty(len(patterns))
    for i, pattern in enumerate(patterns):
        p = counts[pattern] / total
        if use_row and row is not None:
            p = 0.5 * (row.get(pattern, 0) / row_total) + 0.5 * p
        gap = len(pattern) - target
        weights[i] = p * math.exp(-(gap * gap) / (2.0 * DENSITY_SIGMA**2))
    if weights.sum() <= 0.0:
        weights = np.ones(len(patterns))
    return patterns[int(rng.choice(len(patterns), p=weights / weights.sum()))]


def plan_rhythm(
    n_bars: int,
    tempo_bpm: float,
    rng: np.random.Generator,
    stats: TuneStats,
    *,
    final_whole_note: bool = True,
) -> list[BarRhythm]:
    """One :class:`BarRhythm` per bar. With ``final_whole_note`` the last bar is a
    single note on the downbeat; otherwise it draws a cadence pattern like any bar."""
    roles = bar_roles(n_bars)
    lengths = split_phrases(n_bars)
    labels = form_labels(n_bars)
    starts = [sum(lengths[:i]) for i in range(len(lengths))]
    phrase_of = [i for i, length in enumerate(lengths) for _ in range(length)]
    target = _target_onsets(tempo_bpm, stats)

    plan: list[BarRhythm] = []
    for bar in range(n_bars):
        role = roles[bar]
        if final_whole_note and bar == n_bars - 1:
            plan.append(BarRhythm((0,), "cadence", None))
            continue
        phrase = phrase_of[bar]
        offset = bar - starts[phrase]
        if role != "cadence" and labels[phrase] == "A'":
            source = offset  # the matching bar of the first phrase
            if source < lengths[0] and roles[source] == role and rng.random() < P_RESTATE_RHYTHM:
                plan.append(BarRhythm(plan[source].pattern, role, source))
                continue
        previous = plan[-1].pattern if plan else None
        pattern = _draw_pattern(rng, role, previous, target, stats, need_slot_zero=bar == 0)
        plan.append(BarRhythm(pattern, role, None))
    return plan
