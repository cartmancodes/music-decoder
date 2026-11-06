"""One-bar rhythm templates and phrase planning for the melody generator.

Units are eighth notes; every template sums to 8 (one 4/4 bar). The phrase
*layout* depends only on the number of bars (see :func:`phrase_layout`), so
the musicality metrics can evaluate any melody, old or new, against the same
phrase boundaries and echo pairs the planner uses:

- antecedent phrases (even index) play ``[A, B, A, cadence]``; bar 3 echoes
  bar 1's intervals;
- the following consequent phrase reuses the antecedent's body rhythms and
  echoes its intervals bar for bar;
- every phrase ends on a cadence bar whose last note is long.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BODY_RHYTHMS: tuple[tuple[int, ...], ...] = (
    (2, 2, 2, 2),
    (2, 1, 1, 2, 2),
    (1, 1, 2, 2, 2),
    (2, 2, 1, 1, 2),
    (3, 1, 2, 2),
    (2, 2, 3, 1),
    (4, 2, 2),
    (1, 1, 1, 1, 2, 2),
    (2, 1, 1, 1, 1, 2),
    (3, 1, 3, 1),
    (4, 1, 1, 2),
)

# Each ends on a note of at least a half note (4 eighths).
CADENCE_RHYTHMS: tuple[tuple[int, ...], ...] = (
    (2, 2, 4),
    (4, 4),
    (3, 1, 4),
    (2, 1, 1, 4),
    (8,),
)

_FINAL_MIN_LAST_EIGHTHS = 4


@dataclass(frozen=True)
class PhraseLayout:
    """Bar-level phrase structure for a piece of ``n_bars`` bars."""

    phrase_lengths: tuple[int, ...]
    cadence_bars: tuple[int, ...]  # last bar of each phrase (absolute index)
    echo_pairs: tuple[tuple[int, int], ...]  # (source bar, echoing bar)


@dataclass(frozen=True)
class BarPlan:
    """Rhythm and role of one bar of melody."""

    rhythm: tuple[int, ...]  # note lengths in eighths, summing to 8
    cadence: bool  # last bar of a phrase
    final: bool  # last bar of the piece
    motif_from: int | None  # absolute index of the bar whose intervals to echo


def split_phrases(n_bars: int) -> list[int]:
    """Phrase lengths (bars) covering ``n_bars``: 4s, then a 2- or 3-bar tail."""
    if n_bars < 1:
        raise ValueError(f"n_bars must be >= 1, got {n_bars}")
    if n_bars <= 4:
        return [n_bars]
    phrases = [4] * (n_bars // 4)
    remainder = n_bars % 4
    if remainder == 1:
        phrases[-1] = 3
        phrases.append(2)
    elif remainder:
        phrases.append(remainder)
    return phrases


def phrase_layout(n_bars: int) -> PhraseLayout:
    """Phrase boundaries and motif echo pairs (independent of any randomness)."""
    lengths = split_phrases(n_bars)
    cadences: list[int] = []
    echoes: list[tuple[int, int]] = []
    start = 0
    antecedent_start, antecedent_body = 0, 0
    for index, length in enumerate(lengths):
        body = length - 1
        if index % 2 == 0:
            if length == 4:
                echoes.append((start, start + 2))
            antecedent_start, antecedent_body = start, body
        else:
            echoes.extend(
                (antecedent_start + i, start + i) for i in range(min(body, antecedent_body))
            )
        cadences.append(start + length - 1)
        start += length
    return PhraseLayout(tuple(lengths), tuple(cadences), tuple(echoes))


def _draw(rng: np.random.Generator, templates: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    return templates[int(rng.integers(len(templates)))]


def plan_bars(n_bars: int, rng: np.random.Generator) -> list[BarPlan]:
    """One ``BarPlan`` per bar, in order, following :func:`phrase_layout`."""
    layout = phrase_layout(n_bars)
    echo_of = {target: source for source, target in layout.echo_pairs}
    rhythms: list[tuple[int, ...]] = []
    plans: list[BarPlan] = []
    start = 0
    for index, length in enumerate(layout.phrase_lengths):
        final = index == len(layout.phrase_lengths) - 1
        if index % 2 == 0:
            a = _draw(rng, BODY_RHYTHMS)
            b = _draw(rng, BODY_RHYTHMS)
            fresh = [a, b, a]
        for i in range(length - 1):
            bar = start + i
            source = echo_of.get(bar)
            if source is not None:
                rhythm = rhythms[source]
            elif index % 2 == 0:
                rhythm = fresh[i]
            else:
                rhythm = _draw(rng, BODY_RHYTHMS)
            rhythms.append(rhythm)
            plans.append(BarPlan(rhythm=rhythm, cadence=False, final=False, motif_from=source))
        cadences = tuple(
            c for c in CADENCE_RHYTHMS if not final or c[-1] >= _FINAL_MIN_LAST_EIGHTHS
        )
        rhythm = _draw(rng, cadences)
        rhythms.append(rhythm)
        plans.append(BarPlan(rhythm=rhythm, cadence=True, final=final, motif_from=None))
        start += length
    return plans
