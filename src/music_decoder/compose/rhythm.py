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

:func:`form_labels` names each phrase's role in the piece's form (A, A' or
B, v5 spec 5.3); with ``contrast_b`` the planner gives the B phrase body
rhythms of its own.
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
_B_PHRASE = 2  # the B phrase of A A' B A'
_CONTRAST_REDRAWS = 5


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


def form_labels(n_bars: int) -> tuple[str, ...]:
    """The form role of each phrase of :func:`phrase_layout`: A and A' alternate,
    and the third phrase, when there is one, is B (so four phrases are
    A A' B A' and two are A A')."""
    count = len(split_phrases(n_bars))
    return tuple(
        "B" if index == _B_PHRASE else "A" if index % 2 == 0 else "A'" for index in range(count)
    )


def _draw(rng: np.random.Generator, templates: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    return templates[int(rng.integers(len(templates)))]


def _fresh_bodies(rng: np.random.Generator) -> list[tuple[int, ...]]:
    """An antecedent's body rhythms: ``[A, B, A]``."""
    a = _draw(rng, BODY_RHYTHMS)
    b = _draw(rng, BODY_RHYTHMS)
    return [a, b, a]


def plan_bars(n_bars: int, rng: np.random.Generator, contrast_b: bool = False) -> list[BarPlan]:
    """One ``BarPlan`` per bar, in order, following :func:`phrase_layout`.

    With ``contrast_b`` the B phrase (see :func:`form_labels`) redraws its
    fresh body rhythms, up to ``_CONTRAST_REDRAWS`` times, while any it plays
    is one of phrase A's.
    """
    layout = phrase_layout(n_bars)
    echo_of = {target: source for source, target in layout.echo_pairs}
    rhythms: list[tuple[int, ...]] = []
    plans: list[BarPlan] = []
    phrase_a: set[tuple[int, ...]] = set()
    start = 0
    for index, length in enumerate(layout.phrase_lengths):
        final = index == len(layout.phrase_lengths) - 1
        if index % 2 == 0:
            fresh = _fresh_bodies(rng)
            if index == 0:
                phrase_a = set(fresh[: length - 1])
            elif contrast_b and index == _B_PHRASE:
                for _ in range(_CONTRAST_REDRAWS):
                    if not set(fresh[: length - 1]) & phrase_a:
                        break
                    fresh = _fresh_bodies(rng)
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
