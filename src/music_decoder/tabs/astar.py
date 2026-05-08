# src/music_decoder/tab_assignment/astar.py
from __future__ import annotations

import heapq
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from music_decoder.types import TabPosition

from .cost import chord_collides, chord_span_penalty, transition_cost
from .heuristic import remaining_high_fret_penalty

Candidate = tuple[TabPosition, ...]  # singleton for a note, multi for a chord


@dataclass(frozen=True)
class Group:
    candidates: list[Candidate]


def _representative(candidate: Candidate) -> TabPosition:
    """Lowest-fret member of a chord state, or the singleton position."""
    return min(candidate, key=lambda p: p.fret)


def _state_extra_cost(candidate: Candidate, *, weights: dict[str, float]) -> float:
    if len(candidate) <= 1:
        return 0.0
    if chord_collides(candidate):
        return float("inf")
    return weights["w_chord_intra"] * chord_span_penalty(candidate)


def astar_min_cost_path(
    groups: Sequence[Group],
    *,
    weights: dict[str, float],
    hand_anchor_window: int,
) -> tuple[list[Candidate], float]:
    if not groups:
        return [], 0.0
    n = len(groups)
    # Each state: (group_index, candidate_index, recent_anchor_frets_tuple).
    heap: list[tuple[float, int, int, tuple[int, ...]]] = []
    parents: dict[tuple[int, int, tuple[int, ...]], tuple[int, int, tuple[int, ...]]] = {}
    g_score: dict[tuple[int, int, tuple[int, ...]], float] = {}
    for idx, candidate in enumerate(groups[0].candidates):
        extra = _state_extra_cost(candidate, weights=weights)
        rep = _representative(candidate)
        if extra == float("inf"):
            continue
        init_history: tuple[int, ...] = (rep.fret,)
        state: tuple[int, int, tuple[int, ...]] = (0, idx, init_history)
        # Include position cost for first group (high-fret and open-string terms only)
        pos_cost = transition_cost(
            prev=rep,
            curr=rep,
            weights=weights,
            hand_anchor=float(rep.fret),
        )
        g = extra + pos_cost
        g_score[state] = g
        heuristic = remaining_high_fret_penalty(
            [grp.candidates for grp in groups[1:]],
            weights=weights,
        )
        heapq.heappush(heap, (g + heuristic, *state))

    if not heap:
        return [], 0.0

    goal_state: tuple[int, int, tuple[int, ...]] | None = None
    while heap:
        heap_item = heapq.heappop(heap)
        gi, ci = heap_item[1], heap_item[2]
        cur_history: tuple[int, ...] = heap_item[3]
        state = (gi, ci, cur_history)
        if gi == n - 1:
            goal_state = state
            break
        candidate = groups[gi].candidates[ci]
        rep = _representative(candidate)
        for next_idx, next_candidate in enumerate(groups[gi + 1].candidates):
            next_extra = _state_extra_cost(next_candidate, weights=weights)
            if next_extra == float("inf"):
                continue
            next_rep = _representative(next_candidate)
            anchor_window = [*list(cur_history), next_rep.fret]
            anchor_window = anchor_window[-hand_anchor_window:]
            anchor = float(statistics.median(anchor_window))
            step = transition_cost(
                prev=rep,
                curr=next_rep,
                weights=weights,
                hand_anchor=anchor,
            )
            new_g = g_score[state] + step + next_extra
            next_state = (gi + 1, next_idx, tuple(anchor_window))
            if new_g < g_score.get(next_state, float("inf")):
                g_score[next_state] = new_g
                parents[next_state] = state
                remaining = [grp.candidates for grp in groups[gi + 2 :]]
                heapq.heappush(
                    heap,
                    (
                        new_g + remaining_high_fret_penalty(remaining, weights=weights),
                        *next_state,
                    ),
                )

    if goal_state is None:
        return [], float("inf")

    # Reconstruct path
    path_states: list[tuple[int, int, tuple[int, ...]]] = [goal_state]
    while path_states[-1] in parents:
        path_states.append(parents[path_states[-1]])
    path_states.reverse()
    path = [groups[gi].candidates[ci] for gi, ci, _ in path_states]
    return path, g_score[goal_state]
