# src/music_decoder/tab_assignment/assigner.py
from __future__ import annotations

from collections.abc import Iterable

from music_decoder.tabs.tuning import Tuning
from music_decoder.types import (
    TabAssignmentResult,
    TabbedNote,
    TranscribedNote,
)

from .astar import Group, astar_min_cost_path
from .candidates import chord_combinations, note_candidates

# Notes whose onsets fall within this window of a group's first onset form one
# chord (covers strums). Chosen on GuitarSet dev: overlap-based grouping
# 0.584 / 0.550 (tab_gt / tab_e2e) -> 50 ms onset window 0.614 / 0.587.
_CHORD_ONSET_WINDOW_S = 0.05


def _group_simultaneous(
    notes: list[TranscribedNote],
) -> list[list[TranscribedNote]]:
    """Group notes struck together into chord states.

    A note joins the current group when its onset is within
    ``_CHORD_ONSET_WINDOW_S`` of the group's first onset. Notes struck while
    an earlier note is still ringing start a new group — fusing them by
    overlap (v2) merged re-attacks of the same pitch into one "chord" that
    no fingering could satisfy.
    """
    if not notes:
        return []
    sorted_notes = sorted(notes, key=lambda n: (n.start_s, n.pitch))
    groups: list[list[TranscribedNote]] = [[sorted_notes[0]]]
    for n in sorted_notes[1:]:
        if n.start_s - groups[-1][0].start_s <= _CHORD_ONSET_WINDOW_S:
            groups[-1].append(n)
        else:
            groups.append([n])
    return groups


def assign_tab(
    notes: Iterable[TranscribedNote],
    *,
    tuning: Tuning,
    weights: dict[str, float],
    max_fret: int,
) -> TabAssignmentResult:
    note_list = list(notes)
    if not note_list:
        return TabAssignmentResult(
            tabbed_notes=[],
            tuning=tuning,
            total_cost=0.0,
            notes_dropped=[],
        )
    dropped: list[tuple[TranscribedNote, str]] = []
    # Drop notes no string can play *before* grouping, so a single out-of-range
    # artifact (e.g. a sub-bass partial) can't make its whole chord unsatisfiable.
    playable: list[TranscribedNote] = []
    for n in note_list:
        if note_candidates(pitch=n.pitch, tuning=tuning, max_fret=max_fret):
            playable.append(n)
        else:
            dropped.append((n, "out_of_range_for_tuning"))
    groups = _group_simultaneous(playable)

    # Cap each chord group to 6 notes (highest confidence), per spec §6.1
    clamped_groups: list[list[TranscribedNote]] = []
    for chord_group in groups:
        if len(chord_group) > 6:
            sorted_by_conf = sorted(chord_group, key=lambda n: n.confidence, reverse=True)
            keep, drop = sorted_by_conf[:6], sorted_by_conf[6:]
            clamped_groups.append(sorted(keep, key=lambda n: n.pitch))
            for n in drop:
                dropped.append((n, "chord_too_dense_capped_to_6"))
        else:
            clamped_groups.append(chord_group)
    groups = clamped_groups

    candidate_groups: list[Group] = []
    flat_group_to_notes: list[list[TranscribedNote]] = []
    for chord_group in groups:
        if len(chord_group) == 1:
            cands = note_candidates(
                pitch=chord_group[0].pitch,
                tuning=tuning,
                max_fret=max_fret,
            )
            if not cands:
                dropped.append((chord_group[0], "out_of_range_for_tuning"))
                continue
            candidate_groups.append(Group([(c,) for c in cands]))
            flat_group_to_notes.append(chord_group)
        else:
            combos = chord_combinations(
                pitches=[n.pitch for n in chord_group],
                tuning=tuning,
                max_fret=max_fret,
            )
            if not combos:
                for n in chord_group:
                    dropped.append((n, "unsatisfiable_chord"))
                continue
            candidate_groups.append(Group(list(combos)))
            flat_group_to_notes.append(chord_group)

    if not candidate_groups:
        return TabAssignmentResult(
            tabbed_notes=[],
            tuning=tuning,
            total_cost=0.0,
            notes_dropped=dropped,
        )

    path, cost = astar_min_cost_path(
        candidate_groups,
        weights=weights,
        hand_anchor_window=2,
    )
    if cost == float("inf"):
        for chord_group in flat_group_to_notes:
            for n in chord_group:
                dropped.append((n, "no_valid_path"))
        return TabAssignmentResult(
            tabbed_notes=[],
            tuning=tuning,
            total_cost=float("inf"),
            notes_dropped=dropped,
        )

    tabbed: list[TabbedNote] = []
    for chord_group, candidate in zip(flat_group_to_notes, path, strict=True):
        for n, pos in zip(chord_group, candidate, strict=True):
            tabbed.append(
                TabbedNote(
                    note=n,
                    position=pos,
                    cost_breakdown={"path_total": cost},
                )
            )
    return TabAssignmentResult(
        tabbed_notes=tabbed,
        tuning=tuning,
        total_cost=cost,
        notes_dropped=dropped,
    )
