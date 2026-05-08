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


def _group_simultaneous(
    notes: list[TranscribedNote],
) -> list[list[TranscribedNote]]:
    """Group notes whose intervals overlap into chord groups.

    Two notes belong to the same chord state if either's start_s falls within
    the time span of any current group member. This is conservative — slightly
    overlapping notes (e.g. legato) get grouped even if they aren't true chords.
    """
    if not notes:
        return []
    sorted_notes = sorted(notes, key=lambda n: (n.start_s, n.pitch))
    groups: list[list[TranscribedNote]] = [[sorted_notes[0]]]
    for n in sorted_notes[1:]:
        current_max_end = max(m.end_s for m in groups[-1])
        if n.start_s < current_max_end:
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
    groups = _group_simultaneous(note_list)
    dropped: list[tuple[TranscribedNote, str]] = []

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
