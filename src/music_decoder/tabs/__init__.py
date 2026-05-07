"""Tab assignment adapters for the public ``analyze()`` pipeline."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Sequence

from music_decoder.tabs.assigner import assign_tab as _assign_tab_impl
from music_decoder.tabs.tuning import STANDARD_EADGBE, Tuning
from music_decoder.types import TabbedNote, TranscribedNote


_DEFAULT_WEIGHTS: dict[str, float] = {
    "fret": 0.3,
    "string": 0.1,
    "hand_motion": 0.4,
    "open_string_bonus": -0.2,
    "high_fret_penalty": 0.5,
    "string_jump": 0.3,
}


def assign_tabs(
    notes: Iterable[TranscribedNote],
    *,
    tuning: Tuning = STANDARD_EADGBE,
    max_fret: int = 22,
) -> Sequence[TabbedNote]:
    """Assign tab positions for a stream of notes.

    Adapter that calls :func:`assign_tab` with default weight + fret-cap
    hyperparameters and returns just the tabbed-note sequence.
    """
    result = _assign_tab_impl(
        notes, tuning=tuning, weights=_DEFAULT_WEIGHTS, max_fret=max_fret,
    )
    return tuple(result.tabbed_notes)


__all__ = ["assign_tabs"]
