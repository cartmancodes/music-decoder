from __future__ import annotations

from collections.abc import Iterable

from music_decoder.pipeline.contracts import TranscribedNote


def drop_short_notes(
    notes: Iterable[TranscribedNote], *, min_duration_s: float,
) -> list[TranscribedNote]:
    return [n for n in notes if (n.end_s - n.start_s) >= min_duration_s]
