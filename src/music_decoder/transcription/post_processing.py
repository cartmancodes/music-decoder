from __future__ import annotations

from collections.abc import Iterable

from music_decoder.pipeline.contracts import TranscribedNote


def drop_short_notes(
    notes: Iterable[TranscribedNote], *, min_duration_s: float,
) -> list[TranscribedNote]:
    return [n for n in notes if (n.end_s - n.start_s) >= min_duration_s]


def merge_same_pitch(
    notes: Iterable[TranscribedNote], *, gap_s: float,
) -> list[TranscribedNote]:
    sorted_notes = sorted(notes, key=lambda n: (n.pitch, n.start_s))
    out: list[TranscribedNote] = []
    for n in sorted_notes:
        if out and out[-1].pitch == n.pitch and (n.start_s - out[-1].end_s) <= gap_s:
            prev = out[-1]
            total_dur = (prev.end_s - prev.start_s) + (n.end_s - n.start_s)
            mean_conf = (
                prev.confidence * (prev.end_s - prev.start_s)
                + n.confidence * (n.end_s - n.start_s)
            ) / total_dur
            out[-1] = TranscribedNote(
                start_s=prev.start_s, end_s=n.end_s, pitch=prev.pitch,
                velocity=max(prev.velocity, n.velocity),
                confidence=mean_conf,
            )
        else:
            out.append(n)
    return sorted(out, key=lambda n: n.start_s)
