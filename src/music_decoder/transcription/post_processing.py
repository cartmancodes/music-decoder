from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from scipy.signal import medfilt

from music_decoder.config.hyperparameters import PostProcessingParams
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


def median_filter_pitch_contour(
    contour: np.ndarray[Any, np.dtype[np.float64]], *, window: int
) -> np.ndarray[Any, np.dtype[np.float64]]:
    if window % 2 == 0:
        raise ValueError("median filter window must be odd")
    result: np.ndarray[Any, np.dtype[np.float64]] = medfilt(
        np.asarray(contour, dtype=float), kernel_size=window
    )
    return result


def snap_to_beats(
    notes: Iterable[TranscribedNote],
    *,
    beats: np.ndarray[Any, np.dtype[np.float64]],
    confidence_threshold: float,
    max_snap_s: float,
) -> list[TranscribedNote]:
    if len(beats) == 0:
        return list(notes)
    beat_arr = np.asarray(beats)
    out: list[TranscribedNote] = []
    for n in notes:
        if n.confidence < confidence_threshold:
            out.append(n)
            continue
        diffs = np.abs(beat_arr - n.start_s)
        idx = int(np.argmin(diffs))
        if diffs[idx] <= max_snap_s:
            shift = float(beat_arr[idx]) - n.start_s
            out.append(TranscribedNote(
                start_s=float(beat_arr[idx]), end_s=n.end_s + shift,
                pitch=n.pitch, velocity=n.velocity, confidence=n.confidence,
            ))
        else:
            out.append(n)
    return out


def apply_post_processing(
    notes: Iterable[TranscribedNote],
    *,
    params: PostProcessingParams,
    beats: np.ndarray[Any, np.dtype[np.float64]] | None = None,
) -> list[TranscribedNote]:
    pipeline = list(notes)
    pipeline = drop_short_notes(pipeline, min_duration_s=params.min_note_duration_s)
    pipeline = merge_same_pitch(pipeline, gap_s=params.same_pitch_merge_gap_s)
    if beats is not None and len(beats) > 0:
        pipeline = snap_to_beats(
            pipeline, beats=beats,
            confidence_threshold=params.rhythmic_snap_confidence_threshold,
            max_snap_s=0.05,
        )
    return pipeline
