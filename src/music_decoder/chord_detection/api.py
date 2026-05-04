# src/music_decoder/chord_detection/api.py
"""Public chord-detection entry point: chroma + beat grid -> ChordRecognitionResult."""
from __future__ import annotations

import statistics
from typing import Any

import numpy as np

from music_decoder.config.hyperparameters import ChordDetectionParams
from music_decoder.pipeline.contracts import (
    BeatGrid,
    ChordRecognitionResult,
)

from .recognize import (
    beat_sync_chroma,
    merge_segments,
    score_beats,
    viterbi_smooth,
)
from .templates import QUALITIES, label_index

_MIN_BEATS = 4


def detect_chords(
    *,
    chroma: np.ndarray[Any, np.dtype[Any]],
    sr: int,
    hop_length: int,
    beat_grid: BeatGrid,
    params: ChordDetectionParams,
) -> ChordRecognitionResult:
    # Guard against config drift: the hardcoded QUALITIES tuple in templates.py
    # is what actually drives the 48 chord templates. If hyperparameters.yaml
    # is edited to a different list, fail loudly rather than silently ignoring.
    if tuple(params.qualities) != QUALITIES:
        raise ValueError(
            f"chord_detection.qualities {params.qualities!r} does not match "
            f"the templated vocabulary {list(QUALITIES)!r}; v1 supports only "
            "the templated qualities. Edit templates.py to extend the vocabulary."
        )
    if beat_grid.beat_times_s.size < _MIN_BEATS:
        return ChordRecognitionResult(
            segments=[], median_confidence=0.0,
            skipped_reason="degenerate_beat_grid",
        )

    beat_chroma = beat_sync_chroma(
        chroma, sr=sr, hop_length=hop_length,
        beat_times_s=beat_grid.beat_times_s,
    )
    if beat_chroma.shape[1] == 0:
        return ChordRecognitionResult(
            segments=[], median_confidence=0.0,
            skipped_reason="audio_too_short_for_chord_window",
        )

    scores = score_beats(beat_chroma)               # (B, 49)
    # Force "N" where the max similarity is below threshold.
    n_idx = label_index("N", "")
    max_per_beat = scores.max(axis=1)
    below_threshold = max_per_beat < params.no_chord_threshold
    if below_threshold.any():
        # Boost N's score above all others on those beats so Viterbi picks it.
        scores[below_threshold] = 0.0
        scores[below_threshold, n_idx] = 1.0

    state_path = viterbi_smooth(scores, p_self=params.hmm_self_transition_prob)
    segments = merge_segments(
        state_path, beat_grid.beat_times_s, scores,
        min_segment_duration_s=params.min_segment_duration_s,
    )

    confidences = [s.confidence for s in segments] or [0.0]
    return ChordRecognitionResult(
        segments=segments,
        median_confidence=float(statistics.median(confidences)),
        skipped_reason=None,
    )
