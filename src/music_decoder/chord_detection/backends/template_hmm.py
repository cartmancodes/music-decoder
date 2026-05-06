"""Template-matching + 49-state HMM Viterbi backend.

This is the v1 algorithm extracted into a backend class so it can sit behind
the same ``ChordBackend`` protocol as the madmom backend. Behaviour is
unchanged from the previous ``api.detect_chords`` implementation.
"""
from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

import numpy as np

from music_decoder.config.hyperparameters import ChordDetectionParams
from music_decoder.pipeline.contracts import BeatGrid, ChordRecognitionResult

from ..recognize import (
    beat_sync_chroma,
    merge_segments,
    score_beats,
    viterbi_smooth,
)
from ..templates import label_index

_MIN_BEATS = 4


class TemplateHmmBackend:
    """Chroma → 49-template scoring → Viterbi smoothing → segment merge."""

    def detect(
        self,
        *,
        chroma: np.ndarray[Any, np.dtype[Any]],
        sr: int,
        hop_length: int,
        beat_grid: BeatGrid,
        params: ChordDetectionParams,
        audio_path: Path | None,  # unused; kept for protocol parity
    ) -> ChordRecognitionResult:
        if beat_grid.beat_times_s.size < _MIN_BEATS:
            return ChordRecognitionResult(
                segments=[],
                median_confidence=0.0,
                skipped_reason="degenerate_beat_grid",
            )

        beat_chroma = beat_sync_chroma(
            chroma, sr=sr, hop_length=hop_length,
            beat_times_s=beat_grid.beat_times_s,
        )
        if beat_chroma.shape[1] == 0:
            return ChordRecognitionResult(
                segments=[],
                median_confidence=0.0,
                skipped_reason="audio_too_short_for_chord_window",
            )

        scores = score_beats(beat_chroma)  # (B, 49)
        # Force "N" where the max similarity is below threshold.
        n_idx = label_index("N", "")
        max_per_beat = scores.max(axis=1)
        below_threshold = max_per_beat < params.no_chord_threshold
        if below_threshold.any():
            scores[below_threshold] = 0.0
            scores[below_threshold, n_idx] = 1.0

        state_path = viterbi_smooth(
            scores, p_self=params.hmm_self_transition_prob,
        )
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
