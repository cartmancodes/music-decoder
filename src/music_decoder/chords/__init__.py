"""Chord recognition adapters for the public ``analyze()`` pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from music_decoder.types import BeatGrid, ChordSegment

# Minimum beats the template-HMM backend needs (matches its internal threshold).
_MIN_BEATS = 4


def _synthetic_beat_grid(duration_s: float) -> BeatGrid:
    """Build a uniform 4-beat grid spanning *duration_s*.

    Used when the real beat tracker degenerates (silence, sustained chords,
    very short clips) so the chord recognizer can still emit at least one
    segment instead of bailing out with ``degenerate_beat_grid``.
    """
    duration_s = max(float(duration_s), 0.25)
    # Five edges = four equal-width beats spanning [0, duration_s].
    edges = np.linspace(0.0, duration_s, _MIN_BEATS + 1, dtype=float)
    return BeatGrid(
        tempo_bpm=60.0 * _MIN_BEATS / duration_s,
        beat_times_s=edges,
        downbeat_times_s=np.array([edges[0]], dtype=float),
        ts_numerator=4,
        ts_denominator=4,
        ts_confidence=0.0,
        ts_assumed=True,
    )


def recognize_chords(
    samples: np.ndarray[Any, np.dtype[np.float32]],
    sr: int,
    *,
    beat_grid: BeatGrid,
) -> Sequence[ChordSegment]:
    """Run beat-synchronous chord recognition on raw samples.

    Adapter that:
      1. Computes HPSS chroma at the default hop length (512).
      2. Calls the chord-detection API with default :class:`ChordDetectionParams`.
      3. Unwraps :class:`ChordRecognitionResult` to the bare segment tuple.

    Falls back to a synthetic uniform beat grid spanning the audio when the
    supplied grid is degenerate (fewer than the backend-required beats); this
    keeps short / static / sustained-chord clips from silently producing zero
    segments.
    """
    # Lazy imports: ``music_decoder.chords.api`` pulls in the template-HMM
    # backend on import; keep ``import music_decoder.chords`` itself cheap.
    from music_decoder.chords.api import detect_chords
    from music_decoder.chords.templates import QUALITIES
    from music_decoder.config.hyperparameters import ChordDetectionParams
    from music_decoder.dsp.chroma import compute_chroma_with_hpss

    chroma = compute_chroma_with_hpss(samples, sr=sr, hpss_margin=8.0)
    params = ChordDetectionParams(
        qualities=list(QUALITIES),
        hmm_self_transition_prob=0.9,
        no_chord_threshold=0.3,
        min_segment_duration_s=0.25,
        backend="template_hmm",
    )

    # If the beat grid degenerates, swap in a synthetic 4-beat grid spanning
    # the audio. This costs us beat-synchronous accuracy on the segmentation
    # boundaries but lets the recognizer emit a label for short/static clips.
    if beat_grid.beat_times_s.size < _MIN_BEATS:
        duration_s = float(samples.shape[-1]) / float(sr) if samples.size else 1.0
        beat_grid = _synthetic_beat_grid(duration_s)

    result = detect_chords(
        chroma=chroma,
        sr=sr,
        hop_length=512,
        beat_grid=beat_grid,
        params=params,
    )

    if result.skipped_reason is not None and not result.segments:
        # Final safety net: if the backend still bailed (e.g. chroma window
        # too short), emit a single no-chord segment so downstream consumers
        # always see a well-formed progression.
        duration_s = float(samples.shape[-1]) / float(sr) if samples.size else 0.0
        return (
            ChordSegment(
                start_s=0.0,
                end_s=max(duration_s, 0.25),
                root="N",
                quality="no-chord",
                confidence=0.0,
            ),
        )

    return tuple(result.segments)


__all__ = ["recognize_chords"]
