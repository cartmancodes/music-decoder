"""Chord recognition adapters for the public ``analyze()`` pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from music_decoder.types import BeatGrid, ChordSegment


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
    result = detect_chords(
        chroma=chroma,
        sr=sr,
        hop_length=512,
        beat_grid=beat_grid,
        params=params,
    )
    return tuple(result.segments)


__all__ = ["recognize_chords"]
