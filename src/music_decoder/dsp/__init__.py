"""DSP primitives: beat tracking and chroma computation.

Public adapters with the simple ``(samples, sr)`` signature used by the
top-level :func:`music_decoder.api.analyze` orchestrator. The underlying
implementations accept additional hyperparameters; the adapters bind sane
defaults so callers can use them without threading config through.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from music_decoder.dsp.beats import track_beats as _track_beats_impl
from music_decoder.dsp.chroma import compute_chroma_with_hpss
from music_decoder.types import BeatGrid


def track_beats(
    samples: np.ndarray[Any, np.dtype[np.float32]], sr: int,
) -> BeatGrid:
    """Adapter: estimate tempo + beat grid from raw samples."""
    return _track_beats_impl(samples, sr=sr, start_bpm=120.0, tightness=100.0)


def compute_chroma(
    samples: np.ndarray[Any, np.dtype[np.float32]], sr: int,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Adapter: HPSS-based chroma matrix shape (12, T)."""
    return compute_chroma_with_hpss(samples, sr=sr, hpss_margin=8.0)


__all__ = ["track_beats", "compute_chroma"]
