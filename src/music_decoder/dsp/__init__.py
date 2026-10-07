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
from music_decoder.logging_setup import get_logger
from music_decoder.types import BeatGrid

_log = get_logger("dsp.beats")


def _resolve_backend() -> str:
    """Beat backend from ``config/hyperparameters.yaml`` (``"librosa"`` if unreadable)."""
    try:
        from music_decoder.config.hyperparameters import load_hyperparameters

        return load_hyperparameters().beat_tracking.backend
    except Exception as e:  # pragma: no cover - defensive
        _log.warning("hyperparameters_load_failed", extra={"error": str(e)})
        return "librosa"


def track_beats(
    samples: np.ndarray[Any, np.dtype[np.float32]],
    sr: int,
) -> BeatGrid:
    """Adapter: estimate tempo + beat grid from raw samples.

    With ``beat_tracking.backend: madmom`` uses the RNN+DBN downbeat tracker
    (real downbeats and 3/4 vs 4/4); falls back to librosa if madmom fails or
    finds fewer than four beats.
    """
    rms = float(np.sqrt(np.mean(samples**2))) if samples.size else 0.0
    if rms >= 1e-5 and _resolve_backend() == "madmom":
        from music_decoder.dsp import beats_madmom

        try:
            grid = beats_madmom.track_beats_madmom(samples, sr)
        except Exception as e:  # optional model must never break analyze()
            _log.warning("beats_madmom_failed_falling_back", extra={"error": str(e)})
            grid = None
        if grid is not None:
            return grid
    return _track_beats_impl(samples, sr=sr, start_bpm=120.0, tightness=100.0)


def compute_chroma(
    samples: np.ndarray[Any, np.dtype[np.float32]],
    sr: int,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Adapter: HPSS-based chroma matrix shape (12, T)."""
    return compute_chroma_with_hpss(samples, sr=sr, hpss_margin=8.0)


__all__ = ["compute_chroma", "track_beats"]
