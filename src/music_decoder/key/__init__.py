"""Key detection adapters for the public ``analyze()`` pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np

from music_decoder.errors import KeyDetectionError
from music_decoder.key.global_estimator import estimate_global_key
from music_decoder.logging_setup import get_logger
from music_decoder.types import KeyEstimate

_log = get_logger("key.adapter")


def _resolve_backend() -> str:
    """Key backend from ``config/hyperparameters.yaml`` (``"profile"`` if unreadable)."""
    try:
        from music_decoder.config.hyperparameters import load_hyperparameters

        return load_hyperparameters().key_detection.backend
    except Exception as e:  # pragma: no cover - defensive
        _log.warning("hyperparameters_load_failed", extra={"error": str(e)})
        return "profile"


def estimate_key(
    chroma: np.ndarray[Any, np.dtype[Any]],
    *,
    samples: np.ndarray[Any, np.dtype[Any]] | None = None,
    sr: int | None = None,
) -> KeyEstimate:
    """Estimate the global key.

    With ``key_detection.backend: cnn`` and raw *samples*, uses madmom's CNN
    key classifier. Otherwise (or if madmom fails) reduces (12, T) chroma to a
    12-dim pitch-class distribution, runs the Krumhansl-Kessler / Temperley
    consensus, and falls back to the Krumhansl-Kessler top pick if the two
    profiles disagree.
    """
    if samples is not None and sr is not None and _resolve_backend() == "cnn":
        from music_decoder.key import cnn

        try:
            return cnn.estimate_key_cnn(samples, sr)
        except Exception as e:  # optional model must never break analyze()
            _log.warning("key_cnn_failed_falling_back", extra={"error": str(e)})
    if chroma.ndim == 2:
        pc = chroma.mean(axis=1).astype(np.float64)
    else:
        pc = np.asarray(chroma, dtype=np.float64)
    if pc.shape[-1] != 12:
        raise KeyDetectionError(f"chroma must have 12 pitch classes, got shape {chroma.shape!r}")
    result = estimate_global_key(pc)
    if result.consensus_key is not None:
        return result.consensus_key
    # Profiles disagree → fall back to Krumhansl-Kessler top pick.
    kk_top = result.global_top3_per_profile.get("krumhansl_kessler", [])
    if not kk_top:
        raise KeyDetectionError("no key estimate could be produced")
    return kk_top[0]


__all__ = ["estimate_key"]
