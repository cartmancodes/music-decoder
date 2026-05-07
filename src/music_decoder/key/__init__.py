"""Key detection adapters for the public ``analyze()`` pipeline."""
from __future__ import annotations

from typing import Any

import numpy as np

from music_decoder.errors import KeyDetectionError
from music_decoder.key.global_estimator import estimate_global_key
from music_decoder.types import KeyEstimate


def estimate_key(
    chroma: np.ndarray[Any, np.dtype[Any]],
) -> KeyEstimate:
    """Estimate a single consensus :class:`KeyEstimate` from a chroma matrix.

    Reduces (12, T) chroma to a 12-dim pitch-class distribution, runs the
    Krumhansl-Kessler / Temperley consensus, and returns the consensus key.
    Falls back to the Krumhansl-Kessler top pick if the two profiles disagree.
    """
    if chroma.ndim == 2:
        pc = chroma.mean(axis=1).astype(np.float64)
    else:
        pc = np.asarray(chroma, dtype=np.float64)
    if pc.shape[-1] != 12:
        raise KeyDetectionError(
            f"chroma must have 12 pitch classes, got shape {chroma.shape!r}"
        )
    result = estimate_global_key(pc)
    if result.consensus_key is not None:
        return result.consensus_key
    # Profiles disagree → fall back to Krumhansl-Kessler top pick.
    kk_top = result.global_top3_per_profile.get("krumhansl_kessler", [])
    if not kk_top:
        raise KeyDetectionError("no key estimate could be produced")
    return kk_top[0]


__all__ = ["estimate_key"]
