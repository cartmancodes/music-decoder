from __future__ import annotations

from typing import Any

import librosa
import numpy as np


def compute_chroma_with_hpss(
    samples: np.ndarray[Any, np.dtype[np.float32]],
    *,
    sr: int,
    hpss_margin: float,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """CQT-based chroma after harmonic-percussive separation.

    Returns shape (12, T) with values in [0, 1].
    """
    y_h, _ = librosa.effects.hpss(samples.astype(float), margin=hpss_margin)
    chroma: np.ndarray[Any, np.dtype[np.float32]] = librosa.feature.chroma_cqt(y=y_h, sr=sr)
    chroma = chroma / (chroma.max(axis=0, keepdims=True) + 1e-12)
    return chroma.astype(np.float32)
