from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TimeSignatureResult:
    numerator: int
    denominator: int
    confidence: float
    assumed: bool


_CANDIDATES = (3, 4, 6, 5, 7)


def infer_time_signature(
    beat_strengths: np.ndarray[Any, np.dtype[np.float64]],
    *,
    min_confidence: float,
) -> TimeSignatureResult:
    if beat_strengths.size < 4:
        return TimeSignatureResult(numerator=4, denominator=4,
                                   confidence=0.0, assumed=True)
    s = np.asarray(beat_strengths, dtype=float)
    s = s - s.mean()
    n = len(s)
    autocorr = np.correlate(s, s, mode="full")[n - 1:] / max(np.var(s) * n, 1e-12)
    best = (4, 0.0)
    for k in _CANDIDATES:
        if k < len(autocorr):
            peak = autocorr[k]
            if peak > best[1]:
                best = (k, float(peak))
    numerator, peak = best
    confidence = float(max(0.0, min(1.0, peak)))
    if confidence < min_confidence:
        return TimeSignatureResult(numerator=4, denominator=4,
                                   confidence=confidence, assumed=True)
    return TimeSignatureResult(numerator=numerator, denominator=4,
                               confidence=confidence, assumed=False)
