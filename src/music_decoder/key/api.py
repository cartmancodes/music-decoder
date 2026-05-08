from __future__ import annotations

from typing import Any

import numpy as np

from music_decoder.dsp.chroma import compute_chroma_with_hpss
from music_decoder.types import KeyDetectionResult

from .global_estimator import estimate_global_key
from .windowed import detect_windowed_keys

_DEFAULT_HOP_LENGTH = 512


def detect_key(
    samples: np.ndarray[Any, np.dtype[np.float32]],
    *,
    sr: int,
    hpss_margin: float,
    segment_length_s: float,
    hop_s: float,
) -> KeyDetectionResult:
    chroma = compute_chroma_with_hpss(samples, sr=sr, hpss_margin=hpss_margin)
    pc = chroma.mean(axis=1)
    global_result = estimate_global_key(pc)
    windowed = detect_windowed_keys(
        chroma,
        sr=sr,
        hop_length=_DEFAULT_HOP_LENGTH,
        segment_length_s=segment_length_s,
        hop_s=hop_s,
    )
    return KeyDetectionResult(
        global_top3_per_profile=global_result.global_top3_per_profile,
        consensus_key=global_result.consensus_key,
        windowed_segments=windowed,
        confidence=global_result.confidence,
    )
