from __future__ import annotations

from typing import Any

import numpy as np

from music_decoder.types import KeyDetectionResult

from .ks import top_k_estimates


def estimate_global_key(
    pitch_class_distribution: np.ndarray[Any, np.dtype[np.float64]],
) -> KeyDetectionResult:
    kk_top3 = top_k_estimates(pitch_class_distribution,
                              profile="krumhansl_kessler", k=3)
    temp_top3 = top_k_estimates(pitch_class_distribution,
                                profile="temperley", k=3)
    profile_top1 = {
        "krumhansl_kessler": kk_top3[0],
        "temperley": temp_top3[0],
    }
    if (profile_top1["krumhansl_kessler"].tonic == profile_top1["temperley"].tonic
            and profile_top1["krumhansl_kessler"].mode == profile_top1["temperley"].mode):
        consensus = profile_top1["krumhansl_kessler"]
        # confidence = mean margin across profiles, clipped to [0, 1]
        confidence = float(np.clip(
            (kk_top3[0].margin + temp_top3[0].margin) / 2 + 0.5, 0.0, 1.0,
        ))
    else:
        consensus = None
        confidence = float(np.clip(
            (kk_top3[0].margin + temp_top3[0].margin) / 4, 0.0, 0.5,
        ))
    return KeyDetectionResult(
        global_top3_per_profile={
            "krumhansl_kessler": kk_top3, "temperley": temp_top3,
        },
        consensus_key=consensus,
        windowed_segments=[],
        confidence=confidence,
    )
