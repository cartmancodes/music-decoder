from __future__ import annotations

from typing import Any, Literal

import numpy as np

from music_decoder.types import KeyEstimate

from .profiles import (
    KEYS,
    KRUMHANSL_KESSLER_MAJOR,
    KRUMHANSL_KESSLER_MINOR,
    TEMPERLEY_MAJOR,
    TEMPERLEY_MINOR,
)

_NdFloat = np.ndarray[Any, np.dtype[np.float64]]

_PROFILES: dict[str, tuple[_NdFloat, _NdFloat]] = {
    "krumhansl_kessler": (np.array(KRUMHANSL_KESSLER_MAJOR),
                          np.array(KRUMHANSL_KESSLER_MINOR)),
    "temperley":         (np.array(TEMPERLEY_MAJOR),
                          np.array(TEMPERLEY_MINOR)),
}


def _pearson(x: _NdFloat, y: _NdFloat) -> float:
    xc = x - x.mean()
    yc = y - y.mean()
    denom = np.sqrt((xc ** 2).sum() * (yc ** 2).sum())
    if denom == 0:
        return 0.0
    return float((xc * yc).sum() / denom)


def correlate_against_profiles(
    pitch_class_distribution: _NdFloat,
    *,
    profile: Literal["krumhansl_kessler", "temperley"],
) -> list[KeyEstimate]:
    if profile not in _PROFILES:
        raise ValueError(f"unknown profile {profile!r}")
    major, minor = _PROFILES[profile]
    pc = np.asarray(pitch_class_distribution, dtype=float)
    if pc.sum() > 0:
        pc = pc / pc.sum()
    corrs: list[tuple[str, Literal["major", "minor"], float]] = []
    for i in range(12):
        corrs.append((KEYS[i], "major", _pearson(np.roll(major, i), pc)))
        corrs.append((KEYS[i], "minor", _pearson(np.roll(minor, i), pc)))
    sorted_corrs = sorted(corrs, key=lambda x: x[2], reverse=True)
    top = sorted_corrs[0][2]
    second = sorted_corrs[1][2] if len(sorted_corrs) > 1 else top
    return [
        KeyEstimate(
            tonic=tonic, mode=mode, profile=profile,
            correlation=corr, margin=corr - second,
        )
        for (tonic, mode, corr) in sorted_corrs
    ]


def top_k_estimates(
    pitch_class_distribution: _NdFloat,
    *,
    profile: Literal["krumhansl_kessler", "temperley"],
    k: int = 3,
) -> list[KeyEstimate]:
    return correlate_against_profiles(pitch_class_distribution, profile=profile)[:k]
