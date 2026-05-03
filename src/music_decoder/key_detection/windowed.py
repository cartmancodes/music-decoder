from __future__ import annotations

from typing import Any

import numpy as np

from music_decoder.pipeline.contracts import KeyEstimate

from .ks import top_k_estimates


def detect_windowed_keys(
    chroma: np.ndarray[Any, np.dtype[np.float32]],  # shape (12, T)
    *,
    sr: int,
    hop_length: int,
    segment_length_s: float,
    hop_s: float,
) -> list[tuple[float, float, KeyEstimate]]:
    if chroma.size == 0:
        return []
    frames_per_second = sr / hop_length
    seg_frames = max(1, round(segment_length_s * frames_per_second))
    hop_frames = max(1, round(hop_s * frames_per_second))
    n_frames = chroma.shape[1]
    segments: list[tuple[float, float, KeyEstimate]] = []
    for start_frame in range(0, max(1, n_frames - seg_frames + 1), hop_frames):
        end_frame = min(n_frames, start_frame + seg_frames)
        window = chroma[:, start_frame:end_frame].mean(axis=1)
        if window.sum() == 0:
            continue
        top1 = top_k_estimates(window, profile="krumhansl_kessler", k=1)[0]
        start_s = start_frame / frames_per_second
        end_s = end_frame / frames_per_second
        segments.append((start_s, end_s, top1))
    return segments
