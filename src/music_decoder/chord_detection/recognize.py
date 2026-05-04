# src/music_decoder/chord_detection/recognize.py
"""Beat-synchronous chord recognition: chroma + beats → chord segments."""
from __future__ import annotations

from typing import Any

import numpy as np


def beat_sync_chroma(
    chroma: np.ndarray[Any, np.dtype[Any]],
    *,
    sr: int,
    hop_length: int,
    beat_times_s: np.ndarray[Any, np.dtype[Any]],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Average chroma columns over each beat-to-beat interval.

    Returns a (12, max(0, len(beats) - 1)) matrix. The last beat closes the
    final window — if there are N beats, there are N-1 windows.
    """
    if chroma.size == 0 or beat_times_s.size < 2:
        return np.zeros((12, max(0, beat_times_s.size - 1)), dtype=float)
    frame_period = hop_length / sr  # seconds per chroma column
    # Convert beat times to column indices.
    beat_cols = np.clip(
        np.round(beat_times_s / frame_period).astype(int),
        0,
        chroma.shape[1],
    )
    out = np.zeros((12, len(beat_cols) - 1), dtype=float)
    for i in range(len(beat_cols) - 1):
        start, end = beat_cols[i], beat_cols[i + 1]
        if end <= start:
            out[:, i] = 0.0
        else:
            out[:, i] = chroma[:, start:end].mean(axis=1)
    return out
