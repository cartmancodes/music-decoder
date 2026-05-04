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


from .templates import all_templates  # noqa: E402


def score_beats(
    beat_chroma: np.ndarray[Any, np.dtype[Any]],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Cosine-similarity scores between every beat column and every template.

    Returns a (num_beats, 49) matrix.  Beats whose chroma column is all-zero
    receive an all-zero score row (the per-beat scorer never spuriously
    "matches" a silent beat).
    """
    if beat_chroma.size == 0:
        return np.zeros((0, 49), dtype=float)
    templates = all_templates()                # (49, 12)
    template_norms = np.linalg.norm(templates, axis=1, keepdims=True)
    template_norms[template_norms == 0] = 1.0
    templates_n = templates / template_norms

    chroma_norms = np.linalg.norm(beat_chroma, axis=0, keepdims=True)  # (1, B)
    safe_norms = np.where(chroma_norms == 0, 1.0, chroma_norms)
    chroma_n = beat_chroma / safe_norms                                # (12, B)

    # (B, 49) = (B, 12) @ (12, 49)
    scores = chroma_n.T @ templates_n.T

    # Zero out scores where the source chroma was all-zero.
    mask = (chroma_norms == 0).flatten()       # (B,)
    result: np.ndarray[Any, np.dtype[np.float64]] = np.array(scores, dtype=np.float64)
    if mask.any():
        result[mask] = 0.0
    return result
