# src/music_decoder/chord_detection/recognize.py
"""Beat-synchronous chord recognition: chroma + beats -> chord segments."""
from __future__ import annotations

from typing import Any

import numpy as np

from music_decoder.pipeline.contracts import ChordSegment

from .templates import NO_CHORD, QUALITIES, ROOTS, all_templates


def beat_sync_chroma(
    chroma: np.ndarray[Any, np.dtype[Any]],
    *,
    sr: int,
    hop_length: int,
    beat_times_s: np.ndarray[Any, np.dtype[Any]],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Average chroma columns over each beat-to-beat interval.

    Returns a (12, max(0, len(beats) - 1)) matrix. The last beat closes the
    final window -- if there are N beats, there are N-1 windows.
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


def viterbi_smooth(
    scores: np.ndarray[Any, np.dtype[Any]],
    *,
    p_self: float,
) -> np.ndarray[Any, np.dtype[np.int_]]:
    """Standard Viterbi over 49 states with a uniform stay/switch transition.

    `scores` is the (num_beats, 49) per-beat similarity matrix from `score_beats`.
    `p_self` is the self-transition probability; switches are uniform across
    the other 48 states. Decoding is done in log-space.

    Returns an integer array of length num_beats holding the most-likely state
    index per beat.
    """
    if scores.shape[0] == 0:
        return np.zeros(0, dtype=int)
    num_beats, n_states = scores.shape
    # Build log-emission matrix; clip scores to (0, 1] before log.
    eps = 1e-12
    log_emit = np.log(np.clip(scores, eps, 1.0))

    # Transition matrix in log-space.
    p_switch = (1.0 - p_self) / (n_states - 1)
    log_trans_self = np.log(p_self + eps)
    log_trans_other = np.log(p_switch + eps)

    # Viterbi DP.
    dp = np.full((num_beats, n_states), -np.inf, dtype=float)
    back = np.zeros((num_beats, n_states), dtype=int)
    dp[0] = log_emit[0]   # uniform initial -> constant offset, can drop
    for t in range(1, num_beats):
        # For each next-state j, best k is either j (self) or the argmax over k!=j.
        prev = dp[t - 1]
        # The best-of-others is just (max - is_self_correction). Equivalent to:
        # take max over all k for each j (using log_trans_other), and then for
        # k=j replace with prev[j] + log_trans_self if higher.
        best_other_value = prev.max() + log_trans_other
        best_other_idx = int(prev.argmax())
        for j in range(n_states):
            self_score = prev[j] + log_trans_self
            other_score = best_other_value
            other_idx = best_other_idx
            if other_idx == j:
                # The "best-of-others" candidate WAS j; we have to find the
                # second-best for j-as-other, which is the rare case.
                tmp = prev.copy()
                tmp[j] = -np.inf
                if np.isfinite(tmp).any():
                    other_idx = int(tmp.argmax())
                    other_score = tmp[other_idx] + log_trans_other
                else:
                    other_score = -np.inf
            if self_score >= other_score:
                dp[t, j] = self_score + log_emit[t, j]
                back[t, j] = j
            else:
                dp[t, j] = other_score + log_emit[t, j]
                back[t, j] = other_idx

    # Backtrace.
    path = np.zeros(num_beats, dtype=int)
    path[-1] = int(np.argmax(dp[-1]))
    for t in range(num_beats - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def _state_to_root_quality(state_idx: int) -> tuple[str, str]:
    if state_idx == 48:
        return NO_CHORD, ""
    return ROOTS[state_idx // len(QUALITIES)], QUALITIES[state_idx % len(QUALITIES)]


def merge_segments(
    state_path: np.ndarray[Any, np.dtype[Any]],
    beat_times_s: np.ndarray[Any, np.dtype[Any]],
    scores: np.ndarray[Any, np.dtype[Any]],
    *,
    min_segment_duration_s: float,
) -> list[ChordSegment]:
    """Collapse consecutive identical states into ChordSegment records and
    absorb runs shorter than `min_segment_duration_s` into their predecessor."""
    if state_path.size == 0:
        return []

    # First pass: build raw runs.
    raw: list[tuple[int, int, int]] = []   # (start_beat, end_beat, state)
    start = 0
    for i in range(1, len(state_path)):
        if state_path[i] != state_path[start]:
            raw.append((start, i, int(state_path[start])))
            start = i
    raw.append((start, len(state_path), int(state_path[start])))

    # Second pass: drop short segments.
    cleaned: list[tuple[int, int, int]] = []
    for s, e, state in raw:
        duration = float(beat_times_s[e]) - float(beat_times_s[s])
        if cleaned and duration < min_segment_duration_s:
            prev_s, _prev_e, prev_state = cleaned[-1]
            cleaned[-1] = (prev_s, e, prev_state)
        else:
            cleaned.append((s, e, state))

    # Third pass: re-collapse if absorption created adjacent same-state runs.
    final: list[tuple[int, int, int]] = []
    for s, e, state in cleaned:
        if final and final[-1][2] == state:
            ps, _pe, pstate = final[-1]
            final[-1] = (ps, e, pstate)
        else:
            final.append((s, e, state))

    segments: list[ChordSegment] = []
    for s, e, state in final:
        root, quality = _state_to_root_quality(state)
        beat_scores = scores[s:e, state] if scores.size else np.array([0.0])
        confidence = float(beat_scores.mean()) if beat_scores.size else 0.0
        segments.append(ChordSegment(
            start_s=float(beat_times_s[s]),
            end_s=float(beat_times_s[e]),
            root=root,
            quality=quality,
            confidence=confidence,
        ))
    return segments
