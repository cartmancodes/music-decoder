from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import mir_eval
import numpy as np

_PITCH_CLASS = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4,
                "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9,
                "A#": 10, "Bb": 10, "B": 11}

NDArray = np.ndarray[Any, np.dtype[Any]]


def _midi_to_hz(midi: NDArray) -> NDArray:
    return 440.0 * 2 ** ((midi - 69) / 12.0)


def note_f_measure(
    pred_intervals: NDArray, pred_pitches_midi: NDArray,
    gt_intervals: NDArray, gt_pitches_midi: NDArray,
    onset_tolerance_s: float = 0.05, pitch_tolerance_cents: float = 50.0,
) -> float:
    if len(pred_intervals) == 0 and len(gt_intervals) == 0:
        return 1.0
    if len(pred_intervals) == 0 or len(gt_intervals) == 0:
        return 0.0
    pred_hz = _midi_to_hz(np.asarray(pred_pitches_midi))
    gt_hz = _midi_to_hz(np.asarray(gt_pitches_midi))
    _p, _r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        np.asarray(gt_intervals), gt_hz,
        np.asarray(pred_intervals), pred_hz,
        onset_tolerance=onset_tolerance_s,
        pitch_tolerance=pitch_tolerance_cents,
        offset_ratio=None,
    )
    return float(f)


def onset_f_measure(
    pred_intervals: NDArray, gt_intervals: NDArray,
    tolerance_s: float = 0.05,
) -> float:
    if len(pred_intervals) == 0 and len(gt_intervals) == 0:
        return 1.0
    if len(pred_intervals) == 0 or len(gt_intervals) == 0:
        return 0.0
    pred_onsets = np.asarray(pred_intervals)[:, 0]
    gt_onsets = np.asarray(gt_intervals)[:, 0]
    _p, _r, f = mir_eval.onset.f_measure(gt_onsets, pred_onsets, window=tolerance_s)
    return float(f)


def pitch_class_accuracy(
    pred_intervals: NDArray, pred_pitches_midi: NDArray,
    gt_intervals: NDArray, gt_pitches_midi: NDArray,
    frame_rate_hz: float = 100.0,
) -> float:
    if len(gt_intervals) == 0:
        return 1.0 if len(pred_intervals) == 0 else 0.0
    end = max(float(np.asarray(gt_intervals).max()),
              float(np.asarray(pred_intervals).max()) if len(pred_intervals) else 0.0)
    n_frames = max(int(np.ceil(end * frame_rate_hz)) + 1, 1)
    times = np.arange(n_frames) / frame_rate_hz

    def _frame_pcs(intervals: NDArray, pitches: NDArray) -> NDArray:
        out = np.full(n_frames, -1, dtype=int)
        for (s, e), p in zip(intervals, pitches, strict=True):
            mask = (times >= s) & (times < e)
            out[mask] = int(p) % 12
        return out

    gt_pcs = _frame_pcs(np.asarray(gt_intervals), np.asarray(gt_pitches_midi))
    pred_pcs = _frame_pcs(np.asarray(pred_intervals), np.asarray(pred_pitches_midi))
    valid = gt_pcs != -1
    if valid.sum() == 0:
        return 1.0
    matched = ((pred_pcs == gt_pcs) & valid).sum()
    return float(matched / valid.sum())


def key_mirex_score(predicted: tuple[str, str], truth: tuple[str, str]) -> float:
    p_tonic, p_mode = predicted
    t_tonic, t_mode = truth
    p, t = _PITCH_CLASS[p_tonic], _PITCH_CLASS[t_tonic]
    if (p, p_mode) == (t, t_mode):
        return 1.0
    if p_mode == t_mode and (p - t) % 12 == 7:
        return 0.5
    if {p_mode, t_mode} == {"major", "minor"}:
        if p_mode == "minor" and (p - t) % 12 == 9:
            return 0.3
        if t_mode == "minor" and (t - p) % 12 == 9:
            return 0.3
        if p == t:
            return 0.2
    return 0.0


def tab_string_accuracy(
    predicted: Iterable[tuple[int, int, int]],
    truth: Iterable[tuple[int, int, int]],
) -> float:
    pred_list = list(predicted)
    truth_list = list(truth)
    n = max(len(pred_list), len(truth_list))
    if n == 0:
        return 1.0
    correct = 0
    for p, t in zip(pred_list, truth_list, strict=False):
        if p[0] == t[0] and p[1] == t[1]:
            correct += 1
    return correct / n
