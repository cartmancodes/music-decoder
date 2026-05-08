from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import mir_eval
import numpy as np

from music_decoder.types import ChordSegment, KeyEstimate, TabbedNote

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


def _coerce_key(value: tuple[str, str] | KeyEstimate) -> tuple[str, str]:
    if isinstance(value, KeyEstimate):
        return value.tonic, value.mode
    return value


def key_mirex_score(
    predicted: tuple[str, str] | KeyEstimate,
    truth: tuple[str, str] | KeyEstimate,
) -> float:
    p_tonic, p_mode = _coerce_key(predicted)
    t_tonic, t_mode = _coerce_key(truth)
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


def _coerce_tab_row(
    row: tuple[int, int, int] | TabbedNote,
) -> tuple[int, int, int]:
    """Normalize a tab row to ``(pitch, string, fret)``.

    Accepts either the legacy ``(pitch, string, fret)`` tuple or the public
    :class:`TabbedNote` shape used by ``analyze()``.
    """
    if isinstance(row, TabbedNote):
        return int(row.note.pitch), int(row.position.string), int(row.position.fret)
    return row


def tab_string_accuracy(
    predicted: Iterable[tuple[int, int, int] | TabbedNote],
    truth: Iterable[tuple[int, int, int] | TabbedNote],
    *,
    pred_intervals: np.ndarray[Any, np.dtype[np.float64]] | None = None,
    gt_intervals: np.ndarray[Any, np.dtype[np.float64]] | None = None,
    onset_tolerance_s: float = 0.05,
) -> float:
    """String-assignment accuracy.

    Two modes:

    - **Index-aligned (legacy):** when ``pred_intervals`` and ``gt_intervals`` are
      omitted, predicted and truth are aligned 1:1 by order. This mode is only
      meaningful for tightly hand-curated test fixtures where note counts match
      and order is deterministic.
    - **Time-aligned (preferred for real audio):** when intervals are supplied,
      uses ``mir_eval.transcription.match_notes`` to find pitch+onset matches
      between predicted and truth, then reports the fraction of matched pairs
      whose ``string`` index also agrees. Computed conditional on a successful
      pitch+onset match — unmatched notes are excluded from both numerator and
      denominator. The metric collapses to 0.0 when no matches exist (a clear
      signal of pipeline failure rather than a meaningless small number).
    """
    pred_raw = list(predicted)
    truth_raw = list(truth)

    # If callers passed TabbedNote instances and didn't provide explicit
    # intervals, derive them from the note start/end so that time-aligned
    # matching can run automatically.
    if pred_intervals is None and pred_raw and isinstance(pred_raw[0], TabbedNote):
        pred_intervals = np.array(
            [(r.note.start_s, r.note.end_s) for r in pred_raw if isinstance(r, TabbedNote)],
            dtype=float,
        )
    if gt_intervals is None and truth_raw and isinstance(truth_raw[0], TabbedNote):
        gt_intervals = np.array(
            [(r.note.start_s, r.note.end_s) for r in truth_raw if isinstance(r, TabbedNote)],
            dtype=float,
        )

    pred_list = [_coerce_tab_row(r) for r in pred_raw]
    truth_list = [_coerce_tab_row(r) for r in truth_raw]
    if not pred_list and not truth_list:
        return 1.0
    if not pred_list or not truth_list:
        return 0.0

    if pred_intervals is None or gt_intervals is None:
        # Legacy index-aligned mode.
        n = max(len(pred_list), len(truth_list))
        correct = sum(
            1 for p, t in zip(pred_list, truth_list, strict=False)
            if p[0] == t[0] and p[1] == t[1]
        )
        return correct / n

    pred_pitches_hz = _midi_to_hz(np.array([p[0] for p in pred_list], dtype=float))
    gt_pitches_hz = _midi_to_hz(np.array([t[0] for t in truth_list], dtype=float))
    matching = mir_eval.transcription.match_notes(
        np.asarray(gt_intervals), gt_pitches_hz,
        np.asarray(pred_intervals), pred_pitches_hz,
        onset_tolerance=onset_tolerance_s,
        pitch_tolerance=50.0,
        offset_ratio=None,
    )
    if not matching:
        return 0.0
    correct = sum(
        1 for gt_idx, pred_idx in matching
        if pred_list[pred_idx][1] == truth_list[gt_idx][1]
    )
    return correct / len(matching)


def chord_recognition_score(
    predicted: Iterable[ChordSegment | tuple[float, float, str, str]],
    truth: Iterable[ChordSegment | tuple[float, float, str, str]],
    *,
    frame_rate_hz: float = 100.0,
) -> float:
    """MIREX-style chord score over a 10ms frame grid.

    Per-frame:
        1.0 if predicted (root, quality) matches truth exactly
        0.5 if predicted root matches truth root but quality differs
        0.0 otherwise

    Returns the time-weighted mean across the union span.

    Accepts either :class:`ChordSegment` instances or raw
    ``(start_s, end_s, root, quality)`` tuples on either side.
    """
    predicted_list = list(predicted)
    truth_list = list(truth)
    if not predicted_list and not truth_list:
        return 1.0
    if not predicted_list or not truth_list:
        return 0.0

    def _end(seg: ChordSegment | tuple[float, float, str, str]) -> float:
        return seg.end_s if isinstance(seg, ChordSegment) else seg[1]

    end = max(
        max(_end(seg) for seg in predicted_list),
        max(_end(seg) for seg in truth_list),
    )
    n_frames = max(round(end * frame_rate_hz), 1)
    times = (np.arange(n_frames) + 0.5) / frame_rate_hz

    def _label_at(segments_iter: Any, t: float) -> tuple[str, str] | None:
        for seg in segments_iter:
            if isinstance(seg, ChordSegment):
                if seg.start_s <= t < seg.end_s:
                    return seg.root, seg.quality
            else:
                start, end_, root, quality = seg
                if start <= t < end_:
                    return root, quality
        return None

    score_sum = 0.0
    counted = 0
    for t in times:
        truth_label = _label_at(truth_list, float(t))
        if truth_label is None:
            continue
        counted += 1
        pred_label = _label_at(predicted_list, float(t))
        if pred_label is None:
            continue
        if pred_label == truth_label:
            score_sum += 1.0
        elif pred_label[0] == truth_label[0]:
            score_sum += 0.5
    if counted == 0:
        return 1.0
    return score_sum / counted
