import numpy as np

from music_decoder.evaluation.metrics import (
    beat_f_measure,
    chord_majmin_score,
    note_onset_prf,
    segment_to_harte,
)
from music_decoder.types import ChordSegment


def _seg(s: float, e: float, r: str, q: str) -> ChordSegment:
    return ChordSegment(start_s=s, end_s=e, root=r, quality=q, confidence=1.0)


def test_segment_to_harte() -> None:
    assert segment_to_harte(_seg(0, 1, "C#", "min7")) == "C#:min7"
    assert segment_to_harte(_seg(0, 1, "N", "")) == "N"


def test_chord_majmin_half_right() -> None:
    pred = [_seg(0, 2, "D#", "maj"), _seg(2, 4, "C", "maj")]
    ref = [(0.0, 2.0, "D#:maj"), (2.0, 4.0, "G#:maj")]
    assert abs(chord_majmin_score(pred, ref) - 0.5) < 1e-6


def test_chord_majmin_excludes_out_of_vocabulary_reference() -> None:
    # MIREX majmin skips reference frames that are not maj/min/N (e.g. maj6).
    pred = [_seg(0, 2, "D#", "maj"), _seg(2, 4, "C", "maj")]
    ref = [(0.0, 2.0, "D#:maj"), (2.0, 4.0, "G#:maj6(*5)/1")]
    assert chord_majmin_score(pred, ref) == 1.0


def test_chord_majmin_sevenths_reduce_to_triads() -> None:
    pred = [_seg(0, 4, "G", "7")]
    ref = [(0.0, 4.0, "G:maj")]
    assert chord_majmin_score(pred, ref) == 1.0


def test_note_onset_prf_handles_unsorted_reference() -> None:
    gt_iv = np.array([[1.0, 1.5], [0.0, 0.5]])
    gt_p = np.array([60, 62])
    pred_iv = np.array([[0.01, 0.5], [2.0, 2.5]])
    pred_p = np.array([62, 70])
    p, r, f = note_onset_prf(pred_iv, pred_p, gt_iv, gt_p)
    assert (p, r) == (0.5, 0.5)
    assert abs(f - 0.5) < 1e-9


def test_beat_f_measure_perfect_and_empty() -> None:
    ref = np.arange(0, 20, 0.5)
    assert beat_f_measure(ref, ref) == 1.0
    assert beat_f_measure(np.array([]), ref) == 0.0
