import numpy as np
import pytest

from music_decoder.evaluation.metrics import (
    key_mirex_score,
    note_f_measure,
    onset_f_measure,  # noqa: F401
    pitch_class_accuracy,
    tab_string_accuracy,
)


def _intervals_pitches(triples):
    intervals = np.array([(s, e) for (s, e, _) in triples], dtype=float)
    pitches = np.array([p for (_, _, p) in triples], dtype=float)
    return intervals, pitches


def test_note_f_measure_perfect():
    notes = [(0.0, 0.5, 60), (0.5, 1.0, 62), (1.0, 1.5, 64)]
    pi, pp = _intervals_pitches(notes)
    f = note_f_measure(pi, pp, pi.copy(), pp.copy())
    assert f == pytest.approx(1.0)


def test_note_f_measure_one_missed_one_extra():
    truth = [(0.0, 0.5, 60), (0.5, 1.0, 62)]
    pred = [(0.0, 0.5, 60), (0.5, 1.0, 99)]
    ti, tp = _intervals_pitches(truth)
    pi, pp = _intervals_pitches(pred)
    f = note_f_measure(pi, pp, ti, tp)
    assert 0.4 < f < 0.7


def test_pitch_class_accuracy_perfect():
    truth = [(0.0, 1.0, 60)]
    pred = [(0.0, 1.0, 72)]
    ti, tp = _intervals_pitches(truth)
    pi, pp = _intervals_pitches(pred)
    acc = pitch_class_accuracy(pi, pp, ti, tp)
    assert acc > 0.9


def test_key_mirex_correct():
    assert key_mirex_score(("C", "major"), ("C", "major")) == 1.0


def test_key_mirex_perfect_fifth():
    assert key_mirex_score(("G", "major"), ("C", "major")) == pytest.approx(0.5)


def test_key_mirex_relative():
    assert key_mirex_score(("A", "minor"), ("C", "major")) == pytest.approx(0.3)


def test_key_mirex_parallel():
    assert key_mirex_score(("C", "minor"), ("C", "major")) == pytest.approx(0.2)


def test_key_mirex_wrong():
    assert key_mirex_score(("F#", "major"), ("C", "major")) == 0.0


def test_tab_string_accuracy_legacy_index_aligned():
    """Index-aligned mode (no intervals) for hand-curated fixtures."""
    pred = [(60, 4, 1), (62, 4, 3)]
    truth = [(60, 4, 1), (62, 3, 7)]
    assert tab_string_accuracy(pred, truth) == pytest.approx(0.5)


def test_tab_string_accuracy_time_aligned_perfect():
    """Time-aligned mode: matched pitch+onset, matching string."""
    pred_intervals = np.array([[0.0, 0.5], [0.5, 1.0]])
    gt_intervals = np.array([[0.0, 0.5], [0.5, 1.0]])
    pred = [(60, 4, 1), (62, 4, 3)]
    truth = [(60, 4, 1), (62, 4, 3)]
    acc = tab_string_accuracy(
        pred,
        truth,
        pred_intervals=pred_intervals,
        gt_intervals=gt_intervals,
    )
    assert acc == pytest.approx(1.0)


def test_tab_string_accuracy_time_aligned_excludes_unmatched_notes():
    """Time-aligned: a predicted note that doesn't match any GT in time+pitch
    is excluded from the denominator — does not unfairly punish the metric."""
    # Two GT notes; predict 4 notes (2 correct + 2 spurious).
    pred_intervals = np.array([[0.0, 0.5], [0.5, 1.0], [1.0, 1.5], [1.5, 2.0]])
    gt_intervals = np.array([[0.0, 0.5], [0.5, 1.0]])
    pred = [(60, 4, 1), (62, 4, 3), (90, 0, 0), (88, 1, 0)]  # last 2 fake
    truth = [(60, 4, 1), (62, 3, 7)]  # second has different string from pred
    acc = tab_string_accuracy(
        pred,
        truth,
        pred_intervals=pred_intervals,
        gt_intervals=gt_intervals,
    )
    # Only first GT note matches in pitch+time AND string. Second matches in
    # pitch+time but differs in string. Last 2 predictions don't match any GT.
    # So 1 of 2 matched pairs has correct string → 0.5.
    assert acc == pytest.approx(0.5)


def test_tab_string_accuracy_time_aligned_no_matches_returns_zero():
    """When no predicted notes match any GT in time+pitch, return 0.0."""
    pred_intervals = np.array([[10.0, 11.0]])
    gt_intervals = np.array([[0.0, 0.5]])
    pred = [(60, 4, 1)]
    truth = [(60, 4, 1)]
    acc = tab_string_accuracy(
        pred,
        truth,
        pred_intervals=pred_intervals,
        gt_intervals=gt_intervals,
    )
    assert acc == 0.0
