import pytest

from music_decoder.evaluation.metrics import chord_recognition_score
from music_decoder.types import ChordSegment


def _seg(start: float, end: float, root: str, quality: str) -> ChordSegment:
    return ChordSegment(start_s=start, end_s=end, root=root, quality=quality, confidence=1.0)


def test_perfect_match():
    pred = [_seg(0.0, 4.0, "C", "maj"), _seg(4.0, 8.0, "G", "maj")]
    truth = [(0.0, 4.0, "C", "maj"), (4.0, 8.0, "G", "maj")]
    assert chord_recognition_score(pred, truth) == pytest.approx(1.0)


def test_root_match_quality_mismatch_scores_half():
    pred = [_seg(0.0, 4.0, "C", "maj")]
    truth = [(0.0, 4.0, "C", "7")]
    assert chord_recognition_score(pred, truth) == pytest.approx(0.5)


def test_total_mismatch_scores_zero():
    pred = [_seg(0.0, 4.0, "C", "maj")]
    truth = [(0.0, 4.0, "G", "maj")]
    assert chord_recognition_score(pred, truth) == pytest.approx(0.0)


def test_partial_overlap_weighted_by_time():
    # Pred says C maj for the first 2 seconds, then F maj. Truth says C maj entire.
    pred = [_seg(0.0, 2.0, "C", "maj"), _seg(2.0, 4.0, "F", "maj")]
    truth = [(0.0, 4.0, "C", "maj")]
    score = chord_recognition_score(pred, truth)
    assert score == pytest.approx(0.5, abs=0.05)


def test_empty_returns_one():
    assert chord_recognition_score([], []) == 1.0


def test_predicted_empty_returns_zero():
    assert chord_recognition_score([], [(0.0, 4.0, "C", "maj")]) == 0.0
