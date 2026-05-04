import numpy as np

from music_decoder.beat_tracking.time_signature import infer_time_signature


def test_strong_4_4_pattern_detects_4_4():
    # Beat strengths cycling 1.0, 0.4, 0.6, 0.4 (classic 4/4 emphasis on beat 1)
    strengths = np.tile(np.array([1.0, 0.4, 0.6, 0.4]), 8)
    result = infer_time_signature(strengths, min_confidence=0.3)
    assert result.numerator == 4
    assert result.denominator == 4
    assert result.confidence >= 0.3
    assert result.assumed is False


def test_unclear_pattern_falls_back_to_4_4_assumed():
    strengths = np.full(20, 0.5)   # uniform → no autocorrelation peaks
    result = infer_time_signature(strengths, min_confidence=0.5)
    assert result.numerator == 4
    assert result.denominator == 4
    assert result.assumed is True
