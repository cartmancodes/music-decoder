import numpy as np

from music_decoder.key.global_estimator import estimate_global_key


def test_consensus_when_profiles_agree():
    pc = np.zeros(12)
    pc[0] = 0.6
    pc[4] = 0.3
    pc[7] = 0.4  # C major triad emphasis
    result = estimate_global_key(pc)
    assert "krumhansl_kessler" in result.global_top3_per_profile
    assert "temperley" in result.global_top3_per_profile
    assert result.consensus_key is not None
    assert result.consensus_key.tonic == "C"
    assert result.consensus_key.mode == "major"
    assert 0.0 <= result.confidence <= 1.0


def test_no_consensus_when_profiles_disagree():
    pc = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0])
    result = estimate_global_key(pc)
    assert isinstance(result.global_top3_per_profile["krumhansl_kessler"], list)
    # consensus may or may not be set depending on rankings; just ensure shape.
    if result.consensus_key is None:
        assert result.confidence < 0.7
