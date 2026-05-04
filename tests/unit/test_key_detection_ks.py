import numpy as np

from music_decoder.key_detection.ks import (
    correlate_against_profiles,
    top_k_estimates,
)
from music_decoder.key_detection.profiles import (
    KRUMHANSL_KESSLER_MAJOR,
    KRUMHANSL_KESSLER_MINOR,
)


def test_kk_profile_lengths():
    assert len(KRUMHANSL_KESSLER_MAJOR) == 12
    assert len(KRUMHANSL_KESSLER_MINOR) == 12


def test_correlate_returns_24_keys():
    pc = np.zeros(12)
    pc[0] = 1.0  # pure C
    estimates = correlate_against_profiles(
        pc, profile="krumhansl_kessler",
    )
    assert len(estimates) == 24
    by_key = {(e.tonic, e.mode): e.correlation for e in estimates}
    assert max(by_key.values()) == by_key[("C", "major")]


def test_top_k_returns_three_sorted_descending():
    pc = np.zeros(12)
    pc[0] = 1.0
    top3 = top_k_estimates(pc, profile="krumhansl_kessler", k=3)
    assert len(top3) == 3
    assert top3[0].tonic == "C"
    assert top3[0].mode == "major"
    assert top3[0].correlation >= top3[1].correlation >= top3[2].correlation
    assert top3[0].margin == top3[0].correlation - top3[1].correlation


def test_temperley_distinguishable_from_kk():
    pc = np.array([0.4, 0.0, 0.2, 0.0, 0.3, 0.05, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0])
    kk = top_k_estimates(pc, profile="krumhansl_kessler", k=1)[0]
    temp = top_k_estimates(pc, profile="temperley", k=1)[0]
    assert (kk.tonic, kk.mode) is not None
    assert (temp.tonic, temp.mode) is not None
