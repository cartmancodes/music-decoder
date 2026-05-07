import numpy as np

from music_decoder.dsp.chroma import compute_chroma_with_hpss


def test_chroma_shape_and_range():
    sr = 22050
    t = np.arange(sr * 2) / sr
    samples = (0.5 * np.sin(2 * np.pi * 261.63 * t)).astype(np.float32)   # C4
    chroma = compute_chroma_with_hpss(samples, sr=sr, hpss_margin=1.0)
    assert chroma.shape[0] == 12
    assert (chroma >= 0).all()
    assert (chroma <= 1).all()


def test_pure_c_concentrates_mass_on_c():
    sr = 22050
    t = np.arange(sr * 2) / sr
    samples = (0.5 * np.sin(2 * np.pi * 261.63 * t)).astype(np.float32)
    chroma = compute_chroma_with_hpss(samples, sr=sr, hpss_margin=1.0)
    pc_sum = chroma.sum(axis=1)
    pc_sum /= pc_sum.sum()
    assert pc_sum.argmax() == 0  # C
    assert pc_sum[0] > 0.3
