import numpy as np
import pytest

from music_decoder.productionize import dsp
from music_decoder.productionize.qc import (
    Check,
    QcReport,
    first_sound_s,
    full_scale_samples,
    low_band_correlation,
    mono_fold_down_db,
    octave_band_levels,
    off_grid,
    stereo_correlation,
    tail_peak_db,
)

SR = 44_100


def _tone(freq_hz, seconds=2.0, amp=0.5):
    t = np.arange(int(seconds * SR)) / SR
    return dsp.to_stereo(amp * np.sin(2 * np.pi * freq_hz * t))


def test_first_sound():
    x = np.zeros((SR, 2))
    x[5292:] = 0.01  # 120 ms
    assert first_sound_s(x, SR, -60.0) == pytest.approx(0.120, abs=1e-4)
    assert first_sound_s(x * 0.01, SR, -60.0) is None  # -80 dBFS never crosses -60


def test_tail_peak():
    x = np.ones((SR, 2))
    x[-int(0.05 * SR) :] = 1e-5
    assert tail_peak_db(x, SR, 0.050) == pytest.approx(-100.0, abs=0.1)


def test_stereo_measures():
    mono = _tone(440.0)
    assert stereo_correlation(mono) == pytest.approx(1.0)
    assert mono_fold_down_db(mono) == pytest.approx(0.0, abs=1e-6)
    rng = np.random.default_rng(0)
    wide = rng.standard_normal((SR, 2))
    assert abs(stereo_correlation(wide)) < 0.05
    assert mono_fold_down_db(wide) == pytest.approx(-3.0, abs=0.2)
    inverted = mono * np.array([1.0, -1.0])
    assert stereo_correlation(inverted) == pytest.approx(-1.0)
    assert stereo_correlation(np.zeros((100, 2))) == 1.0


def test_low_band_correlation_ignores_the_highs():
    low = _tone(60.0)
    highs = np.random.default_rng(0).standard_normal((len(low), 2)) * 0.2
    highs = dsp.butter(highs, "highpass", 2000.0, SR, order=4)
    assert low_band_correlation(low + highs, SR, 110.0) > 0.99
    assert stereo_correlation(low + highs) < 0.99


def test_full_scale_samples():
    x = np.array([[0.5, -0.99], [1.0, 0.2], [-1.2, 0.99999]])
    assert full_scale_samples(x) == 3
    assert full_scale_samples(x * 0.5) == 0


def test_off_grid():
    eighth = 0.25
    onsets = [0.0, 0.2515, 0.5035, 0.7499, 1.1]
    assert off_grid(onsets, eighth, 0.003) == [0.5035, 1.1]
    assert off_grid([], eighth, 0.003) == []


def test_octave_bands_put_a_tone_in_its_band():
    centres = (62.5, 125.0, 250.0, 500.0, 1000.0, 2000.0)
    x = _tone(1000.0, seconds=4.0)
    levels = octave_band_levels(x, SR, centres, -10.0, 2048)
    assert int(np.argmax(levels)) == 4
    assert levels[4] - np.delete(levels, 4).max() > 30.0
    # Loudness-matched: the same tone at another level reads the same.
    quieter = octave_band_levels(x * 0.1, SR, centres, -10.0, 2048)
    assert np.allclose(levels[4], quieter[4], atol=0.05)


def test_octave_bands_measure_the_mono_sum():
    centres = (500.0, 1000.0, 2000.0)
    x = _tone(1000.0, seconds=4.0)
    side = _tone(1000.0, seconds=4.0, amp=0.4) * np.array([1.0, -1.0])  # cancels in mono
    assert np.allclose(
        octave_band_levels(x + side, SR, centres, -10.0, 2048),
        octave_band_levels(x, SR, centres, -10.0, 2048),
    )


def test_report():
    report = QcReport(
        (
            Check("loudness", True, "-10.5 LUFS", "-10.5 +-0.3"),
            Check("section RMS", None, "-17.0", "no reference master"),
            Check("true peak", False, "-0.5 dBTP", "at or below -1.93"),
        )
    )
    assert not report.passed
    assert [c.name for c in report.failures] == ["true peak"]
    lines = report.format().splitlines()
    assert (
        lines[0].startswith("PASS") and lines[1].startswith("info") and lines[2].startswith("FAIL")
    )
    as_dict = report.to_dict()
    assert as_dict["passed"] is False and len(as_dict["checks"]) == 3
    assert QcReport((Check("a", True, "", ""), Check("b", None, "", ""))).passed
