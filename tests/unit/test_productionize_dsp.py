import numpy as np
import pytest
from scipy import signal

from music_decoder.productionize import dsp
from music_decoder.productionize.config import HALL, PLATE, CompressorConfig, EqBand

SR = 44_100


def _response_db(band: EqBand, freq_hz: float) -> float:
    _, h = signal.sosfreqz(dsp.eq_sos([band], SR), worN=[freq_hz], fs=SR)
    return float(20 * np.log10(abs(h[0])))


def _sine(freq_hz: float, seconds: float = 1.0, amp: float = 1.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return dsp.to_stereo(amp * np.sin(2 * np.pi * freq_hz * t))


def test_db_conversions_round_trip():
    assert dsp.db_to_gain(-6.0) == pytest.approx(0.5012, abs=1e-4)
    assert dsp.gain_to_db(dsp.db_to_gain(-13.0)) == pytest.approx(-13.0)


def test_peak_band_has_its_gain_at_the_centre_and_none_far_away():
    band = EqBand("peak", 1000.0, 3.0, 1.0)
    assert _response_db(band, 1000.0) == pytest.approx(3.0, abs=0.01)
    assert _response_db(band, 50.0) == pytest.approx(0.0, abs=0.1)
    assert _response_db(band, 15_000.0) == pytest.approx(0.0, abs=0.1)


def test_shelves_reach_their_gain_on_their_side_only():
    high = EqBand("high_shelf", 3000.0, 9.5)
    assert _response_db(high, 18_000.0) == pytest.approx(9.5, abs=0.2)
    assert _response_db(high, 100.0) == pytest.approx(0.0, abs=0.1)
    assert _response_db(high, 3000.0) == pytest.approx(4.75, abs=0.1)
    low = EqBand("low_shelf", 200.0, -4.0)
    assert _response_db(low, 20.0) == pytest.approx(-4.0, abs=0.2)
    assert _response_db(low, 5000.0) == pytest.approx(0.0, abs=0.1)


def test_apply_eq_without_bands_is_the_input():
    x = _sine(440.0, 0.1)
    assert dsp.apply_eq(x, (), SR) is x


def test_butter_highpass_removes_lows_and_keeps_highs():
    low = dsp.butter(_sine(30.0), "highpass", 300.0, SR)
    high = dsp.butter(_sine(3000.0), "highpass", 300.0, SR)
    assert dsp.rms_db(low[SR // 2 :]) < -35.0
    assert dsp.rms_db(high[SR // 2 :]) == pytest.approx(dsp.rms_db(_sine(3000.0)), abs=0.1)


def test_compressor_is_unity_below_threshold():
    cfg = CompressorConfig(rms_multiple=1.0, ratio=4.0, attack_s=0.005, release_s=0.1)
    x = _sine(220.0, amp=0.1)
    assert np.allclose(dsp.compress(x, 0.5, cfg, SR), x)


def test_compressor_settles_on_the_ratio_above_threshold():
    cfg = CompressorConfig(rms_multiple=1.0, ratio=4.0, attack_s=0.005, release_s=0.1)
    gain = dsp.compressor_gain(_sine(220.0, amp=1.0), 0.25, cfg, SR)
    # envelope 1.0 over a 0.25 threshold at 4:1 -> (0.25 + 0.75 / 4) / 1.0
    assert gain[-1000:].mean() == pytest.approx(0.4375, abs=0.01)
    assert gain[0] > 0.9  # the attack has not bitten yet


def test_compressor_is_stereo_linked():
    cfg = CompressorConfig(rms_multiple=1.0, ratio=4.0, attack_s=0.005, release_s=0.1)
    x = _sine(220.0)
    x[:, 1] *= 0.1
    out = dsp.compress(x, 0.25, cfg, SR)
    # One gain for both channels: the quiet channel keeps its 10:1 distance.
    ratio = np.abs(out[SR // 2 :, 0]).max() / np.abs(out[SR // 2 :, 1]).max()
    assert ratio == pytest.approx(10.0, rel=1e-3)


def test_saturation_shapes():
    x = dsp.to_stereo(np.array([0.0, 0.01, 1.0, -1.0]))
    peak_kept = dsp.saturate(x, 1.2, normalize=True)
    assert peak_kept[2, 0] == pytest.approx(1.0)
    assert peak_kept[3, 0] == pytest.approx(-1.0)
    bus = dsp.saturate(x, 1.5, normalize=False)
    assert bus[1, 0] == pytest.approx(0.01, rel=1e-3)  # unity for small signals
    assert abs(bus[2, 0]) < 1 / 1.5


def test_soft_limit_never_exceeds_the_ceiling():
    x = dsp.to_stereo(np.linspace(-5.0, 5.0, 1001))
    out = dsp.soft_limit(x, 0.8)
    assert np.abs(out).max() < 0.8
    assert dsp.soft_limit(dsp.to_stereo(np.array([0.01])), 0.8)[0, 0] == pytest.approx(
        0.01, rel=1e-3
    )


@pytest.mark.parametrize("cfg", [HALL, PLATE])
def test_impulse_response_is_unit_energy_and_deterministic(cfg):
    ir = dsp.impulse_response(cfg, SR)
    pre_delay = int(cfg.pre_delay_s * SR)
    assert ir.shape == (pre_delay + int(cfg.length_t60 * cfg.t60_s * SR), 2)
    assert np.allclose((ir**2).sum(axis=0), 1.0)
    assert np.all(ir[:pre_delay] == 0.0)
    assert not np.allclose(ir[:, 0], ir[:, 1])  # separate seeds left and right
    dsp.impulse_response.cache_clear()
    assert np.array_equal(dsp.impulse_response(cfg, SR), ir)


def test_reverb_decays_by_about_60_db_over_t60():
    ir = dsp.impulse_response(HALL, SR)[:, 0]
    start = int((HALL.pre_delay_s + 0.1) * SR)
    window = int(0.1 * SR)
    early = np.sqrt(np.mean(ir[start : start + window] ** 2))
    late_at = start + int(1.5 * SR)
    late = np.sqrt(np.mean(ir[late_at : late_at + window] ** 2))
    expected = 60.0 * 1.5 / HALL.t60_s
    assert 20 * np.log10(early / late) == pytest.approx(expected, abs=8.0)


def test_reverb_output_has_the_input_length():
    x = _sine(440.0, 0.5)
    wet = dsp.reverb(x, PLATE, SR)
    assert wet.shape == x.shape
    assert np.all(wet[: int(PLATE.pre_delay_s * SR)] == pytest.approx(0.0, abs=1e-9))


def test_moving_average_matches_a_zero_padded_convolution():
    x = np.random.default_rng(0).standard_normal(500)
    expected = np.convolve(x, np.ones(31) / 31, mode="same")
    assert np.allclose(dsp.moving_average(x, 31), expected)
    assert dsp.moving_average(x, 1) is x


def test_constant_power_pan():
    mono = np.ones(4)
    centre = dsp.constant_power_pan(mono, 0.0)
    assert np.allclose(centre, 1.0)
    left = dsp.constant_power_pan(mono, -1.0)
    assert np.allclose(left[:, 1], 0.0) and np.allclose(left[:, 0], np.sqrt(2.0))
    slightly_left = dsp.constant_power_pan(mono, -0.35)
    assert slightly_left[0, 0] > slightly_left[0, 1]
    assert (slightly_left[0] ** 2).sum() == pytest.approx(2.0)


def test_mix_into_clips_to_the_bus():
    bus = np.zeros((10, 2))
    dsp.mix_into(bus, np.ones((4, 2)), start_s=8 / SR, sr=SR, gain=0.5)
    assert bus[:, 0].tolist() == [0.0] * 8 + [0.5, 0.5]
    dsp.mix_into(bus, np.ones((4, 2)), start_s=-2 / SR, sr=SR)
    assert bus[:3, 0].tolist() == [1.0, 1.0, 0.0]
    dsp.mix_into(bus, np.ones((4, 2)), start_s=1.0, sr=SR)  # entirely outside: no error


def test_levels_over_spans():
    x = np.zeros((1000, 2))
    x[100:200] = 0.5
    assert dsp.rms_over(x, [(100, 200)]) == pytest.approx(0.5)
    assert dsp.rms_over(x, [(0, 100)]) == 0.0
    assert dsp.rms_over(x, []) == 0.0
    gain = dsp.gain_for_level(x, -20.0, [(100, 200)])
    assert dsp.rms_over(x * gain, [(100, 200)]) == pytest.approx(0.1)
    assert dsp.gain_for_level(np.zeros((10, 2)), -20.0, [(0, 10)]) == 1.0


def test_true_peak_sees_inter_sample_peaks():
    t = np.arange(4410) / SR
    # A sine at fs/4 with 45 degree phase: samples at +-0.707, true peak 1.0.
    x = dsp.to_stereo(np.sin(2 * np.pi * (SR / 4) * t + np.pi / 4))
    assert np.abs(x).max() == pytest.approx(0.7071, abs=1e-3)
    assert dsp.true_peak(x[200:-200]) == pytest.approx(1.0, abs=0.03)
    assert dsp.percentile_peak(np.zeros((0, 2)), 99.9) == 0.0
