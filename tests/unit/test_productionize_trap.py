import numpy as np
import pytest

from music_decoder.productionize.config import TrapDrumsConfig
from music_decoder.productionize.instruments.bass import midi_to_hz
from music_decoder.productionize.instruments.trap import (
    clap,
    crash,
    hat,
    kick_punch,
    riser,
    sub_808,
)

SR = 44_100
CFG = TrapDrumsConfig()


def _rms(x):
    return float(np.sqrt(np.mean(x**2)))


def test_sub_808_sounds_at_its_pitch():
    x = sub_808(37, 1.0, CFG, SR, np.random.default_rng(1))
    assert len(x) == int((1.0 + CFG.sub_tail_s) * SR)
    seg = x[int(0.1 * SR) : int(0.9 * SR)]
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), n=8 * len(seg)))
    freqs = np.fft.rfftfreq(8 * len(seg), 1 / SR)
    band = (freqs >= 20) & (freqs <= 200)
    peak = freqs[band][np.argmax(spec[band])]
    assert peak == pytest.approx(midi_to_hz(37), abs=2.0)


def test_sub_808_is_choked_after_its_length():
    x = sub_808(37, 1.0, CFG, SR, np.random.default_rng(1))
    tail = _rms(x[-int(0.1 * SR) :])
    body = _rms(x[int(0.1 * SR) : int(0.5 * SR)])
    assert 20 * np.log10(tail / body) <= -25.0


HITS = {
    "sub_808": lambda rng: sub_808(37, 0.5, CFG, SR, rng),
    "kick_punch": lambda rng: kick_punch(SR, rng),
    "clap": lambda rng: clap(SR, rng),
    "closed_hat": lambda rng: hat(False, 0.8, SR, rng),
    "open_hat": lambda rng: hat(True, 0.8, SR, rng),
    "crash": lambda rng: crash(SR, rng),
    "riser": lambda rng: riser(1.0, SR, rng),
}


@pytest.mark.parametrize("name", HITS)
def test_hits_are_deterministic_per_generator(name):
    a = HITS[name](np.random.default_rng(5))
    b = HITS[name](np.random.default_rng(5))
    assert np.array_equal(a, b)


@pytest.mark.parametrize(
    ("make", "seconds"),
    [
        (lambda rng: clap(SR, rng), 0.45),
        (lambda rng: hat(False, 1.0, SR, rng), 0.045),
        (lambda rng: hat(True, 1.0, SR, rng), 0.32),
        (lambda rng: crash(SR, rng), 2.5),
        (lambda rng: riser(2.0, SR, rng), 2.0),
    ],
)
def test_hit_lengths(make, seconds):
    x = make(np.random.default_rng(2))
    assert x.dtype == np.float64
    assert x.ndim == 1
    assert len(x) == int(seconds * SR)
    assert np.all(np.isfinite(x))
    assert np.abs(x).max() > 0.0


def test_riser_rises():
    x = riser(2.0, SR, np.random.default_rng(2))
    q = len(x) // 4
    assert _rms(x[-q:]) > 4.0 * _rms(x[:q])
