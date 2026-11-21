"""Synthesized trap kit: the 808 sub, a short kick, clap, hats, crash and riser.

Every clip is a mono float64 array. Noise comes from the caller's generator, so a
clip is reproducible per generator. The 808's shape is configurable
(:class:`TrapDrumsConfig`); the other hits keep their sound design as module
constants below.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from music_decoder.productionize import dsp
from music_decoder.productionize.config import TrapDrumsConfig
from music_decoder.productionize.dsp import Audio
from music_decoder.productionize.instruments.bass import midi_to_hz

# Filter design: coefficient places kept, to hide ulp-level noise in scipy's design.
_COEFF_DECIMALS = 12

# 808 click: a 4 ms decaying noise burst on the first samples.
_SUB_CLICK_S = 0.004
_SUB_CLICK_GAIN = 0.25
_SUB_ATTACK_S = 0.002

# Kick punch: a 0.2 s kick that falls from 212 Hz to 52 Hz, so the hit reads apart
# from the 808's note.
_PUNCH_LENGTH_S = 0.2
_PUNCH_BASE_HZ = 52.0
_PUNCH_SWEEP_HZ = 160.0
_PUNCH_SWEEP_S = 0.018
_PUNCH_DECAY_S = 0.07
_PUNCH_CLICK_S = 0.003
_PUNCH_CLICK_GAIN = 0.4
_PUNCH_DRIVE = 1.8

# Clap: three noise hands then a tail, band-passed, plus a short snare-body sine.
_CLAP_LENGTH_S = 0.45
_CLAP_HAND_OFFSETS_S = (0.0, 0.009, 0.019)
_CLAP_HAND_GAINS = (0.8, 0.8, 1.0)
_CLAP_HAND_DECAY_S = 0.006
_CLAP_TAIL_GAIN = 0.55
_CLAP_TAIL_DECAY_S = 0.11
_CLAP_BAND_HZ = (900.0, 5200.0)
_CLAP_BODY_HZ = 190.0
_CLAP_BODY_DECAY_S = 0.05
_CLAP_BODY_GAIN = 0.35

# Hats: high-passed noise with an exponential decay over a quarter of the length.
_HAT_CLOSED_S = 0.045
_HAT_OPEN_S = 0.32
_HAT_HIGHPASS_HZ = 7500.0
_HAT_GAIN = 1.0 / 3.0

# Crash: high-passed noise with a slow decay.
_CRASH_LENGTH_S = 2.5
_CRASH_HIGHPASS_HZ = 4000.0
_CRASH_DECAY_S = 0.7
_CRASH_GAIN = 0.5

# Riser: a noise band that sweeps up 40x from 300 Hz in 64 steps under a quadratic
# swell from 0.05 to 1.
_RISER_BLOCKS = 64
_RISER_START_HZ = 300.0
_RISER_SWEEP_RATIO = 40.0
_RISER_BAND = (0.7, 1.4)
_RISER_CEILING_HZ = 20000.0
_RISER_START_LEVEL = 0.05


def _butter(order: int, wn: float | tuple[float, float], btype: str, sr: int) -> Audio:
    """Butterworth second-order sections with coefficients rounded to 12 places.

    ``signal.butter`` is not bit-reproducible here: a coefficient can differ by one
    ulp between calls (it depends on memory layout), which would make a clip differ
    between two renders from the same generator. Rounding hides that noise.
    """
    sos = signal.butter(order, wn, btype, fs=sr, output="sos")
    return np.asarray(np.round(sos, _COEFF_DECIMALS), dtype=np.float64)


def sub_808(
    pitch: int, length_s: float, cfg: TrapDrumsConfig, sr: int, rng: np.random.Generator
) -> Audio:
    """An 808: a sine that drops fast into ``pitch``, then decays and is choked.

    The clip is ``length_s + cfg.sub_tail_s`` long; the choke starts at ``length_s``
    (the next hit). Mix gain (``cfg.sub_gain``) is applied where it is placed.
    """
    n = int((length_s + cfg.sub_tail_s) * sr)
    t = np.arange(n) / sr
    f = midi_to_hz(pitch)
    freq = f + (cfg.sub_punch_ratio * f) * np.exp(-t / cfg.sub_punch_s)
    body = np.sin(2.0 * np.pi * np.cumsum(freq) / sr)
    amp = np.exp(-t / cfg.sub_decay_s) * np.minimum(1.0, t / _SUB_ATTACK_S)
    amp *= np.where(t > length_s, np.exp(-(t - length_s) / cfg.sub_choke_s), 1.0)
    x = dsp.saturate(body * amp, cfg.sub_drive, normalize=True)
    k = int(_SUB_CLICK_S * sr)
    x[:k] += rng.standard_normal(k) * np.linspace(1.0, 0.0, k) * _SUB_CLICK_GAIN
    return x


def kick_punch(sr: int, rng: np.random.Generator) -> Audio:
    """A short kick for the drum bus, so the hit reads apart from the 808's note."""
    n = int(_PUNCH_LENGTH_S * sr)
    t = np.arange(n) / sr
    freq = _PUNCH_BASE_HZ + _PUNCH_SWEEP_HZ * np.exp(-t / _PUNCH_SWEEP_S)
    x = np.sin(2.0 * np.pi * np.cumsum(freq) / sr) * np.exp(-t / _PUNCH_DECAY_S)
    k = int(_PUNCH_CLICK_S * sr)
    x[:k] += rng.standard_normal(k) * np.linspace(1.0, 0.0, k) * _PUNCH_CLICK_GAIN
    return dsp.saturate(x, _PUNCH_DRIVE, normalize=True)


def clap(sr: int, rng: np.random.Generator) -> Audio:
    n = int(_CLAP_LENGTH_S * sr)
    t = np.arange(n) / sr
    noise = rng.standard_normal(n)
    env = np.zeros(n)
    for off, gain in zip(_CLAP_HAND_OFFSETS_S, _CLAP_HAND_GAINS, strict=True):
        env += np.where(t >= off, np.exp(-(t - off) / _CLAP_HAND_DECAY_S), 0.0) * gain
    tail_from = _CLAP_HAND_OFFSETS_S[-1]
    env += np.where(
        t >= tail_from, _CLAP_TAIL_GAIN * np.exp(-(t - tail_from) / _CLAP_TAIL_DECAY_S), 0.0
    )
    sos = _butter(2, _CLAP_BAND_HZ, "bandpass", sr)
    x = signal.sosfilt(sos, noise * env)
    body = np.sin(2.0 * np.pi * _CLAP_BODY_HZ * t) * np.exp(-t / _CLAP_BODY_DECAY_S)
    return np.asarray(x / np.abs(x).max() + body * _CLAP_BODY_GAIN, dtype=np.float64)


def hat(open_: bool, velocity: float, sr: int, rng: np.random.Generator) -> Audio:
    dur = _HAT_OPEN_S if open_ else _HAT_CLOSED_S
    n = int(dur * sr)
    t = np.arange(n) / sr
    sos = _butter(4, _HAT_HIGHPASS_HZ, "highpass", sr)
    x = signal.sosfilt(sos, rng.standard_normal(n))
    return np.asarray(x * np.exp(-t / (dur / 4.0)) * velocity * _HAT_GAIN, dtype=np.float64)


def crash(sr: int, rng: np.random.Generator) -> Audio:
    n = int(_CRASH_LENGTH_S * sr)
    t = np.arange(n) / sr
    sos = _butter(2, _CRASH_HIGHPASS_HZ, "highpass", sr)
    x = signal.sosfilt(sos, rng.standard_normal(n))
    return np.asarray(x * np.exp(-t / _CRASH_DECAY_S) * _CRASH_GAIN, dtype=np.float64)


def riser(length_s: float, sr: int, rng: np.random.Generator) -> Audio:
    """A band of noise sweeping up, swelling toward its end."""
    n = int(length_s * sr)
    noise = rng.standard_normal(n)
    out = np.zeros(n)
    for b in range(_RISER_BLOCKS):
        lo, hi = b * n // _RISER_BLOCKS, (b + 1) * n // _RISER_BLOCKS
        fc = _RISER_START_HZ * (_RISER_SWEEP_RATIO ** (b / _RISER_BLOCKS))
        band = (fc * _RISER_BAND[0], min(fc * _RISER_BAND[1], _RISER_CEILING_HZ))
        sos = _butter(2, band, "bandpass", sr)
        out[lo:hi] = signal.sosfilt(sos, noise[lo:hi])
    return np.asarray(out * np.linspace(_RISER_START_LEVEL, 1.0, n) ** 2, dtype=np.float64)
