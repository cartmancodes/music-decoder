"""Shared DSP building blocks: filters, EQ, compressor, reverb.

Every buffer is a float64 array of shape ``(samples, 2)`` unless a function
says otherwise. Filters are causal (``sosfilt``), as in the reference.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy import signal
from scipy.ndimage import uniform_filter1d

from music_decoder.productionize.config import CompressorConfig, EqBand, ReverbConfig

Audio = npt.NDArray[np.float64]

_EPS = 1e-12


# ---- Levels ----------------------------------------------------------------


def db_to_gain(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def gain_to_db(gain: float) -> float:
    return float(20.0 * np.log10(max(gain, _EPS)))


def rms(x: Audio) -> float:
    """Root mean square over every sample of every channel."""
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x))))


def rms_db(x: Audio) -> float:
    return gain_to_db(rms(x))


def percentile_peak(x: Audio, percentile: float) -> float:
    """The ``percentile``-th percentile of ``|x|`` (a robust peak)."""
    if x.size == 0:
        return 0.0
    return float(np.percentile(np.abs(x), percentile))


def to_stereo(mono: npt.NDArray[np.float64]) -> Audio:
    return np.repeat(np.asarray(mono, dtype=np.float64)[:, None], 2, axis=1)


def moving_average(x: npt.NDArray[np.float64], width: int) -> npt.NDArray[np.float64]:
    """Centred moving average along axis 0 (zeros beyond the ends); the
    output has the input's length."""
    if width <= 1:
        return x
    return np.asarray(uniform_filter1d(x, size=width, axis=0, mode="constant"), dtype=np.float64)


def constant_power_pan(mono: npt.NDArray[np.float64], pan: float) -> Audio:
    """``pan`` in [-1, 1]; unity gain on both channels at centre."""
    angle = (pan + 1.0) * np.pi / 4.0
    scale = np.sqrt(2.0)
    return np.stack([mono * np.cos(angle) * scale, mono * np.sin(angle) * scale], axis=1)


def mix_into(bus: Audio, clip: Audio, start_s: float, sr: int, gain: float = 1.0) -> None:
    """Add ``clip`` into ``bus`` at ``start_s``; whatever falls outside is dropped."""
    start = round(start_s * sr)
    lo, hi = max(start, 0), min(start + len(clip), len(bus))
    if hi > lo:
        bus[lo:hi] += gain * clip[lo - start : hi - start]


def rms_over(x: Audio, spans: Sequence[tuple[int, int]]) -> float:
    """RMS over the given ``(start, stop)`` sample spans taken together."""
    parts = [x[a:b] for a, b in spans if b > a]
    return rms(np.concatenate(parts)) if parts else 0.0


def gain_for_level(x: Audio, target_dbfs: float, spans: Sequence[tuple[int, int]]) -> float:
    """Gain that puts the RMS of ``x`` over ``spans`` at ``target_dbfs``.

    A silent input has no level to set, so its gain is 1.
    """
    level = rms_over(x, spans)
    return db_to_gain(target_dbfs) / level if level > _EPS else 1.0


# ---- Filters and EQ --------------------------------------------------------


def butter(
    x: Audio,
    kind: Literal["highpass", "lowpass", "bandpass"],
    freq_hz: float | tuple[float, float],
    sr: int,
    order: int = 2,
    *,
    zero_phase: bool = False,
) -> Audio:
    """Butterworth filter along axis 0: causal, or forward-backward
    (``zero_phase``) where the output must stay time-aligned with the input."""
    sos = signal.butter(order, freq_hz, btype=kind, fs=sr, output="sos")
    run = signal.sosfiltfilt if zero_phase else signal.sosfilt
    return np.asarray(run(sos, x, axis=0), dtype=np.float64)


def _biquad(band: EqBand, sr: int) -> npt.NDArray[np.float64]:
    """RBJ Audio EQ Cookbook coefficients as one second-order section."""
    a = 10.0 ** (band.gain_db / 40.0)
    w0 = 2.0 * np.pi * band.freq_hz / sr
    cos, sin = np.cos(w0), np.sin(w0)
    if band.kind == "peak":
        alpha = sin / (2.0 * band.q)
        b = [1.0 + alpha * a, -2.0 * cos, 1.0 - alpha * a]
        den = [1.0 + alpha / a, -2.0 * cos, 1.0 - alpha / a]
    else:
        alpha = sin / 2.0 * np.sqrt(2.0)  # shelf slope S = 1
        k = 2.0 * np.sqrt(a) * alpha
        if band.kind == "high_shelf":
            b = [
                a * ((a + 1.0) + (a - 1.0) * cos + k),
                -2.0 * a * ((a - 1.0) + (a + 1.0) * cos),
                a * ((a + 1.0) + (a - 1.0) * cos - k),
            ]
            den = [
                (a + 1.0) - (a - 1.0) * cos + k,
                2.0 * ((a - 1.0) - (a + 1.0) * cos),
                (a + 1.0) - (a - 1.0) * cos - k,
            ]
        elif band.kind == "low_shelf":
            b = [
                a * ((a + 1.0) - (a - 1.0) * cos + k),
                2.0 * a * ((a - 1.0) - (a + 1.0) * cos),
                a * ((a + 1.0) - (a - 1.0) * cos - k),
            ]
            den = [
                (a + 1.0) + (a - 1.0) * cos + k,
                -2.0 * ((a - 1.0) + (a + 1.0) * cos),
                (a + 1.0) + (a - 1.0) * cos - k,
            ]
        else:  # pragma: no cover - guarded by the EqKind literal
            raise ValueError(f"Unknown EQ band kind {band.kind!r}")
    return np.array(
        [b[0] / den[0], b[1] / den[0], b[2] / den[0], 1.0, den[1] / den[0], den[2] / den[0]]
    )


def eq_sos(bands: Sequence[EqBand], sr: int) -> npt.NDArray[np.float64]:
    """The bands as an ``(n, 6)`` second-order-sections array."""
    return np.array([_biquad(b, sr) for b in bands]).reshape(-1, 6)


def apply_eq(x: Audio, bands: Sequence[EqBand], sr: int) -> Audio:
    if not bands:
        return x
    return np.asarray(signal.sosfilt(eq_sos(bands, sr), x, axis=0), dtype=np.float64)


# ---- Dynamics --------------------------------------------------------------


def block_envelope(
    peak: npt.NDArray[np.float64], block: int, attack_s: float, release_s: float, sr: int
) -> npt.NDArray[np.float64]:
    """One-pole attack/release follower of the per-block peak of ``peak``."""
    n_blocks = -(-len(peak) // block)
    padded = np.zeros(n_blocks * block)
    padded[: len(peak)] = peak
    blocks = padded.reshape(n_blocks, block).max(axis=1)
    attack = float(np.exp(-block / (sr * attack_s)))
    release = float(np.exp(-block / (sr * release_s)))
    env = np.empty(n_blocks)
    level = 0.0
    for i, p in enumerate(blocks.tolist()):
        coef = attack if p > level else release
        level = coef * level + (1.0 - coef) * p
        env[i] = level
    return env


def compressor_gain(
    x: Audio, threshold: float, cfg: CompressorConfig, sr: int
) -> npt.NDArray[np.float64]:
    """Per-sample gain of a stereo-linked feed-forward compressor."""
    if threshold <= 0.0 or len(x) == 0:
        return np.ones(len(x))
    env = block_envelope(np.max(np.abs(x), axis=1), cfg.block, cfg.attack_s, cfg.release_s, sr)
    gain = np.ones_like(env)
    over = env > threshold
    gain[over] = (threshold + (env[over] - threshold) / cfg.ratio) / env[over]
    centres = np.arange(len(env)) * cfg.block + cfg.block / 2.0
    return np.asarray(np.interp(np.arange(len(x)), centres, gain), dtype=np.float64)


def compress(x: Audio, threshold: float, cfg: CompressorConfig, sr: int) -> Audio:
    return x * compressor_gain(x, threshold, cfg, sr)[:, None]


def saturate(x: Audio, drive: float, *, normalize: bool) -> Audio:
    """``tanh(drive x)`` scaled by ``1/tanh(drive)`` (peak-preserving) or ``1/drive``."""
    scale = np.tanh(drive) if normalize else drive
    return np.asarray(np.tanh(drive * x) / scale, dtype=np.float64)


def soft_limit(x: Audio, ceiling: float) -> Audio:
    """``tanh(x / c) * c``: transparent well below ``ceiling``, never above it."""
    if ceiling <= 0.0:
        return x
    return np.asarray(np.tanh(x / ceiling) * ceiling, dtype=np.float64)


# ---- Reverb ----------------------------------------------------------------


def _impulse(cfg: ReverbConfig, seed: int, sr: int) -> npt.NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    n = int(cfg.length_t60 * cfg.t60_s * sr)
    t = np.arange(n) / sr
    body = np.zeros(n)
    for i, k in enumerate(cfg.band_decay):
        lo, hi = cfg.band_edges_hz[i], cfg.band_edges_hz[i + 1]
        sos = signal.butter(2, (lo, hi), btype="bandpass", fs=sr, output="sos")
        noise = signal.sosfilt(sos, rng.standard_normal(n))
        decay = k * (cfg.bright if i in cfg.bright_bands else 1.0)
        body += noise * np.exp(-6.91 * t / (cfg.t60_s * decay))
    body *= np.minimum(1.0, t / cfg.build_up_s) ** cfg.build_up_curve
    body /= max(float(np.max(np.abs(body))), _EPS)
    times = np.sort(rng.uniform(cfg.early_min_s, cfg.early_max_s, cfg.early_reflections))
    signs = rng.choice([-1.0, 1.0], cfg.early_reflections)
    for j, (when, sign) in enumerate(zip(times, signs, strict=True)):
        body[int(when * sr)] += sign * cfg.early_gain * (1.0 - 0.5 * j / cfg.early_reflections)
    ir = np.concatenate([np.zeros(int(cfg.pre_delay_s * sr)), body])
    return np.asarray(ir / np.sqrt(np.sum(np.square(ir))), dtype=np.float64)


@lru_cache(maxsize=8)
def impulse_response(cfg: ReverbConfig, sr: int) -> Audio:
    """Stereo impulse response, unit energy per channel, ``(samples, 2)``."""
    return np.stack([_impulse(cfg, cfg.seed_left, sr), _impulse(cfg, cfg.seed_right, sr)], axis=1)


def reverb(x: Audio, cfg: ReverbConfig, sr: int) -> Audio:
    """Wet signal only, the same length as ``x`` (the tail is cut)."""
    ir = impulse_response(cfg, sr)
    wet = np.empty_like(x)
    for c in range(2):
        wet[:, c] = signal.fftconvolve(x[:, c], ir[:, c])[: len(x)]
    return wet


# ---- Resampling ------------------------------------------------------------


def oversample(x: npt.NDArray[np.float64], factor: int) -> npt.NDArray[np.float64]:
    return np.asarray(signal.resample_poly(x, factor, 1, axis=0), dtype=np.float64)


def downsample(x: npt.NDArray[np.float64], factor: int) -> npt.NDArray[np.float64]:
    return np.asarray(signal.resample_poly(x, 1, factor, axis=0), dtype=np.float64)


def true_peak(x: Audio, factor: int = 4) -> float:
    """Largest inter-sample peak, estimated by ``factor``-times oversampling."""
    if x.size == 0:
        return 0.0
    return float(np.max(np.abs(oversample(x, factor))))
