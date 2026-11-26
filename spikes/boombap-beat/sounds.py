"""THROWAWAY spike: kick, snare and bass built to the numbers measured on the second reference
(youtube tTA4d3__n28, 88.0 bpm). None of its audio is used.

kick   an 808-style boom: ~90 Hz at the hit sliding to 40 Hz by 130 ms, level flat for ~140 ms
       then gone by 300 ms, no click, second harmonic 17 dB under the fundamental
snare  short, fat and dry: body at ~200 Hz, noise roughly flat from 150 Hz to 6 kHz, 20 dB down
       after ~110 ms
bass   a held sub note with a strong second harmonic (-5 dB), cut dead on the snare
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt

SUB_HARMONICS_DB = (0.0, -5.0, -17.0, -24.0, -23.0, -32.0)
_H = [10 ** (d / 20) for d in SUB_HARMONICS_DB]


def _bp(x, sr, lo, hi, order=2):
    return sosfilt(butter(order, [lo, hi], btype="band", fs=sr, output="sos"), x)


def kick(sr: int, rng: np.random.Generator) -> np.ndarray:
    n = int(0.5 * sr)
    t = np.arange(n) / sr
    f = 42.0 + 50.0 * np.exp(-((t / 0.042) ** 1.3))
    ph = 2 * np.pi * np.cumsum(f) / sr
    top = np.exp(-((t / 0.150) ** 8))
    tail = 0.28 * np.exp(-np.maximum(t - 0.14, 0.0) / 0.085)
    amp = np.maximum(top, tail) * np.minimum(1.0, t / 0.003)
    x = (np.sin(ph) + 0.14 * np.sin(2 * ph + 0.6) + 0.035 * np.sin(3 * ph) + 0.02 * np.sin(4 * ph)) * amp
    return x / np.abs(x).max()


def snare(sr: int, rng: np.random.Generator) -> np.ndarray:
    n = int(0.3 * sr)
    t = np.arange(n) / sr

    def env(fast, t_fast, t_slow):  # a crack, then a short tail
        return fast * np.exp(-t / t_fast) + (1.0 - fast) * np.exp(-t / t_slow)

    f = 200.0 + 60.0 * np.exp(-t / 0.012)
    body = np.sin(2 * np.pi * np.cumsum(f) / sr) + 0.45 * np.sin(2 * np.pi * 331.0 * t + 1.0)
    w = rng.standard_normal(n)
    x = (
        0.8 * body * env(0.5, 0.005, 0.042)
        + 3.2 * _bp(w, sr, 230, 850) * env(0.4, 0.008, 0.048)
        + 1.5 * _bp(w, sr, 850, 3000) * env(0.2, 0.01, 0.058)
        + 0.8 * _bp(w, sr, 3000, 5500) * env(0.0, 0.01, 0.055)
    )
    x *= np.minimum(1.0, t / 0.0008) * np.exp(-((t / 0.17) ** 4))
    x = np.tanh(1.3 * x / np.abs(x).max())
    x = sosfilt(butter(2, 9500, btype="low", fs=sr, output="sos"), x)
    return x / np.abs(x).max()


def hat(sr: int, rng: np.random.Generator) -> np.ndarray:
    n = int(0.06 * sr)
    t = np.arange(n) / sr
    x = _bp(rng.standard_normal(n), sr, 3000, 9000) * np.exp(-t / 0.012)
    return x / np.abs(x).max()


def bass(pitch: float, length_s: float, sr: int, start_pitch: float | None = None) -> np.ndarray:
    n = int((length_s + 0.08) * sr)
    t = np.arange(n) / sr
    start = pitch if start_pitch is None else start_pitch
    note = pitch + (start - pitch) * np.exp(-t / 0.03)
    ph = 2 * np.pi * np.cumsum(440.0 * 2.0 ** ((note - 69.0) / 12.0)) / sr
    tone = sum(a * np.sin(k * ph) for k, a in enumerate(_H, start=1))
    amp = np.minimum(1.0, t / 0.02) * np.where(t > length_s, np.exp(-(t - length_s) / 0.012), 1.0)
    return tone / np.abs(tone).max() * amp
