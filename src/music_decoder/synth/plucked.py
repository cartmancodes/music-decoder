"""String-aware plucked-guitar renderer (composer v5, spec 5.1).

An extended Karplus-Strong voice per tabbed note: a noise burst, darker at low
velocity, shaped by a pluck-position comb, excites a string loop whose loss
depends on the string and the fret. A slight stiffness all-pass and a
fractional-delay tuning all-pass sit in the loop. Each note is muted over
:data:`MUTE_S` at its end. The summed voices are mixed with a three-biquad body
resonance, then peak-normalised like
:func:`~music_decoder.synth.fluidsynth_wrapper.synthesize_midi_to_wav`.

Everything is pinned here; the output depends only on the notes, the tuning,
the seed and the sample rate.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile
from numpy.typing import NDArray
from scipy.signal import lfilter

from music_decoder.types import Note, TabbedNote, TabPosition, Tuning

PLUCK_POSITION = 0.18  # pluck-position comb delay, as a fraction of the period
MUTE_S = 0.040  # finger-mute fade at a note's end
RELEASE_S = 0.5  # tail after the last note's end (the mute and the body ring)
T60_LOW_OPEN_S = 4.5  # open low string
T60_HIGH_FRETTED_S = 2.0  # top string, above fret 7
FRET_SATURATION = 8  # frets at or above this count as "high"
MAX_LOOP_GAIN = 0.9995
STIFFNESS = -0.12  # stiffness all-pass coefficient (negative: upper partials sharpen)
BRIGHT_POLE = 0.10  # excitation low-pass pole at the loudest velocity
DARK_POLE = 0.70  # ... and at the softest
MIN_VELOCITY, MAX_VELOCITY = 30, 118
BODY_MODES: tuple[tuple[float, float, float], ...] = (  # (Hz, Q, gain)
    (100.0, 6.0, 0.45),
    (200.0, 6.0, 0.30),
    (400.0, 5.0, 0.20),
)
PEAK = 0.9


def tab_for_midi_notes(notes: Sequence[Note], tuning: Tuning) -> tuple[TabbedNote, ...]:
    """Each note on the string giving it the lowest fret (the lower string on
    a tie), for rendering when no tab exists. Raises ``ValueError`` for a
    pitch below the lowest open string."""
    out: list[TabbedNote] = []
    for n in notes:
        frets = [(n.pitch - p, s) for s, p in enumerate(tuning.open_pitches) if n.pitch >= p]
        if not frets:
            raise ValueError(f"pitch {n.pitch} is below every open string of {tuning.name}")
        fret, string = min(frets)
        out.append(TabbedNote(note=n, position=TabPosition(string=string, fret=fret)))
    return tuple(out)


def _allpass_phase_delay(c: float, w: float) -> float:
    """Phase delay in samples of ``(c + z^-1) / (1 + c z^-1)`` at ``w`` rad/sample."""
    return 1.0 - 2.0 * math.atan2(c * math.sin(w), 1.0 + c * math.cos(w)) / w


def _allpass_for_delay(d: float, w: float) -> float:
    """The first-order all-pass coefficient whose phase delay at ``w`` is ``d``."""
    return math.sin((1.0 - d) * w / 2.0) / math.sin((1.0 + d) * w / 2.0)


def _t60(position: TabPosition, n_strings: int) -> float:
    """Decay time of the fundamental: long on open low strings, short high up
    the neck on the top strings (geometric in between)."""
    w = 0.5 * position.string / max(1, n_strings - 1)
    w += 0.5 * min(position.fret, FRET_SATURATION) / FRET_SATURATION
    return T60_LOW_OPEN_S * math.pow(T60_HIGH_FRETTED_S / T60_LOW_OPEN_S, w)


def _voice(
    f0: float, t60: float, velocity: int, n_samples: int, rng: np.random.Generator, sr: int
) -> NDArray[np.float64]:
    """One plucked string, ``n_samples`` long, before the end-of-note mute."""
    period = sr / f0
    n = round(period - 0.5)
    w0 = 2.0 * math.pi * f0 / sr

    # excitation: one period of noise, low-passed more at low velocity, then
    # the pluck-position comb
    soft = 1.0 - (min(max(velocity, MIN_VELOCITY), MAX_VELOCITY) - MIN_VELOCITY) / (
        MAX_VELOCITY - MIN_VELOCITY
    )
    pole = BRIGHT_POLE + (DARK_POLE - BRIGHT_POLE) * soft
    burst = rng.uniform(-1.0, 1.0, size=n)
    burst = np.asarray(lfilter([1.0 - pole], [1.0, -pole], burst - burst.mean()))
    comb = max(1, round(PLUCK_POSITION * n))
    burst[comb:] -= burst[: n - comb].copy()
    burst *= (velocity / 127.0) ** 1.5
    x = np.zeros(n_samples)
    x[: min(n, n_samples)] = burst[:n_samples]

    # string loop: z^-m * g(1 + z^-1)/2 * stiffness all-pass * tuning all-pass.
    # The averaging loss filter delays 0.5 samples; the tuning all-pass takes
    # up what the integer delay and the stiffness leave of the period.
    stiff_delay = _allpass_phase_delay(STIFFNESS, w0)
    rest = period - 0.5 - stiff_delay
    m = math.floor(rest - 0.5)  # leaves the tuning all-pass a delay in [0.5, 1.5)
    tune = _allpass_for_delay(rest - m, w0)
    # loop gain g*cos(w0/2) per period gives the T60; g < 1 keeps DC stable
    g = min(MAX_LOOP_GAIN, 10.0 ** (-3.0 / (f0 * t60)) / math.cos(w0 / 2.0))
    allpass_num = np.convolve([STIFFNESS, 1.0], [tune, 1.0])
    allpass_den = np.convolve([1.0, STIFFNESS], [1.0, tune])
    den = np.zeros(m + 4)
    den[m:] = -g / 2.0 * np.convolve([1.0, 1.0], allpass_num)
    den[: len(allpass_den)] += allpass_den
    return np.asarray(lfilter(allpass_den, den, x), dtype=np.float64)


def _bandpass(f: float, q: float, sr: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """RBJ band-pass biquad (0 dB peak)."""
    w = 2.0 * math.pi * f / sr
    alpha = math.sin(w) / (2.0 * q)
    b = np.array([alpha, 0.0, -alpha])
    a = np.array([1.0 + alpha, -2.0 * math.cos(w), 1.0 - alpha])
    return b / a[0], a / a[0]


def render_plucked(
    notes: Sequence[TabbedNote],
    tuning: Tuning,
    out_path: Path,
    *,
    seed: int = 0,
    sr: int = 44100,
) -> Path:
    """Render ``notes`` with the plucked-string voice to a 16-bit mono WAV.

    The pitch comes from the position (``tuning.open_pitches[string] +
    fret``); note ``i`` draws its excitation from
    ``numpy.random.default_rng(seed + i)``. The WAV lasts to the last note's
    end plus :data:`RELEASE_S`.
    """
    end_s = max((t.note.end_s for t in notes), default=0.0)
    total = max(1, round((end_s + RELEASE_S) * sr))
    mix = np.zeros(total)
    mute = 0.5 * (1.0 + np.cos(np.linspace(0.0, math.pi, round(MUTE_S * sr))))
    for i, t in enumerate(notes):
        pitch = tuning.open_pitches[t.position.string] + t.position.fret
        f0 = 440.0 * 2.0 ** ((pitch - 69) / 12.0)
        start = round(t.note.start_s * sr)
        stop = max(start + 1, round(t.note.end_s * sr))
        length = min(total - start, stop - start + len(mute))
        if length <= 0:
            continue
        y = _voice(
            f0,
            _t60(t.position, len(tuning.open_pitches)),
            t.note.velocity,
            length,
            np.random.default_rng(seed + i),
            sr,
        )
        tail = y[stop - start :]
        tail *= mute[: len(tail)]
        mix[start : start + length] += y

    body = np.zeros_like(mix)
    for f, q, gain in BODY_MODES:
        b, a = _bandpass(f, q, sr)
        body += gain * np.asarray(lfilter(b, a, mix))
    audio = (mix + body).astype(np.float32)
    peak = float(np.max(np.abs(audio)) or 1.0)
    audio = (audio / peak * PEAK).astype(np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(out_path), sr, (audio * 32767).astype(np.int16))
    return out_path
