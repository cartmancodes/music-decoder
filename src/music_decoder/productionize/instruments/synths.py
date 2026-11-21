"""Synthesized pads, plucks and leads for the pop-rap style.

Every clip is mono float64. The pad takes its detune phases from the caller's
generator, so a clip is reproducible per generator. The guitar lead is the one
voice rendered by FluidSynth (see :mod:`fluid`); the rest are oscillators.
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy import signal

from music_decoder.productionize import dsp
from music_decoder.productionize.config import (
    GuitarLeadConfig,
    PadConfig,
    PluckConfig,
    SynthLeadConfig,
)
from music_decoder.productionize.dsp import Audio
from music_decoder.productionize.instruments import fluid
from music_decoder.productionize.instruments.bass import midi_to_hz
from music_decoder.productionize.song import NoteEvent, ProductionError

# The synth lead's pitch before its first note, so a glide never starts from silence's zero.
_LEAD_REST_PITCH = 61
_ENVELOPE_FLOOR_S = 1e-4
_PLUCK_OCTAVE = 12  # the square wave sits an octave above the triangle
_GUITAR_STEM = "guitar_lead"
_EPS = 1e-9


def saw(freq_hz: npt.NDArray[np.float64], sr: int, phase0: float = 0.0) -> npt.NDArray[np.float64]:
    """Falling saw in [-1, 1], band-limited with PolyBLEP, over a (gliding) frequency.

    ``freq_hz`` is per sample; ``phase0`` is the starting phase in radians.
    """
    phase = phase0 / (2.0 * np.pi) + np.cumsum(freq_hz) / sr
    p = phase - np.floor(phase)
    dt = np.maximum(freq_hz / sr, _EPS)
    y = 1.0 - 2.0 * p
    after = p < dt  # just past the wrap: the saw jumps up here
    t = np.where(after, p / dt, 0.0)
    y += np.where(after, 2.0 * t - t * t - 1.0, 0.0)
    before = p > 1.0 - dt  # just before it
    t = np.where(before, (p - 1.0) / dt, 0.0)
    y += np.where(before, t * t + 2.0 * t + 1.0, 0.0)
    return np.asarray(y, dtype=np.float64)


def _adsr(
    n: int, attack_s: float, decay_s: float, sustain: float, release_s: float, hold: int, sr: int
) -> npt.NDArray[np.float64]:
    """Attack, exponential decay to ``sustain``, then exponential release from sample ``hold``."""
    t = np.arange(n) / sr
    env = np.where(
        t < attack_s,
        t / max(attack_s, _ENVELOPE_FLOOR_S),
        sustain + (1.0 - sustain) * np.exp(-(t - attack_s) / max(decay_s, _ENVELOPE_FLOOR_S)),
    )
    since_release = (np.arange(n) - hold) / sr
    released = env[min(hold, n - 1)] * np.exp(
        -np.maximum(since_release, 0.0) / max(release_s, _ENVELOPE_FLOOR_S)
    )
    return np.asarray(np.where(np.arange(n) >= hold, released, env), dtype=np.float64)


def pad_chord(
    pitches: Sequence[int], length_s: float, cfg: PadConfig, sr: int, rng: np.random.Generator
) -> Audio:
    """A supersaw chord: every pitch gets one detuned saw per ``cfg.detune_semitones``.

    The clip is ``length_s + cfg.release_tail_s`` long; the release starts at ``length_s``.
    """
    n = int((length_s + cfg.release_tail_s) * sr)
    x = np.zeros(n)
    for pitch in pitches:
        for detune in cfg.detune_semitones:
            freq = midi_to_hz(pitch + detune)
            x += saw(np.full(n, freq), sr, rng.uniform(0.0, 2.0 * np.pi))
    x = dsp.butter(x, "lowpass", cfg.lowpass_hz, sr, stable=True)
    envelope = _adsr(
        n, cfg.attack_s, cfg.decay_s, cfg.sustain, cfg.release_s, int(length_s * sr), sr
    )
    return np.asarray(x * envelope, dtype=np.float64)


def pluck(pitch: int, length_s: float, velocity: float, cfg: PluckConfig, sr: int) -> Audio:
    """A soft stab: a triangle and a quiet square an octave up, decaying and low-passed.

    ``velocity`` is a 0-1 gain. The clip is ``min(length_s, cfg.max_length_s) + cfg.tail_s`` long.
    """
    n = int((min(length_s, cfg.max_length_s) + cfg.tail_s) * sr)
    t = np.arange(n) / sr
    triangle = signal.sawtooth(2.0 * np.pi * midi_to_hz(pitch) * t, 0.5)
    square = signal.square(2.0 * np.pi * midi_to_hz(pitch + _PLUCK_OCTAVE) * t) * cfg.square_level
    x = (triangle + square) * np.exp(-t / cfg.decay_s) * velocity
    return dsp.butter(x, "lowpass", cfg.lowpass_hz, sr, stable=True)


def _one_pole(x: npt.NDArray[np.float64], time_s: float, sr: int) -> npt.NDArray[np.float64]:
    """Exponential smoothing with time constant ``time_s``, starting at ``x[0]``."""
    a = float(np.exp(-1.0 / (time_s * sr)))
    smoothed, _ = signal.lfilter([1.0 - a], [1.0, -a], x, zi=[x[0] * a])
    return np.asarray(smoothed, dtype=np.float64)


def synth_lead(notes: Sequence[NoteEvent], total_s: float, cfg: SynthLeadConfig, sr: int) -> Audio:
    """A legato lead: two detuned saws and a sub square, gliding between notes.

    Notes are at the melody's pitch; the lead plays ``cfg.octave`` above them. The
    pitch holds through rests (so each glide starts from the last note) and a short
    gate smooths the note edges. Long notes get a little vibrato.
    """
    n = int(total_s * sr)
    target = np.full(n, np.nan)
    gate = np.zeros(n)
    for note in notes:
        a, b = int(note.start_s * sr), min(int(note.end_s * sr), n)
        target[a:b] = midi_to_hz(note.pitch + cfg.octave)
        gate[a:b] = 1.0
    held = np.where(~np.isnan(target), np.arange(n), 0)
    np.maximum.accumulate(held, out=held)
    freq = np.nan_to_num(target[held], nan=midi_to_hz(_LEAD_REST_PITCH))
    freq = _one_pole(freq, cfg.glide_s, sr)
    t = np.arange(n) / sr
    freq *= 2.0 ** (cfg.vibrato_semitones / 12.0 * np.sin(2.0 * np.pi * cfg.vibrato_hz * t))
    x = saw(freq, sr)
    x += saw(freq * 2.0 ** (cfg.detune_semitones / 12.0), sr, 1.0) * cfg.second_saw
    x += signal.square(np.cumsum(2.0 * np.pi * freq / 2.0) / sr) * cfg.sub_square
    x *= _one_pole(gate, cfg.gate_s, sr)
    return dsp.butter(x, "lowpass", cfg.lowpass_hz, sr, stable=True)


def guitar_lead(
    notes: Sequence[NoteEvent],
    total_s: float,
    cfg: GuitarLeadConfig,
    tempo_bpm: float,
    work_dir: Path,
    sr: int,
) -> Audio:
    """The melody on a soundfont guitar, then a small amp: a mid push, drive and a speaker roll-off.

    FluidSynth's left and right are averaged to mono and normalized before the amp, so
    ``cfg.fluidsynth_gain`` only sets the render level, not the result.
    """
    if shutil.which("fluidsynth") is None:
        raise ProductionError(
            "the guitar lead needs the fluidsynth command-line tool "
            "(brew install fluidsynth / apt install fluidsynth)"
        )
    soundfont = fluid.find_soundfont(cfg.soundfont, "guitar lead")
    n = int(total_s * sr)
    played = [
        NoteEvent(note.start_s, note.end_s, note.pitch + cfg.octave, cfg.velocity) for note in notes
    ]
    stereo = fluid.render_notes(
        played,
        cfg.program,
        tempo_bpm,
        soundfont,
        cfg.fluidsynth_gain,
        work_dir,
        _GUITAR_STEM,
        sr,
        n,
    )
    x = stereo.mean(axis=1)
    x = x / (np.abs(x).max() + _EPS)
    mid = dsp.butter(x, "bandpass", cfg.mid_band_hz, sr, stable=True)
    x = mid * cfg.mid_gain + x * cfg.dry_gain
    x = dsp.saturate(x, cfg.drive, normalize=True)
    x = dsp.butter(x, "lowpass", cfg.lowpass_hz, sr, stable=True)
    return dsp.butter(x, "highpass", cfg.highpass_hz, sr, stable=True)
