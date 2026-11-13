"""Synthesized bass: one sine oscillator with continuous phase.

Notes join without restarts or clicks: pitch is a smoothed step curve, the
phase is its running sum, and a single gate opens at the first note and
closes with a decay where the bass sections end.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from music_decoder.productionize import dsp
from music_decoder.productionize.config import BassConfig
from music_decoder.productionize.dsp import Audio
from music_decoder.productionize.song import Song


@dataclass(frozen=True)
class BassEvent:
    time_s: float
    pitch: int
    accent: float


def midi_to_hz(pitch: float) -> float:
    return float(440.0 * 2.0 ** ((pitch - 69.0) / 12.0))


def bass_events(song: Song, cfg: BassConfig) -> list[BassEvent]:
    """One note per configured step in every bar of every bass section."""
    g = song.grid
    return [
        BassEvent(g.time(index, bar, step), song.bar_roots[bar], accent)
        for index, section in enumerate(song.sections)
        if section.bass
        for bar in range(g.bars_per_section)
        for step, accent in cfg.events
    ]


def bass_runs(song: Song) -> list[tuple[float, float]]:
    """(start, end) seconds of each unbroken stretch of bass sections."""
    runs: list[tuple[float, float]] = []
    start: int | None = None
    for index, section in enumerate([*song.sections, None]):
        playing = section is not None and section.bass
        if playing and start is None:
            start = index
        elif not playing and start is not None:
            runs.append((song.grid.time(start), song.grid.time(index)))
            start = None
    return runs


def kick_envelope(
    kick_times_s: Sequence[float], cfg: BassConfig, n_samples: int, sr: int
) -> npt.NDArray[np.float64]:
    """Maximum over all kicks of a short exponential; drives the ducking."""
    env = np.zeros(n_samples)
    shape = np.exp(-np.arange(int(cfg.duck_length_s * sr)) / (cfg.duck_tau_s * sr))
    for when in kick_times_s:
        start = round(when * sr)
        lo, hi = max(start, 0), min(start + len(shape), n_samples)
        if hi > lo:
            np.maximum(env[lo:hi], shape[lo - start : hi - start], out=env[lo:hi])
    return env


def render_bass(
    song: Song, cfg: BassConfig, kick_times_s: Sequence[float], n_samples: int
) -> Audio:
    """The bass, mono on both channels (unlevelled)."""
    sr = song.grid.sample_rate
    events = bass_events(song, cfg)
    if not events:
        return np.zeros((n_samples, 2))

    # Pitch: a step curve of root frequencies with a short glide.
    freq = np.full(n_samples, midi_to_hz(events[0].pitch))
    accent = np.zeros(n_samples)
    tail = np.exp(-np.arange(int(cfg.accent_length_s * sr)) / (cfg.accent_tau_s * sr))
    for event in events:
        start = min(max(round(event.time_s * sr), 0), n_samples)
        freq[start:] = midi_to_hz(event.pitch)
        stop = min(start + len(tail), n_samples)
        np.maximum(accent[start:stop], event.accent * tail[: stop - start], out=accent[start:stop])
    freq = dsp.moving_average(freq, int(cfg.glide_s * sr))
    phase = 2.0 * np.pi * np.cumsum(freq) / sr

    # Gate: open through each run of bass sections, then a decaying release.
    gate = np.zeros(n_samples)
    release_len = int(cfg.release_beats * song.grid.beat_s * sr)
    t = np.arange(release_len) / sr
    release = np.exp(-t / cfg.release_tau_s) * (1.0 - t / (release_len / sr)) ** cfg.release_curve
    for start_s, end_s in bass_runs(song):
        start, end = round(start_s * sr), min(round(end_s * sr), n_samples)
        gate[start:end] = 1.0
        stop = min(end + release_len, n_samples)
        gate[end:stop] = np.maximum(gate[end:stop], release[: stop - end])
    window = np.hanning(max(int(cfg.gate_smooth_s * sr), 3))
    gate = np.convolve(gate, window / window.sum(), mode="same")

    tone = np.sin(phase) + cfg.second_harmonic * np.sin(2.0 * phase)
    tone = np.tanh(cfg.drive * tone) / np.tanh(cfg.drive)
    mono = tone * (cfg.amp_base + cfg.amp_accent * accent) * gate

    out = dsp.to_stereo(mono)
    out = dsp.butter(out, "highpass", cfg.highpass_hz, sr)
    out = dsp.butter(out, "lowpass", cfg.lowpass_hz, sr)
    duck = 1.0 - cfg.duck_depth * kick_envelope(kick_times_s, cfg, n_samples, sr)
    return out * duck[:, None]
