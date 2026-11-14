"""Piano sampler over the Salamander Grand Piano V3.

Nearest sampled pitch shifted by resampling, 13 velocity layers, a
sustain-pedal rule (a note rings to the end of its bar) and re-strike
damping (a repeated key cuts its previous note).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from music_decoder.productionize import dsp
from music_decoder.productionize.config import PianoConfig
from music_decoder.productionize.dsp import Audio
from music_decoder.productionize.samples import SampleStore, piano_ref, resample
from music_decoder.productionize.song import NoteEvent, Song

# Extra audio read past a note's length: the silence before a sample's onset.
_LOAD_MARGIN_S = 0.5


@dataclass(frozen=True)
class Voice:
    """One note as the sampler plays it."""

    start_s: float
    hold_s: float  # seconds from the onset to the start of the release
    tau_s: float
    sample_pitch: int
    shift: int  # semitones above the sampled pitch
    layer: int


def velocity_layer(velocity: int, cfg: PianoConfig) -> int:
    """Velocity layer (1-based) for a MIDI velocity."""
    scaled = min(max(round(velocity * cfg.velocity_scale), 1), cfg.velocity_max)
    for layer, bound in enumerate(cfg.layer_bounds, start=1):
        if scaled <= bound:
            return layer
    return len(cfg.layer_bounds)


def nearest_sample(pitch: int, cfg: PianoConfig) -> tuple[int, int]:
    """(sampled pitch, semitone shift) for a MIDI pitch."""
    sample = min(cfg.sample_pitches, key=lambda p: (abs(pitch - p), p))
    return sample, pitch - sample


def plan_voices(song: Song, cfg: PianoConfig) -> list[Voice]:
    """Pedal and re-strike rules applied to the song's piano notes."""
    bar = song.grid.bar_s
    tail_start = song.tail_start_s - cfg.bar_snap_s
    notes: list[NoteEvent] = sorted(song.piano, key=lambda n: (n.start_s, n.pitch))
    holds: list[tuple[float, float]] = []  # (hold end, tau) per note
    for note in notes:
        if note.start_s >= tail_start:
            holds.append((note.end_s, cfg.tail_release_tau_s))
        else:
            bar_end = (np.floor((note.start_s + cfg.bar_snap_s) / bar) + 1.0) * bar
            holds.append((float(bar_end) - cfg.pedal_lift_s, cfg.release_tau_s))
    # A key struck again while it still sounds damps its previous note.
    sounding: dict[int, int] = {}
    for i, note in enumerate(notes):
        prev = sounding.get(note.pitch)
        if prev is not None:
            end, tau = holds[prev]
            if end + cfg.release_length_taus * tau > note.start_s:
                cut = note.start_s + cfg.restrike_delay_s
                if cut < end:
                    holds[prev] = (cut, cfg.restrike_tau_s)
        sounding[note.pitch] = i
    voices = []
    for note, (end, tau) in zip(notes, holds, strict=True):
        sample, shift = nearest_sample(note.pitch, cfg)
        voices.append(
            Voice(
                start_s=note.start_s,
                hold_s=max(end - note.start_s, 0.0),
                tau_s=tau,
                sample_pitch=sample,
                shift=shift,
                layer=velocity_layer(note.velocity, cfg),
            )
        )
    return voices


def _load(store: SampleStore, cfg: PianoConfig, voice: Voice, seconds: float, sr: int) -> Audio:
    """The pitch-shifted sample, trimmed to just before its onset.

    Only as much of the file as ``seconds`` of output needs is read and
    resampled (never more than ``max_sample_s``).
    """
    stretch = 2.0 ** (voice.shift / 12.0)  # a sample shifted up plays faster
    wanted = min(cfg.max_sample_s, (seconds + _LOAD_MARGIN_S) * stretch)
    raw, rate = store.load(piano_ref(voice.sample_pitch, voice.layer), max_seconds=wanted)
    if raw.shape[1] == 1:
        raw = np.repeat(raw, 2, axis=1)
    shifted = resample(
        raw, rate, sr, semitones=voice.shift, max_denominator=cfg.resample_max_denominator
    )
    level = np.max(np.abs(shifted), axis=1)
    above = np.nonzero(level > cfg.trim_threshold * level.max())[0]
    start = max(0, int(above[0]) - int(cfg.trim_pre_roll_s * sr)) if len(above) else 0
    return shifted[start:]


def _envelope(voice: Voice, cfg: PianoConfig, length: int, sr: int) -> npt.NDArray[np.float64]:
    """1 until the hold ends, then ``exp(-t/tau) * (1 - t/L)^curve`` over L."""
    hold = min(int(voice.hold_s * sr), length)
    env = np.ones(length)
    t = np.arange(length - hold) / sr
    span = cfg.release_length_taus * voice.tau_s
    env[hold:] = np.exp(-t / voice.tau_s) * np.clip(1.0 - t / span, 0.0, None) ** cfg.release_curve
    return env


def render_piano(song: Song, cfg: PianoConfig, store: SampleStore, n_samples: int) -> Audio:
    """The dry piano after EQ, compression and soft limit (unlevelled)."""
    sr = song.grid.sample_rate
    out = np.zeros((n_samples, 2))
    voices = plan_voices(song, cfg)
    if not voices:
        return out
    store.ensure({piano_ref(v.sample_pitch, v.layer) for v in voices})

    def sounding_s(voice: Voice) -> float:
        return voice.hold_s + cfg.release_length_taus * voice.tau_s

    longest: dict[tuple[int, int, int], float] = {}
    for voice in voices:
        key = (voice.sample_pitch, voice.layer, voice.shift)
        longest[key] = max(longest.get(key, 0.0), sounding_s(voice))
    cache: dict[tuple[int, int, int], Audio] = {}
    for voice in voices:
        key = (voice.sample_pitch, voice.layer, voice.shift)
        if key not in cache:
            cache[key] = _load(store, cfg, voice, longest[key], sr)
        clip = cache[key][: int(sounding_s(voice) * sr)]
        dsp.mix_into(out, clip * _envelope(voice, cfg, len(clip), sr)[:, None], voice.start_s, sr)

    mid = (out[:, 0] + out[:, 1]) / 2.0
    side = (out[:, 0] - out[:, 1]) / 2.0 * cfg.side_gain
    out = np.stack([mid + side, mid - side], axis=1)
    out = dsp.apply_eq(dsp.butter(out, "highpass", cfg.highpass_hz, sr), cfg.eq, sr)

    verses = [song.section_span(i) for i in song.verse_sections]
    threshold = cfg.compressor.rms_multiple * dsp.rms_over(out, verses)
    out = dsp.compress(out, threshold, cfg.compressor, sr)
    peak = dsp.percentile_peak(out, cfg.limit_percentile)
    return dsp.soft_limit(out, cfg.limit_headroom * peak)


def reverb_send_gain(song: Song, cfg: PianoConfig, n_samples: int) -> npt.NDArray[np.float64]:
    """Send level over time: louder in sections with ``reverb_boost``,
    crossfaded with a moving average."""
    sr = song.grid.sample_rate
    boost = np.zeros(n_samples)
    for index, section in enumerate(song.sections):
        if section.reverb_boost:
            start, stop = song.section_span(index)
            boost[start:stop] = 1.0
    boost = dsp.moving_average(boost, int(cfg.reverb_boost_smooth_s * sr))
    return dsp.db_to_gain(cfg.reverb_send_db) * (1.0 + (cfg.reverb_boost - 1.0) * boost)


def reverb_feed(dry: Audio, song: Song, cfg: PianoConfig) -> Audio:
    """What the hall hears: the piano band-limited, at the send level."""
    sr = song.grid.sample_rate
    low, high = cfg.reverb_band_hz
    band = dsp.butter(dsp.butter(dry, "highpass", low, sr), "lowpass", high, sr)
    feed: Audio = band * reverb_send_gain(song, cfg, len(dry))[:, None]
    return feed
