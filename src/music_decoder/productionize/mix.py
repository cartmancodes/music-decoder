"""Pre-master mix: each element levelled on its own, then summed.

Every bus is rendered and processed alone, set to a fixed RMS level over
the verses, and summed. The sum is made mono below a crossover, gets a
little air, and is faded out. The master is the only stage that sees the
whole song.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from music_decoder.productionize import dsp
from music_decoder.productionize.config import MixConfig, ProductionConfig
from music_decoder.productionize.dsp import Audio
from music_decoder.productionize.instruments.bass import render_bass
from music_decoder.productionize.instruments.drums import render_drums
from music_decoder.productionize.instruments.percussion import render_percussion
from music_decoder.productionize.instruments.piano import render_piano, reverb_feed
from music_decoder.productionize.instruments.strings import render_strings
from music_decoder.productionize.samples import SampleStore
from music_decoder.productionize.song import Song
from music_decoder.progress import NullProgress, ProgressCallback

STEM_NAMES = ("drums", "bass", "piano", "strings", "kick", "snare", "hats", "crash")


@dataclass(frozen=True)
class MixResult:
    mix: Audio
    stems: dict[str, Audio]
    drum_onsets_s: tuple[float, ...]


def mix_length(song: Song, cfg: MixConfig) -> int:
    """Samples in the render buffer: the sections plus the tail."""
    return song.grid.samples(song.body_s + cfg.tail_s)


def mono_low_end(mix: Audio, cfg: MixConfig, sr: int) -> Audio:
    """Replace everything below the crossover with its mono average.

    The low-pass is zero-phase: subtracting a phase-shifted low band would
    leave most of the stereo difference in place instead of removing it.
    """
    low = dsp.butter(
        mix, "lowpass", cfg.mono_below_hz, sr, order=cfg.mono_filter_order, zero_phase=True
    )
    out: Audio = mix - low + low.mean(axis=1, keepdims=True)
    return out


def fade_out(mix: Audio, cfg: MixConfig, sr: int) -> Audio:
    n = min(int(cfg.fade_out_s * sr), len(mix))
    out = mix.copy()
    if n:
        out[-n:] *= ((1.0 - np.arange(n) / n) ** cfg.fade_out_curve)[:, None]
    return out


def finish(total: Audio, cfg: MixConfig, sr: int) -> Audio:
    """Mono low end, air shelf and fade-out on the summed buses."""
    out = mono_low_end(total, cfg, sr)
    out = dsp.apply_eq(out, (cfg.air,), sr)
    return fade_out(out, cfg, sr)


def render_mix(
    song: Song,
    cfg: ProductionConfig,
    store: SampleStore,
    work_dir: Path,
    progress: ProgressCallback | None = None,
) -> MixResult:
    """Render every element, level it, and sum to the pre-master mix."""
    cb: ProgressCallback = progress if progress is not None else NullProgress()
    sr = song.grid.sample_rate
    n = mix_length(song, cfg.mix)
    verses = [song.section_span(i) for i in song.verse_sections]

    cb("drums", 0.0)
    drums = render_drums(song, cfg.drums, cfg.plate, store, n)
    drum_gain = dsp.gain_for_level(drums.bus, cfg.mix.drums_dbfs, verses)

    cb("bass", 0.0)
    bass = render_bass(song, cfg.bass, drums.kick_times_s, n)
    bass = bass * dsp.gain_for_level(bass, cfg.mix.bass_dbfs, verses)

    cb("piano", 0.0)
    piano = render_piano(song, cfg.piano, store, n)
    piano = piano * dsp.gain_for_level(piano, cfg.mix.piano_dbfs, verses)
    piano = piano + dsp.reverb(reverb_feed(piano, song, cfg.piano), cfg.hall, sr)

    cb("strings", 0.0)
    strings = render_strings(song, cfg.strings, work_dir, n)
    string_spans = [song.section_span(i) for i in song.strings_sections[:1]]
    strings = strings * dsp.gain_for_level(strings, cfg.mix.strings_dbfs, string_spans)
    strings = strings + dsp.db_to_gain(cfg.strings.reverb_send_db) * dsp.reverb(
        strings, cfg.hall, sr
    )

    stems: dict[str, Audio] = {
        "drums": drums.bus * drum_gain,
        "bass": bass,
        "piano": piano,
        "strings": strings,
        "kick": drums.kick * drum_gain,
        "snare": drums.snare * drum_gain,
        "hats": drums.hats * drum_gain,
        "crash": drums.crash * drum_gain,
    }
    total = stems["drums"] + bass + piano + strings
    if cfg.percussion.layers:
        cb("percussion", 0.0)
        stems["percussion"] = render_percussion(song, cfg.percussion, cfg.drums, cfg.hall, store, n)
        total = total + stems["percussion"]
    cb("mix", 0.0)
    return MixResult(finish(total, cfg.mix, sr), stems, drums.onsets_s)
