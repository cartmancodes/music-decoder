"""Optional percussion layers (shaker, tambourine, congas); off by default.

Close-mic mono samples on the 16th grid, with the drums' round robin and
humanizing. Hand drums are aligned at their onset; shakes at their loudest
point, so the swell lands on the grid.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from music_decoder.productionize import dsp
from music_decoder.productionize.config import (
    PERCUSSION_OPTIONS,
    DrumsConfig,
    PercussionConfig,
    PercussionLayer,
    PercussionOption,
    PercussionVoice,
    ReverbConfig,
)
from music_decoder.productionize.dsp import Audio
from music_decoder.productionize.samples import (
    RoundRobin,
    SampleStore,
    percussion_ref,
    percussion_refs,
    resample,
)
from music_decoder.productionize.song import Song


def prepare(
    raw: npt.NDArray[np.float64], voice: PercussionVoice, cfg: PercussionConfig, sr: int
) -> tuple[npt.NDArray[np.float64], int]:
    """(mono clip cut, faded and normalized to peak 1; samples from the clip's
    start to the point that goes on the grid)."""
    mono = raw.mean(axis=1) if raw.ndim == 2 else raw
    level = np.abs(mono)
    if voice.align == "peak":
        anchor = int(np.argmax(dsp.moving_average(level, int(cfg.peak_window_s * sr))))
        start = 0
    else:
        above = np.nonzero(level > cfg.onset_threshold * level.max())[0]
        start = max(0, int(above[0]) - int(cfg.onset_pre_roll_s * sr)) if len(above) else 0
        anchor = 0
    clip = mono[start : start + int(voice.cut_s * sr)].copy()
    fade = min(int(cfg.fade_s * sr), len(clip))
    if fade:
        clip[-fade:] *= np.linspace(1.0, 0.0, fade)
    clip /= max(float(np.max(np.abs(clip))), 1e-12)
    return clip, min(anchor, len(clip))


def _takes(
    store: SampleStore, voice: PercussionVoice, cfg: PercussionConfig, sr: int
) -> list[tuple[npt.NDArray[np.float64], int]]:
    """Every round robin of ``voice``, prepared."""
    takes = []
    for rr in range(1, voice.round_robins + 1):
        raw, rate = store.load(percussion_ref(voice, rr))
        takes.append(prepare(resample(raw, rate, sr), voice, cfg, sr))
    return takes


def render_layer(
    song: Song,
    layer: PercussionLayer,
    option: PercussionOption,
    cfg: PercussionConfig,
    drums: DrumsConfig,
    hall: ReverbConfig,
    store: SampleStore,
    rng: np.random.Generator,
    n_samples: int,
) -> Audio:
    """One percussion option over its sections: panned, levelled, with reverb."""
    sr = song.grid.sample_rate
    sections = [i for i in layer.sections if 0 <= i < len(song.sections)]
    if not sections:
        return np.zeros((n_samples, 2))
    clips = [_takes(store, voice, cfg, sr) for voice in option.voices]
    robins = [RoundRobin(len(takes), rng) for takes in clips]
    mono = np.zeros(n_samples)
    for index in sections:
        for bar in range(song.grid.bars_per_section):
            for hit in option.pattern:
                clip, anchor = clips[hit.voice][robins[hit.voice].next()]
                when = song.grid.time(index, bar, hit.step) + float(
                    rng.uniform(-drums.humanize_time_s, drums.humanize_time_s)
                )
                gain = dsp.db_to_gain(
                    hit.gain_db
                    + float(rng.uniform(-drums.humanize_gain_db, drums.humanize_gain_db))
                )
                start = round(when * sr) - anchor
                lo, hi = max(start, 0), min(start + len(clip), n_samples)
                if hi > lo:
                    mono[lo:hi] += gain * clip[lo - start : hi - start]
    out = dsp.to_stereo(mono)
    out = dsp.apply_eq(dsp.butter(out, "highpass", option.highpass_hz, sr), option.eq, sr)
    out = dsp.constant_power_pan(out[:, 0], option.pan)
    out = out * dsp.gain_for_level(out, option.level_dbfs, [song.section_span(i) for i in sections])
    return out + dsp.db_to_gain(option.hall_send_db) * dsp.reverb(out, hall, sr)


def render_percussion(
    song: Song,
    cfg: PercussionConfig,
    drums: DrumsConfig,
    hall: ReverbConfig,
    store: SampleStore,
    n_samples: int,
) -> Audio:
    """Every enabled percussion layer, summed (silence when none is on)."""
    out = np.zeros((n_samples, 2))
    if not cfg.layers:
        return out
    store.ensure(percussion_refs(layer.option for layer in cfg.layers))
    rng = np.random.default_rng(drums.seed)
    for layer in cfg.layers:
        option = PERCUSSION_OPTIONS[layer.option]
        out += render_layer(song, layer, option, cfg, drums, hall, store, rng, n_samples)
    return out
