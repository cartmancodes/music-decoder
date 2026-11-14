"""Acoustic boom-bap kit: every hit on an 8th note, groove through bar 4.

:func:`drum_pattern` is the arrangement (pure, no audio). :func:`render_drums`
builds each hit from the multi-mic samples, humanizes it, and processes the
kick, snare, hat and crash buses into one compressed drum bus.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from music_decoder.productionize import dsp
from music_decoder.productionize.config import DrumPiece, DrumsConfig, ReverbConfig, ThumpConfig
from music_decoder.productionize.dsp import Audio
from music_decoder.productionize.samples import (
    RoundRobin,
    SampleStore,
    drum_ref,
    drum_refs,
    drum_takes,
    resample,
)
from music_decoder.productionize.song import Song

Piece = Literal["kick", "snare", "hat", "open_hat", "crash"]


@dataclass(frozen=True)
class DrumHit:
    """One scheduled hit. ``layers`` are the velocity layers it may use."""

    piece: Piece
    section: int
    time_s: float
    gain_db: float
    layers: tuple[int, ...]
    choke_s: float | None = None  # open hat: when its choke starts


@dataclass(frozen=True)
class DrumRender:
    bus: Audio
    kick: Audio
    snare: Audio
    hats: Audio
    crash: Audio
    kick_times_s: tuple[float, ...]
    onsets_s: tuple[float, ...]  # every hit as played (humanized)


def drum_pattern(song: Song, cfg: DrumsConfig) -> list[DrumHit]:
    """Every drum hit of the song on the grid, before humanizing."""
    g = song.grid
    hits: list[DrumHit] = []
    for index, section in enumerate(song.sections):
        mode = section.drums
        if mode == "none":
            continue
        light = mode == "light"
        if index > 0:
            into_hook = section.role == "hook"
            hits.append(
                DrumHit(
                    "crash",
                    index,
                    g.time(index),
                    (cfg.crash_light_hook_db if light else cfg.crash_hook_db)
                    if into_hook
                    else cfg.crash_other_db,
                    (cfg.crash_hook_layer if into_hook else cfg.crash_other_layer,),
                )
            )
        if mode == "outro":
            hits.append(
                DrumHit("kick", index, g.time(index), cfg.outro_kick_db, (cfg.kick_other_layer,))
            )
            continue
        for bar in range(g.bars_per_section):
            for step in cfg.kick_steps[bar % len(cfg.kick_steps)]:
                main = step == cfg.kick_main_step
                gain = (0.0 if main else cfg.kick_other_db) + (cfg.kick_light_db if light else 0.0)
                layer = cfg.kick_main_layer if main and not light else cfg.kick_other_layer
                hits.append(DrumHit("kick", index, g.time(index, bar, step), gain, (layer,)))
            for step in cfg.snare_steps:
                hits.append(
                    DrumHit(
                        "snare",
                        index,
                        g.time(index, bar, step),
                        cfg.snare_light_db if light else 0.0,
                        cfg.snare_light_layers if light else cfg.snare_layers,
                    )
                )
            last_bar = bar == g.bars_per_section - 1
            for step, gain in zip(cfg.hat_steps, cfg.hat_gains_db, strict=True):
                trim = cfg.hat_light_db if light else 0.0
                when = g.time(index, bar, step)
                if last_bar and step == cfg.open_hat_step:
                    hits.append(
                        DrumHit(
                            "open_hat",
                            index,
                            when,
                            cfg.open_hat_db + trim,
                            (cfg.hat_open_layer,),
                            choke_s=g.time(index + 1) + cfg.open_hat_choke_after_s,
                        )
                    )
                else:
                    layer = cfg.hat_light_layer if light else cfg.hat_layer
                    hits.append(DrumHit("hat", index, when, gain + trim, (layer,)))
    return sorted(hits, key=lambda h: (h.time_s, h.piece))


def swell_times(song: Song) -> list[float]:
    """Downbeats that get a reverse crash swell: every hook after the start."""
    return [
        song.grid.time(i)
        for i, s in enumerate(song.sections)
        if i > 0 and s.role == "hook" and s.drums != "none"
    ]


def shape(hit: Audio, piece: DrumPiece, sr: int) -> Audio:
    """Hold, exponential decay, then a short fade into the cut."""
    if piece.cut_s <= 0.0:
        return hit
    out = hit[: int(piece.cut_s * sr)].copy()
    t = np.arange(len(out)) / sr
    env = np.where(t < piece.hold_s, 1.0, np.exp(-(t - piece.hold_s) / piece.tau_s))
    fade = min(int(piece.fade_s * sr), len(out))
    if fade:
        env[-fade:] *= np.linspace(1.0, 0.0, fade)
    shaped: Audio = out * env[:, None]
    return shaped


def thump(cfg: ThumpConfig, sr: int) -> Audio:
    """Pitch-dropping sine under the kick (peak about 1)."""
    t = np.arange(int(cfg.length_s * sr)) / sr
    freq = cfg.base_hz + cfg.sweep_hz * np.exp(-t / cfg.sweep_tau_s)
    phase = 2.0 * np.pi * np.cumsum(freq) / sr
    env = np.minimum(1.0, t / cfg.attack_s) * np.exp(-t / cfg.decay_tau_s)
    return dsp.to_stereo(np.sin(phase) * env)


class _Kit:
    """Builds and caches hits; keeps one round robin per (piece, layers)."""

    def __init__(self, store: SampleStore, cfg: DrumsConfig, rng: np.random.Generator, sr: int):
        self._store, self._cfg, self._rng, self._sr = store, cfg, rng, sr
        self._hits: dict[tuple[str, int, int | None], Audio] = {}
        self._robins: dict[tuple[str, tuple[int, ...]], RoundRobin] = {}
        self.pieces: dict[Piece, DrumPiece] = {
            "kick": cfg.kick,
            "snare": cfg.snare,
            "hat": cfg.hat_closed,
            "open_hat": cfg.hat_open,
            "crash": cfg.crash,
        }

    def hit(self, piece: DrumPiece, layer: int, round_robin: int | None) -> Audio:
        """One stroke: mics summed in phase, peak 1, shaped."""
        key = (piece.articulation, layer, round_robin)
        if key not in self._hits:
            cfg, sr = self._cfg, self._sr
            mics = [
                resample(
                    self._store.load(drum_ref(piece, mic, layer, round_robin))[0],
                    cfg.source_rate,
                    sr,
                )
                for mic, _ in piece.mic_weights
            ]
            # One trim index for every mic keeps them in phase.
            first = np.max(np.abs(mics[0]), axis=1)
            above = np.nonzero(first > cfg.onset_threshold * first.max())[0]
            start = max(0, int(above[0]) - int(cfg.onset_pre_roll_s * sr)) if len(above) else 0
            length = max(len(m) for m in mics) - start
            total = np.zeros((length, 2))
            for mic, (_, weight) in zip(mics, piece.mic_weights, strict=True):
                part = mic[start:]
                total[: len(part)] += weight * (part if part.shape[1] == 2 else part[:, :1])
            total /= max(float(np.max(np.abs(total))), 1e-12)
            self._hits[key] = shape(total, piece, sr)
        return self._hits[key]

    def next_take(self, name: Piece, layers: tuple[int, ...]) -> Audio:
        piece = self.pieces[name]
        takes = drum_takes(piece, layers)
        robin = self._robins.setdefault((name, layers), RoundRobin(len(takes), self._rng))
        layer, rr = takes[robin.next()]
        return self.hit(piece, layer, rr)


def _choke(hit: Audio, start_s: float, fade_s: float, sr: int) -> Audio:
    """Squared fade to silence starting ``start_s`` into the hit."""
    start = max(0, int(start_s * sr))
    if start >= len(hit):
        return hit
    out = hit[: start + int(fade_s * sr)].copy()
    n = len(out) - start
    out[start:] *= ((1.0 - np.arange(n) / max(n, 1)) ** 2)[:, None]
    return out


def _swell(kit: _Kit, cfg: DrumsConfig, beat_s: float, sr: int) -> Audio:
    """The start of a crash, reversed, its last beat faded in with t squared."""
    crash = kit.hit(cfg.crash, cfg.crash_hook_layer, cfg.swell_round_robin)
    rev = crash[: int(cfg.swell_source_s * sr)][::-1]
    out = rev[-int(cfg.swell_beats * beat_s * sr) :].copy()
    return out * (np.linspace(0.0, 1.0, len(out)) ** 2)[:, None]


def render_drums(
    song: Song, cfg: DrumsConfig, plate: ReverbConfig, store: SampleStore, n_samples: int
) -> DrumRender:
    """The processed drum bus and its four element stems (unlevelled)."""
    sr = song.grid.sample_rate
    silence = np.zeros((n_samples, 2))
    pattern = drum_pattern(song, cfg)
    if not pattern:
        return DrumRender(silence, silence, silence, silence, silence, (), ())
    store.ensure(drum_refs(cfg))
    rng = np.random.default_rng(cfg.seed)
    kit = _Kit(store, cfg, rng, sr)
    buses: dict[str, Audio] = {
        k: np.zeros((n_samples, 2)) for k in ("kick", "snare", "hats", "crash")
    }
    bus_of: dict[Piece, str] = {
        "kick": "kick",
        "snare": "snare",
        "hat": "hats",
        "open_hat": "hats",
        "crash": "crash",
    }
    sub = thump(cfg.thump, sr)
    kick_times: list[float] = []
    onsets: list[float] = []
    for hit in pattern:
        when = hit.time_s + float(rng.uniform(-cfg.humanize_time_s, cfg.humanize_time_s))
        gain = dsp.db_to_gain(
            hit.gain_db + float(rng.uniform(-cfg.humanize_gain_db, cfg.humanize_gain_db))
        )
        clip = kit.next_take(hit.piece, hit.layers)
        if hit.choke_s is not None:
            clip = _choke(clip, hit.choke_s - when, cfg.open_hat_choke_fade_s, sr)
        dsp.mix_into(buses[bus_of[hit.piece]], clip, when, sr, gain)
        onsets.append(when)
        if hit.piece == "kick":
            dsp.mix_into(buses["kick"], sub, when, sr, cfg.thump.gain * gain)
            kick_times.append(when)
    swell = _swell(kit, cfg, song.grid.beat_s, sr)
    for downbeat in swell_times(song):
        dsp.mix_into(buses["crash"], swell, downbeat - len(swell) / sr, sr, cfg.swell_gain)

    verses = [song.section_span(i) for i in song.verse_sections]

    kick = dsp.apply_eq(
        dsp.butter(buses["kick"], "highpass", cfg.kick_highpass_hz, sr), cfg.kick_eq, sr
    )
    peak = dsp.percentile_peak(kick, cfg.kick_peak_percentile)
    if peak > 0.0:
        kick = kick * (dsp.db_to_gain(cfg.kick_peak_dbfs) / peak)
    kick_level = dsp.rms_over(kick, verses)

    def relative(x: Audio, db: float) -> Audio:
        """``x`` with its verse RMS ``db`` away from the kick's."""
        level = dsp.rms_over(x, verses)
        if kick_level <= 0.0 or level <= 0.0:
            return x
        return x * (kick_level * dsp.db_to_gain(db) / level)

    snare = dsp.apply_eq(
        dsp.butter(buses["snare"], "highpass", cfg.snare_highpass_hz, sr), cfg.snare_eq, sr
    )
    snare = relative(dsp.saturate(snare, cfg.snare_drive, normalize=True), cfg.snare_db)
    snare = snare + dsp.db_to_gain(cfg.snare_plate_send_db) * dsp.reverb(snare, plate, sr)

    hats = dsp.apply_eq(
        dsp.butter(buses["hats"], "highpass", cfg.hats_highpass_hz, sr), cfg.hats_eq, sr
    )
    hats = relative(dsp.butter(hats, "lowpass", cfg.hats_lowpass_hz, sr), cfg.hats_db)
    hats = hats * np.array([cfg.hats_left_gain, 1.0])

    crash = dsp.apply_eq(
        dsp.butter(buses["crash"], "highpass", cfg.crash_highpass_hz, sr), cfg.crash_eq, sr
    )
    crash = relative(crash, cfg.crash_db)

    bus = kick + snare + hats + crash
    bus = dsp.compress(bus, cfg.bus_compressor.rms_multiple * dsp.rms(bus), cfg.bus_compressor, sr)
    bus = dsp.saturate(bus, cfg.bus_drive, normalize=False)
    return DrumRender(bus, kick, snare, hats, crash, tuple(kick_times), tuple(onsets))
