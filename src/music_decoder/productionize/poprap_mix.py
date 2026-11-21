"""Pop-rap pre-master mix: the layout rendered to buses, levelled and summed.

Every bus (drums, 808, pad, pluck, fx, lead) is rendered alone, given its
sends, set to a fixed RMS level and summed. The sum is made mono below the
crossover and faded out; the master is the only stage that sees the whole song.
Every random draw comes from its own generator, keyed by the config seed, the
stream and the event's position in it, so a render is reproducible.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from music_decoder.productionize import dsp
from music_decoder.productionize.config import (
    HALL,
    PLATE,
    MixConfig,
    PopRapConfig,
    PopRapFormConfig,
)
from music_decoder.productionize.dsp import Audio
from music_decoder.productionize.instruments import synths, trap
from music_decoder.productionize.mix import MixResult, fade_out, mono_low_end
from music_decoder.productionize.poprap import (
    PopRapSong,
    drum_hits,
    lead_notes,
    pad_events,
    pluck_notes,
)
from music_decoder.productionize.song import NoteEvent
from music_decoder.progress import NullProgress, ProgressCallback

POPRAP_STEMS = ("drums", "808", "pad", "pluck", "fx", "lead")

# Random streams: the first number of each generator's seed sequence.
_KICK, _PUNCH, _CLAP, _HAT, _CRASH, _RISER, _PAD, _PLUCK_PAN = range(8)


def poprap_length(song: PopRapSong, cfg: PopRapFormConfig) -> int:
    """Samples in the render buffer: the sections plus the tail."""
    return song.grid.samples(song.body_s + cfg.tail_s)


class _Buses:
    """The stereo buses being built, and the placing of clips on them."""

    def __init__(self, n: int, sr: int, seed: int) -> None:
        self.sr = sr
        self.seed = seed
        self.audio: dict[str, Audio] = {
            k: np.zeros((n, 2)) for k in ("drums", "808", "pad", "pluck", "fx")
        }

    def rng(self, stream: int, index: int) -> np.random.Generator:
        return np.random.default_rng((self.seed, stream, index))

    def put(
        self, bus: str, clip: Audio, start_s: float, pan: float = 0.0, gain: float = 1.0
    ) -> None:
        dsp.mix_into(self.audio[bus], dsp.constant_power_pan(clip, pan), start_s, self.sr, gain)


def _render_drums(
    song: PopRapSong, cfg: PopRapConfig, buses: _Buses, progress: ProgressCallback
) -> Audio:
    """Place drums, 808 and riser; return the kick envelope (1 at a kick, 0 after its release)."""
    sr, drums = buses.sr, cfg.drums
    n = len(buses.audio["drums"])
    kick_env = np.zeros(n)
    release = np.linspace(1.0, 0.0, int(drums.sidechain_release_s * sr))
    counts = dict.fromkeys(("kick", "clap", "hat", "crash", "riser"), 0)
    hits = drum_hits(song, drums, riser_bars=cfg.form.riser_bars)
    progress("drums", 0.0)
    progress("808", 0.0)  # the 808 is placed in the same pass
    for hit in hits:
        stream = "hat" if hit.kind == "open_hat" else hit.kind
        i = counts[stream]
        counts[stream] += 1
        t = hit.time_s
        if hit.kind == "kick":
            sub = trap.sub_808(hit.pitch, hit.length_s, drums, sr, buses.rng(_KICK, i))
            buses.put("808", sub, t, gain=hit.velocity * drums.sub_gain)
            buses.put("drums", trap.kick_punch(sr, buses.rng(_PUNCH, i)), t, gain=hit.velocity)
            a = round(t * sr)
            if a < n:
                seg = kick_env[a : a + len(release)]
                np.maximum(seg, release[: len(seg)], out=seg)
        elif hit.kind == "clap":
            buses.put("drums", trap.clap(sr, buses.rng(_CLAP, i)), t, hit.pan, hit.velocity)
        elif hit.kind in ("hat", "open_hat"):
            clip = trap.hat(hit.kind == "open_hat", hit.velocity, sr, buses.rng(_HAT, i))
            buses.put("drums", clip, t, hit.pan)
        elif hit.kind == "crash":
            buses.put("drums", trap.crash(sr, buses.rng(_CRASH, i)), t, hit.pan)
        else:
            riser = trap.riser(hit.length_s, sr, buses.rng(_RISER, i))
            buses.put("fx", riser, t, gain=hit.velocity)
    return kick_env


def _render_pad_and_pluck(
    song: PopRapSong, cfg: PopRapConfig, buses: _Buses, progress: ProgressCallback
) -> None:
    sr = buses.sr
    progress("pad", 0.0)
    for i, event in enumerate(pad_events(song, cfg)):
        clip = synths.pad_chord(event.pitches, event.length_s, cfg.pad, sr, buses.rng(_PAD, i))
        buses.put("pad", clip, event.start_s, gain=event.gain)
    progress("pluck", 0.0)
    spread = cfg.pluck.pan_spread
    for i, note in enumerate(pluck_notes(song)):
        pan = buses.rng(_PLUCK_PAN, i).uniform(-spread, spread)
        clip = synths.pluck(
            note.pitch, note.end_s - note.start_s, note.velocity / 127, cfg.pluck, sr
        )
        buses.put("pluck", clip, note.start_s, pan)


def _lead(
    kind: str,
    notes: list[NoteEvent],
    total_s: float,
    song: PopRapSong,
    cfg: PopRapConfig,
    work_dir: Path,
) -> Audio:
    """One lead as a mono clip of ``total_s``; silence when it has no notes."""
    sr = song.grid.sample_rate
    if not notes:
        return np.zeros(int(total_s * sr))
    if kind == "synth":
        return synths.synth_lead(notes, total_s, cfg.synth_lead, sr)
    return synths.guitar_lead(notes, total_s, cfg.guitar_lead, song.grid.tempo_bpm, work_dir, sr)


def _render_lead(song: PopRapSong, cfg: PopRapConfig, n: int, work_dir: Path) -> Audio:
    """The main lead, the hook-2 double, then the ping-pong delay and the plate."""
    sr, bus = song.grid.sample_rate, cfg.lead_bus
    total_s = n / sr
    main, double = lead_notes(song)
    other = "guitar" if cfg.lead == "synth" else "synth"
    lead = dsp.constant_power_pan(_lead(cfg.lead, main, total_s, song, cfg, work_dir), 0.0)
    if double:
        doubled = _lead(other, double, total_s, song, cfg, work_dir)
        lead += bus.double_gain * dsp.constant_power_pan(doubled, bus.double_pan)
    d = int(bus.delay_beats * song.grid.beat_s * sr)
    echo = np.zeros_like(lead)
    if 0 < d < n:
        echo[d:, 0] = lead[: n - d, 1] * bus.delay_first
    if 0 < 2 * d < n:
        echo[2 * d :, 1] = lead[: n - 2 * d, 0] * bus.delay_second
    return lead + echo + bus.plate_send * dsp.reverb(lead, PLATE, sr)


def _level(x: Audio, target_dbfs: float, silence: float, peak_ratio: float | None) -> Audio:
    """``x`` set to ``target_dbfs`` RMS over its audible samples, optionally soft-limited."""
    audible = x[np.abs(x).sum(axis=1) > silence]
    level = dsp.rms(audible)
    if level <= 0.0:
        return np.zeros_like(x)
    target = dsp.db_to_gain(target_dbfs)
    out: Audio = x * (target / level)
    if peak_ratio is not None:
        out = dsp.soft_limit(out, peak_ratio * target)
    return out


def render_poprap_mix(
    song: PopRapSong,
    cfg: PopRapConfig,
    work_dir: Path,
    progress: ProgressCallback | None = None,
) -> MixResult:
    """Render every bus, level it, and sum to the pre-master mix."""
    cb: ProgressCallback = progress if progress is not None else NullProgress()
    sr = song.grid.sample_rate
    n = poprap_length(song, cfg.form)
    buses = _Buses(n, sr, cfg.seed)

    kick_env = _render_drums(song, cfg, buses, cb)
    _render_pad_and_pluck(song, cfg, buses, cb)
    duck = 1.0 - cfg.drums.sidechain_depth * kick_env
    for name in ("pad", "pluck"):
        buses.audio[name] *= duck[:, None]

    cb("lead", 0.0)
    lead = _render_lead(song, cfg, n, work_dir)

    cb("mix", 0.0)
    m, a = cfg.mix, buses.audio
    dry = {
        "drums": a["drums"] + m.drums_plate_send * dsp.reverb(a["drums"], PLATE, sr),
        "808": dsp.butter(a["808"], "highpass", m.sub_highpass_hz, sr, stable=True),
        "pad": a["pad"] + m.pad_hall_send * dsp.reverb(a["pad"], HALL, sr),
        "pluck": a["pluck"] + m.pluck_hall_send * dsp.reverb(a["pluck"], HALL, sr),
        "fx": a["fx"] + m.fx_hall_send * dsp.reverb(a["fx"], HALL, sr),
        "lead": lead,
    }
    targets = {
        "drums": m.drums_dbfs,
        "808": m.sub_dbfs,
        "pad": m.pad_dbfs,
        "pluck": m.pluck_dbfs,
        "fx": m.fx_dbfs,
        "lead": m.lead_guitar_dbfs if cfg.lead == "guitar" else m.lead_synth_dbfs,
    }
    limited = {"drums", "808"}
    stems = {
        name: _level(dry[name], targets[name], m.silence, m.peak_ratio if name in limited else None)
        for name in POPRAP_STEMS
    }
    total = sum(stems.values(), np.zeros((n, 2)))
    fade = replace(
        MixConfig(), fade_out_s=cfg.form.fade_out_s, fade_out_curve=cfg.form.fade_out_curve
    )
    mix_ = fade_out(mono_low_end(total, MixConfig(), sr), fade, sr)
    onsets = sorted(
        h.time_s
        for h in drum_hits(song, cfg.drums, riser_bars=cfg.form.riser_bars)
        if h.kind in ("kick", "clap")
    )
    return MixResult(mix_, stems, tuple(onsets))
