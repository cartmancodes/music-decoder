"""THROWAWAY spike: the pop-rap style (form, pad, pluck, piano lead, master) with a beat
modelled on the reference (youtube dqEWV6if_6s): a harder kick and snare, a saturated 808
that slides between roots and sits near the drums' level, louder hats, and a synthesised
tabla theka. None of the reference's audio is used.

Usage: render.py TAKE_DIR "PROGRESSION" KEY OUT_DIR NAME TITLE
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from music_decoder.productionize import PopRapConfig, ProductionConfig, dsp, poprap_mix, productionize
from music_decoder.productionize.config import PLATE, PopRapMixConfig, TrapDrumsConfig
from music_decoder.productionize.instruments import trap
from music_decoder.productionize.instruments.bass import midi_to_hz
from music_decoder.types import ChordSymbol

TAKE, PROG, KEY, OUT, NAME, TITLE = sys.argv[1:7]
SR = 44100
rng = np.random.default_rng(7)
TONIC_PC = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"].index(KEY.rstrip("m"))
DAYAN_PITCH = 68 + (TONIC_PC - 68) % 12  # the dayan is tuned to the tonic, G#4..G5
DAYAN_HZ = midi_to_hz(DAYAN_PITCH)
TABLA_BELOW_DRUMS_DB = 9.0

# ---- harder drums ---------------------------------------------------------------
_clap = trap.clap
_hat = trap.hat


def kick(sr, rng_):
    n = int(0.45 * sr)
    t = np.arange(n) / sr
    f = 55 + 200 * np.exp(-t / 0.03)
    x = np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-t / 0.16)
    k = int(0.004 * sr)
    x[:k] += rng_.standard_normal(k) * 0.6
    return dsp.saturate(x, 3.0, normalize=True)


def snare(sr, rng_):
    c = _clap(sr, rng_)
    n = len(c)
    t = np.arange(n) / sr
    body = np.sin(2 * np.pi * 190 * t) * np.exp(-t / 0.06) * 0.9
    noise = dsp.butter(rng_.standard_normal(n), "highpass", 1500, sr, 2, stable=True) * np.exp(-t / 0.09) * 0.5
    return dsp.saturate(c + body + noise, 2.0, normalize=True)


def hat(open_, velocity, sr, rng_):
    return _hat(open_, velocity, sr, rng_) * 1.5


_last_pitch: list[float | None] = [None]


def sub_808(pitch, length_s, cfg, sr, rng_):
    """Gritty (lopsided drive: even harmonics), sliding in from the previous note."""
    n = int((length_s + cfg.sub_tail_s) * sr)
    t = np.arange(n) / sr
    start = _last_pitch[0] if _last_pitch[0] is not None else pitch
    _last_pitch[0] = pitch
    note = pitch + (start - pitch) * np.exp(-t / 0.045)
    hz = 440.0 * 2.0 ** ((note - 69.0) / 12.0) * (1 + 2.5 * np.exp(-t / 0.02))
    ph = 2 * np.pi * np.cumsum(hz) / sr
    raw = np.tanh(5.0 * (np.sin(ph) + 0.5))
    raw -= raw.mean()
    tone = dsp.butter(raw / np.abs(raw).max(), "lowpass", 1800.0, sr, 2, stable=True)
    amp = np.exp(-t / 0.6) * np.minimum(1.0, t / 0.004)
    amp *= np.where(t > length_s, np.exp(-(t - length_s) / cfg.sub_choke_s), 1.0)
    return tone * amp


trap.kick_punch, trap.clap, trap.hat, trap.sub_808 = kick, snare, hat, sub_808

# ---- tabla (synthesised) ----------------------------------------------------------
PARTIALS = (1.0, 2.0, 3.0, 4.0, 5.0)


def click(n_ms, band, gain):
    n = int(n_ms / 1000 * SR)
    x = dsp.butter(rng.standard_normal(n), "bandpass", band, SR, 2, stable=True)
    return x * np.exp(-np.arange(n) / (n / 4)) * gain


def dayan(kind, vel):
    dur = {"na": 0.7, "tin": 1.1, "te": 0.18}[kind]
    n = int(dur * SR)
    t = np.arange(n) / SR
    amps = {"na": (0.5, 1.0, 0.7, 0.45, 0.3), "tin": (1.0, 0.45, 0.3, 0.15, 0.1),
            "te": (0.6, 0.5, 0.4, 0.3, 0.2)}[kind]
    decay = {"na": 0.22, "tin": 0.45, "te": 0.03}[kind]
    x = sum(a * np.sin(2 * np.pi * DAYAN_HZ * k * t * (1 + 0.002 * k)) * np.exp(-t * k**0.7 / decay)
            for a, k in zip(amps, PARTIALS))
    c = click(6 if kind != "te" else 12, (1800, 6000), 0.5 if kind != "te" else 1.2)
    x[: len(c)] += c
    x = dsp.butter(x, "highpass", 300.0, SR, 2, stable=True)
    return x / np.abs(x).max() * vel


def bayan(kind, vel):
    if kind == "ka":
        x = click(40, (120, 900), 1.0)
        return x / np.abs(x).max() * vel * 0.7
    n = int(0.9 * SR)
    t = np.arange(n) / SR
    f = 82.0 * (1 + 0.32 * (1 - np.exp(-t / 0.09)))  # the heel presses: the pitch swoops up
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = (np.sin(ph) + 0.35 * np.sin(2 * ph) + 0.12 * np.sin(3 * ph)) * np.exp(-t / 0.16)
    c = click(10, (150, 700), 0.6)
    x[: len(c)] += c
    x = dsp.butter(x / np.abs(x).max(), "highpass", 70.0, SR, 2, stable=True)
    return x / np.abs(x).max() * vel * 0.6


# a bouncy theka on 16ths (stroke, velocity); "dha" = ge + na together
THEKA = {0: [("ge", 1.0), ("na", 0.7)], 2: [("na", 0.55)], 3: [("te", 0.45)], 5: [("na", 0.6)],
         6: [("ge", 0.8)], 8: [("tin", 0.75)], 10: [("ge", 0.85), ("na", 0.5)], 11: [("te", 0.5)],
         12: [("na", 0.65)], 13: [("ka", 0.6)], 14: [("tin", 0.6)], 15: [("te", 0.4)]}
# the last bar of the verse, where the drums drop out: a short tihai-like fill
FILL = {0: [("ge", 1.0), ("na", 0.8)], 2: [("te", 0.5)], 3: [("te", 0.55)], 4: [("na", 0.7)],
        6: [("ge", 0.9), ("na", 0.7)], 8: [("te", 0.5)], 9: [("te", 0.55)], 10: [("na", 0.75)],
        12: [("ge", 1.0), ("na", 0.8)], 13: [("te", 0.6)], 14: [("te", 0.7)], 15: [("tin", 0.9)]}


def rms_active(x):
    a = x[np.abs(x).sum(axis=1) > 1e-5]
    return float(np.sqrt(np.mean(a**2))) if len(a) else 0.0


_render_drums = poprap_mix._render_drums


def render_drums_with_tabla(song, cfg, buses, progress):
    kick_env = _render_drums(song, cfg, buses, progress)
    drums = dsp.saturate(buses.audio["drums"], 1.6, normalize=True)
    tabla = np.zeros_like(drums)
    grid = song.grid
    for section in song.sections:
        if section.role == "outro":
            continue
        for b in range(section.bars):
            if section.role == "intro" and b < 2:
                continue
            last_verse_bar = section.drums == "drop" and b == section.bars - 1
            soft = 0.8 if section.role == "intro" else 1.0
            for step, strokes in (FILL if last_verse_bar else THEKA).items():
                t = (section.start_bar + b) * grid.bar_s + step * grid.step_s
                for s, v in strokes:
                    v *= rng.uniform(0.85, 1.05) * soft
                    clip = bayan(s, v) if s in ("ge", "ka") else dayan(s, v)
                    pan = -0.15 if s in ("ge", "ka") else 0.2
                    dsp.mix_into(tabla, dsp.constant_power_pan(clip, pan), t, buses.sr, 1.0)
    tabla = tabla + 0.12 * dsp.reverb(tabla, PLATE, buses.sr)
    gain = rms_active(drums) / (rms_active(tabla) + 1e-12) * 10 ** (-TABLA_BELOW_DRUMS_DB / 20)
    buses.audio["drums"] = drums + gain * tabla
    return kick_env


poprap_mix._render_drums = render_drums_with_tabla

# ---- render ------------------------------------------------------------------------
cfg = ProductionConfig(
    style="pop-rap",
    poprap=PopRapConfig(
        lead="piano",
        drums=replace(TrapDrumsConfig(), kick_steps_even=(0, 7, 10), kick_steps_odd=(0, 3, 10, 14)),
        mix=replace(PopRapMixConfig(), drums_dbfs=-14.0, sub_dbfs=-15.5),
    ),
)
result = productionize(Path(TAKE), out_dir=Path(OUT), name=NAME, config=cfg,
                       progression=[ChordSymbol.parse(c) for c in PROG.split()], title=TITLE, key=KEY)
qc = result.qc
print("QC passed:", qc.passed if qc else None)
if qc:
    for c in qc.checks:
        if c.passed is False:
            print("FAIL", c.name, c.value, "expected", c.expected)
print("dayan pitch", DAYAN_PITCH)
