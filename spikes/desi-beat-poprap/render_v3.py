"""THROWAWAY spike v3 (v2 plus: bass drums ~11 dB over the music as in the reference (+15), hats 8 dB
down, a harder kick attack, the 808 swelling in later, a sparse tabla, a drier drum bus).

v2 notes: pop-rap form and piano lead over a hip-hop beat built to the numbers
measured on the reference (youtube dqEWV6if_6s, 91.1 bpm). None of its audio is used.

Measured there, and copied here:
  groove  snare/clap on beats 2 and 4 (16th steps 4, 12); kick on 0, 6, 10 with a ghost on 3;
          hats on 8ths; the 808 re-triggers with the kicks on 0, 6, 10 and rings through beat 4
  kick    round and deep: 75 Hz falling to ~50 Hz in 50 ms, -6 dB after 88 ms, -20 dB after
          128 ms, next to no click
  808     deep (the reference sits on F1, 44 Hz), sustained, swelling in ~100 ms after the kick,
          harmonics H2 -10, H3 -15, H4 -19, H5 -24, H6 -33 dB, sliding between notes
  levels  drums and 808 equally loud; below 45 Hz carries about a quarter of the mix's energy

Usage: render_v2.py TAKE_DIR "PROGRESSION" KEY OUT_DIR NAME TITLE [TEMPO]
       TEMPO re-times the take's notes to that tempo first (the tune's notes are unchanged).
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pretty_midi

from music_decoder.productionize import PopRapConfig, ProductionConfig, dsp, poprap_mix, productionize
from music_decoder.productionize.config import PLATE, MasterConfig, PopRapMixConfig, TrapDrumsConfig
from music_decoder.productionize.instruments import trap
from music_decoder.productionize.instruments.bass import midi_to_hz
from music_decoder.productionize.poprap import DrumHit
from music_decoder.types import ChordSymbol

TAKE, PROG, KEY, OUT, NAME, TITLE = sys.argv[1:7]
TEMPO = float(sys.argv[7]) if len(sys.argv) > 7 else None
SR = 44100
rng = np.random.default_rng(7)
TONIC_PC = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"].index(KEY.rstrip("m"))
DAYAN_HZ = midi_to_hz(68 + (TONIC_PC - 68) % 12)  # the dayan is tuned to the tonic, G#4..G5
TABLA_BELOW_DRUMS_DB = 14.0
HAT_GAIN = 0.65  # the reference's hats sit lower in its drums than ours did
LEAD_DUCK = 0.25  # the music dips ~5 dB at each kick in the reference; pad and pluck duck more
TARGET_LUFS = -14.0  # the reference measures -14.4 LUFS, crest 10 dB: its kicks keep their peaks
SUB_LOW = 25  # 808 notes fold into C#1..C2 (35-65 Hz)
SUB_HARMONICS_DB = (0.0, -10.0, -15.0, -19.0, -24.0, -33.0)
GHOST_KICK_STEP, GHOST_KICK_GAIN = 3, 0.5
CLAP_STEPS = (4, 12)

# ---- drums ---------------------------------------------------------------------
_clap = trap.clap
_hat = trap.hat


def kick(sr, rng_):
    n = int(0.28 * sr)
    t = np.arange(n) / sr
    f = 47.0 + 55.0 * np.exp(-t / 0.03)  # ~100 Hz at the hit, 60 Hz by 40 ms, then ~48 Hz
    amp = np.exp(-((t / 0.125) ** 3)) * np.minimum(1.0, t / 0.002)
    x = dsp.saturate(np.sin(2 * np.pi * np.cumsum(f) / sr) * amp, 1.8, normalize=True)
    return x * (1.0 + 0.5 * np.exp(-t / 0.025))  # the reference's attack is ~4 dB over its body


def snare(sr, rng_):
    c = _clap(sr, rng_)
    n = len(c)
    t = np.arange(n) / sr
    body = np.sin(2 * np.pi * 190 * t) * np.exp(-t / 0.06) * 0.9
    noise = dsp.butter(rng_.standard_normal(n), "highpass", 1500, sr, 2, stable=True) * np.exp(-t / 0.09) * 0.25
    x = dsp.saturate(c + body + noise, 2.0, normalize=True)
    return dsp.butter(x, "lowpass", 7000.0, sr, 2, stable=True)


_last_pitch: list[float | None] = [None]
_sub_count = [0]
KICKS_PER_BAR = 3
ANSWER_GAIN, ANSWER_DECAY_S = (float(x) for x in __import__("os").environ.get("ANSWER", "0.85,1.6").split(","))
_H = [10 ** (d / 20) for d in SUB_HARMONICS_DB]


def sub_808(pitch, length_s, cfg, sr, rng_):
    """Deep and sustained: swells in after the kick, slides in from the previous note."""
    pitch = SUB_LOW + (pitch - SUB_LOW) % 12
    n = int((length_s + cfg.sub_tail_s) * sr)
    t = np.arange(n) / sr
    start = _last_pitch[0] if _last_pitch[0] is not None else pitch
    _last_pitch[0] = pitch
    note = pitch + (start - pitch) * np.exp(-t / 0.045)
    ph = 2 * np.pi * np.cumsum(440.0 * 2.0 ** ((note - 69.0) / 12.0)) / sr
    tone = sum(a * np.sin(k * ph) for k, a in enumerate(_H, start=1))
    swell = 0.5 - 0.5 * np.cos(np.pi * np.clip((t - 0.05) / 0.11, 0.0, 1.0))  # in after the kick's thump
    # as in the reference, every second bar's 808 is shorter and softer: the bounce
    answer_bar = (_sub_count[0] // KICKS_PER_BAR) % 2 == 1
    _sub_count[0] += 1
    amp = swell * (ANSWER_GAIN * np.exp(-t / ANSWER_DECAY_S) if answer_bar else np.exp(-t / 1.6))
    amp *= np.where(t > length_s, np.exp(-(t - length_s) / 0.05), 1.0)
    return tone / np.abs(tone).max() * amp


def hat(open_, velocity, sr, rng_):
    return _hat(open_, velocity, sr, rng_) * HAT_GAIN


trap.kick_punch, trap.clap, trap.hat, trap.sub_808 = kick, snare, hat, sub_808

# ---- the groove: claps on 2 and 4 ----------------------------------------------------
_drum_hits = poprap_mix.drum_hits


def hiphop_hits(song, cfg, **kw):
    step = song.grid.step_s
    out = []
    for h in _drum_hits(song, cfg, **kw):
        out.append(h)
        if h.kind == "clap" and h.velocity == cfg.clap_gain:  # the main clap, on CLAP_STEPS[0]
            out.append(replace(h, time_s=h.time_s + (CLAP_STEPS[1] - CLAP_STEPS[0]) * step))
    return sorted(out, key=lambda h: h.time_s)


poprap_mix.drum_hits = hiphop_hits

# ---- tabla (synthesised), kept out of the low end --------------------------------------
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
    f = 82.0 * (1 + 0.32 * (1 - np.exp(-t / 0.09)))
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = (np.sin(ph) + 0.35 * np.sin(2 * ph) + 0.12 * np.sin(3 * ph)) * np.exp(-t / 0.16)
    c = click(10, (150, 700), 0.6)
    x[: len(c)] += c
    return x / np.abs(x).max() * vel * 0.6


# sparse, as in the reference: the space after each kick stays clear. Bars alternate A, B.
THEKA_A = {7: [("te", 0.4)], 14: [("na", 0.6)], 15: [("te", 0.45)]}
THEKA_B = {2: [("na", 0.5)], 13: [("ka", 0.5)], 14: [("tin", 0.65)], 15: [("te", 0.5)]}
# the intro has no drums yet, so the tabla plays a full pattern there
THEKA_INTRO = {0: [("ge", 1.0), ("na", 0.7)], 2: [("na", 0.55)], 3: [("te", 0.45)], 5: [("na", 0.6)],
               6: [("ge", 0.8)], 8: [("tin", 0.75)], 10: [("ge", 0.85), ("na", 0.5)], 11: [("te", 0.5)],
               12: [("na", 0.65)], 13: [("ka", 0.6)], 14: [("tin", 0.6)], 15: [("te", 0.4)]}
FILL = {0: [("ge", 1.0), ("na", 0.8)], 2: [("te", 0.5)], 3: [("te", 0.55)], 4: [("na", 0.7)],
        6: [("ge", 0.9), ("na", 0.7)], 8: [("te", 0.5)], 9: [("te", 0.55)], 10: [("na", 0.75)],
        12: [("ge", 1.0), ("na", 0.8)], 13: [("te", 0.6)], 14: [("te", 0.7)], 15: [("tin", 0.9)]}


def rms_active(x):
    a = x[np.abs(x).sum(axis=1) > 1e-5]
    return float(np.sqrt(np.mean(a**2))) if len(a) else 0.0


_render_drums = poprap_mix._render_drums


def render_drums(song, cfg, buses, progress):
    kick_env = _render_drums(song, cfg, buses, progress)
    grid, drums = song.grid, buses.audio["drums"]
    tabla = np.zeros_like(drums)
    for section in song.sections:
        for b in range(section.bars):
            bar_t = (section.start_bar + b) * grid.bar_s
            drum_bar = section.drums == "full" or (section.drums == "drop" and b != section.bars - 1)
            if drum_bar:
                dsp.mix_into(drums, dsp.constant_power_pan(kick(buses.sr, rng), 0.0),
                             bar_t + GHOST_KICK_STEP * grid.step_s, buses.sr, GHOST_KICK_GAIN)
            if section.role == "outro" or (section.role == "intro" and b < 2):
                continue
            last_verse_bar = section.drums == "drop" and b == section.bars - 1
            soft = 0.8 if section.role == "intro" else 1.0
            pattern = (FILL if last_verse_bar else THEKA_INTRO if section.role == "intro"
                       else THEKA_B if b % 2 else THEKA_A)
            for step, strokes in pattern.items():
                for s, v in strokes:
                    v *= rng.uniform(0.85, 1.05) * soft
                    clip = bayan(s, v) if s in ("ge", "ka") else dayan(s, v)
                    pan = -0.15 if s in ("ge", "ka") else 0.2
                    dsp.mix_into(tabla, dsp.constant_power_pan(clip, pan), bar_t + step * grid.step_s, buses.sr, 1.0)
    tabla = dsp.butter(tabla, "highpass", 120.0, buses.sr, 2, stable=True)
    tabla = tabla + 0.12 * dsp.reverb(tabla, PLATE, buses.sr)
    gain = rms_active(drums) / (rms_active(tabla) + 1e-12) * 10 ** (-TABLA_BELOW_DRUMS_DB / 20)
    buses.audio["drums"] = drums + gain * tabla
    _kick_env[0] = kick_env
    return kick_env


poprap_mix._render_drums = render_drums
_kick_env: list = [None]
_render_lead = poprap_mix._render_lead


def render_lead(song, cfg, n, work_dir, store=None):
    lead = _render_lead(song, cfg, n, work_dir, store)
    return lead * (1.0 - LEAD_DUCK * _kick_env[0][: len(lead), None])


poprap_mix._render_lead = render_lead


# ---- the take, re-timed if asked ----------------------------------------------------------
def retimed(take: Path, tempo: float) -> Path:
    meta = json.loads((take / "composition.json").read_text())
    k = float(meta["tempo_bpm"]) / tempo
    src = pretty_midi.PrettyMIDI(str(take / "composition.mid"))
    dst = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    for inst in src.instruments:
        new = pretty_midi.Instrument(program=inst.program, is_drum=inst.is_drum, name=inst.name)
        new.notes = [pretty_midi.Note(velocity=n.velocity, pitch=n.pitch, start=n.start * k, end=n.end * k)
                     for n in inst.notes]
        dst.instruments.append(new)
    out = Path(tempfile.mkdtemp(prefix="retimed-take-"))
    dst.write(str(out / "composition.mid"))
    meta["tempo_bpm"] = tempo
    (out / "composition.json").write_text(json.dumps(meta, indent=2))
    return out


take = Path(TAKE)
tmp = None
if TEMPO is not None:
    take = tmp = retimed(take, TEMPO)

cfg = ProductionConfig(
    style="pop-rap",
    master=replace(MasterConfig(), target_lufs=TARGET_LUFS),
    poprap=PopRapConfig(
        lead="piano",
        drums=replace(TrapDrumsConfig(), kick_steps_even=(0, 6, 10), kick_steps_odd=(0, 6, 10),
                      clap_step=CLAP_STEPS[0], clap_gain=float(os.environ.get("CLAP", "0.6")),
                      sidechain_depth=0.6),
        mix=replace(PopRapMixConfig(), drums_dbfs=-14.0, sub_dbfs=-15.0, lead_piano_dbfs=-22.0,
                    pad_dbfs=-31.0, pluck_dbfs=-32.0, fx_dbfs=-33.0, drums_plate_send=0.04,
                    peak_ratio=float(os.environ.get("PEAK", "5.0"))),  # 2.6 flattened the kick
    ),
)
result = productionize(take, out_dir=Path(OUT), name=NAME, config=cfg,
                       progression=[ChordSymbol.parse(c) for c in PROG.split()], title=TITLE, key=KEY)
if tmp is not None:
    shutil.rmtree(tmp)
qc = result.qc
print("QC passed:", qc.passed if qc else None)
if qc:
    for c in qc.checks:
        if c.passed is False:
            print("FAIL", c.name, c.value, "expected", c.expected)
_ = DrumHit  # the hit type hiphop_hits copies
