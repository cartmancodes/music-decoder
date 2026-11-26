"""THROWAWAY spike: the pop-rap form and piano lead over a boom-bap beat built to the numbers
measured on the second reference (youtube tTA4d3__n28, 88.0 bpm). None of its audio is used.

Measured there, and copied here:
  groove  kick on 16th steps 0, 7, 8 (beat 1, and a double into beat 3); snare on beats 2 and 4
          (steps 4, 12); straight 16ths, no swing; next to no hats (18 dB under the snare)
  kick    an 808-style boom, 90 Hz sliding to 40 Hz, flat for ~140 ms, no click (sounds.py)
  snare   short, fat, dry, body at 200 Hz, gone in ~120 ms (sounds.py)
  bass    one held note per bar: beat 1 only (steps 0-4, cut on the snare), on the root around
          60 Hz; a two-16th pickup into every second bar, and once in four bars a two-note walk
          down into the next root
  levels  drums lead; bass 5 dB under the drums; the voice 3 dB under (here: the piano tune);
          the rest of the music 8 dB under in hooks, 16 dB under in verses
  master  -9.4 LUFS with a crest of 12 dB

Usage: render.py TAKE_DIR "PROGRESSION" KEY OUT_DIR NAME TITLE [TEMPO]
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

import sounds
from music_decoder.productionize import PopRapConfig, ProductionConfig, dsp, poprap_mix, productionize
from music_decoder.productionize.config import MasterConfig, PopRapFormConfig, PopRapMixConfig, TrapDrumsConfig
from music_decoder.productionize.instruments import trap
from music_decoder.productionize.poprap import DrumHit

from music_decoder.types import ChordSymbol

TAKE, PROG, KEY, OUT, NAME, TITLE = sys.argv[1:7]
TEMPO = float(sys.argv[7]) if len(sys.argv) > 7 else None
ENV = os.environ.get

KICK_STEPS = (0, 7, 8)
SNARE_STEPS = (4, 12)
HAT_STEPS = (2, 6, 10, 14)
HAT_GAIN = float(ENV("HAT", "0.07"))
INTRO_HAT_GAIN = 0.3
SNARE_GAIN = float(ENV("SNARE", "0.8"))
CRASH_GAIN = 0.35
BASS_LOW = 31  # bass notes fold into G1..F#2 (49-92 Hz); the reference sits on B1, 62 Hz
BASS_BEAT_STEPS = 4  # held through beat 1, cut on the snare
INTRO_BASS_GAIN = 0.8
INTRO_BARS, INTRO_HATS_FROM_BAR = 4, 2  # the reference's intro is 4 bars
# -12: the limiter takes ~1 dB off the kicks; at the product's -10.5 it takes 4 dB
TARGET_LUFS = float(ENV("LUFS", "-12.0"))

NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
TONIC = NAMES.index(KEY.rstrip("m"))
SCALE = {(TONIC + i) % 12 for i in ((0, 2, 3, 5, 7, 8, 10) if KEY.endswith("m") else (0, 2, 4, 5, 7, 9, 11))}

# ---- the sounds ------------------------------------------------------------------
_crash = trap.crash
trap.kick_punch = sounds.kick
trap.clap = sounds.snare
trap.hat = lambda open_, velocity, sr, rng_: sounds.hat(sr, rng_) * velocity
trap.crash = lambda sr, rng_: _crash(sr, rng_) * CRASH_GAIN
trap.sub_808 = lambda pitch, length_s, cfg, sr, rng_: np.zeros(8)  # the bass is placed below


# ---- the groove --------------------------------------------------------------------
def is_drum_bar(section, b):
    return section.drums == "full" or (section.drums == "drop" and b != section.bars - 1)


def boombap_hits(song, cfg, *, riser_bars=2):
    grid = song.grid
    hits = []
    for section in song.sections:
        for b in range(section.bars):
            bar_t = (section.start_bar + b) * grid.bar_s
            if is_drum_bar(section, b):
                root = song.parts.bar_roots[song.pass_bar(b)]
                hits += [DrumHit(bar_t + s * grid.step_s, "kick", pitch=root, length_s=grid.step_s)
                         for s in KICK_STEPS]
                hits += [DrumHit(bar_t + s * grid.step_s, "clap", cfg.clap_gain) for s in SNARE_STEPS]
                hits += [DrumHit(bar_t + s * grid.step_s, "hat", HAT_GAIN, pan=cfg.hat_pan) for s in HAT_STEPS]
            elif section.hats_from_bar is not None and b >= section.hats_from_bar:
                hits += [DrumHit(bar_t + s * grid.step_s, "hat", INTRO_HAT_GAIN, pan=cfg.hat_pan)
                         for s in range(0, 16, 2)]
            if section.crash and b == 0:
                hits.append(DrumHit(bar_t, "crash", pan=cfg.crash_pan))
        if section.riser:
            hits.append(DrumHit((section.start_bar + section.bars - riser_bars) * grid.bar_s, "riser",
                                cfg.riser_gain, length_s=riser_bars * grid.bar_s))
    return sorted(hits, key=lambda h: h.time_s)


poprap_mix.drum_hits = boombap_hits


# ---- the bass ------------------------------------------------------------------------
def fold(pitch):
    return BASS_LOW + (pitch - BASS_LOW) % 12


def in_scale(pitch):
    return pitch % 12 in SCALE


def walk_into(root):
    """Two notes falling into ``root``: a fourth (or fifth) above, then the scale step above."""
    far = root + 5 if in_scale(root + 5) else root + 7
    near = root + 1 if in_scale(root + 1) else root + 2
    return far, near


_render_drums = poprap_mix._render_drums


def render_drums(song, cfg, buses, progress):
    kick_env = _render_drums(song, cfg, buses, progress)
    grid, sr = song.grid, buses.sr
    step = grid.step_s
    bars = [(section, b) for section in song.sections for b in range(section.bars)]
    for i, (section, b) in enumerate(bars):
        intro = section.role == "intro"  # as in the reference, the bass plays before the drums do
        if not (is_drum_bar(section, b) or intro):
            continue
        bar_t = (section.start_bar + b) * grid.bar_s
        root = fold(song.parts.bar_roots[song.pass_bar(b)])
        buses.put("808", sounds.bass(root, BASS_BEAT_STEPS * step, sr), bar_t, gain=INTRO_BASS_GAIN if intro else 1.0)
        if i + 1 == len(bars) or not is_drum_bar(*bars[i + 1]):
            continue
        nxt_section, nxt_b = bars[i + 1]
        nxt = fold(song.parts.bar_roots[song.pass_bar(nxt_b)])
        if b % 4 in (1, 3):  # the pickup: the next bar's note arrives two 16ths early
            buses.put("808", sounds.bass(nxt, 2 * step, sr), bar_t + 14 * step)
        elif b % 4 == 2:  # the walk down into the next root
            far, near = walk_into(nxt)
            buses.put("808", sounds.bass(far, 2 * step, sr), bar_t + 12 * step)
            buses.put("808", sounds.bass(near, 2 * step, sr, start_pitch=far), bar_t + 14 * step)
    return kick_env


poprap_mix._render_drums = render_drums


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

DRUMS_DBFS = -14.0
cfg = ProductionConfig(
    style="pop-rap",
    master=replace(MasterConfig(), target_lufs=TARGET_LUFS),
    poprap=PopRapConfig(
        lead="piano",
        form=replace(PopRapFormConfig(), intro_bars=INTRO_BARS, intro_hats_from_bar=INTRO_HATS_FROM_BAR),
        drums=replace(TrapDrumsConfig(), clap_gain=SNARE_GAIN, sidechain_depth=0.4),
        mix=replace(
            PopRapMixConfig(),
            drums_dbfs=DRUMS_DBFS,
            sub_dbfs=DRUMS_DBFS + float(ENV("BASS", "-1.0")),
            lead_piano_dbfs=DRUMS_DBFS + float(ENV("LEAD", "-3.5")),
            pad_dbfs=DRUMS_DBFS + float(ENV("PAD", "-12.0")),
            pluck_dbfs=DRUMS_DBFS + float(ENV("PLUCK", "-13.0")),
            fx_dbfs=DRUMS_DBFS - 17.0,
            drums_plate_send=0.03,
            peak_ratio=float(ENV("PEAK", "5.0")),  # the default 2.6 flattens the kick
        ),
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
