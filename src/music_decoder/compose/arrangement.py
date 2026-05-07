"""Combine melody + chord voicings into a 2-track PrettyMIDI file."""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

import pretty_midi

from music_decoder.tabs.tuning import STANDARD_EADGBE
from music_decoder.types import Note, Tuning, VoicedChord

GUITAR_PROGRAM = 25  # Steel Acoustic Guitar
Style = Literal["strum", "arpeggio", "fingerstyle"]


def _voiced_pitches(voicing: VoicedChord, tuning: Tuning) -> list[int]:
    return sorted(
        tuning.open_pitches[p.string] + p.fret
        for p in voicing.positions if p.fret >= 0
    )


def _accompaniment_notes(
    voicings: Sequence[VoicedChord],
    bars_per_chord: int,
    tempo_bpm: float,
    style: Style,
    tuning: Tuning,
) -> list[pretty_midi.Note]:
    seconds_per_beat = 60.0 / tempo_bpm
    seconds_per_bar = seconds_per_beat * 4.0
    out: list[pretty_midi.Note] = []
    t = 0.0
    for voicing in voicings:
        for _ in range(bars_per_chord):
            pitches = _voiced_pitches(voicing, tuning)
            if not pitches:
                t += seconds_per_bar
                continue
            if style == "strum":
                # block chord on beats 1 and 3
                for offset in (0.0, 2.0 * seconds_per_beat):
                    for p in pitches:
                        out.append(pretty_midi.Note(
                            velocity=70, pitch=int(p),
                            start=t + offset,
                            end=t + offset + 1.9 * seconds_per_beat,
                        ))
            elif style == "arpeggio":
                # 8 eighth notes ascending then descending across pitches
                seq = pitches + list(reversed(pitches[:-1]))
                step = seconds_per_bar / max(len(seq), 1)
                for k, p in enumerate(seq):
                    out.append(pretty_midi.Note(
                        velocity=68, pitch=int(p),
                        start=t + k * step, end=t + (k + 1) * step * 0.95,
                    ))
            elif style == "fingerstyle":
                # bass on 1+3, treble cluster on 2+4
                bass = pitches[0]
                treble = pitches[1:]
                for k in range(4):
                    beat_start = t + k * seconds_per_beat
                    if k % 2 == 0:
                        out.append(pretty_midi.Note(
                            velocity=72, pitch=int(bass),
                            start=beat_start,
                            end=beat_start + 0.95 * seconds_per_beat,
                        ))
                    else:
                        for p in treble:
                            out.append(pretty_midi.Note(
                                velocity=64, pitch=int(p),
                                start=beat_start,
                                end=beat_start + 0.95 * seconds_per_beat,
                            ))
            t += seconds_per_bar
    return out


def build_midi(
    *,
    melody: Sequence[Note],
    voicings: Sequence[VoicedChord],
    bars_per_chord: int,
    tempo_bpm: float,
    style: Style,
    out_path: Path,
    tuning: Tuning = STANDARD_EADGBE,
) -> Path:
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo_bpm)
    melody_inst = pretty_midi.Instrument(program=GUITAR_PROGRAM, name="melody")
    for n in melody:
        melody_inst.notes.append(pretty_midi.Note(
            velocity=int(n.velocity), pitch=int(n.pitch),
            start=float(n.start_s), end=float(n.end_s),
        ))
    accomp_inst = pretty_midi.Instrument(program=GUITAR_PROGRAM, name="accompaniment")
    accomp_inst.notes.extend(_accompaniment_notes(
        voicings, bars_per_chord, tempo_bpm, style, tuning,
    ))
    pm.instruments.extend([melody_inst, accomp_inst])
    pm.write(str(out_path))
    return out_path
