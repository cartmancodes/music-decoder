"""Combine melody + chord voicings into a 2-track PrettyMIDI file.

Accompaniment patterns sit on a straight 4/4 grid and are melody-aware: at
each attack, voicing pitches that would reach within a minor third of the
melody are dropped so the tune stays on top, except that the bass is always
kept and a chord's defining tones (third, seventh, altered fifth, fourth)
are never lost unless the melody itself is sounding them. The last bar is
always a single strummed chord ringing to the end.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Literal

import pretty_midi

from music_decoder.compose.harmony import essential_pcs
from music_decoder.tabs.tuning import STANDARD_EADGBE
from music_decoder.types import Note, Tuning, VoicedChord

GUITAR_PROGRAM = 25  # Steel Acoustic Guitar
Style = Literal["strum", "arpeggio", "fingerstyle"]

THIN_MARGIN_SEMITONES = 3  # accompaniment stays at least a minor third under the melody
STRUM_STAGGER_S = 0.008  # per string, low -> high on a down stroke
FINAL_STAGGER_S = 0.010  # <= 50 ms across six strings
# (eighth-note slot, direction): the "D . D U . U D U" pop strum
STRUM_STROKES: tuple[tuple[int, Literal["down", "up"]], ...] = (
    (0, "down"),
    (2, "down"),
    (3, "up"),
    (5, "up"),
    (6, "down"),
    (7, "up"),
)
UP_STROKE_STRINGS = 3


def _voiced_pitches(voicing: VoicedChord, tuning: Tuning) -> list[int]:
    return sorted(tuning.open_pitches[p.string] + p.fret for p in voicing.positions if p.fret >= 0)


def _thin_below(
    pitches: Sequence[int],
    essential: Collection[int],
    melody: Sequence[Note],
    t0: float,
    t1: float,
) -> list[int]:
    """The (sorted) ``pitches`` to play in [t0, t1).

    Register: pitches within a minor third of the lowest melody note sounding
    anywhere in the window are dropped, but the lowest pitch is always kept.
    Identity: each ``essential`` pitch class is then put back at its lowest
    voicing pitch unless a kept pitch has it or the melody note sounding at
    the end of the window does. Only the note that lasts to the end counts as
    supplying it: the accompaniment rings through the window, so a passing
    melody note earlier in it must not stand in for the chord's third or
    seventh under the held note that follows (for the final stroke, the
    cadence note).
    """
    overlapping = [n for n in melody if n.start_s < t1 and n.end_s > t0]
    if not overlapping:
        return list(pitches)
    ceiling = min(n.pitch for n in overlapping) - THIN_MARGIN_SEMITONES
    kept = [p for p in pitches if p <= ceiling] or list(pitches[:1])
    last = max(overlapping, key=lambda n: (n.start_s, n.pitch))
    heard = {p % 12 for p in kept} | {last.pitch % 12}
    for pc in sorted(essential):
        if pc not in heard:
            same = [p for p in pitches if p % 12 == pc]
            if same:
                kept.append(min(same))
                heard.add(pc)
    return sorted(kept)


def _stroke(
    pitches: Sequence[int], start: float, end: float, velocity: int, stagger_s: float
) -> list[pretty_midi.Note]:
    return [
        pretty_midi.Note(velocity=velocity, pitch=int(p), start=start + k * stagger_s, end=end)
        for k, p in enumerate(pitches)
    ]


def _arpeggio_bar(
    pitches: list[int], essential: Collection[int], melody: Sequence[Note], t: float, eighth: float
) -> list[pretty_midi.Note]:
    """Bass on the downbeat, then seven eighths cycling up and down the treble."""
    treble = pitches[1:]
    cycle = treble + treble[-2:0:-1] if len(treble) > 1 else treble
    sequence = [pitches[0]] + [cycle[i % len(cycle)] for i in range(7)] if cycle else pitches[:1]
    out: list[pretty_midi.Note] = []
    for k, p in enumerate(sequence):
        start = t + k * eighth
        if k > 0 and p not in _thin_below(pitches, essential, melody, start, start + eighth):
            continue
        out.append(
            pretty_midi.Note(
                velocity=70 if k == 0 else 62,
                pitch=int(p),
                start=start,
                end=start + 0.95 * eighth,
            )
        )
    return out


def _strum_bar(
    pitches: list[int], essential: Collection[int], melody: Sequence[Note], t: float, eighth: float
) -> list[pretty_midi.Note]:
    """Down strokes play the whole (thinned) voicing; up strokes the top strings."""
    out: list[pretty_midi.Note] = []
    for i, (slot, direction) in enumerate(STRUM_STROKES):
        start = t + slot * eighth
        next_slot = STRUM_STROKES[i + 1][0] if i + 1 < len(STRUM_STROKES) else 8
        end = t + next_slot * eighth
        chord = _thin_below(pitches, essential, melody, start, end)
        if direction == "up":
            out.extend(_stroke(chord[-UP_STROKE_STRINGS:][::-1], start, end, 60, STRUM_STAGGER_S))
        else:
            out.extend(_stroke(chord, start, end, 80 if slot == 0 else 72, STRUM_STAGGER_S))
    return out


def _fingerstyle_bar(
    pitches: list[int], essential: Collection[int], melody: Sequence[Note], t: float, beat: float
) -> list[pretty_midi.Note]:
    """Alternating bass on beats 1 and 3, treble pinch on beats 2 and 4."""
    alternate = len(pitches) >= 4
    alt_bass = pitches[1] if alternate else pitches[0]
    out: list[pretty_midi.Note] = []
    for k in range(4):
        start = t + k * beat
        end = start + 0.95 * beat
        if k % 2 == 0:
            bass = pitches[0] if k == 0 else alt_bass
            out.append(pretty_midi.Note(velocity=72, pitch=int(bass), start=start, end=end))
            continue
        for p in _thin_below(pitches[1:] or pitches, essential, melody, start, start + beat):
            if alternate and p == alt_bass:
                continue
            out.append(pretty_midi.Note(velocity=64, pitch=int(p), start=start, end=end))
    return out


def _accompaniment_notes(
    voicings: Sequence[VoicedChord],
    bars_per_chord: int,
    tempo_bpm: float,
    style: Style,
    tuning: Tuning,
    melody: Sequence[Note] = (),
) -> list[pretty_midi.Note]:
    seconds_per_beat = 60.0 / tempo_bpm
    seconds_per_bar = seconds_per_beat * 4.0
    eighth = seconds_per_beat / 2.0
    n_bars = len(voicings) * bars_per_chord
    out: list[pretty_midi.Note] = []
    for bar in range(n_bars):
        voicing = voicings[bar // bars_per_chord]
        pitches = _voiced_pitches(voicing, tuning)
        essential = essential_pcs(voicing.chord)
        t = bar * seconds_per_bar
        if not pitches:
            continue
        if bar == n_bars - 1:
            chord = _thin_below(pitches, essential, melody, t, t + seconds_per_bar)
            out.extend(_stroke(chord, t, t + seconds_per_bar, 80, FINAL_STAGGER_S))
        elif style == "arpeggio":
            out.extend(_arpeggio_bar(pitches, essential, melody, t, eighth))
        elif style == "strum":
            out.extend(_strum_bar(pitches, essential, melody, t, eighth))
        elif style == "fingerstyle":
            out.extend(_fingerstyle_bar(pitches, essential, melody, t, seconds_per_beat))
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
        melody_inst.notes.append(
            pretty_midi.Note(
                velocity=int(n.velocity),
                pitch=int(n.pitch),
                start=float(n.start_s),
                end=float(n.end_s),
            )
        )
    accomp_inst = pretty_midi.Instrument(program=GUITAR_PROGRAM, name="accompaniment")
    accomp_inst.notes.extend(
        _accompaniment_notes(
            voicings,
            bars_per_chord,
            tempo_bpm,
            style,
            tuning,
            melody=melody,
        )
    )
    pm.instruments.extend([melody_inst, accomp_inst])
    pm.write(str(out_path))
    return out_path
