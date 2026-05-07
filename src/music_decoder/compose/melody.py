"""Diatonic + Markov melody generator over a chord progression."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from music_decoder.types import VALID_TONICS, ChordSymbol, Note, Scale

# Pitch-class sets (relative to tonic = 0)
_MAJOR_PCS = (0, 2, 4, 5, 7, 9, 11)
_MINOR_PCS = (0, 2, 3, 5, 7, 8, 10)

# Chord-quality → relative pitch classes (root = 0)
_CHORD_PCS: dict[str, tuple[int, ...]] = {
    "maj":  (0, 4, 7),
    "min":  (0, 3, 7),
    "maj7": (0, 4, 7, 11),
    "min7": (0, 3, 7, 10),
    "7":    (0, 4, 7, 10),
    "dim":  (0, 3, 6),
    "sus4": (0, 5, 7),
    "aug":  (0, 4, 8),
}


def _scale_pcs(scale: Scale) -> tuple[int, ...]:
    base = _MAJOR_PCS if scale.mode == "major" else _MINOR_PCS
    tonic_pc = VALID_TONICS.index(scale.tonic)
    return tuple((p + tonic_pc) % 12 for p in base)


def _chord_tones(chord: ChordSymbol) -> tuple[int, ...]:
    rel = _CHORD_PCS[chord.quality]
    root_pc = VALID_TONICS.index(chord.root)
    return tuple((r + root_pc) % 12 for r in rel)


def _scale_pitches_in_range(scale_pcs: tuple[int, ...],
                            low: int, high: int) -> list[int]:
    return [p for p in range(low, high + 1) if (p % 12) in scale_pcs]


def generate_melody(
    *,
    scale: Scale,
    progression: Sequence[ChordSymbol],
    bars_per_chord: int = 1,
    tempo_bpm: float = 100.0,
    notes_per_bar: int = 4,
    pitch_low: int = 60,            # C4
    pitch_high: int = 84,           # C6
    seed: int | None = None,
    strong_beat_chord_tone_prob: float = 0.7,
    weak_beat_chord_tone_prob: float = 0.3,
    markov_max_interval_semitones: int = 4,
    velocity_strong: int = 80,
    velocity_weak: int = 70,
) -> list[Note]:
    """Generate a list of Notes spanning the given progression."""
    rng = np.random.default_rng(seed)
    scale_pcs = _scale_pcs(scale)
    scale_pitches = _scale_pitches_in_range(scale_pcs, pitch_low, pitch_high)
    if not scale_pitches:
        raise ValueError("Scale has no pitches in the requested range.")

    seconds_per_beat = 60.0 / tempo_bpm
    seconds_per_bar = seconds_per_beat * 4.0
    seconds_per_note = seconds_per_bar / notes_per_bar

    out: list[Note] = []
    prev_pitch = scale_pitches[len(scale_pitches) // 2]
    t = 0.0

    for chord in progression:
        chord_pcs = set(_chord_tones(chord))
        for _bar in range(bars_per_chord):
            for nidx in range(notes_per_bar):
                strong = (nidx % (notes_per_bar // 2) == 0)
                p_chord = strong_beat_chord_tone_prob if strong else weak_beat_chord_tone_prob
                use_chord_tone = rng.random() < p_chord
                pool = [
                    p for p in scale_pitches
                    if (p % 12) in chord_pcs
                ] if use_chord_tone else scale_pitches

                # Markov constraint: prefer pitches within max-interval of prev_pitch.
                near = [p for p in pool if abs(p - prev_pitch) <= markov_max_interval_semitones]
                pool_final = near if near else pool

                pitch = int(rng.choice(pool_final))
                velocity = velocity_strong if strong else velocity_weak
                start = t
                end = t + seconds_per_note * 0.95
                out.append(Note(
                    start_s=start, end_s=end, pitch=pitch,
                    velocity=velocity, confidence=1.0,
                ))
                prev_pitch = pitch
                t += seconds_per_note
    return out
