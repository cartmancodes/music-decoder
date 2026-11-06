"""Phrase-structured melody generator over a chord progression.

Rhythm comes from :func:`music_decoder.compose.rhythm.plan_bars` (one-bar
templates arranged in antecedent/consequent phrases that end on cadences).
Each pitch is sampled from a softmax over musical scores: interval size,
chord fit weighted by metric position, leap recovery, leading-tone
resolution, motif echo, register gravity and, in the final bar, a pull
toward the cadence target. The only randomness is
``numpy.random.default_rng(seed)``, so output is deterministic for a seed.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence
from dataclasses import dataclass

import numpy as np

from music_decoder.compose.harmony import (
    chord_pcs,
    chord_scale_pcs,
    leading_tone_pc,
    scale_pcs,
    tonic_pc,
)
from music_decoder.compose.rhythm import plan_bars
from music_decoder.types import ChordSymbol, Note, Scale

# Log-weight by interval size in semitones: steps preferred, repeats and wide
# leaps discouraged, tritone strongly avoided. Intervals > 12 are never chosen.
INTERVAL_WEIGHT: dict[int, float] = {
    0: -0.8,
    1: 0.8,
    2: 0.8,
    3: -0.2,
    4: -0.6,
    5: -1.2,
    6: -4.0,
    7: -1.8,
    8: -3.5,
    9: -3.5,
    10: -4.0,
    11: -4.5,
    12: -3.0,
}
MAX_INTERVAL = 12  # hard limit, including the move into the final note
STRONG_CHORD_TONE = 2.0  # beats 1 and 3
STRONG_NON_CHORD_TONE = -3.0
LONG_EIGHTHS = 3  # notes at least a dotted quarter long
LONG_CHORD_TONE = 1.0
LONG_NON_CHORD_TONE = -1.5
QUARTER_EIGHTHS = 2
QUARTER_SEMITONE_CLASH = -1.0  # quarter+ non-chord tone a semitone off a chord tone
WEAK_CHORD_TONE = 0.3
LEAP_SEMITONES = 5  # a leap of a fourth or more wants recovery
LEAP_RECOVERY_BONUS = 2.5  # opposite-direction step after a leap
LEAP_CONTINUE_PENALTY = -2.5  # same direction (or repeat) after a leap
LEADING_TONE_RESOLUTION = 3.0  # leading tone -> tonic a semitone up, into a tonic chord
MOTIF_SIGN_BONUS = 1.5  # same direction as the echoed interval
MOTIF_SIZE_BONUS = 1.0  # within a semitone of the echoed interval
GRAVITY_WEIGHT = 0.5  # register pull toward the tessitura centre
GRAVITY_WIDTH = 6.0
CADENCE_PULL = 0.25  # per semitone from the final target, final bar only
CADENCE_END_NON_CHORD_TONE = -3.0  # last note of a cadence bar
MIN_FLOOR_CANDIDATES = 5  # a chord floor applies only if it leaves this many pitches


def _scale_pitches_in_range(pcs: Collection[int], low: int, high: int) -> list[int]:
    return [p for p in range(low, high + 1) if (p % 12) in pcs]


def cadence_pc(scale: Scale, chord: ChordSymbol) -> int:
    """Pitch class of the final note: the tonic if ``chord`` contains it, else its root."""
    tones = chord_pcs(chord)
    return tonic_pc(scale) if tonic_pc(scale) in tones else tones[0]


def final_target(
    scale: Scale, chord: ChordSymbol, candidates: Sequence[int], reference: int
) -> int:
    """Final pitch: the candidate of :func:`cadence_pc` nearest ``reference``
    (ties -> lower)."""
    target_pc = cadence_pc(scale, chord)
    matches = [p for p in candidates if p % 12 == target_pc]
    if not matches:
        raise ValueError("No cadence target pitch in the requested range.")
    return min(matches, key=lambda p: (abs(p - reference), p))


@dataclass(frozen=True)
class NoteContext:
    """Everything the scorer needs to know about the note being chosen."""

    strong: bool
    eighths: float  # nominal length in eighth notes
    chord_pcs: frozenset[int]
    prev_pitch: int | None
    prev_interval: int
    motif_interval: int | None
    target: int | None  # final-bar cadence target
    cadence_end: bool  # last note of a cadence bar
    resolve_up: bool = False  # previous note is a leading tone entering a tonic chord


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def score_pitch(candidate: int, ctx: NoteContext, tessitura_center: int) -> float:
    """Log-weight of ``candidate``; ``-inf`` excludes it."""
    if ctx.target is not None and abs(candidate - ctx.target) > MAX_INTERVAL:
        return -math.inf  # the final note must stay reachable
    score = 0.0
    is_chord_tone = candidate % 12 in ctx.chord_pcs
    if ctx.prev_pitch is not None:
        interval = candidate - ctx.prev_pitch
        size = abs(interval)
        if size > MAX_INTERVAL:
            return -math.inf
        score += INTERVAL_WEIGHT[size]
        if abs(ctx.prev_interval) >= LEAP_SEMITONES:
            if interval * ctx.prev_interval < 0 and 1 <= size <= 2:
                score += LEAP_RECOVERY_BONUS
            elif interval * ctx.prev_interval >= 0:
                score += LEAP_CONTINUE_PENALTY
        if ctx.resolve_up and interval == 1:
            score += LEADING_TONE_RESOLUTION
        if ctx.motif_interval is not None:
            if _sign(interval) == _sign(ctx.motif_interval):
                score += MOTIF_SIGN_BONUS
            if abs(interval - ctx.motif_interval) <= 1:
                score += MOTIF_SIZE_BONUS
    if ctx.strong:
        score += STRONG_CHORD_TONE if is_chord_tone else STRONG_NON_CHORD_TONE
    elif ctx.eighths >= LONG_EIGHTHS:
        score += LONG_CHORD_TONE if is_chord_tone else LONG_NON_CHORD_TONE
    elif (
        ctx.eighths >= QUARTER_EIGHTHS
        and not is_chord_tone
        and any((candidate - c) % 12 in (1, 11) for c in ctx.chord_pcs)
    ):
        score += QUARTER_SEMITONE_CLASH
    elif is_chord_tone:
        score += WEAK_CHORD_TONE
    score -= GRAVITY_WEIGHT * ((candidate - tessitura_center) / GRAVITY_WIDTH) ** 2
    if ctx.target is not None:
        score -= CADENCE_PULL * abs(candidate - ctx.target)
    if ctx.cadence_end and not is_chord_tone:
        score += CADENCE_END_NON_CHORD_TONE
    return score


def _sample(
    candidates: Sequence[int], scores: Sequence[float], rng: np.random.Generator, prev: int | None
) -> int:
    arr = np.array(scores, dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():  # nothing reachable: take the nearest pitch
        ref = prev if prev is not None else candidates[len(candidates) // 2]
        return min(candidates, key=lambda p: (abs(p - ref), p))
    weights = np.where(finite, np.exp(arr - arr[finite].max()), 0.0)
    weights /= weights.sum()
    return int(candidates[int(rng.choice(len(candidates), p=weights))])


@dataclass(frozen=True)
class _Slot:
    offset_s: float  # onset within the bar
    length_s: float  # nominal (pre-articulation) length
    eighths: float
    strong: bool
    velocity: int


@dataclass(frozen=True)
class _Bar:
    slots: tuple[_Slot, ...]
    cadence: bool
    final: bool
    motif_from: int | None


def _phrase_bars(
    n_bars: int, beat_s: float, rng: np.random.Generator, v_strong: int, v_mid: int, v_weak: int
) -> list[_Bar]:
    eighth = beat_s / 2.0
    bars: list[_Bar] = []
    for plan in plan_bars(n_bars, rng):
        slots: list[_Slot] = []
        pos = 0
        for value in plan.rhythm:
            velocity = v_strong if pos == 0 else v_mid if pos == 4 else v_weak
            slots.append(_Slot(pos * eighth, value * eighth, value, pos in (0, 4), velocity))
            pos += value
        bars.append(_Bar(tuple(slots), plan.cadence, plan.final, plan.motif_from))
    return bars


def _grid_bars(
    n_bars: int, beat_s: float, notes_per_bar: int, v_strong: int, v_mid: int, v_weak: int
) -> list[_Bar]:
    """Legacy uniform grid: ``notes_per_bar`` equal notes per bar."""
    step = 4.0 * beat_s / notes_per_bar
    half = max(notes_per_bar // 2, 1)
    slots = tuple(
        _Slot(
            k * step,
            step,
            8.0 / notes_per_bar,
            k % half == 0,
            v_strong if k == 0 else v_mid if k % half == 0 else v_weak,
        )
        for k in range(notes_per_bar)
    )
    return [_Bar(slots, False, b == n_bars - 1, None) for b in range(n_bars)]


def _candidates(
    scale: Scale,
    chord: ChordSymbol,
    low: int,
    high: int,
    floor: int | None,
    prev: int | None,
    target_pc: int | None,
) -> list[int]:
    """Chord-scale pitches in [low, high], at or above ``floor`` when allowed.

    Precedence: the octave limit and the cadence come before register
    clearance. The floor is dropped for the bar if it would leave fewer than
    ``MIN_FLOOR_CANDIDATES`` pitches, no pitch within ``MAX_INTERVAL`` of the
    previous note, or (final bar, ``target_pc`` given) no cadence pitch within
    ``MAX_INTERVAL`` of it. The unfloored window always has one: it spans an
    octave or more, so every pitch class occurs within 12 semitones of any
    pitch in it. Thinning in the accompaniment still keeps the chord's
    identity tones when the melody dips below them.
    """
    pitches = _scale_pitches_in_range(chord_scale_pcs(scale, chord), low, high)
    if floor is None:
        return pitches
    above = [p for p in pitches if p >= floor]

    def reachable(p: int) -> bool:
        return prev is None or abs(p - prev) <= MAX_INTERVAL

    if len(above) < MIN_FLOOR_CANDIDATES or not any(reachable(p) for p in above):
        return pitches
    if target_pc is not None and not any(p % 12 == target_pc and reachable(p) for p in above):
        return pitches
    return above


def generate_melody(
    *,
    scale: Scale,
    progression: Sequence[ChordSymbol],
    bars_per_chord: int = 1,
    tempo_bpm: float = 100.0,
    notes_per_bar: int | None = None,
    pitch_low: int = 62,  # D4
    pitch_high: int = 79,  # G5
    seed: int | None = None,
    velocity_strong: int = 80,
    velocity_weak: int = 70,
    tessitura_center: int = 70,  # A#4
    chord_floors: Sequence[int | None] | None = None,
) -> list[Note]:
    """Generate a list of Notes spanning the given progression.

    ``notes_per_bar=None`` (default) uses phrase-planned rhythms; an integer
    uses the legacy uniform grid of that many notes per bar.
    ``chord_floors`` (one entry per chord) is the lowest melody pitch allowed
    over that chord, so the melody can sit above the accompaniment's
    defining tones; it yields to the octave limit and the cadence (see
    ``_candidates``).
    """
    rng = np.random.default_rng(seed)
    if not _scale_pitches_in_range(scale_pcs(scale), pitch_low, pitch_high):
        raise ValueError("Scale has no pitches in the requested range.")
    if chord_floors is not None and len(chord_floors) != len(progression):
        raise ValueError("chord_floors needs one entry per chord.")

    beat_s = 60.0 / tempo_bpm
    bar_s = 4.0 * beat_s
    n_bars = len(progression) * bars_per_chord
    v_mid = (velocity_strong + velocity_weak) // 2
    if notes_per_bar is None:
        bars = _phrase_bars(n_bars, beat_s, rng, velocity_strong, v_mid, velocity_weak)
    else:
        bars = _grid_bars(n_bars, beat_s, notes_per_bar, velocity_strong, v_mid, velocity_weak)
    lead, tonic = leading_tone_pc(scale), tonic_pc(scale)

    out: list[Note] = []
    prev: int | None = None
    prev_interval = 0
    prev_chord: ChordSymbol | None = None
    bar_intervals: list[list[int]] = []
    for bar_idx, bar in enumerate(bars):
        chord_idx = min(bar_idx // bars_per_chord, len(progression) - 1)
        chord = progression[chord_idx]
        tones = frozenset(chord_pcs(chord))
        floor = chord_floors[chord_idx] if chord_floors is not None else None
        candidates = _candidates(
            scale,
            chord,
            pitch_low,
            pitch_high,
            floor,
            prev,
            cadence_pc(scale, chord) if bar.final else None,
        )
        if not candidates:
            raise ValueError("Scale has no pitches in the requested range.")
        target = (
            final_target(scale, chord, candidates, prev if prev is not None else tessitura_center)
            if bar.final
            else None
        )
        motif = bar_intervals[bar.motif_from] if bar.motif_from is not None else None
        here: list[int] = []
        for k, slot in enumerate(bar.slots):
            last = k == len(bar.slots) - 1
            resolve_up = (
                prev is not None
                and prev_chord is not None
                and prev % 12 == lead
                and lead in chord_pcs(prev_chord)
                and chord != prev_chord
                and tonic in tones
            )
            if target is not None and last:
                pitch = target
            else:
                ctx = NoteContext(
                    strong=slot.strong,
                    eighths=slot.eighths,
                    chord_pcs=tones,
                    prev_pitch=prev,
                    prev_interval=prev_interval,
                    motif_interval=motif[k] if motif is not None and 0 < k < len(motif) else None,
                    target=target,
                    cadence_end=bar.cadence and last,
                    resolve_up=resolve_up,
                )
                scores = [score_pitch(c, ctx, tessitura_center) for c in candidates]
                pitch = _sample(candidates, scores, rng, prev)
            interval = pitch - prev if prev is not None else 0
            here.append(interval)
            start = bar_idx * bar_s + slot.offset_s
            out.append(
                Note(
                    start_s=start,
                    end_s=start + slot.length_s * 0.95,
                    pitch=pitch,
                    velocity=velocity_strong if (target is not None and last) else slot.velocity,
                    confidence=1.0,
                )
            )
            prev_interval = interval
            prev = pitch
            prev_chord = chord
        bar_intervals.append(here)
    return out
