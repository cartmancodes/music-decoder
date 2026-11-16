"""Phrase-structured melody generator over a chord progression.

Rhythm comes from :func:`music_decoder.compose.rhythm.plan_bars` (one-bar
templates arranged in antecedent/consequent phrases that end on cadences).
Each pitch is sampled from a softmax over musical scores: interval size,
chord fit weighted by metric position, leap recovery, leading-tone
resolution, motif echo, register gravity and, in the final bar, a pull
toward the cadence target. With ``phrase_shaping`` (v5 spec 5.3) each phrase
also restates the opening hook, follows a rise-and-fall arch in place of the
register gravity and saves its unique high note for the peak window, in an
A A' B A' form. Hard rules keep the line singable (no tritones
or sevenths, large leaps only as recovered arpeggiations, every chord change
and the final approach within a fifth); see :func:`admissible`. The only
randomness is ``numpy.random.default_rng(seed)``, so output is deterministic
for a seed.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from music_decoder.compose.harmony import (
    chord_pcs,
    chord_scale_pcs,
    leading_tone_pc,
    scale_pcs,
    tonic_pc,
)
from music_decoder.compose.metrics import ARCH_PEAK, PEAK_WINDOW, hook_cell
from music_decoder.compose.rhythm import form_labels, phrase_layout, plan_bars
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
SMOOTH_LIMIT = 7  # largest interval allowed without justification: a perfect fifth
FORBIDDEN_INTERVALS = frozenset({6, 10, 11})  # tritone, minor and major seventh
CHORD_CHANGE_STEP_BONUS = 1.0  # step (1-2 semitones) into a new chord
# Phrase shaping (v5 spec 5.3), only with generate_melody(phrase_shaping=True).
# Tuned from the plan's starting values (hook 3.0/1.5, variation 2, contour
# 0.8, span 7, foot -3, peak -1.5/+1.0/-2.0) to meet the feature-3 targets
# while the v3 singability guards hold (v5 ruling F3b): light enough that
# chord fit and leap recovery still win on strong beats. Re-tuned for ruling
# F3d (the varied A' slot withholds the original interval's reward).
HOOK_EXACT = 4.0  # a phrase opening restates the hook interval exactly
HOOK_NEAR = 4.0  # ... or within a semitone of it (transposed to fit the chord)
# The varied interval of an A' restatement widens by this much; 1 keeps the
# restatement within metrics.hook_restatements' semitone tolerance (ruling F3a).
HOOK_VARIATION = 1
CONTOUR_WEIGHT = 0.35  # pull toward the phrase's arch (replaces register gravity)
CONTOUR_WIDTH = 4.0
ARCH_SPAN = 23  # semitones from the arch's foot to its apex
ARCH_BASE_OFFSET = -5  # the arch's foot, relative to the tessitura centre
B_REGISTER_LIFT = 2  # the B phrase's arch sits this much higher
PEAK_TIE = -2.25  # reaching the phrase's running maximum before the peak window
PEAK_BONUS = 2.5  # a new phrase maximum inside the peak window
# Tying the phrase maximum inside the window (the final note included), or
# reaching it after the window.
PEAK_AFTER = -7.0

PeakPhase = Literal["before", "inside", "after"]


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
    chord_change: bool = False  # first note over a new chord
    prev_chord_tone: bool = False  # the previous pitch is a tone of this chord
    # The next note's candidates, when it is in the same unconstrained bar;
    # None forbids a large leap here.
    recovery_pool: frozenset[int] | None = None
    # Constrained bar: the pitches from which it can still finish smoothly.
    allowed: frozenset[int] | None = None
    # Phrase shaping (all None without it): the hook interval this note's
    # interval restates, the arch's target pitch (replacing register gravity),
    # and the phrase's highest pitch so far with where this note falls
    # relative to the peak window.
    hook_interval: int | None = None
    # An A' phrase's varied slot: the hook's own interval there, which earns
    # no hook reward, so the restatement really changes (v5 ruling F3d).
    hook_original: int | None = None
    contour_target: float | None = None
    phrase_max: int | None = None
    peak_phase: PeakPhase | None = None
    # The next note is the one-note final bar: the cadence pitches it takes,
    # the one nearest this note.
    final_choices: tuple[int, ...] | None = None


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
        if ctx.chord_change and 1 <= size <= 2:
            score += CHORD_CHANGE_STEP_BONUS
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
        if ctx.hook_interval is not None and interval != ctx.hook_original:
            if interval == ctx.hook_interval:
                score += HOOK_EXACT
            elif abs(interval - ctx.hook_interval) <= 1:
                score += HOOK_NEAR
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
    if ctx.contour_target is None:
        score -= GRAVITY_WEIGHT * ((candidate - tessitura_center) / GRAVITY_WIDTH) ** 2
    else:
        score -= CONTOUR_WEIGHT * ((candidate - ctx.contour_target) / CONTOUR_WIDTH) ** 2
    if ctx.phrase_max is not None:
        if ctx.peak_phase == "inside":
            if candidate > ctx.phrase_max:
                score += PEAK_BONUS
            elif candidate == ctx.phrase_max:
                score += PEAK_AFTER  # a tie would share the peak
        elif candidate >= ctx.phrase_max:
            score += PEAK_TIE if ctx.peak_phase == "before" else PEAK_AFTER
        if ctx.final_choices:
            final = min(ctx.final_choices, key=lambda p: (abs(p - candidate), p))
            if final == max(ctx.phrase_max, candidate):
                score += PEAK_AFTER  # the final note would tie the phrase's peak
    if ctx.target is not None:
        score -= CADENCE_PULL * abs(candidate - ctx.target)
    if ctx.cadence_end and not is_chord_tone:
        score += CADENCE_END_NON_CHORD_TONE
    return score


def _melodic(candidate: int, ctx: NoteContext) -> bool:
    """Rules R1 (no forbidden interval) and R2 (a large leap is recovered)."""
    if ctx.prev_pitch is None:
        return True
    interval = candidate - ctx.prev_pitch
    size = abs(interval)
    if size in FORBIDDEN_INTERVALS or size > MAX_INTERVAL:
        return False
    return abs(ctx.prev_interval) <= SMOOTH_LIMIT or (
        interval * ctx.prev_interval < 0 and 1 <= size <= 2
    )


def admissible(candidate: int, ctx: NoteContext) -> bool:
    """Hard singability rules R1-R4 (spec section 5.3).

    R1 no tritone, seventh or leap over an octave; R2 a leap larger than a
    fifth is followed by a contrary step; R3 such a leap is an arpeggiation
    inside one chord (not at a chord change, not in the final or approach
    bar, from a step or repeat) with a recovery pitch available next; R4 in a
    constrained bar the candidate is in the feasibility set.
    """
    if ctx.allowed is not None and candidate not in ctx.allowed:
        return False
    if not _melodic(candidate, ctx):
        return False
    if ctx.prev_pitch is None:
        return True
    interval = candidate - ctx.prev_pitch
    if abs(interval) <= SMOOTH_LIMIT:
        return True
    return (
        not ctx.chord_change
        and ctx.target is None
        and ctx.prev_chord_tone
        and candidate % 12 in ctx.chord_pcs
        and abs(ctx.prev_interval) <= 2
        and ctx.recovery_pool is not None
        and any(1 <= (candidate - r) * _sign(interval) <= 2 for r in ctx.recovery_pool)
    )


def smooth(a: int, b: int) -> bool:
    """A move of at most a fifth that is not a tritone."""
    return abs(b - a) <= SMOOTH_LIMIT and abs(b - a) != 6


def feasible_sets(
    candidates: Sequence[int], n_free: int, goal: Collection[int]
) -> list[frozenset[int]]:
    """Per free slot, the candidates from which ``goal`` stays reachable by smooth moves.

    Built backward: the last free slot must move smoothly into ``goal``, each
    earlier slot smoothly into the next slot's set.
    """
    sets: list[frozenset[int]] = []
    nxt: Collection[int] = goal
    for _ in range(n_free):
        here = frozenset(c for c in candidates if any(smooth(c, x) for x in nxt))
        sets.append(here)
        nxt = here
    return sets[::-1]


def ranked_scores(
    candidates: Sequence[int], ctx: NoteContext, tessitura_center: int
) -> tuple[list[float], int]:
    """Scores at the lowest ladder level with any finite score, and that level.

    Level 0 keeps admissible candidates; level 1 only R1 and R2; level 2 is
    the plain :func:`score_pitch`.
    """
    plain = [score_pitch(c, ctx, tessitura_center) for c in candidates]
    for level, keep in enumerate((admissible, _melodic)):
        scores = [s if keep(c, ctx) else -math.inf for c, s in zip(candidates, plain, strict=True)]
        if any(math.isfinite(s) for s in scores):
            return scores, level
    return plain, 2


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
    n_bars: int,
    beat_s: float,
    rng: np.random.Generator,
    v_strong: int,
    v_mid: int,
    v_weak: int,
    contrast_b: bool = False,
) -> list[_Bar]:
    eighth = beat_s / 2.0
    bars: list[_Bar] = []
    for plan in plan_bars(n_bars, rng, contrast_b=contrast_b):
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


def _plan_bar(
    options: Sequence[list[int]],
    final: bool,
    approach: bool,
    n_slots: int,
    prev: int | None,
    ref: int,
    cadence: int | None,
    final_approach: Collection[int],
) -> tuple[list[int], int | None, list[frozenset[int]] | None] | None:
    """The first option (candidates, target, feasibility sets) that can be entered.

    Final bar: cadence pitches nearest ``ref`` first (the first try is
    :func:`final_target`); approach bar: aim at ``final_approach``; any other
    bar: some candidate a smooth move from ``prev``. ``None`` when no option
    works.
    """

    def enters(feas: list[frozenset[int]]) -> bool:
        return prev is None or any(smooth(prev, c) for c in feas[0])

    for option in options:
        if final:
            targets = sorted(
                (p for p in option if p % 12 == cadence), key=lambda p: (abs(p - ref), p)
            )
            for target in targets:
                feas = feasible_sets(option, n_slots - 1, {target})
                if (prev is None or smooth(prev, target)) if n_slots == 1 else enters(feas):
                    return option, target, feas
        elif approach:
            feas = feasible_sets(option, n_slots, final_approach)
            if enters(feas):
                return option, None, feas
        elif prev is None or any(smooth(prev, p) for p in option):
            return option, None, None
    return None


@dataclass(frozen=True)
class _Phrase:
    """Where a bar sits in the phrase layout, for phrase shaping."""

    start: int  # first bar of the phrase
    length: int  # bars
    label: str  # "A", "A'" or "B" (see rhythm.form_labels)


def _phrase_of_bar(n_bars: int) -> list[_Phrase]:
    phrases: list[_Phrase] = []
    start = 0
    for length, label in zip(
        phrase_layout(n_bars).phrase_lengths, form_labels(n_bars), strict=True
    ):
        phrases.extend([_Phrase(start, length, label)] * length)
        start += length
    return phrases


def _arch(x: float) -> float:
    """The arch's height (0 to 1) at ``x``, a fraction of the phrase: a straight
    rise to ``ARCH_PEAK`` and a straight fall to the phrase's end."""
    return x / ARCH_PEAK if x <= ARCH_PEAK else (1.0 - x) / (1.0 - ARCH_PEAK)


def _peak_phase(x: float) -> PeakPhase:
    if x < PEAK_WINDOW[0]:
        return "before"
    return "inside" if x <= PEAK_WINDOW[1] else "after"


def _shaped(
    ctx: NoteContext, phrase: _Phrase, bar_idx: int, bar_fraction: float, tessitura_center: int
) -> NoteContext:
    """``ctx`` with the arch target and the peak-window phase of this note."""
    x = (bar_idx - phrase.start + bar_fraction) / phrase.length
    base = tessitura_center + ARCH_BASE_OFFSET + (B_REGISTER_LIFT if phrase.label == "B" else 0)
    return replace(ctx, contour_target=base + ARCH_SPAN * _arch(x), peak_phase=_peak_phase(x))


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
    allowed_pitches: Sequence[frozenset[int]] | None = None,
    final_whole_note: bool = False,
    phrase_shaping: bool = False,
) -> list[Note]:
    """Generate a list of Notes spanning the given progression.

    ``notes_per_bar=None`` (default) uses phrase-planned rhythms; an integer
    uses the legacy uniform grid of that many notes per bar.
    ``chord_floors`` (one entry per chord) is the lowest melody pitch allowed
    over that chord, so the melody can sit above the accompaniment's
    defining tones; it yields to the octave limit and the cadence (see
    ``_candidates``).
    ``allowed_pitches`` (one set per bar) limits each bar to those pitches,
    narrowed to the chord's scale when that leaves any; the pitch range and
    ``chord_floors`` then play no part and the latter may not be given.
    ``final_whole_note`` makes the last bar one held note (phrase rhythms only).
    ``phrase_shaping`` (phrase rhythms only; v5 spec 5.3) shapes every phrase:
    after the first, each opens by restating the hook (bar 0's first
    intervals; an A' phrase varies one of them by ``HOOK_VARIATION``, and the
    original interval earns nothing there), pitches follow the phrase's arch
    instead of the register gravity (the B phrase's sits ``B_REGISTER_LIFT``
    higher and draws rhythms of its own; the A' after it echoes phrase A),
    and the phrase's high note is saved for the peak window. The hard rules
    still win.
    """
    rng = np.random.default_rng(seed)
    n_bars = len(progression) * bars_per_chord
    if allowed_pitches is None:
        if not _scale_pitches_in_range(scale_pcs(scale), pitch_low, pitch_high):
            raise ValueError("Scale has no pitches in the requested range.")
    else:
        if len(allowed_pitches) != n_bars:
            raise ValueError("allowed_pitches needs one set per bar.")
        if any(not pitches for pitches in allowed_pitches):
            raise ValueError("allowed_pitches has an empty set.")
        if chord_floors is not None:
            raise ValueError("allowed_pitches and chord_floors cannot be combined.")
    if chord_floors is not None and len(chord_floors) != len(progression):
        raise ValueError("chord_floors needs one entry per chord.")
    if final_whole_note and notes_per_bar is not None:
        raise ValueError("final_whole_note needs phrase rhythms, not notes_per_bar.")
    if phrase_shaping and notes_per_bar is not None:
        raise ValueError("phrase_shaping needs phrase rhythms, not notes_per_bar.")

    beat_s = 60.0 / tempo_bpm
    bar_s = 4.0 * beat_s
    v_mid = (velocity_strong + velocity_weak) // 2
    if notes_per_bar is None:
        bars = _phrase_bars(
            n_bars, beat_s, rng, velocity_strong, v_mid, velocity_weak, contrast_b=phrase_shaping
        )
        if final_whole_note and bars:
            whole = _Slot(0.0, 8 * beat_s / 2.0, 8, True, velocity_strong)
            bars[-1] = replace(bars[-1], slots=(whole,), motif_from=None)
    else:
        bars = _grid_bars(n_bars, beat_s, notes_per_bar, velocity_strong, v_mid, velocity_weak)
    if not bars:  # legacy grid over no bars: nothing to play, as in v2
        return []
    lead, tonic = leading_tone_pc(scale), tonic_pc(scale)
    # A one-note final bar is entered from the bar before it, so that bar
    # aims at the final chord's cadence pitches, at or above its floor if any.
    final_pc = cadence_pc(scale, progression[-1])
    if allowed_pitches is not None:
        final_approach = sorted(p for p in allowed_pitches[-1] if p % 12 == final_pc)
    else:
        final_approach = [p for p in range(pitch_low, pitch_high + 1) if p % 12 == final_pc]
        final_floor = chord_floors[-1] if chord_floors is not None else None
        if final_floor is not None and any(p >= final_floor for p in final_approach):
            final_approach = [p for p in final_approach if p >= final_floor]
    one_note_final = len(bars[-1].slots) == 1

    out: list[Note] = []
    prev: int | None = None
    prev_interval = 0
    prev_chord: ChordSymbol | None = None
    bar_intervals: list[list[int]] = []
    phrases = _phrase_of_bar(n_bars) if phrase_shaping else []
    hook: list[int] = []  # bar 0's opening intervals (metrics.hook_cell)
    phrase_max: int | None = None
    for bar_idx, bar in enumerate(bars):
        chord_idx = min(bar_idx // bars_per_chord, len(progression) - 1)
        chord = progression[chord_idx]
        tones = frozenset(chord_pcs(chord))
        floor = chord_floors[chord_idx] if chord_floors is not None else None
        n = len(bar.slots)
        approach_bar = one_note_final and bar_idx == len(bars) - 2
        cadence = cadence_pc(scale, chord) if bar.final else None
        if allowed_pitches is not None:
            in_scale = chord_scale_pcs(scale, chord)
            floored = sorted(p for p in allowed_pitches[bar_idx] if p % 12 in in_scale)
            floored = floored or sorted(allowed_pitches[bar_idx])
            options = [floored]
        else:
            floored = _candidates(scale, chord, pitch_low, pitch_high, floor, prev, cadence)
            if not floored:
                raise ValueError("Scale has no pitches in the requested range.")
            unfloored = _candidates(scale, chord, pitch_low, pitch_high, None, prev, cadence)
            options = [floored] if unfloored == floored else [floored, unfloored]
        ref = prev if prev is not None else tessitura_center
        choice = _plan_bar(options, bar.final, approach_bar, n, prev, ref, cadence, final_approach)
        if choice is None:  # no smooth way through: v2's bar, and the ladder per note
            choice = (
                floored,
                final_target(scale, chord, floored, ref) if bar.final else None,
                None,
            )
        candidates, target, feas = choice
        motif = bar_intervals[bar.motif_from] if bar.motif_from is not None else None
        phrase = phrases[bar_idx] if phrase_shaping else None
        hook_here: list[int] = []
        # An A' restatement varies one hook interval: the first, from a random
        # index on, whose varied form the candidates can play there (if none
        # can, the restatement stays plain).
        varied: int | None = None
        if phrase is not None and phrase.start == bar_idx:
            phrase_max = None
            if bar_idx > 0 and hook:
                hook_here = list(hook)
                if phrase.label == "A'":
                    varied = int(rng.integers(len(hook)))
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
                    chord_change=prev_chord is not None and chord != prev_chord,
                    prev_chord_tone=prev is not None and prev % 12 in tones,
                    recovery_pool=(
                        frozenset(candidates)
                        if k + 1 < n and not bar.final and not approach_bar
                        else None
                    ),
                    allowed=feas[k] if feas else None,
                )
                if phrase is not None:
                    ctx = _shaped(ctx, phrase, bar_idx, slot.offset_s / bar_s, tessitura_center)
                    ctx = replace(
                        ctx,
                        hook_interval=hook_here[k - 1] if 0 < k <= len(hook_here) else None,
                        phrase_max=phrase_max,
                        final_choices=(tuple(final_approach) if approach_bar and last else None),
                    )
                scores, _ = ranked_scores(candidates, ctx, tessitura_center)
                h = ctx.hook_interval
                if varied is not None and h is not None and prev is not None and k > varied:
                    v = h + HOOK_VARIATION * (_sign(h) or 1)
                    if any(
                        c - prev == v and math.isfinite(x)
                        for c, x in zip(candidates, scores, strict=True)
                    ):
                        ctx = replace(ctx, hook_interval=v, hook_original=h)
                        scores, _ = ranked_scores(candidates, ctx, tessitura_center)
                        varied = None
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
            phrase_max = pitch if phrase_max is None else max(phrase_max, pitch)
        bar_intervals.append(here)
        if phrase_shaping and bar_idx == 0:
            hook = list(hook_cell(out, tempo_bpm))
    return out
