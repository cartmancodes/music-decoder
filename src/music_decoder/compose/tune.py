"""The v6 tune engine: a melody built from the statistics of real tunes.

Two halves, both driven by ``tune_stats.TuneStats`` and a passed-in
``numpy.random.Generator``:

* Rhythm plan (:func:`plan_rhythm`): each bar draws a real one-bar onset
  pattern for its role (open, body, cadence), weighted by how often that
  pattern follows the previous bar's and by closeness to a target density
  taken from the tempo. A' phrases often restate the matching bar of phrase A.
* Pitch plan (:func:`generate_tune`): notes chosen left to right on those
  onsets, each weighted by the real move tables (in scale steps), chord-tone
  share, degree shares, register and an optional pitch prior, after the hard
  interval rules; a repeat keeps the real share the move table gives it. A bar
  whose pattern was used before often replays that bar's moves (cell memory);
  phrases end on chord tones, the piece on the cadence pitch.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from music_decoder.compose.harmony import chord_pcs, chord_scale_pcs, tonic_pc
from music_decoder.compose.melody import (
    PRIOR_WEIGHT,
    PRIOR_WIDTH,
    _prior_target,
    cadence_pc,
    feasible_sets,
    final_target,
)
from music_decoder.compose.rhythm import form_labels, split_phrases
from music_decoder.compose.tune_stats import (
    START,
    Pattern,
    Role,
    TuneStats,
    load_tune_stats,
    metric_class,
)
from music_decoder.types import ChordSymbol, Note, Scale

P_RESTATE_RHYTHM = 0.5  # chance an A' bar copies the matching bar of phrase A
DENSITY_SIGMA = 2.5  # width of the onsets-per-bar preference, in onsets (tuned, v6 spec 5.5)
DENSITY_MIN, DENSITY_MAX = 3.0, 6.5  # clamp on the target onsets per bar
PAIR_MIN_COUNT = 20  # following bars a previous pattern needs for its own row to count

P_RESTATE_CELL = 0.75  # chance a bar with a used pattern replays that bar's moves (tuned)
DEGREE_EXPONENT = 0.8  # tuned (v6 spec 5.5)
REGISTER_SIGMA = 6.0  # semitones (tuned, v6 spec 5.5)
CONTEXT_MIN_COUNT = 20  # observations a previous move needs for its own row to count
MAX_REPLAY_REPLACEMENTS = 2  # replaced notes after which a replay gives up for the bar

MAX_LEAP = 7  # no move larger than a fifth (ruling R10; L2 and L3 hold by construction)
CHANGE_LEAP = 7  # into a new chord and into the final note: at most a fifth
FORBIDDEN_SIZES = frozenset({6, 10, 11})  # tritone, sevenths
MAX_STEPS = 7  # scale-step moves are clipped to this size
MISSING_COUNT = 0.5  # count given to a move the table never saw
DEGREE_FLOOR = 1e-3  # an unseen degree's share, as a fraction of the mean share
LENGTH_FACTOR = 0.95  # a note sounds this share of the time to the next onset
STRONG_SLOTS = (0, 4)


# --- Rhythm plan -----------------------------------------------------------------------


@dataclass(frozen=True)
class BarRhythm:
    pattern: Pattern
    role: Role
    restated_from: int | None  # the bar whose pattern this bar copied


def bar_roles(n_bars: int) -> list[Role]:
    """Role of each bar: a phrase opens on its first bar and cadences on its last
    (a one-bar phrase is a cadence); the rest are body."""
    roles: list[Role] = []
    for length in split_phrases(n_bars):
        for k in range(length):
            if k == length - 1:
                roles.append("cadence")
            elif k == 0:
                roles.append("open")
            else:
                roles.append("body")
    return roles


def _target_onsets(tempo_bpm: float, stats: TuneStats) -> float:
    target = stats.notes_per_second.mid * 240.0 / tempo_bpm
    return min(max(target, DENSITY_MIN), DENSITY_MAX)


def _draw_pattern(
    rng: np.random.Generator,
    role: Role,
    previous: Pattern | None,
    target: float,
    stats: TuneStats,
    *,
    need_slot_zero: bool,
) -> Pattern:
    counts = stats.rhythm[role]
    patterns = sorted(p for p in counts if not need_slot_zero or 0 in p)
    if not patterns:
        return (0,)
    total = sum(counts.values())  # shares are over the whole role, even when restricted
    row = stats.rhythm_pairs[role].get(previous) if previous is not None else None
    row_total = sum(row.values()) if row else 0
    use_row = row is not None and row_total >= PAIR_MIN_COUNT
    weights = np.empty(len(patterns))
    for i, pattern in enumerate(patterns):
        p = counts[pattern] / total
        if use_row and row is not None:
            p = 0.5 * (row.get(pattern, 0) / row_total) + 0.5 * p
        gap = len(pattern) - target
        weights[i] = p * math.exp(-(gap * gap) / (2.0 * DENSITY_SIGMA**2))
    if weights.sum() <= 0.0:
        weights = np.ones(len(patterns))
    return patterns[int(rng.choice(len(patterns), p=weights / weights.sum()))]


def plan_rhythm(
    n_bars: int,
    tempo_bpm: float,
    rng: np.random.Generator,
    stats: TuneStats,
    *,
    final_whole_note: bool = True,
) -> list[BarRhythm]:
    """One :class:`BarRhythm` per bar. With ``final_whole_note`` the last bar is a
    single note on the downbeat; otherwise it draws a cadence pattern like any bar."""
    roles = bar_roles(n_bars)
    lengths = split_phrases(n_bars)
    labels = form_labels(n_bars)
    starts = [sum(lengths[:i]) for i in range(len(lengths))]
    phrase_of = [i for i, length in enumerate(lengths) for _ in range(length)]
    target = _target_onsets(tempo_bpm, stats)

    plan: list[BarRhythm] = []
    for bar in range(n_bars):
        role = roles[bar]
        if final_whole_note and bar == n_bars - 1:
            plan.append(BarRhythm((0,), "cadence", None))
            continue
        phrase = phrase_of[bar]
        offset = bar - starts[phrase]
        if role != "cadence" and labels[phrase] == "A'":
            source = offset  # the matching bar of the first phrase
            if source < lengths[0] and roles[source] == role and rng.random() < P_RESTATE_RHYTHM:
                plan.append(BarRhythm(plan[source].pattern, role, source))
                continue
        previous = plan[-1].pattern if plan else None
        pattern = _draw_pattern(rng, role, previous, target, stats, need_slot_zero=bar == 0)
        plan.append(BarRhythm(pattern, role, None))
    return plan


# --- Pitch plan ------------------------------------------------------------------------


def scale_steps(prev: int, cand: int, pcs: Collection[int]) -> int:
    """Signed count of ladder pitches (any octave of ``pcs``) passed from ``prev``
    to ``cand``, ``cand`` included when on the ladder; clipped to +-MAX_STEPS."""
    if cand >= prev:
        n = sum(1 for p in range(prev + 1, cand + 1) if p % 12 in pcs)
    else:
        n = -sum(1 for p in range(cand, prev) if p % 12 in pcs)
    return max(-MAX_STEPS, min(MAX_STEPS, n))


def _step_by(pitch: int, steps: int, pcs: Collection[int]) -> int:
    """The ladder pitch ``steps`` scale steps from ``pitch`` (the inverse of
    :func:`scale_steps`)."""
    direction = 1 if steps > 0 else -1
    left = abs(steps)
    while left:
        pitch += direction
        if pitch % 12 in pcs:
            left -= 1
    return pitch


@dataclass(frozen=True)
class _Piece:
    """What every note of the piece is chosen against."""

    scale: Scale
    stats: TuneStats
    center: int
    prior: Sequence[Note] | None
    eighth_s: float


@dataclass(frozen=True)
class _Spot:
    """One onset and the rules its note obeys."""

    slot: int
    onset_s: float
    chord: ChordSymbol
    ladder: frozenset[int]
    change: bool  # first note over a new chord: at most CHANGE_LEAP
    feasible: frozenset[int] | None  # approach to the final note, on the chord scale
    feasible_wide: frozenset[int] | None  # the same over all allowed pitches
    phrase_end: bool


@dataclass
class _Voice:
    """The line so far: its last pitch, the move into it and its chord."""

    prev: int | None = None
    prev_move: int | None = None  # None before the second note; START for it
    prev_chord: ChordSymbol | None = None

    def ref(self, center: int) -> int:
        return self.prev if self.prev is not None else center

    def advance(self, pitch: int, spot: _Spot) -> None:
        # The second note's previous move is START (the first note has none).
        prev = self.prev
        self.prev_move = START if prev is None else scale_steps(prev, pitch, spot.ladder)
        self.prev, self.prev_chord = pitch, spot.chord


def _bar_candidates(scale: Scale, chord: ChordSymbol, allowed: Collection[int]) -> list[int]:
    ladder = chord_scale_pcs(scale, chord)
    return sorted(p for p in allowed if p % 12 in ladder) or sorted(allowed)


def _fits(prev: int | None, cand: int, limit: int) -> bool:
    if prev is None:
        return True
    size = abs(cand - prev)
    return size <= limit and size not in FORBIDDEN_SIZES


def _phrase_end_pool(pool: list[int], scale: Scale, chord: ChordSymbol) -> list[int]:
    """Chord tones if any, without the tonic when another chord tone is there."""
    tones = chord_pcs(chord)
    chordal = [c for c in pool if c % 12 in tones]
    open_end = [c for c in chordal if c % 12 != tonic_pc(scale)]
    return open_end or chordal or pool


def _pool(spot: _Spot, allowed: Collection[int], prev: int | None, scale: Scale) -> list[int]:
    """Candidates after the hard filters. The chord-scale restriction is dropped
    first, then the chord-change limit, then the approach to the final note. When
    nothing at all fits, the last resort is the allowed pitch nearest ``prev`` with
    the interval limits dropped: ``allowed_pitches`` (what the hand can reach)
    outranks the interval rules (ruling R12)."""
    on_ladder = _bar_candidates(scale, spot.chord, allowed)
    everything = sorted(allowed)
    change = CHANGE_LEAP if spot.change else MAX_LEAP
    tiers = (
        (on_ladder, change, spot.feasible),
        (everything, change, spot.feasible_wide),
        # Inert while MAX_LEAP == CHANGE_LEAP (ruling R10); kept for a wider MAX_LEAP.
        (everything, MAX_LEAP, spot.feasible_wide),
        (everything, MAX_LEAP, None),
    )
    for cands, limit, feasible in tiers:
        pool = [c for c in cands if _fits(prev, c, limit) and (feasible is None or c in feasible)]
        if pool:
            return _phrase_end_pool(pool, scale, spot.chord) if spot.phrase_end else pool
    assert prev is not None  # with no previous pitch every candidate fits
    return [min(everything, key=lambda p: (abs(p - prev), p))]


def _move_row(stats: TuneStats, slot: int, prev_move: int) -> Mapping[int, int]:
    cls = metric_class(slot)
    row = stats.steps[cls].get(prev_move)
    if row is not None and sum(row.values()) >= CONTEXT_MIN_COUNT:
        return row
    return stats.steps_base[cls]


def _degree_factors(shares: Sequence[float]) -> list[float]:
    mean = sum(shares) / len(shares)
    return [(max(x, DEGREE_FLOOR * mean) / mean) ** DEGREE_EXPONENT for x in shares]


def _weights(pool: Sequence[int], spot: _Spot, voice: _Voice, piece: _Piece) -> list[float]:
    """Move x harmony x degree x register x prior, per candidate (spec 5.2). The
    repeat's weight is set per rulings R11 and R11b (:func:`_hold_repeat_share`)."""
    stats, mode = piece.stats, piece.scale.mode
    tones = chord_pcs(spot.chord)
    n_tones = sum(c % 12 in tones for c in pool)
    share = stats.chord_tone[metric_class(spot.slot)]
    degrees = _degree_factors((stats.final_degrees if spot.phrase_end else stats.degrees)[mode])
    tonic = tonic_pc(piece.scale)
    row = None
    if voice.prev is not None and voice.prev_move is not None:
        row = _move_row(stats, spot.slot, voice.prev_move)
    target = None
    if piece.prior is not None:
        target = _prior_target(piece.prior, spot.onset_s, piece.eighth_s)
    weights: list[float] = []
    factors: list[float] = []  # the non-move factors: harmony x degree x register x prior
    for c in pool:
        f = share / n_tones if c % 12 in tones else (1.0 - share) / (len(pool) - n_tones)
        f *= degrees[(c - tonic) % 12]
        f *= math.exp(-((c - piece.center) ** 2) / (2.0 * REGISTER_SIGMA**2))
        if target is not None:
            r = (c - target) % 12
            f *= math.exp(PRIOR_WEIGHT * math.exp(-min(r, 12 - r) / PRIOR_WIDTH))
        move = 1.0
        if row is not None and voice.prev is not None:
            move = row.get(scale_steps(voice.prev, c, spot.ladder)) or MISSING_COUNT
        factors.append(f)
        weights.append(move * f)
    if row is not None and voice.prev is not None:
        _hold_repeat_share(weights, factors, pool, voice.prev, row)
    return weights


def _hold_repeat_share(
    weights: list[float],
    factors: Sequence[float],
    pool: Sequence[int],
    prev: int,
    row: Mapping[int, int],
) -> None:
    """Set the weight of repeating ``prev``, in place (rulings R11, R11b).

    The harmony, degree and register factors favour the pitch the line is on
    (usually a chord tone on a common degree near the centre), and dropping
    the moves the pool cannot make leaves their share to the rest; together
    they inflated repeats far past real tunes'. So the repeat starts from the
    move row's own repeat odds against the other candidates' total weight,
    and its non-move factors (``factors``) may only lower that: by their ratio
    to the plain mean of the others' when it is below one (a non-chord tone on
    a new chord's downbeat repeats less), never raising it above the row's
    share (the line already sits on pitches the factors favour).
    """
    if prev not in pool:
        return
    i = pool.index(prev)
    others = sum(weights) - weights[i]
    n_others = len(pool) - 1
    if others <= 0.0 or n_others == 0:
        return
    share = (row.get(0) or MISSING_COUNT) / sum(row.values())
    if share >= 1.0:  # a row of repeats only leaves the weights as they are
        return
    mean_factor = (sum(factors) - factors[i]) / n_others
    fit = min(1.0, factors[i] / mean_factor)
    weights[i] = others * share / (1.0 - share) * fit


def _sample(rng: np.random.Generator, pool: Sequence[int], weights: Sequence[float]) -> int:
    if len(pool) == 1:
        return pool[0]
    p = np.asarray(weights, dtype=float)
    return int(pool[int(rng.choice(len(pool), p=p / p.sum()))])


def _replay_starts(
    pool: Sequence[int], moves: Sequence[int], pattern: Pattern, spot: _Spot
) -> list[int]:
    """The starts giving the most chord tones on slots 0 and 4 when ``moves`` are
    replayed from them. The caller draws one by the usual weights, so the move
    into a replayed bar is as real as any other (a nearest-pitch tie-break
    made it a repeat about half the time)."""
    tones = chord_pcs(spot.chord)

    def strong_tones(start: int) -> int:
        pitch, hits = start, 0
        for k, slot in enumerate(pattern):
            if k:
                pitch = _step_by(pitch, moves[k - 1], spot.ladder)
            hits += slot in STRONG_SLOTS and pitch % 12 in tones
        return hits

    best = max(strong_tones(p) for p in pool)
    return [p for p in pool if strong_tones(p) == best]


def _approach_sets(
    plan: Sequence[BarRhythm],
    candidates: Sequence[list[int]],
    allowed: Sequence[frozenset[int]],
    goal: Collection[int],
) -> dict[tuple[int, int], tuple[frozenset[int], frozenset[int]]]:
    """Per (bar, note) leading into the final note, the pitches from which it
    stays reachable by smooth moves (as v5's approach bar): over the chord-scale
    candidates, and over all allowed pitches for when the chord scale cannot.
    The bar before the approach bar gets sets too, leading into the approach
    bar's first note: with no move over a fifth (ruling R10) a line could
    otherwise enter the approach bar where no smooth path to the final is left."""
    last = len(plan) - 1
    n_last = len(plan[last].pattern)
    bar, n_free = (last, n_last - 1) if n_last > 1 else (last - 1, len(plan[last - 1].pattern))
    if bar < 0:
        return {}
    narrow = feasible_sets(candidates[bar], n_free, goal)
    wide = feasible_sets(sorted(allowed[bar]), n_free, goal)
    out = {(bar, k): pair for k, pair in enumerate(zip(narrow, wide, strict=True))}
    if bar >= 1:
        before, n_before = bar - 1, len(plan[bar - 1].pattern)
        narrow = feasible_sets(candidates[before], n_before, narrow[0])
        wide = feasible_sets(sorted(allowed[before]), n_before, wide[0])
        out.update({(before, k): pair for k, pair in enumerate(zip(narrow, wide, strict=True))})
    return out


def _replay_moves(
    rng: np.random.Generator,
    bar: int,
    rhythm: BarRhythm,
    first_two: bool,
    bar_moves: Sequence[list[int]],
    last_with: Mapping[Pattern, int],
) -> list[int] | None:
    """Cell memory: the earlier bar's moves to replay here, or None."""
    source = rhythm.restated_from
    if source is None:
        source = last_with.get(rhythm.pattern)
    if source is None or source >= bar:
        return None
    if (rhythm.restated_from is not None and first_two) or rng.random() < P_RESTATE_CELL:
        return bar_moves[source]
    return None


def _notes(bar: int, pattern: Pattern, pitches: Sequence[int], eighth_s: float) -> list[Note]:
    out: list[Note] = []
    for k, (slot, pitch) in enumerate(zip(pattern, pitches, strict=True)):
        nxt = pattern[k + 1] if k + 1 < len(pattern) else 8
        start = (bar * 8 + slot) * eighth_s
        velocity = 80 if slot == 0 else 75 if slot == 4 else 70
        end = start + LENGTH_FACTOR * (nxt - slot) * eighth_s
        out.append(Note(start_s=start, end_s=end, pitch=pitch, velocity=velocity, confidence=1.0))
    return out


def _check_allowed(allowed_pitches: Sequence[frozenset[int]], n_bars: int) -> None:
    if len(allowed_pitches) != n_bars:
        raise ValueError("allowed_pitches needs one set per bar.")
    if any(not pitches for pitches in allowed_pitches):
        raise ValueError("allowed_pitches has an empty set.")


def _phrase_offsets(n_bars: int) -> list[int]:
    return [k for length in split_phrases(n_bars) for k in range(length)]


def generate_tune(
    *,
    scale: Scale,
    progression: Sequence[ChordSymbol],
    bars_per_chord: int = 1,
    tempo_bpm: float = 100.0,
    seed: int | None = None,
    tessitura_center: int,
    allowed_pitches: Sequence[frozenset[int]],
    final_whole_note: bool = True,
    pitch_prior: Sequence[Note] | None = None,
    stats: TuneStats | None = None,
) -> list[Note]:
    """A melody over ``progression`` from the statistics of real tunes (spec 5.2).

    Notes sit on the eighth grid, every note of bar ``b`` in
    ``allowed_pitches[b]``, none crossing a bar line; with ``final_whole_note``
    the last bar is one held cadence note. Deterministic for a seed.
    """
    n_bars = len(progression) * bars_per_chord
    _check_allowed(allowed_pitches, n_bars)
    stats = stats if stats is not None else load_tune_stats()
    rng = np.random.default_rng(seed)
    plan = plan_rhythm(n_bars, tempo_bpm, rng, stats, final_whole_note=final_whole_note)
    eighth_s = 30.0 / tempo_bpm
    center = tessitura_center
    piece = _Piece(scale, stats, center, pitch_prior, eighth_s)
    chords = [progression[b // bars_per_chord] for b in range(n_bars)]
    candidates = [_bar_candidates(scale, chords[b], allowed_pitches[b]) for b in range(n_bars)]
    final_pc = cadence_pc(scale, progression[-1])
    goal = {p for p in candidates[-1] if p % 12 == final_pc}
    approach = _approach_sets(plan, candidates, allowed_pitches, goal)
    offsets = _phrase_offsets(n_bars)

    voice = _Voice()
    out: list[Note] = []
    bar_moves: list[list[int]] = []
    last_with: dict[Pattern, int] = {}
    for bar, rhythm in enumerate(plan):
        chord, final_bar = chords[bar], bar == n_bars - 1
        ladder = chord_scale_pcs(scale, chord)
        replay = None
        if not final_bar:
            replay = _replay_moves(rng, bar, rhythm, offsets[bar] < 2, bar_moves, last_with)
        replaced = 0
        pitches: list[int] = []
        for k, slot in enumerate(rhythm.pattern):
            last = k == len(rhythm.pattern) - 1
            spot = _Spot(
                slot=slot,
                onset_s=(bar * 8 + slot) * eighth_s,
                chord=chord,
                ladder=ladder,
                change=voice.prev_chord is not None and chord != voice.prev_chord,
                feasible=approach[bar, k][0] if (bar, k) in approach else None,
                feasible_wide=approach[bar, k][1] if (bar, k) in approach else None,
                phrase_end=last and rhythm.role == "cadence" and not final_bar,
            )
            if final_bar and last:
                pitch = final_target(scale, progression[-1], candidates[bar], voice.ref(center))
            else:
                pool = _pool(spot, allowed_pitches[bar], voice.prev, scale)
                want = None
                if replay is not None and replaced < MAX_REPLAY_REPLACEMENTS:
                    if k == 0:
                        pool = _replay_starts(pool, replay, rhythm.pattern, spot)
                    else:
                        want = _step_by(voice.ref(center), replay[k - 1], spot.ladder)
                        replaced += want not in pool
                if want is not None and want in pool:
                    pitch = want
                else:
                    pitch = _sample(rng, pool, _weights(pool, spot, voice, piece))
            voice.advance(pitch, spot)
            pitches.append(pitch)
        bar_moves.append([scale_steps(a, b, ladder) for a, b in pairwise(pitches)])
        last_with[rhythm.pattern] = bar
        out.extend(_notes(bar, rhythm.pattern, pitches, eighth_s))
    return out
