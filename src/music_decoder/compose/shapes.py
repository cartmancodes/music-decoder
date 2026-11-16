"""Per-bar chord shapes for the fingerstyle part (spec 5.2, feature 2).

Each bar may hold a different shape of its chord. :func:`shape_candidates`
lists the shapes a bar may use; :func:`choose_bar_shapes` picks one per bar by
Viterbi over the bars, scoring each shape with :func:`bar_cost` (identity
safety, melody reach, rank, register) and each change of shape with
:func:`move_cost` (hand movement). After the part is arranged,
:func:`repair_identity` re-voices the body bars that missed an essential tone.
"""

from __future__ import annotations

import dataclasses
import functools
import itertools
import math
import statistics
from collections.abc import Sequence

from music_decoder.compose import metrics
from music_decoder.compose.fingerstyle import FingerstylePart, HandShape, body_bar, hand_shape
from music_decoder.compose.harmony import essential_pcs
from music_decoder.compose.voicings import ranked_shapes, voicings_for
from music_decoder.types import ChordSymbol, TabbedNote, TabPosition, Tuning, VoicedChord

MAX_CANDIDATES = 8
W_IDENTITY = 3.0
IDENTITY_FILL = 0.3  # an essential tone on a fill string below the top string
IDENTITY_TOP = 1.0  # an essential tone only the top string (the melody's) sounds
W_REACH = 0.15  # per reachable melody pitch (subtracted)
W_RANK = 0.4  # per candidate rank; the v4 shape is rank 0
W_REGISTER = 0.05  # per semitone the reach's median sits from REGISTER_CENTRE
REGISTER_CENTRE = 67  # G4
W_MOVE = 0.25  # per fret the hand window's lowest fret moves
_TIE_EPS = 1e-9  # costs this close are a tie; the lower rank wins
_EPS_S = 1e-4  # as in fingerstyle: a note starting this close to a bar line is in that bar


def _hashable(tuning: Tuning) -> Tuning:
    # A Tuning built with a list must still hash for the caches.
    return Tuning(tuning.name, tuple(tuning.open_pitches))


def _voiced(chord: ChordSymbol, frets: Sequence[int]) -> VoicedChord:
    return VoicedChord(
        chord=chord, positions=tuple(TabPosition(string=s, fret=f) for s, f in enumerate(frets))
    )


def _frets(voicing: VoicedChord) -> tuple[int, ...]:
    return tuple(p.fret for p in voicing.positions)


def _usable_hand(voicing: VoicedChord, tuning: Tuning) -> HandShape | None:
    try:
        hand = hand_shape(voicing, tuning)
    except ValueError:  # no sounding string
        return None
    return hand if hand.allowed else None


@functools.cache  # both arguments are frozen dataclasses
def _candidate_hands(chord: ChordSymbol, tuning: Tuning) -> tuple[HandShape, ...]:
    """The hands of :func:`shape_candidates`, in rank order."""
    v4 = voicings_for(chord, tuning)[0]
    v4_hand = _usable_hand(v4, tuning)
    hands = [] if v4_hand is None else [v4_hand]
    seen = {_frets(v4)}
    for frets in ranked_shapes(chord, tuning):
        if len(hands) == MAX_CANDIDATES:
            break
        if frets in seen:
            continue
        seen.add(frets)
        hand = _usable_hand(_voiced(chord, frets), tuning)
        if hand is not None:
            hands.append(hand)
    # Nothing reaches a melody pitch (only in exotic custom tunings): keep the
    # v4 shape, as v4 did, so every chord has a candidate.
    return tuple(hands) if hands else (hand_shape(v4, tuning),)


def shape_candidates(chord: ChordSymbol, tuning: Tuning) -> tuple[VoicedChord, ...]:
    """The shapes a bar of ``chord`` may hold, best rank first (spec 5.2).

    The v4 shape (``voicings_for``) first, then :func:`ranked_shapes` in
    order, without duplicate frets and keeping only shapes whose hand reaches
    some melody pitch; at most ``MAX_CANDIDATES``. If no shape reaches one
    (only in exotic custom tunings), the v4 shape alone.
    """
    return tuple(h.voicing for h in _candidate_hands(chord, _hashable(tuning)))


def _identity_risk(hand: HandShape, essential: frozenset[int]) -> float:
    open_pitches = hand.tuning.open_pitches

    def pc(p: TabPosition) -> int:
        return (open_pitches[p.string] + p.fret) % 12

    sounding = [p for p in hand.voicing.positions if p.fret >= 0]
    top = max(p.string for p in sounding)
    thumb = {pc(hand.root), pc(hand.alternate)}
    fills = {pc(p) for p in sounding if p.string in hand.fill_strings and p.string < top}
    risk = 0.0
    for e in essential:
        if e in thumb:
            continue
        # Top-string-only, or (outside a faithful shape) not sounded at all.
        risk += IDENTITY_FILL if e in fills else IDENTITY_TOP
    return W_IDENTITY * risk


def bar_cost(hand: HandShape, essential: frozenset[int], rank: int) -> float:
    """Cost of holding ``hand`` for one bar; lower is better (spec 5.2).

    Identity risk of the ``essential`` pitch classes, minus melody reach,
    plus candidate ``rank`` and the reach's distance from G4. A hand that
    reaches no melody pitch cannot be used: infinite cost.
    """
    if not hand.allowed:
        return math.inf
    reach = -W_REACH * len(hand.allowed)
    register = W_REGISTER * abs(statistics.median(sorted(hand.allowed)) - REGISTER_CENTRE)
    return _identity_risk(hand, essential) + reach + W_RANK * rank + register


def move_cost(a: HandShape, b: HandShape) -> float:
    """Cost of changing from ``a`` to ``b`` between bars: the hand window's
    lowest-fret movement (0 for the same shape)."""
    return W_MOVE * abs(a.window[0] - b.window[0])


@functools.cache  # both arguments are frozen dataclasses
def _scored(chord: ChordSymbol, tuning: Tuning) -> tuple[tuple[HandShape, float], ...]:
    essential = essential_pcs(chord)
    return tuple(
        (hand, bar_cost(hand, essential, rank))
        for rank, hand in enumerate(_candidate_hands(chord, tuning))
    )


def choose_bar_shapes(
    progression: Sequence[ChordSymbol], *, bars_per_chord: int, tuning: Tuning
) -> tuple[VoicedChord, ...]:
    """One shape per bar, ``bars_per_chord`` bars per chord (spec 5.2).

    Viterbi over the bars minimises the summed :func:`bar_cost` plus the
    :func:`move_cost` between consecutive bars. Ties go to the lower
    candidate rank.
    """
    tuning = _hashable(tuning)
    bars = [_scored(c, tuning) for c in progression for _ in range(bars_per_chord)]
    if not bars:
        return ()
    totals = [cost for _, cost in bars[0]]
    back: list[list[int]] = []
    for prev, cur in itertools.pairwise(bars):
        step_totals: list[float] = []
        step_back: list[int] = []
        for hand, cost in cur:
            best_i = 0
            best = totals[0] + move_cost(prev[0][0], hand)
            for i in range(1, len(prev)):
                total = totals[i] + move_cost(prev[i][0], hand)
                if total < best - _TIE_EPS:
                    best_i, best = i, total
            step_totals.append(best + cost)
            step_back.append(best_i)
        totals = step_totals
        back.append(step_back)
    state = 0
    for i in range(1, len(totals)):
        if totals[i] < totals[state] - _TIE_EPS:
            state = i
    path = [state]
    for step_back in reversed(back):
        state = step_back[state]
        path.append(state)
    path.reverse()
    return tuple(bars[b][s][0].voicing for b, s in enumerate(path))


@functools.cache  # both arguments are frozen dataclasses
def _repair_hands(chord: ChordSymbol, tuning: Tuning) -> tuple[HandShape, ...]:
    """The hands :func:`repair_identity` tries: the :func:`shape_candidates`
    in :func:`bar_cost` order, then every other usable :func:`ranked_shapes`
    shape in :func:`bar_cost` order (v5 ruling F2b)."""
    essential = essential_pcs(chord)
    scored = _scored(chord, tuning)
    first = [h for _, h in sorted(enumerate(scored), key=lambda x: (x[1][1], x[0]))]
    seen = {_frets(h.voicing) for h, _ in scored}
    rest: list[tuple[float, int, HandShape]] = []
    for frets in ranked_shapes(chord, tuning):
        if frets in seen:
            continue
        seen.add(frets)
        hand = _usable_hand(_voiced(chord, frets), tuning)
        if hand is not None:
            rank = len(scored) + len(rest)
            rest.append((bar_cost(hand, essential, rank), rank, hand))
    return (*(h for h, _ in first), *(h for _, _, h in sorted(rest, key=lambda x: x[:2])))


def _bar_of(t: TabbedNote, bar_s: float) -> int:
    return int((t.note.start_s + _EPS_S) // bar_s)


def _bar_holds(
    melody: Sequence[TabbedNote], accompaniment: Sequence[TabbedNote], t0: float, t1: float
) -> bool:
    """P1 and P2 hold at every onset in ``[t0, t1)``: the sounding notes are
    playable, and each melody note starting there is on top."""
    part = [*melody, *accompaniment]
    for onset in sorted({round(t.note.start_s, 6) for t in part}):
        if not t0 - _EPS_S <= onset < t1 - _EPS_S:
            continue
        at = onset + _EPS_S
        if not metrics.moment_playable(
            [t.position for t in part if t.note.start_s <= at and t.note.end_s > at]
        ):
            return False
    starting = [t.note for t in melody if t0 - _EPS_S <= t.note.start_s < t1 - _EPS_S]
    ratio = metrics.melody_onset_on_top_ratio(starting, [t.note for t in accompaniment])
    return ratio is None or ratio == 1.0


def repair_identity(
    part: FingerstylePart,
    bar_voicings: Sequence[VoicedChord],
    progression: Sequence[ChordSymbol],
    *,
    bars_per_chord: int,
    tempo_bpm: float,
    tuning: Tuning,
) -> tuple[FingerstylePart, tuple[VoicedChord, ...]]:
    """Re-voice the body bars that miss an essential tone (spec 5.2, repair).

    For each body bar in ``part.identity_misses``, the chord's other shapes
    are tried: its :func:`shape_candidates` in :func:`bar_cost` order, then
    every other :func:`ranked_shapes` shape in that order (v5 ruling F2b).
    The first whose hand reaches every melody pitch of the bar, whose
    re-arranged bar meets identity and which keeps P1 and P2 for the bar
    replaces the bar's notes and its shape. Melody pitches and timing never change, only their
    positions. Returns the part, with ``identity_misses`` recomputed, and
    the bar shapes.
    """
    beat_s = 60.0 / tempo_bpm
    bar_s = 4.0 * beat_s
    tuning = _hashable(tuning)
    voicings = list(bar_voicings)
    melody = list(part.melody_tab)
    accompaniment = list(part.accompaniment_tab)
    for bar in part.identity_misses:
        if bar == len(voicings) - 1:  # the final bar is not a body bar
            continue
        chord = progression[bar // bars_per_chord]
        essential = essential_pcs(chord)
        start = bar * bar_s
        notes = [t.note for t in melody if _bar_of(t, bar_s) == bar]
        current = _frets(voicings[bar])
        for hand in _repair_hands(chord, tuning):
            if _frets(hand.voicing) == current or any(n.pitch not in hand.allowed for n in notes):
                continue
            new = body_bar(hand, notes, bar_start_s=start, beat_s=beat_s, essential=essential)
            if not new.identity_met:
                continue
            new_melody = sorted(
                [*(t for t in melody if _bar_of(t, bar_s) != bar), *new.melody_tab],
                key=lambda t: t.note.start_s,
            )
            new_accompaniment = sorted(
                [*(t for t in accompaniment if _bar_of(t, bar_s) != bar), *new.accompaniment_tab],
                key=lambda t: (t.note.start_s, t.position.string),
            )
            end = max([start + bar_s, *(t.note.end_s for t in new.melody_tab)])
            if _bar_holds(new_melody, new_accompaniment, start, end):
                melody, accompaniment = new_melody, new_accompaniment
                voicings[bar] = hand.voicing
                break
    heard = metrics.bar_identity(
        [t.note for t in accompaniment], progression, bars_per_chord, tempo_bpm
    )
    repaired = dataclasses.replace(
        part,
        melody_tab=tuple(melody),
        accompaniment_tab=tuple(accompaniment),
        identity_misses=tuple(bar for bar, ok in enumerate(heard) if not ok),
    )
    return repaired, tuple(voicings)
