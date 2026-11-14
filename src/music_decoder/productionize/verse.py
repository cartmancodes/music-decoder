"""Verse writing for composer songs: piano voicings and sparse verse lines.

The composer writes one melody. Played in every section it gets no
contrast, so a verse can instead carry:

- **voice-led voicings**: piano chords (a low root and fifth, then the third
  and seventh with the other chord tones in the middle register) chosen so
  each voice moves as little as possible from one bar to the next, instead
  of the composer's guitar shapes;
- a **guide line**: one held note per bar, searched over each bar's chord
  tones and in-scale ninth so the line moves by step (the approved Hisaab
  verse: one melody note per bar);
- the **skeleton** of the hook melody: only the notes on beats 1 and 3, so
  the verse still quotes the tune at half its density.

Everything here is pure: notes in, notes out.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations, pairwise

from music_decoder.compose.harmony import chord_pcs, chord_scale_pcs, essential_pcs
from music_decoder.productionize.config import ComposerArrangeConfig
from music_decoder.types import VALID_TONICS, ChordSymbol, Scale

_NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


@dataclass(frozen=True)
class BarHarmony:
    """The chord of one bar, as pitch classes."""

    root: int
    tones: frozenset[int]  # chord tones
    essential: frozenset[int]  # third, seventh (the tones that name the chord)
    scale: frozenset[int]  # the scale adapted to this chord

    def role(self, pc: int) -> str:
        """``third``, ``seventh``, ``root``, ``fifth``, ``ninth`` or ``other``."""
        interval = (pc - self.root) % 12
        if interval == 0:
            return "root"
        if interval in (3, 4) and pc in self.tones:
            return "third"
        if interval in (10, 11) and pc in self.tones:
            return "seventh"
        if interval in (6, 7, 8) and pc in self.tones:
            return "fifth"
        if interval == 2 and pc in self.scale:
            return "ninth"
        return "other"

    @property
    def line_tones(self) -> frozenset[int]:
        """Pitch classes a sustained verse note may use: the chord tones and
        the ninth when the scale has it (so never a flat ninth)."""
        ninth = (self.root + 2) % 12
        return self.tones | ({ninth} if ninth in self.scale else set())


def pitch_name(pitch: int) -> str:
    return f"{_NOTE_NAMES[pitch % 12]}{pitch // 12 - 1}"


def bar_harmonies(
    bar_chords: Sequence[Sequence[int]],
    root_pcs: Sequence[int],
    *,
    progression: Sequence[ChordSymbol] | None,
    scale: Scale | None,
    all_pitches: Sequence[int],
) -> list[BarHarmony]:
    """Each bar's harmony: from the chord symbols and the scale when known,
    else from the pitches the accompaniment plays and the whole file uses."""
    used = frozenset(p % 12 for p in all_pitches)
    out = []
    n_bars = len(bar_chords)
    for b in range(n_bars):
        if progression:
            chord = progression[b * len(progression) // n_bars]
            tones = frozenset(chord_pcs(chord))
            essential = essential_pcs(chord)
            pool = chord_scale_pcs(scale, chord) if scale is not None else used | tones
            root = VALID_TONICS.index(chord.root)
        else:
            root = root_pcs[b]
            tones = frozenset(p % 12 for p in bar_chords[b])
            essential = frozenset(pc for pc in tones if (pc - root) % 12 in (3, 4, 10, 11))
            pool = used | tones
        out.append(BarHarmony(root, tones, essential, frozenset(pool)))
    return out


# ---- Voicings --------------------------------------------------------------


def _fold_up(pc: int, low: int) -> int:
    return low + (pc - low) % 12


def left_hand(h: BarHarmony, cfg: ComposerArrangeConfig) -> tuple[int, ...]:
    """Root in the low register, with its fifth when the chord has one."""
    root = _fold_up(h.root, cfg.lh_low)
    fifths = [root + i for i in (7, 6, 8) if (h.root + i) % 12 in h.tones]
    return (root, fifths[0]) if fifths else (root,)


def _movement(a: Sequence[int], b: Sequence[int]) -> int:
    """Semitones the voices move from ``a`` to ``b`` (both sorted, voice by voice)."""
    return sum(abs(x - y) for x, y in zip(a, b, strict=True))


def _voicing_cost(
    voicing: tuple[int, ...], h: BarHarmony, prev: tuple[int, ...] | None, centre: float
) -> float:
    doubled = len(voicing) - len({p % 12 for p in voicing})
    span = voicing[-1] - voicing[0]
    shape = 1.5 * doubled + 0.3 * max(0, span - 12)
    if prev is None:
        return shape + abs(sum(voicing) / len(voicing) - centre) / 2.0
    return shape + _movement(prev, voicing)


def right_hand_options(
    h: BarHarmony, cfg: ComposerArrangeConfig, ceiling: int | None = None
) -> list[tuple[int, ...]]:
    """Every allowed right-hand voicing of ``h`` in the configured register
    (no higher than ``ceiling``): the chord's third and seventh, no two notes
    a semitone apart."""
    high = cfg.rh_high if ceiling is None else max(cfg.rh_low + 7, min(cfg.rh_high, ceiling))
    pool = [p for p in range(cfg.rh_low, high + 1) if p % 12 in h.tones]
    size = min(cfg.rh_voices, len(pool))
    options = []
    for combo in combinations(pool, size):
        if any(b - a < 2 for a, b in pairwise(combo)):
            continue
        if not h.essential <= {p % 12 for p in combo}:
            continue
        options.append(combo)
    return options or [tuple(pool[:size])]


def voice_led_voicings(
    harmonies: Sequence[BarHarmony],
    cfg: ComposerArrangeConfig,
    ceilings: Sequence[int | None] | None = None,
) -> list[tuple[int, ...]]:
    """One right-hand voicing per bar, each moving least from the one before.

    ``ceilings`` caps each bar's top note (to stay under a melody). The
    section repeats, so the first bar is then re-chosen to follow the last
    and the rest re-run from it.
    """
    centre = (cfg.rh_low + cfg.rh_high) / 2.0
    caps = list(ceilings) if ceilings is not None else [None] * len(harmonies)
    options = [right_hand_options(h, cfg, cap) for h, cap in zip(harmonies, caps, strict=True)]

    def run(first_prev: tuple[int, ...] | None) -> list[tuple[int, ...]]:
        chosen: list[tuple[int, ...]] = []
        prev = first_prev
        for h, opts in zip(harmonies, options, strict=True):
            same_size = [o for o in opts if prev is None or len(o) == len(prev)] or opts
            best = min(
                same_size,
                key=lambda o: _voicing_cost(
                    o, h, prev if prev is None or len(o) == len(prev) else None, centre
                ),
            )
            chosen.append(best)
            prev = best
        return chosen

    first_pass = run(None)
    return run(first_pass[-1]) if len(harmonies) > 1 else first_pass


def resolve_voicing(
    h: BarHarmony, prev: tuple[int, ...], cfg: ComposerArrangeConfig
) -> tuple[int, ...]:
    """The right-hand voicing of ``h`` that moves least from ``prev`` (a final
    chord resolving from the last chord of the loop)."""
    centre = (cfg.rh_low + cfg.rh_high) / 2.0
    options = right_hand_options(h, cfg)
    same_size = [o for o in options if len(o) == len(prev)] or options
    return min(
        same_size,
        key=lambda o: _voicing_cost(o, h, prev if len(o) == len(prev) else None, centre),
    )


def tonic_harmony(
    harmonies: Sequence[BarHarmony], tonic_pc: int, *, minor: bool, scale_pcs: frozenset[int]
) -> BarHarmony:
    """The home chord: the loop's chord on the tonic as written (so a maj7
    stays a maj7), else the tonic triad of the key."""
    for h in harmonies:
        if h.root == tonic_pc:
            return h
    third = (tonic_pc + (3 if minor else 4)) % 12
    tones = frozenset({tonic_pc, third, (tonic_pc + 7) % 12})
    return BarHarmony(tonic_pc, tones, frozenset({third}), scale_pcs | tones)


# ---- Guide line ------------------------------------------------------------

_STEP_COST = {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.35, 4: 0.5}
_ROLE_COST = {"third": 0.0, "seventh": 0.0, "fifth": 0.25, "ninth": 0.3, "root": 0.35}
# Coming back to the note of two bars before: a line that rocks between two
# notes (C B C B) goes nowhere.
_RETURN_COST = 0.6


def _interval_cost(a: int, b: int) -> float:
    d = abs(a - b)
    return _STEP_COST.get(d, 0.9 + 0.25 * (d - 5))


def _return_cost(a: int, c: int) -> float:
    return _RETURN_COST if a == c else 0.0


@dataclass(frozen=True)
class GuideLine:
    pitches: tuple[int, ...]
    cost: float

    def names(self) -> str:
        return " ".join(pitch_name(p) for p in self.pitches)


def line_candidates(h: BarHarmony, low: int, high: int) -> list[int]:
    return [p for p in range(low, high + 1) if p % 12 in h.line_tones]


def guide_lines(
    harmonies: Sequence[BarHarmony],
    floors: Sequence[int],
    high: int,
    hook_pcs: Sequence[frozenset[int]],
    *,
    limit: int = 3,
) -> list[GuideLine]:
    """The best one-note-per-bar lines, cheapest first, one per starting note.

    A line moves by step where it can (repeats and leaps cost more), does
    not rock back to the note of two bars before, prefers the third and
    seventh over the fifth, ninth and root, slightly prefers a note the hook
    melody uses in the same bar, and is scored as a loop (the last note
    leads back into the first). ``floors`` keeps each bar's note above that
    bar's chord; ``high`` is the top of the hook melody.
    """
    cands = [
        line_candidates(h, low, max(high, low)) or line_candidates(h, low, low + 12)
        for h, low in zip(harmonies, floors, strict=True)
    ]
    n = len(cands)

    def note_cost(b: int, p: int) -> float:
        role = _ROLE_COST.get(harmonies[b].role(p % 12), 0.6)
        return role - (0.15 if p % 12 in hook_pcs[b] else 0.0)

    if n == 1:
        lines = [GuideLine((p,), round(note_cost(0, p), 3)) for p in cands[0]]
        lines.sort(key=lambda line: (line.cost, line.pitches))
        return lines[:limit]

    best_per_start: dict[int, GuideLine] = {}
    for first in cands[0]:
        for second in cands[1]:
            # Viterbi over pairs (previous note, note) with the first two fixed.
            start_cost = note_cost(0, first) + note_cost(1, second) + _interval_cost(first, second)
            states: dict[tuple[int, int], tuple[float, tuple[int, ...]]] = {
                (first, second): (start_cost, (first, second))
            }
            for b in range(2, n):
                nxt: dict[tuple[int, int], tuple[float, tuple[int, ...]]] = {}
                for p in cands[b]:
                    for (q0, q1), (cost, path) in states.items():
                        total = cost + _interval_cost(q1, p) + _return_cost(q0, p) + note_cost(b, p)
                        if (q1, p) not in nxt or total < nxt[(q1, p)][0]:
                            nxt[(q1, p)] = (total, (*path, p))
                states = nxt
            for (q0, q1), (cost, path) in states.items():
                # Close the loop: last -> first, and the returns across the seam.
                total = cost + _interval_cost(q1, first) + _return_cost(q0, first)
                total += _return_cost(q1, second) if n > 2 else 0.0
                line = GuideLine(path, round(total, 3))
                kept = best_per_start.get(first)
                if kept is None or (line.cost, line.pitches) < (kept.cost, kept.pitches):
                    best_per_start[first] = line
    lines = sorted(best_per_start.values(), key=lambda line: (line.cost, line.pitches))
    return lines[:limit]


# ---- Skeleton of the hook melody --------------------------------------------


def skeleton_pitches(
    melody: Sequence[tuple[float, float, int]], bar_s: float, n_bars: int, steps: Sequence[float]
) -> list[tuple[int, float, int]]:
    """(bar, step, pitch) of the melody note sounding at each of ``steps``
    (16ths) in every bar: the tune at half density or less. A note that
    covers several of the steps is kept once."""
    step_s = bar_s / 16.0
    out: list[tuple[int, float, int]] = []
    for b in range(n_bars):
        last: tuple[float, int] | None = None
        for step in steps:
            t = b * bar_s + step * step_s + step_s / 2.0
            sounding = [n for n in melody if n[0] <= t]
            if not sounding:
                continue
            start, end, pitch = max(sounding, key=lambda n: (n[0], n[2]))
            if start < b * bar_s - step_s / 2.0 and end <= t:
                continue  # nothing new in this bar yet
            if last == (start, pitch):
                continue
            last = (start, pitch)
            out.append((b, step, pitch))
    return out
