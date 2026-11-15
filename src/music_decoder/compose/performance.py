"""Expressive performance of the fingerstyle part (composer v5, spec 5.1).

:func:`shape_dynamics` writes the velocities (melody over accompaniment, a
swell over each phrase, a louder thumb on beats 1 and 3, soft fills) and
:func:`let_ring` the lengths (accompaniment notes ring on until their string
is plucked again or the bar ends, unless ringing would break playability or
put them at or above the melody). Onsets, pitches and the tab's positions
never change, so the MIDI and the tab stay on the eighth-note grid.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

from music_decoder.compose.arrangement import THIN_MARGIN_SEMITONES
from music_decoder.compose.fingerstyle import THUMB_EIGHTHS, FingerstylePart
from music_decoder.compose.harmony import essential_pcs
from music_decoder.compose.metrics import moment_playable
from music_decoder.compose.rhythm import phrase_layout
from music_decoder.types import ChordSymbol, TabbedNote

MELODY_VELOCITY = 78
STRONG_BEAT_BOOST = 6  # melody notes on eighths 0 and 4 (beats 1 and 3)
STRONG_EIGHTHS = (0, 4)
SWELL_VELOCITY = 10  # added at a phrase's peak note
FINAL_MELODY_VELOCITY = 84
THUMB_STRONG_VELOCITY = 66  # thumb on beats 1 and 3
THUMB_WEAK_VELOCITY = 58  # thumb on beats 2 and 4
FILL_VELOCITY = 50
ROLL_VELOCITY = 70
MELODY_MARGIN = 8  # melody at least this much louder than the accompaniment under it
MIN_VELOCITY = 30
MAX_VELOCITY = 118
_EPS_S = 1e-4  # notes must share more than this much time to overlap


def _with_velocity(t: TabbedNote, velocity: int) -> TabbedNote:
    v = max(MIN_VELOCITY, min(MAX_VELOCITY, velocity))
    return dataclasses.replace(t, note=dataclasses.replace(t.note, velocity=v))


def _with_end(t: TabbedNote, end_s: float) -> TabbedNote:
    return dataclasses.replace(t, note=dataclasses.replace(t.note, end_s=end_s))


def _sounding_at(notes: Sequence[TabbedNote], t: float) -> list[TabbedNote]:
    """The notes sounding at time ``t`` (those starting then included)."""
    return [x for x in notes if x.note.start_s <= t + _EPS_S and x.note.end_s > t + _EPS_S]


def _swell(times: Sequence[float], pitches: Sequence[int]) -> list[float]:
    """The phrase swell weight ``w`` of each note of one phrase (time-sorted).

    ``w`` rises linearly in onset time from 0 at the first note to 1 at the
    peak (the highest note, the first one if tied), then falls linearly to 0
    at the last note.
    """
    peak = max(range(len(pitches)), key=lambda i: (pitches[i], -i))
    t0, tp, tl = times[0], times[peak], times[-1]
    out: list[float] = []
    for t in times:
        if t <= tp:
            out.append((t - t0) / (tp - t0) if tp > t0 else 1.0)
        else:
            out.append((tl - t) / (tl - tp))
    return out


def shape_dynamics(part: FingerstylePart, *, n_bars: int, tempo_bpm: float) -> FingerstylePart:
    """Write the v5 velocities into ``part``. Spec 5.1, dynamics.

    Accompaniment roles come from string and timing: in a body bar the
    lowest-string note starting on each of :data:`THUMB_EIGHTHS` is the thumb
    (the hand's root or alternate string; an alternate-bass substitute sits
    no higher), every other body note is a fill, and the final bar is the
    roll. The final melody note is :data:`FINAL_MELODY_VELOCITY`; every other
    melody note is :data:`MELODY_VELOCITY` plus the strong-beat boost and its
    phrase's swell (phrases from :func:`phrase_layout`). Then every melody
    note is raised to at least :data:`MELODY_MARGIN` over the loudest
    accompaniment note sounding at its onset, and every velocity is clamped.
    """
    beat_s = 60.0 / tempo_bpm
    eighth_s, bar_s = beat_s / 2.0, 4.0 * beat_s
    final_start = (n_bars - 1) * bar_s

    def bar_of(t: float) -> int:
        return int((t + _EPS_S) // bar_s)

    def eighth_of(t: float) -> int:
        return round((t - bar_of(t) * bar_s) / eighth_s)

    # ---- accompaniment
    lowest: dict[tuple[int, int], int] = {}  # (bar, thumb eighth) -> lowest string
    for t in part.accompaniment_tab:
        key = (bar_of(t.note.start_s), eighth_of(t.note.start_s))
        if key[1] in THUMB_EIGHTHS:
            lowest[key] = min(lowest.get(key, t.position.string), t.position.string)

    def accompaniment_velocity(t: TabbedNote) -> int:
        start = t.note.start_s
        if start >= final_start - _EPS_S:
            return ROLL_VELOCITY
        key = (bar_of(start), eighth_of(start))
        if lowest.get(key) == t.position.string:
            return THUMB_STRONG_VELOCITY if key[1] in STRONG_EIGHTHS else THUMB_WEAK_VELOCITY
        return FILL_VELOCITY

    accompaniment = tuple(
        _with_velocity(t, accompaniment_velocity(t)) for t in part.accompaniment_tab
    )

    # ---- melody
    melody = list(part.melody_tab)
    velocities = [MELODY_VELOCITY] * len(melody)
    first_bar = 0
    for length in phrase_layout(n_bars).phrase_lengths:
        idx = [
            i
            for i, t in enumerate(melody)
            if first_bar <= bar_of(t.note.start_s) < first_bar + length
        ]
        first_bar += length
        if not idx:
            continue
        weights = _swell([melody[i].note.start_s for i in idx], [melody[i].note.pitch for i in idx])
        for i, w in zip(idx, weights, strict=True):
            velocities[i] += round(SWELL_VELOCITY * w)
    for i, t in enumerate(melody):
        start = t.note.start_s
        if start >= final_start - _EPS_S:
            velocities[i] = FINAL_MELODY_VELOCITY
        elif eighth_of(start) in STRONG_EIGHTHS:
            velocities[i] += STRONG_BEAT_BOOST
        under = [a.note.velocity for a in _sounding_at(accompaniment, start)]
        if under and velocities[i] < max(under) + MELODY_MARGIN:
            velocities[i] = max(under) + MELODY_MARGIN
    return dataclasses.replace(
        part,
        melody_tab=tuple(_with_velocity(t, v) for t, v in zip(melody, velocities, strict=True)),
        accompaniment_tab=accompaniment,
    )


def let_ring(
    part: FingerstylePart,
    progression: Sequence[ChordSymbol],
    *,
    bars_per_chord: int,
    tempo_bpm: float,
) -> FingerstylePart:
    """Let the accompaniment ring. Spec 5.1, let-ring.

    Each accompaniment note outside the final bar, in time order, rings to
    the next onset on its string (any note of the part) or its bar's end,
    whichever comes first, but is cut at the first onset ``t`` in between
    where (a) the notes sounding at ``t`` with it ringing are not
    :func:`moment_playable`, (b) a melody note starting at ``t`` is not above
    it, or (c) a melody note starting at ``t`` is less than
    ``THIN_MARGIN_SEMITONES`` above it and it is not an essential tone of
    the bar's chord. No note gets shorter than it was; the final bar and the
    melody are unchanged.
    """
    bar_s = 4.0 * 60.0 / tempo_bpm
    final_start = (len(progression) * bars_per_chord - 1) * bar_s
    melody = list(part.melody_tab)
    accompaniment = list(part.accompaniment_tab)
    onsets = sorted({round(t.note.start_s, 6) for t in (*melody, *accompaniment)})

    for i, a in enumerate(accompaniment):
        start, p, string = a.note.start_s, a.note.pitch, a.position.string
        if start >= final_start - _EPS_S:
            continue
        bar = int((start + _EPS_S) // bar_s)
        essential = essential_pcs(progression[bar // bars_per_chord])
        end = min(
            [
                x.note.start_s
                for x in (*melody, *accompaniment)
                if x.position.string == string and x.note.start_s > start + _EPS_S
            ]
            + [(bar + 1) * bar_s]
        )
        for t in onsets:
            if not start + _EPS_S < t < end - _EPS_S:
                continue
            others = [*melody, *accompaniment[:i], *accompaniment[i + 1 :]]
            sounding = [*_sounding_at(others, t), a]
            starting = [m.note.pitch for m in melody if abs(m.note.start_s - t) < _EPS_S]
            if (
                not moment_playable([x.position for x in sounding])
                or any(p >= m for m in starting)
                or any(p > m - THIN_MARGIN_SEMITONES and p % 12 not in essential for m in starting)
            ):
                end = t
                break
        if end > a.note.end_s:
            accompaniment[i] = _with_end(a, end)
    return dataclasses.replace(part, accompaniment_tab=tuple(accompaniment))
