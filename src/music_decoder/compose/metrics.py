"""Objective musicality metrics for composer output.

Pure functions over melody / accompaniment notes. They are applied to the
pre-2026-11 composer (baseline, ``tests/fixtures/compose/musicality_baseline.json``)
and to the current one on the same inputs by
``music_decoder.evaluation.compose_musicality``; definitions follow
``docs/superpowers/specs/2026-11-06-compose-improvements-design.md`` section 4;
the leap and shape metrics (L0-L5, S2) follow
``docs/superpowers/specs/2026-11-07-compose-voicings-leaps-design.md`` section 4.3.

Times are in seconds and ``tempo_bpm`` fixes the 4/4 beat grid. Ratio metrics
return ``None`` when nothing qualifies to be counted, which is distinct from
a score of 0.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from itertools import pairwise

from music_decoder.compose.harmony import (
    chord_pcs,
    essential_pcs,
    leading_tone_pc,
    scale_pcs,
    tonic_pc,
)
from music_decoder.compose.rhythm import phrase_layout
from music_decoder.types import ChordSymbol, Note, Scale, Tuning, VoicedChord

_GRID_TOL_S = 1e-3  # onset counts as "on the grid" within 1 ms
_STROKE_WINDOW_S = 0.06  # onsets this close to a stroke's first onset belong to it
_OVERLAP_EPS_S = 1e-4  # notes must share more than this much time to overlap
_FORBIDDEN_SIZES = frozenset({6, 10, 11})  # tritone, minor and major seventh
_FIFTH = 7  # leaps larger than this are "large"
_OCTAVE = 12  # intervals larger than this are forbidden


def _beat_s(tempo_bpm: float) -> float:
    return 60.0 / tempo_bpm


def chord_at(
    t: float, progression: Sequence[ChordSymbol], bars_per_chord: int, tempo_bpm: float
) -> ChordSymbol:
    """Chord sounding at time ``t`` (bar index // bars_per_chord, clamped)."""
    bar = math.floor(t / (4.0 * _beat_s(tempo_bpm)) + 1e-6)
    return progression[min(bar // bars_per_chord, len(progression) - 1)]


def metric_position(start_s: float, tempo_bpm: float) -> int:
    """Eighth-note slot within the bar (0-7); 0 and 4 are beats 1 and 3."""
    return round(start_s / (_beat_s(tempo_bpm) / 2.0)) % 8


def _is_exposed(n: Note, tempo_bpm: float) -> bool:
    """On beat 1 or 3, or at least 0.9 beats long."""
    return metric_position(n.start_s, tempo_bpm) in (0, 4) or (
        n.end_s - n.start_s >= 0.9 * _beat_s(tempo_bpm)
    )


def intervals(melody: Sequence[Note]) -> list[int]:
    return [b.pitch - a.pitch for a, b in pairwise(melody)]


def _ratio(hits: int, total: int) -> float | None:
    return hits / total if total else None


# ---- melody ---------------------------------------------------------------


def strong_beat_chord_tone_ratio(
    melody: Sequence[Note],
    progression: Sequence[ChordSymbol],
    bars_per_chord: int,
    tempo_bpm: float,
) -> float | None:
    """M1: share of beat-1/beat-3 onsets whose pitch class is a chord tone."""
    total = hits = 0
    for n in melody:
        if metric_position(n.start_s, tempo_bpm) not in (0, 4):
            continue
        total += 1
        chord = chord_at(n.start_s, progression, bars_per_chord, tempo_bpm)
        hits += n.pitch % 12 in chord_pcs(chord)
    return _ratio(hits, total)


def step_ratio(melody: Sequence[Note]) -> float | None:
    """M2: share of intervals of 1-2 semitones."""
    ivs = intervals(melody)
    return _ratio(sum(1 <= abs(i) <= 2 for i in ivs), len(ivs))


def repeat_ratio(melody: Sequence[Note]) -> float | None:
    """M3: share of repeated-pitch intervals."""
    ivs = intervals(melody)
    return _ratio(sum(i == 0 for i in ivs), len(ivs))


def large_leap_ratio(melody: Sequence[Note]) -> float | None:
    """M4: share of intervals larger than a perfect fifth."""
    ivs = intervals(melody)
    return _ratio(sum(abs(i) > 7 for i in ivs), len(ivs))


def max_interval(melody: Sequence[Note]) -> int:
    return max((abs(i) for i in intervals(melody)), default=0)


def leap_recovery_counts(melody: Sequence[Note]) -> tuple[int, int]:
    """M5: (leaps >= 5 semitones followed by another note, of those recovered).

    Recovered = the next interval moves the opposite way by 1-2 semitones.
    Counts, so callers can pool them across compositions.
    """
    ivs = intervals(melody)
    leaps = recovered = 0
    for i, j in pairwise(ivs):
        if abs(i) >= 5:
            leaps += 1
            recovered += i * j < 0 and 1 <= abs(j) <= 2
    return leaps, recovered


def forbidden_interval_count(melody: Sequence[Note]) -> int:
    """L1: intervals of 6, 10 or 11 semitones, or larger than an octave."""
    return sum(abs(i) in _FORBIDDEN_SIZES or abs(i) > _OCTAVE for i in intervals(melody))


def large_leap_recovery_counts(melody: Sequence[Note]) -> tuple[int, int]:
    """L2: (leaps > a fifth followed by another note, of those recovered).

    M5 with a "larger than a fifth" threshold; recovered = the next interval
    moves the opposite way by 1-2 semitones.
    """
    leaps = recovered = 0
    for i, j in pairwise(intervals(melody)):
        if abs(i) > _FIFTH:
            leaps += 1
            recovered += i * j < 0 and 1 <= abs(j) <= 2
    return leaps, recovered


def unjustified_large_leap_count(
    melody: Sequence[Note],
    progression: Sequence[ChordSymbol],
    bars_per_chord: int,
    tempo_bpm: float,
) -> int:
    """L3: leaps > a fifth that are not an arpeggiation inside one chord.

    A leap is unjustified when its two notes sound over different chords,
    either note is not a tone of the chord under it, or it lands on the
    final note.
    """
    count = 0
    last = len(melody) - 1
    for k, (a, b) in enumerate(pairwise(melody), start=1):
        if abs(b.pitch - a.pitch) <= _FIFTH:
            continue
        ca = chord_at(a.start_s, progression, bars_per_chord, tempo_bpm)
        cb = chord_at(b.start_s, progression, bars_per_chord, tempo_bpm)
        count += (
            ca != cb
            or a.pitch % 12 not in chord_pcs(ca)
            or b.pitch % 12 not in chord_pcs(cb)
            or k == last
        )
    return count


def chord_change_moves(
    melody: Sequence[Note],
    progression: Sequence[ChordSymbol],
    bars_per_chord: int,
    tempo_bpm: float,
) -> list[int]:
    """L4: absolute intervals between consecutive notes over different chords."""
    return [
        abs(b.pitch - a.pitch)
        for a, b in pairwise(melody)
        if chord_at(a.start_s, progression, bars_per_chord, tempo_bpm)
        != chord_at(b.start_s, progression, bars_per_chord, tempo_bpm)
    ]


def ambitus(melody: Sequence[Note]) -> int:
    """M6: melody range in semitones."""
    pitches = [n.pitch for n in melody]
    return max(pitches) - min(pitches) if pitches else 0


def final_resolution(
    melody: Sequence[Note], scale: Scale, progression: Sequence[ChordSymbol]
) -> bool:
    """M7: last note is the tonic if the final chord holds it, else the final root."""
    if not melody:
        return False
    last_chord = progression[-1]
    if tonic_pc(scale) in chord_pcs(last_chord):
        target = tonic_pc(scale)
    else:
        target = chord_pcs(last_chord)[0]
    return melody[-1].pitch % 12 == target


def final_note_beats(melody: Sequence[Note], tempo_bpm: float) -> float:
    """M8: sounding length of the last melody note in beats."""
    if not melody:
        return 0.0
    return (melody[-1].end_s - melody[-1].start_s) / _beat_s(tempo_bpm)


def _durations_in_eighths(melody: Sequence[Note], tempo_bpm: float) -> list[int]:
    eighth = _beat_s(tempo_bpm) / 2.0
    out = [round((b.start_s - a.start_s) / eighth) for a, b in pairwise(melody)]
    if melody:
        out.append(round((melody[-1].end_s - melody[-1].start_s) / eighth))
    return out


def distinct_durations(melody: Sequence[Note], tempo_bpm: float) -> int:
    """M9: number of distinct note lengths (onset spacing, in eighths)."""
    return len(set(_durations_in_eighths(melody, tempo_bpm)))


def ioi_entropy(melody: Sequence[Note], tempo_bpm: float) -> float:
    """M10: Shannon entropy (bits) of inter-onset intervals quantised to eighths."""
    iois = _durations_in_eighths(melody, tempo_bpm)[:-1]
    if not iois:
        return 0.0
    counts = Counter(iois)
    total = len(iois)
    return -sum(c / total * math.log2(c / total) for c in counts.values())


def _bar_notes(melody: Sequence[Note], bar: int, tempo_bpm: float) -> list[Note]:
    bar_s = 4.0 * _beat_s(tempo_bpm)
    lo, hi = bar * bar_s - 1e-6, (bar + 1) * bar_s - 1e-6
    return [n for n in melody if lo <= n.start_s < hi]


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _echo_pairs(
    melody: Sequence[Note], tempo_bpm: float, n_bars: int
) -> list[tuple[list[Note], list[Note]]]:
    return [
        (_bar_notes(melody, source, tempo_bpm), _bar_notes(melody, target, tempo_bpm))
        for source, target in phrase_layout(n_bars).echo_pairs
    ]


def motif_pair_coverage(melody: Sequence[Note], tempo_bpm: float, n_bars: int) -> float | None:
    """M11a: share of the layout's echo pairs whose two bars have the same
    number of notes (>= 2), i.e. that can carry a repeated motif."""
    pairs = _echo_pairs(melody, tempo_bpm, n_bars)
    covered = sum(len(a) == len(b) >= 2 for a, b in pairs)
    return _ratio(covered, len(pairs))


def motif_contour_match(melody: Sequence[Note], tempo_bpm: float, n_bars: int) -> float | None:
    """M11b: interval-direction agreement, averaged over the covered echo pairs."""
    scores: list[float] = []
    for a, b in _echo_pairs(melody, tempo_bpm, n_bars):
        if len(a) != len(b) or len(a) < 2:
            continue
        sa = [_sign(i) for i in intervals(a)]
        sb = [_sign(i) for i in intervals(b)]
        scores.append(sum(x == y for x, y in zip(sa, sb, strict=True)) / len(sa))
    return sum(scores) / len(scores) if scores else None


def phrase_end_long_ratio(melody: Sequence[Note], tempo_bpm: float, n_bars: int) -> float:
    """M12: share of the layout's phrase-final bars whose last note lasts >= 1.5 beats."""
    beat = _beat_s(tempo_bpm)
    cadences = phrase_layout(n_bars).cadence_bars
    hits = 0
    for bar in cadences:
        notes = _bar_notes(melody, bar, tempo_bpm)
        hits += bool(notes) and notes[-1].end_s - notes[-1].start_s >= 1.5 * beat
    return hits / len(cadences)


def exposed_chromatic_clash_ratio(
    melody: Sequence[Note],
    scale: Scale,
    progression: Sequence[ChordSymbol],
    bars_per_chord: int,
    tempo_bpm: float,
) -> float | None:
    """M13: over chords with an out-of-scale tone, the share of exposed melody
    notes (beat 1/3 or >= 0.9 beat) that are non-chord tones a semitone from
    one of those foreign tones (e.g. a held G against E7's G# in A minor).
    Chord tones and short unaccented passing/neighbour notes never count.
    ``None`` when no exposed note sounds over such a chord."""
    scale_set = set(scale_pcs(scale))
    total = hits = 0
    for n in melody:
        chord = chord_at(n.start_s, progression, bars_per_chord, tempo_bpm)
        tones = set(chord_pcs(chord))
        foreign = tones - scale_set
        if not foreign or not _is_exposed(n, tempo_bpm):
            continue
        total += 1
        pc = n.pitch % 12
        hits += pc not in tones and any((pc - f) % 12 in (1, 11) for f in foreign)
    return _ratio(hits, total)


def leading_tone_resolution_counts(
    melody: Sequence[Note],
    scale: Scale,
    progression: Sequence[ChordSymbol],
    bars_per_chord: int,
    tempo_bpm: float,
) -> tuple[int, int]:
    """M15: (leading tones sounding as a chord tone that are followed by a note
    in a new, tonic-containing chord; of those, resolved up a semitone)."""
    lead, tonic = leading_tone_pc(scale), tonic_pc(scale)
    eligible = resolved = 0
    for a, b in pairwise(melody):
        chord_a = chord_at(a.start_s, progression, bars_per_chord, tempo_bpm)
        chord_b = chord_at(b.start_s, progression, bars_per_chord, tempo_bpm)
        if a.pitch % 12 != lead or lead not in chord_pcs(chord_a):
            continue
        if chord_b == chord_a or tonic not in chord_pcs(chord_b):
            continue
        eligible += 1
        resolved += b.pitch == a.pitch + 1
    return eligible, resolved


def onset_grid_ratio(onsets: Sequence[float], tempo_bpm: float, per_beat: int) -> float | None:
    """M14 / A1: share of onsets within 1 ms of a 1/``per_beat``-beat grid."""
    step = _beat_s(tempo_bpm) / per_beat
    hits = sum(abs(t / step - round(t / step)) * step <= _GRID_TOL_S for t in onsets)
    return _ratio(hits, len(onsets))


# ---- accompaniment ----------------------------------------------------------


def stroke_heads(onsets: Sequence[float]) -> list[float]:
    """First onset of each stroke (onsets within 60 ms of a stroke's first onset)."""
    heads: list[float] = []
    for t in sorted(onsets):
        if not heads or t - heads[-1] > _STROKE_WINDOW_S:
            heads.append(t)
    return heads


def _overlapping(accompaniment: Sequence[Note], n: Note) -> list[int]:
    """Accompaniment pitches sounding at any time during melody note ``n``."""
    return [
        a.pitch
        for a in accompaniment
        if a.start_s < n.end_s - _OVERLAP_EPS_S and a.end_s > n.start_s + _OVERLAP_EPS_S
    ]


def melody_above_accompaniment_ratio(
    melody: Sequence[Note], accompaniment: Sequence[Note]
) -> float | None:
    """A2: share of melody notes higher than every accompaniment pitch that
    overlaps them at any point (late-entering strum voices included)."""
    hits = 0
    for n in melody:
        overlapping = _overlapping(accompaniment, n)
        hits += not overlapping or n.pitch > max(overlapping)
    return _ratio(hits, len(melody))


def exposed_semitone_clash_ratio(
    melody: Sequence[Note], accompaniment: Sequence[Note], tempo_bpm: float
) -> float | None:
    """A3: share of exposed melody notes (beat 1/3 or >= 0.9 beat) that overlap
    an accompaniment pitch a minor second / major seventh away (pitch-class
    distance 1 or 11) at any point while they sound."""
    total = hits = 0
    for n in melody:
        if not _is_exposed(n, tempo_bpm):
            continue
        total += 1
        hits += any((n.pitch - p) % 12 in (1, 11) for p in _overlapping(accompaniment, n))
    return _ratio(hits, total)


def final_chord_rings(accompaniment: Sequence[Note], n_bars: int, tempo_bpm: float) -> bool:
    """A4: final-bar accompaniment is one attack (within 60 ms) ringing to the bar end."""
    bar_s = 4.0 * _beat_s(tempo_bpm)
    start, end = (n_bars - 1) * bar_s, n_bars * bar_s
    last = [a for a in accompaniment if a.start_s >= start - 1e-6]
    if not last:
        return False
    return all(a.start_s - start <= _STROKE_WINDOW_S for a in last) and all(
        a.end_s >= end - _GRID_TOL_S for a in last
    )


def _voicing_pitches(voicing: VoicedChord, tuning: Tuning) -> list[int]:
    return [tuning.open_pitches[p.string] + p.fret for p in voicing.positions if p.fret >= 0]


def essential_tone_coverage(
    melody: Sequence[Note],
    accompaniment: Sequence[Note],
    voicings: Sequence[VoicedChord],
    bars_per_chord: int,
    tempo_bpm: float,
    tuning: Tuning,
) -> list[bool]:
    """H1: per bar, whether every essential tone (third, seventh, altered
    fifth, suspended fourth) that the chord's *shape* contains is heard during
    the bar in the accompaniment or the melody.

    This measures what the arrangement preserves, relative to the supplied
    shape; a tone the shape itself lacks is reported by
    :func:`shape_tone_mismatch`, not here."""
    bar_s = 4.0 * _beat_s(tempo_bpm)
    out: list[bool] = []
    for bar in range(len(voicings) * bars_per_chord):
        voicing = voicings[bar // bars_per_chord]
        shape_pcs = {p % 12 for p in _voicing_pitches(voicing, tuning)}
        required = essential_pcs(voicing.chord) & shape_pcs
        t0, t1 = bar * bar_s, (bar + 1) * bar_s
        heard = {
            n.pitch % 12
            for n in (*accompaniment, *melody)
            if n.start_s < t1 - _OVERLAP_EPS_S and n.end_s > t0 + _OVERLAP_EPS_S
        }
        out.append(required <= heard)
    return out


_CADENCE_COVER = 0.9  # tolerates the 5% articulation gaps between re-struck notes


def _covered_fraction(notes: Sequence[Note], t0: float, t1: float) -> float:
    """Fraction of [t0, t1) during which at least one of ``notes`` sounds."""
    clipped = ((max(n.start_s, t0), min(n.end_s, t1)) for n in notes)
    spans = sorted((start, end) for start, end in clipped if end > start)  # drop disjoint notes
    covered, reach = 0.0, t0
    for start, end in spans:
        if end <= reach:
            continue
        covered += end - max(start, reach)
        reach = end
    return covered / (t1 - t0) if t1 > t0 else 0.0


def cadence_identity(
    melody: Sequence[Note],
    accompaniment: Sequence[Note],
    voicings: Sequence[VoicedChord],
    tuning: Tuning,
) -> bool | None:
    """H2: whether every essential tone of the final chord's shape sounds
    under the whole final melody note: as that note's own pitch class, or in
    accompaniment notes of that pitch class covering >= 90% of it.
    ``None`` without a melody.

    Unlike H1's last bar, a tone heard only earlier in the bar does not count;
    this is the harmony under the held cadence note."""
    if not melody or not voicings:
        return None
    final = melody[-1]
    shape = {p % 12 for p in _voicing_pitches(voicings[-1], tuning)}
    for pc in essential_pcs(voicings[-1].chord) & shape:
        if pc == final.pitch % 12:
            continue
        same = [a for a in accompaniment if a.pitch % 12 == pc]
        if _covered_fraction(same, final.start_s, final.end_s) < _CADENCE_COVER:
            return False
    return True


# ---- tab and voicings -----------------------------------------------------


def shape_tone_mismatch(
    voicing: VoicedChord, tuning: Tuning
) -> tuple[frozenset[int], frozenset[int]]:
    """S1: (essential tones of the chord missing from its shape, shape tones
    that are not chord tones). Both empty for a faithful shape."""
    shape = {p % 12 for p in _voicing_pitches(voicing, tuning)}
    return (
        frozenset(essential_pcs(voicing.chord) - shape),
        frozenset(shape - set(chord_pcs(voicing.chord))),
    )


def bass_is_root(voicing: VoicedChord, tuning: Tuning) -> bool:
    """S2: the lowest sounding pitch is the chord root (False for an empty shape)."""
    pitches = _voicing_pitches(voicing, tuning)
    return bool(pitches) and min(pitches) % 12 == chord_pcs(voicing.chord)[0]


def hand_shift_rate(frets: Sequence[int]) -> float | None:
    """T2: hand-position shifts per fretted transition (window spans <= 4 frets)."""
    fretted = [f for f in frets if f > 0]
    if len(fretted) < 2:
        return None
    lo = hi = fretted[0]
    shifts = 0
    for f in fretted[1:]:
        new_lo, new_hi = min(lo, f), max(hi, f)
        if new_hi - new_lo > 4:
            shifts += 1
            lo = hi = f
        else:
            lo, hi = new_lo, new_hi
    return shifts / (len(fretted) - 1)


def nearest_pitch_proximity(voicings: Sequence[VoicedChord], tuning: Tuning) -> float | None:
    """V1: mean distance (semitones) from each pitch of a chord shape to the
    nearest pitch of the previous, different shape.

    A proximity statistic only, not a voice-leading score: several voices may
    share one nearest source, dropped voices cost nothing and crossings are
    ignored. Used solely as a guard that the (unchanged) shapes stay close.
    """
    moves: list[float] = []
    for a, b in pairwise(voicings):
        pa, pb = _voicing_pitches(a, tuning), _voicing_pitches(b, tuning)
        if a.chord == b.chord or not pa or not pb:
            continue
        moves.append(sum(min(abs(x - y) for y in pa) for x in pb) / len(pb))
    return sum(moves) / len(moves) if moves else None
