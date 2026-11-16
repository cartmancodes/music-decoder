"""The symbolic score that picks the best of N fingerstyle candidates (v5 spec 5.4).

:func:`score_arrangement` is a weighted sum of normalised terms, each one of
the musicality metrics in :mod:`music_decoder.compose.metrics`. A term whose
metric is ``None`` (nothing to count) contributes its neutral value instead.
The weights are pinned here, so a seed always picks the same candidate:

======  ==========================================================  =======
Weight  Term                                                        Neutral
======  ==========================================================  =======
+3.0    P3: share of bars whose accompaniment meets ``bar_identity``
+1.0    A2: ``melody_above_accompaniment_ratio``                    1
+2.0    M1: ``strong_beat_chord_tone_ratio``                        1
+1.0    M2: ``step_ratio``                                          0.5
+1.0    ``min(ioi_entropy / 2, 1)``
-1.5    M3: ``repeat_ratio``                                        0
+1.0    M5: recovered / leaps from ``leap_recovery_counts``         1
+1.0    M11b: ``motif_contour_match``                               0.5
+1.0    hook: ``hook_restatements``                                 0.5
+1.0    peak: ``peak_uniqueness``                                   0.5
+1.0    ``max(contour_arch, 0)``                                    0
+1.0    ``min(ambitus / 14, 1)``
-0.5    ``shape_moves``                                             0
-3.0    1 if ``final_fallback``
======  ==========================================================  =======

The plan pinned the rhythm and range weights at +0.5. On the best-of-12
harness that let validation A3.fingerstyle rise to 0.0771, over the
``test_relative_clash_reduction`` limit of 0.0769 (0.7 x baseline 0.1098);
both are raised to +1.0, which keeps every musicality gate, and the F.M6
targets with more room than the single candidate had (the plan lets this
feature tune its own weights when a gate fails).

That table is the score of the ``"rules"`` melody engine. For the v6
``"tune"`` engine (v6 spec 5.4) the melody terms (M1, M2, rhythm, M3, M5,
M11b, hook, peak, arch, range) are replaced by
``-W_CORPUS x band_distance(tune_features(melody), bands[mode])``, the
melody's distance from the 25-75% bands of real melodies in the scale's
mode; the identity, melody-above, shape-move and fallback terms are kept.

:data:`MelodyEngine` lives here, not in :mod:`music_decoder.compose.api`
(which re-exports it), because ``api`` imports this module.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np

from music_decoder.compose import metrics
from music_decoder.compose.tune_metrics import band_distance, tune_features
from music_decoder.compose.tune_stats import load_tune_stats
from music_decoder.types import ChordSymbol, Scale, Tuning

if TYPE_CHECKING:
    from music_decoder.compose.api import Arrangement

W_IDENTITY = 3.0
W_ABOVE = 1.0
W_CHORD_TONES = 2.0
W_STEPS = 1.0
W_RHYTHM = 1.0
W_REPEATS = -1.5
W_RECOVERY = 1.0
W_MOTIF = 1.0
W_HOOK = 1.0
W_PEAK = 1.0
W_ARCH = 1.0
W_AMBITUS = 1.0
W_SHAPE_MOVES = -0.5
W_FALLBACK = -3.0
W_CORPUS = 2.0  # tune engine: the score falls by W_CORPUS per band width outside the bands

# The fingerstyle melody generator: the v6 tune engine, or v5's rules engine.
MelodyEngine = Literal["tune", "rules"]

RHYTHM_FULL_BITS = 2.0  # IOI entropy earning the whole rhythm term
AMBITUS_FULL_SEMITONES = 14  # ambitus earning the whole range term


def _or(value: float | None, neutral: float) -> float:
    return neutral if value is None else value


def candidate_seeds(seed: int, n: int) -> tuple[int, ...]:
    """The seeds of candidates k = 0..n-1: ``seed`` itself for k = 0, then a
    :class:`numpy.random.SeedSequence` of ``(seed, k)``. Independent of
    ``PYTHONHASHSEED``."""
    return tuple(
        seed if k == 0 else int(np.random.SeedSequence([seed, k]).generate_state(1)[0])
        for k in range(n)
    )


def score_arrangement(
    arr: Arrangement,
    progression: Sequence[ChordSymbol],
    *,
    bars_per_chord: int,
    tempo_bpm: float,
    tuning: Tuning,
    scale: Scale | None = None,
    melody_engine: MelodyEngine = "rules",
) -> float:
    """The symbolic score of a fingerstyle arrangement (higher is better).

    ``melody_engine`` picks the melody terms: v5's for ``"rules"``, the
    distance from real melodies' bands for ``"tune"``, which needs ``scale``
    (``ValueError`` without it).
    """
    if melody_engine == "tune":
        if scale is None:
            raise ValueError('score_arrangement(melody_engine="tune") needs the scale')
        return _tune_score(arr, progression, scale, bars_per_chord, tempo_bpm, tuning)
    melody, accompaniment = arr.melody, arr.accompaniment
    n_bars = len(progression) * bars_per_chord
    identity = metrics.bar_identity(accompaniment, progression, bars_per_chord, tempo_bpm)
    leaps, recovered = metrics.leap_recovery_counts(melody)
    m1 = metrics.strong_beat_chord_tone_ratio(melody, progression, bars_per_chord, tempo_bpm)
    terms = (
        (W_IDENTITY, sum(identity) / len(identity)),
        (W_ABOVE, _or(metrics.melody_above_accompaniment_ratio(melody, accompaniment), 1.0)),
        (W_CHORD_TONES, _or(m1, 1.0)),
        (W_STEPS, _or(metrics.step_ratio(melody), 0.5)),
        (W_RHYTHM, min(metrics.ioi_entropy(melody, tempo_bpm) / RHYTHM_FULL_BITS, 1.0)),
        (W_REPEATS, _or(metrics.repeat_ratio(melody), 0.0)),
        (W_RECOVERY, recovered / leaps if leaps else 1.0),
        (W_MOTIF, _or(metrics.motif_contour_match(melody, tempo_bpm, n_bars), 0.5)),
        (W_HOOK, _or(metrics.hook_restatements(melody, tempo_bpm, n_bars), 0.5)),
        (W_PEAK, _or(metrics.peak_uniqueness(melody, tempo_bpm, n_bars), 0.5)),
        (W_ARCH, max(_or(metrics.contour_arch(melody, tempo_bpm, n_bars), 0.0), 0.0)),
        (W_AMBITUS, min(metrics.ambitus(melody) / AMBITUS_FULL_SEMITONES, 1.0)),
        (W_SHAPE_MOVES, _or(metrics.shape_moves(arr.bar_voicings, tuning), 0.0)),
        (W_FALLBACK, float(arr.final_fallback)),
    )
    return float(sum(weight * value for weight, value in terms))


def _tune_score(
    arr: Arrangement,
    progression: Sequence[ChordSymbol],
    scale: Scale,
    bars_per_chord: int,
    tempo_bpm: float,
    tuning: Tuning,
) -> float:
    """The ``"tune"`` engine's score: v5's part terms minus ``W_CORPUS`` times
    the melody's band distance (v6 spec 5.4)."""
    melody, accompaniment = arr.melody, arr.accompaniment
    identity = metrics.bar_identity(accompaniment, progression, bars_per_chord, tempo_bpm)
    features = tune_features(melody, scale, progression, bars_per_chord, tempo_bpm)
    distance = band_distance(features, load_tune_stats().bands[scale.mode])
    terms = (
        (W_IDENTITY, sum(identity) / len(identity)),
        (W_ABOVE, _or(metrics.melody_above_accompaniment_ratio(melody, accompaniment), 1.0)),
        (W_SHAPE_MOVES, _or(metrics.shape_moves(arr.bar_voicings, tuning), 0.0)),
        (W_FALLBACK, float(arr.final_fallback)),
        (-W_CORPUS, distance),
    )
    return float(sum(weight * value for weight, value in terms))
