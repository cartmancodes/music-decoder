"""The tune engine's gate harness (composer v6 spec 5.5).

Builds fingerstyle pieces with ``melody_engine="tune"`` over two fixed
matrices: ``dev``, which the engine is tuned on, and ``holdout`` (other keys
and seeds), evaluated once at the end. Each matrix holds every key x its
mode's three progressions x three tempos x two seeds at two bars per chord
(8 bars), plus the first four keys of each mode x the first progression at
96 BPM x the two seeds at four bars per chord (16 bars). Every piece is
arranged best of :data:`CANDIDATES`, as ``compose()`` does.

A :class:`TuneCase` keeps the best candidate (``arrangement``, what T3
checks) and the tune features of candidate 0 (``features``, what T1 and T2
check): the score picks by the same bands, so gating its pick would pass by
construction; the engine has to land in them on its own.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from music_decoder.compose import metrics
from music_decoder.compose.api import Arrangement, arrange_candidates
from music_decoder.compose.tune_metrics import FEATURES, tune_features
from music_decoder.evaluation.compose_musicality import SEVENTH_QUALITIES, realize
from music_decoder.tabs.tuning import STANDARD_EADGBE, Tuning
from music_decoder.types import ChordSymbol, Scale

Matrix = Literal["dev", "holdout"]

DEV_KEYS_MAJOR = ("C", "G", "D", "A#", "E", "F")
DEV_KEYS_MINOR = ("A", "E", "D", "C", "F#", "B")
HOLDOUT_KEYS_MAJOR = ("A", "D#", "B", "G#")
HOLDOUT_KEYS_MINOR = ("G", "C#", "F", "A#")
MAJOR_PROGRESSIONS = ("I V vi IV", "I vi IV V", "I IV V I")
MINOR_PROGRESSIONS = ("i VI III VII", "i iv V i", "i VII VI V")
TEMPOS = (72.0, 96.0, 120.0)
DEV_SEEDS, HOLDOUT_SEEDS = (5000, 5001), (6000, 6001)
CANDIDATES = 12  # compose()'s default (v5 spec 5.4)
LONG_TEMPO = 96.0
LONG_KEYS = 4  # the 16-bar pieces use the first four keys of each mode

# T3 limits (v6 spec 5.5)
MAX_MOVE = 12  # no move over an octave
MAX_CHANGE_MOVE = 7  # chord-change and final-approach moves at most a fifth
MIN_FINAL_BEATS = 1.5


@dataclass(frozen=True)
class TuneInputs:
    scale: Scale
    progression: tuple[ChordSymbol, ...]
    bars_per_chord: int
    tempo_bpm: float
    seed: int


@dataclass(frozen=True)
class TuneCase:
    scale: Scale
    progression: tuple[ChordSymbol, ...]
    bars_per_chord: int
    tempo_bpm: float
    seed: int
    arrangement: Arrangement  # the best of CANDIDATES, as compose() writes it
    features: dict[str, float]  # tune_features of candidate 0, before any selection


def _keys_and_seeds(
    matrix: Matrix,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[int, ...]]:
    if matrix == "dev":
        return DEV_KEYS_MAJOR, DEV_KEYS_MINOR, DEV_SEEDS
    if matrix == "holdout":
        return HOLDOUT_KEYS_MAJOR, HOLDOUT_KEYS_MINOR, HOLDOUT_SEEDS
    raise ValueError(f"unknown matrix {matrix!r}; choose 'dev' or 'holdout'")


def tune_inputs(matrix: Matrix) -> list[TuneInputs]:
    """The inputs of one matrix, in a fixed order (8-bar pieces, then 16-bar)."""
    major_keys, minor_keys, seeds = _keys_and_seeds(matrix)
    modes: tuple[tuple[Literal["major", "minor"], tuple[str, ...], tuple[str, ...]], ...] = (
        ("major", major_keys, MAJOR_PROGRESSIONS),
        ("minor", minor_keys, MINOR_PROGRESSIONS),
    )
    out: list[TuneInputs] = []
    for mode, keys, progressions in modes:
        for key in keys:
            scale = Scale(key, mode)
            for numerals in progressions:
                progression = tuple(realize(scale, n) for n in numerals.split())
                for tempo in TEMPOS:
                    out.extend(TuneInputs(scale, progression, 2, tempo, s) for s in seeds)
    for mode, keys, progressions in modes:
        for key in keys[:LONG_KEYS]:
            scale = Scale(key, mode)
            progression = tuple(realize(scale, n) for n in progressions[0].split())
            out.extend(TuneInputs(scale, progression, 4, LONG_TEMPO, s) for s in seeds)
    return out


def build_tune_case(inputs: TuneInputs, tuning: Tuning = STANDARD_EADGBE) -> TuneCase:
    """Arrange one piece best of :data:`CANDIDATES` with the tune engine."""
    ranked = arrange_candidates(
        inputs.scale,
        inputs.progression,
        bars_per_chord=inputs.bars_per_chord,
        tempo_bpm=inputs.tempo_bpm,
        style="fingerstyle",
        tuning=tuning,
        seed=inputs.seed,
        candidates=CANDIDATES,
        melody_engine="tune",
    )
    first = next(a for a in ranked if a.chosen_candidate == 0)
    features = tune_features(
        first.melody, inputs.scale, inputs.progression, inputs.bars_per_chord, inputs.tempo_bpm
    )
    return TuneCase(
        scale=inputs.scale,
        progression=inputs.progression,
        bars_per_chord=inputs.bars_per_chord,
        tempo_bpm=inputs.tempo_bpm,
        seed=inputs.seed,
        arrangement=ranked[0],
        features=features,
    )


def tune_cases(matrix: Matrix) -> list[TuneCase]:
    """Every case of one matrix (``"dev"`` or ``"holdout"``), built on demand."""
    return [build_tune_case(i) for i in tune_inputs(matrix)]


def tune_report(cases: Sequence[TuneCase]) -> dict[str, dict[str, dict[str, float]]]:
    """Per mode and per feature, ``{"q25", "median", "q75"}`` over the cases'
    ``features`` (a case without a feature is left out of that feature)."""
    out: dict[str, dict[str, dict[str, float]]] = {}
    for mode in ("major", "minor"):
        rows = [c.features for c in cases if c.scale.mode == mode]
        per_feature: dict[str, dict[str, float]] = {}
        for name in FEATURES:
            values = [r[name] for r in rows if name in r]
            if values:
                q25, median, q75 = np.percentile(values, [25, 50, 75])
                per_feature[name] = {"q25": float(q25), "median": float(median), "q75": float(q75)}
        if per_feature:
            out[mode] = per_feature
    return out


def triad_identity(case: TuneCase) -> list[bool]:
    """P3 per triad-family bar: the accompaniment alone sounds every essential
    tone (as ``compose_musicality`` computes ``P3t``)."""
    a, prog, bars = case.arrangement, case.progression, case.bars_per_chord
    identity = metrics.bar_identity(a.accompaniment, prog, bars, case.tempo_bpm)
    return [
        met
        for bar, met in enumerate(identity)
        if prog[bar // bars].quality not in SEVENTH_QUALITIES
    ]


def guarantee_failures(case: TuneCase, tuning: Tuning = STANDARD_EADGBE) -> list[str]:
    """The T3 guitar guarantees one case breaks (P3 is pooled separately, see
    :func:`triad_identity`); empty when it keeps them all."""
    a = case.arrangement
    part = (*a.melody_tab, *a.accompaniment_tab)
    failures: list[str] = []
    if metrics.playable_moment_ratio(part) != 1.0:
        failures.append(f"P1={metrics.playable_moment_ratio(part)}")
    if metrics.melody_onset_on_top_ratio(a.melody, a.accompaniment) != 1.0:
        failures.append(f"P2={metrics.melody_onset_on_top_ratio(a.melody, a.accompaniment)}")
    if not metrics.tab_consistent((*a.melody, *a.accompaniment), part, tuning):
        failures.append("P4")
    if metrics.forbidden_interval_count(a.melody):
        failures.append(f"forbidden={metrics.forbidden_interval_count(a.melody)}")
    if metrics.max_interval(a.melody) > MAX_MOVE:
        failures.append(f"max_interval={metrics.max_interval(a.melody)}")
    moves = metrics.chord_change_moves(
        a.melody, case.progression, case.bars_per_chord, case.tempo_bpm
    )
    if max(moves, default=0) > MAX_CHANGE_MOVE:
        failures.append(f"chord_change_move={max(moves)}")
    ivs = metrics.intervals(a.melody)
    if ivs and abs(ivs[-1]) > MAX_CHANGE_MOVE:
        failures.append(f"final_approach={abs(ivs[-1])}")
    if metrics.final_note_beats(a.melody, case.tempo_bpm) < MIN_FINAL_BEATS:
        failures.append(f"final_beats={metrics.final_note_beats(a.melody, case.tempo_bpm):.3f}")
    if not metrics.final_resolution(a.melody, case.scale, case.progression):
        failures.append("final_resolution")
    return failures


def describe(case: TuneCase) -> str:
    """A short name for a case: key, progression, bars per chord, tempo, seed."""
    chords = " ".join(c.to_label() for c in case.progression)
    return (
        f"{case.scale.tonic}:{case.scale.mode} [{chords}] x{case.bars_per_chord}"
        f" @{case.tempo_bpm:g} seed={case.seed}"
    )
