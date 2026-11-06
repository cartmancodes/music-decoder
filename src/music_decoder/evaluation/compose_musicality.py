"""Musicality evaluation of the composer over fixed, named input matrices.

Shared by ``scripts/compose_musicality_report.py`` (which wrote the committed
pre-change baseline, ``tests/fixtures/compose/musicality_baseline.json``) and
``tests/unit/test_compose_musicality.py`` (which compares the current
composer against it on exactly the same inputs).

Seeds 1000+ are disjoint from the seeds 0-19 used while tuning the
generator's weights, so the matrices are held-out validation data.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Literal, cast

from music_decoder.compose import metrics
from music_decoder.compose.api import Arrangement, accompany, arrange
from music_decoder.tabs.tuning import STANDARD_EADGBE
from music_decoder.types import VALID_TONICS, ChordSymbol, Note, Scale

Style = Literal["arpeggio", "strum", "fingerstyle"]
STYLES: tuple[Style, ...] = ("arpeggio", "strum", "fingerstyle")
TEMPO_BPM = 120.0
VALIDATION_SEEDS = tuple(range(1000, 1005))
EDGE_SEEDS = tuple(range(1000, 1010))
MAJOR_KEYS = ("C", "G", "A#", "E")
MINOR_KEYS = ("A", "E", "C", "F#")
MAJOR_PROGRESSIONS = (
    ("I", "V", "vi", "IV"),
    ("Imaj7", "vi7", "ii7", "V7"),
    ("I", "IV", "V", "I"),
    ("ii7", "V7", "Imaj7", "Imaj7"),
    ("I", "vi", "IV", "V"),
)
MINOR_PROGRESSIONS = (
    ("i", "iv", "V7", "i"),
    ("i", "VI", "III", "VII"),
    ("i", "iv", "v", "i"),
    ("i", "VII", "VI", "V"),
    ("iidim", "V7", "i", "i"),
)
# Odd lengths (3/5/7/9 bars), a single chord, secondary dominants (including
# B7 -> Em in C), a borrowed chord and final suspended chords; one bar per chord.
EDGE_PROGRESSIONS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("C", "major", ("C", "F", "G")),
    ("G", "major", ("G", "C", "D", "Em", "G")),
    ("A", "minor", ("Am", "Dm", "E7", "Am", "F", "Dm", "E7")),
    ("D", "major", ("D", "Bm", "G", "A", "D", "G", "Em", "A", "D")),
    ("D", "major", ("D",)),
    ("C", "major", ("C", "D7", "G", "C")),
    ("C", "major", ("C", "Fm", "C", "G7")),
    ("E", "minor", ("Em", "B7", "Em", "C", "B7", "Em")),
    ("C", "major", ("C", "B7", "Em", "G7", "C")),
    ("A", "major", ("A", "Dsus4")),
    ("A", "minor", ("Am", "Dsus4")),
)

_MAJOR_DEGREES = {"I": 0, "ii": 2, "iii": 4, "IV": 5, "V": 7, "vi": 9, "vii": 11}
_MINOR_DEGREES = {"i": 0, "ii": 2, "III": 3, "iv": 5, "v": 7, "V": 7, "VI": 8, "VII": 10}
_LABEL_SUFFIX = {"maj": "", "min": "m", "min7": "m7", "7": "7", "maj7": "maj7", "dim": "dim"}


def realize(scale: Scale, numeral: str) -> ChordSymbol:
    """Roman numeral (``I``, ``vi7``, ``V7``, ``iidim``…) to a chord in ``scale``."""
    match = re.fullmatch(r"([ivIV]+)(.*)", numeral)
    if match is None:
        raise ValueError(f"bad numeral {numeral!r}")
    degree, suffix = match.groups()
    table = _MAJOR_DEGREES if scale.mode == "major" else _MINOR_DEGREES
    root = VALID_TONICS[(VALID_TONICS.index(scale.tonic) + table[degree]) % 12]
    upper = degree.isupper()
    if suffix in ("dim", "maj7"):
        quality = suffix
    elif suffix == "7":
        quality = "7" if upper else "min7"
    else:
        quality = "maj" if upper else "min"
    return ChordSymbol.parse(root + _LABEL_SUFFIX[quality])


@dataclass(frozen=True)
class Case:
    """One generated composition plus its accompaniment in every style."""

    scale: Scale
    progression: tuple[ChordSymbol, ...]
    bars_per_chord: int
    seed: int
    arrangement: Arrangement
    accompaniment: dict[str, tuple[Note, ...]]

    @property
    def n_bars(self) -> int:
        return len(self.progression) * self.bars_per_chord


Inputs = tuple[Scale, tuple[ChordSymbol, ...], int, int]


def validation_inputs() -> list[Inputs]:
    out: list[Inputs] = []
    for mode, keys, progressions in (
        ("major", MAJOR_KEYS, MAJOR_PROGRESSIONS),
        ("minor", MINOR_KEYS, MINOR_PROGRESSIONS),
    ):
        for key in keys:
            scale = Scale(key, cast(Literal["major", "minor"], mode))
            for numerals in progressions:
                progression = tuple(realize(scale, n) for n in numerals)
                for bars in (1, 2):
                    out.extend((scale, progression, bars, seed) for seed in VALIDATION_SEEDS)
    return out


def edge_inputs() -> list[Inputs]:
    out: list[Inputs] = []
    for key, mode, labels in EDGE_PROGRESSIONS:
        scale = Scale(key, cast(Literal["major", "minor"], mode))
        progression = tuple(ChordSymbol.parse(c) for c in labels)
        out.extend((scale, progression, 1, seed) for seed in EDGE_SEEDS)
    return out


def build_cases(inputs: Iterable[Inputs]) -> list[Case]:
    cases: list[Case] = []
    for scale, progression, bars, seed in inputs:
        arr = arrange(
            scale,
            progression,
            bars_per_chord=bars,
            tempo_bpm=TEMPO_BPM,
            style="fingerstyle",
            seed=seed,
        )
        acc: dict[str, tuple[Note, ...]] = {
            style: arr.accompaniment
            if style == "fingerstyle"
            else accompany(
                arr.voicings,
                arr.melody,
                bars_per_chord=bars,
                tempo_bpm=TEMPO_BPM,
                style=style,
                tuning=STANDARD_EADGBE,
            )
            for style in STYLES
        }
        cases.append(Case(scale, progression, bars, seed, arr, acc))
    return cases


Value = float | int | bool | None


def case_metrics(case: Case) -> dict[str, Value]:
    """Every metric for one case; accompaniment metrics are keyed ``A2.strum`` etc."""
    mel, prog = case.arrangement.melody, case.progression
    bars, tempo = case.bars_per_chord, TEMPO_BPM
    frets = [tn.position.fret for tn in case.arrangement.melody_tab]
    leaps, recovered = metrics.leap_recovery_counts(mel)
    lt_eligible, lt_resolved = metrics.leading_tone_resolution_counts(
        mel, case.scale, prog, bars, tempo
    )
    out: dict[str, Value] = {
        "M1": metrics.strong_beat_chord_tone_ratio(mel, prog, bars, tempo),
        "M2": metrics.step_ratio(mel),
        "M3": metrics.repeat_ratio(mel),
        "M4": metrics.large_leap_ratio(mel),
        "M4max": metrics.max_interval(mel),
        "M5.leaps": leaps,
        "M5.recovered": recovered,
        "M6": metrics.ambitus(mel),
        "M7": metrics.final_resolution(mel, case.scale, prog),
        "M8": metrics.final_note_beats(mel, tempo),
        "M9": metrics.distinct_durations(mel, tempo),
        "M10": metrics.ioi_entropy(mel, tempo),
        "M11a": metrics.motif_pair_coverage(mel, tempo, case.n_bars),
        "M11b": metrics.motif_contour_match(mel, tempo, case.n_bars),
        "M12": metrics.phrase_end_long_ratio(mel, tempo, case.n_bars),
        "M13": metrics.exposed_chromatic_clash_ratio(mel, case.scale, prog, bars, tempo),
        "M14": metrics.onset_grid_ratio([n.start_s for n in mel], tempo, 2),
        "M15.eligible": lt_eligible,
        "M15.resolved": lt_resolved,
        "T1": max(frets) if frets else None,
        "T2": metrics.hand_shift_rate(frets),
        "Tcount": len(frets) == len(mel),
        "V1": metrics.nearest_pitch_proximity(case.arrangement.voicings, STANDARD_EADGBE),
    }
    mismatches = [
        metrics.shape_tone_mismatch(v, STANDARD_EADGBE) for v in case.arrangement.voicings
    ]
    out["S1.missing"] = sum(bool(missing) for missing, _ in mismatches) / len(mismatches)
    out["S1.foreign"] = sum(bool(foreign) for _, foreign in mismatches) / len(mismatches)
    for style, acc in case.accompaniment.items():
        coverage = metrics.essential_tone_coverage(
            mel, acc, case.arrangement.voicings, bars, tempo, STANDARD_EADGBE
        )
        heads = metrics.stroke_heads([n.start_s for n in acc])
        out[f"A1.{style}"] = metrics.onset_grid_ratio(heads, tempo, 4)
        out[f"A2.{style}"] = metrics.melody_above_accompaniment_ratio(mel, acc)
        out[f"A3.{style}"] = metrics.exposed_semitone_clash_ratio(mel, acc, tempo)
        out[f"A4.{style}"] = metrics.final_chord_rings(acc, case.n_bars, tempo)
        out[f"H1.{style}"] = sum(coverage) / len(coverage)
        out[f"H2.{style}"] = metrics.cadence_identity(
            mel, acc, case.arrangement.voicings, STANDARD_EADGBE
        )
    return out


def _aggregate(rows: Sequence[dict[str, Value]]) -> dict[str, dict[str, float]]:
    """Mean/min/max of every metric over ``rows`` (``None`` values skipped),
    plus pooled ratios for the count-based M5 and M15."""
    out: dict[str, dict[str, float]] = {}
    for key in rows[0]:
        if key.startswith(("M5.", "M15.")):
            continue
        values = [float(r[key]) for r in rows if r[key] is not None]  # type: ignore[arg-type]
        if values:
            out[key] = {
                "mean": round(statistics.fmean(values), 4),
                "min": round(min(values), 4),
                "max": round(max(values), 4),
            }
    for name, total_key, hit_key in (
        ("M5", "M5.leaps", "M5.recovered"),
        ("M15", "M15.eligible", "M15.resolved"),
    ):
        total = sum(int(r[total_key] or 0) for r in rows)
        hits = sum(int(r[hit_key] or 0) for r in rows)
        out[name] = {"pooled": round(hits / total, 4) if total else -1.0, "n": float(total)}
    return out


def summarize(cases: Sequence[Case]) -> dict[str, dict[str, dict[str, float]]]:
    """Aggregates overall and per subgroup (mode, bars per chord)."""
    rows = [case_metrics(c) for c in cases]
    groups: dict[str, Callable[[Case], bool]] = {
        "overall": lambda c: True,
        "major": lambda c: c.scale.mode == "major",
        "minor": lambda c: c.scale.mode == "minor",
        "bars1": lambda c: c.bars_per_chord == 1,
        "bars2": lambda c: c.bars_per_chord == 2,
    }
    out: dict[str, dict[str, dict[str, float]]] = {}
    for name, keep in groups.items():
        selected = [r for c, r in zip(cases, rows, strict=True) if keep(c)]
        if selected:
            out[name] = _aggregate(selected)
    return out


def evaluate() -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    """Summaries for the validation and edge matrices."""
    return {
        "validation": summarize(build_cases(validation_inputs())),
        "edge": summarize(build_cases(edge_inputs())),
    }
