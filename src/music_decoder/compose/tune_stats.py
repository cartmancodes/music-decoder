"""Statistics of real melodies, shipped with the package (composer v6).

``data/tune_stats.json`` is built from POP909 by ``scripts/build_tune_stats.py``
(see ``tune_corpus``). In the JSON a pattern is its slots joined by commas and
a move is a decimal string; :func:`tune_stats_from_dict` turns them back into
tuples and ints.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from importlib import resources
from typing import Any, Literal, cast, get_args

from music_decoder.compose.tune_metrics import Band, Mode

Pattern = tuple[int, ...]  # eighth slots with an onset, ascending, never empty
MetricClass = Literal["strong", "beat", "off"]  # slots 0,4 / 2,6 / odd
Role = Literal["open", "body", "cadence"]
START = 99  # "previous move" of a chain's first move

ROLES: tuple[Role, ...] = get_args(Role)
METRIC_CLASSES: tuple[MetricClass, ...] = get_args(MetricClass)
MODES: tuple[Mode, ...] = get_args(Mode)


def metric_class(slot: int) -> MetricClass:
    """Metric class of an eighth slot within its bar (slot is taken mod 8)."""
    slot %= 8
    if slot in (0, 4):
        return "strong"
    if slot in (2, 6):
        return "beat"
    return "off"


@dataclass(frozen=True)
class TuneStats:
    rhythm: Mapping[Role, Mapping[Pattern, int]]
    rhythm_pairs: Mapping[Role, Mapping[Pattern, Mapping[Pattern, int]]]  # previous -> this -> n
    steps: Mapping[MetricClass, Mapping[int, Mapping[int, int]]]  # previous move -> move -> n
    steps_base: Mapping[MetricClass, Mapping[int, int]]
    chord_tone: Mapping[MetricClass, float]
    degrees: Mapping[Mode, tuple[float, ...]]  # 12 shares, by semitone above the tonic
    final_degrees: Mapping[Mode, tuple[float, ...]]  # same, for phrase-final notes
    notes_per_second: Band
    bands: Mapping[Mode, Mapping[str, Band]]
    meta: Mapping[str, Any]


def pattern_key(pattern: Pattern) -> str:
    return ",".join(str(s) for s in pattern)


def parse_pattern(key: str) -> Pattern:
    return tuple(int(s) for s in key.split(","))


def _band(raw: Mapping[str, float]) -> Band:
    return Band(low=float(raw["low"]), mid=float(raw["mid"]), high=float(raw["high"]))


def _counts(raw: Mapping[str, int]) -> dict[int, int]:
    return {int(k): int(v) for k, v in raw.items()}


def tune_stats_from_dict(raw: Mapping[str, Any]) -> TuneStats:
    """Typed view of the dict ``tune_corpus.build_stats`` produces."""
    rhythm: dict[Role, dict[Pattern, int]] = {
        role: {parse_pattern(k): int(v) for k, v in raw["rhythm"][role].items()} for role in ROLES
    }
    rhythm_pairs: dict[Role, dict[Pattern, dict[Pattern, int]]] = {
        role: {
            parse_pattern(prev): {parse_pattern(k): int(v) for k, v in row.items()}
            for prev, row in raw["rhythm_pairs"][role].items()
        }
        for role in ROLES
    }
    steps: dict[MetricClass, dict[int, dict[int, int]]] = {
        cls: {int(prev): _counts(row) for prev, row in raw["steps"][cls].items()}
        for cls in METRIC_CLASSES
    }
    return TuneStats(
        rhythm=rhythm,
        rhythm_pairs=rhythm_pairs,
        steps=steps,
        steps_base={cls: _counts(raw["steps_base"][cls]) for cls in METRIC_CLASSES},
        chord_tone={cls: float(raw["chord_tone"][cls]) for cls in METRIC_CLASSES},
        degrees={m: tuple(float(x) for x in raw["degrees"][m]) for m in MODES},
        final_degrees={m: tuple(float(x) for x in raw["final_degrees"][m]) for m in MODES},
        notes_per_second=_band(raw["notes_per_second"]),
        bands={m: {f: _band(b) for f, b in raw["bands"][m].items()} for m in MODES},
        meta=dict(raw["meta"]),
    )


@cache
def load_tune_stats() -> TuneStats:
    """The shipped statistics, read once from ``compose/data/tune_stats.json``."""
    path = resources.files("music_decoder.compose") / "data" / "tune_stats.json"
    raw = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    return tune_stats_from_dict(raw)
