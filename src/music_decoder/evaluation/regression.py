from __future__ import annotations

import json
import statistics
from pathlib import Path

import yaml

from .runner import EvaluationReport


def load_thresholds(path: Path) -> dict[str, float]:
    return yaml.safe_load(path.read_text())  # type: ignore[no-any-return]


def load_baseline(path: Path) -> dict[str, float]:
    raw: dict[str, object] = json.loads(path.read_text())
    return {
        k: float(v)
        for k, v in raw.items()
        if not k.startswith("_") and isinstance(v, (int, float))
    }


def _aggregate(report: EvaluationReport) -> dict[str, float]:
    aggregates: dict[str, float] = {}
    for metric in (
        "note_f_measure",
        "onset_f_measure",
        "pitch_class_accuracy",
        "key_mirex_score",
        "tab_string_accuracy",
    ):
        values = [
            getattr(r, metric)
            for r in report.per_fixture
            if getattr(r, metric) is not None
        ]
        if values:
            aggregates[metric] = statistics.mean(values)
    return aggregates


def check_against_thresholds(
    report: EvaluationReport, thresholds: dict[str, float]
) -> list[str]:
    failures: list[str] = []
    aggregates = _aggregate(report)
    for metric, value in aggregates.items():
        threshold = thresholds.get(metric)
        if threshold is None:
            continue
        if value < threshold:
            failures.append(f"{metric}={value:.4f} below threshold {threshold:.4f}")
    return failures


def check_against_baseline(
    report: EvaluationReport,
    baseline: dict[str, float],
    tolerance: float = 0.02,
) -> list[str]:
    failures: list[str] = []
    aggregates = _aggregate(report)
    for metric, current in aggregates.items():
        prev = baseline.get(metric)
        if prev is None:
            continue
        if current + tolerance < prev:
            failures.append(
                f"{metric}: regressed from {prev:.4f} to {current:.4f} "
                f"(beyond tolerance {tolerance})"
            )
    return failures
