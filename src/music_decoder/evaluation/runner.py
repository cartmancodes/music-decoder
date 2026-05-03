from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .fixtures.base import Fixture
from .metrics import (
    key_mirex_score,
    note_f_measure,
    onset_f_measure,
    pitch_class_accuracy,
    tab_string_accuracy,
)


@dataclass(frozen=True)
class FixtureMetrics:
    name: str
    source: str
    note_f_measure: float | None
    onset_f_measure: float | None
    pitch_class_accuracy: float | None
    key_mirex_score: float | None
    tab_string_accuracy: float | None


@dataclass(frozen=True)
class EvaluationReport:
    timestamp: str
    per_fixture: list[FixtureMetrics]


PipelineFn = Callable[[Path, Fixture], dict[str, object]]


def _safe(fn, *args, **kw) -> float | None:  # type: ignore[no-untyped-def]
    try:
        result: float = fn(*args, **kw)
        return result
    except Exception:
        return None


def run_evaluation(
    pipeline: PipelineFn,
    fixtures: Iterable[Fixture],
    *,
    report_path: Path | None = None,
) -> EvaluationReport:
    rows: list[FixtureMetrics] = []
    for fx in fixtures:
        prediction = pipeline(fx.audio_path, fx)
        gt = fx.ground_truth
        f_note = _safe(
            note_f_measure,
            np.asarray(prediction["intervals"]),
            np.asarray(prediction["pitches_midi"]),
            gt.intervals,
            gt.pitches_midi,
        )
        f_onset = _safe(
            onset_f_measure,
            np.asarray(prediction["intervals"]),
            gt.intervals,
        )
        pc = _safe(
            pitch_class_accuracy,
            np.asarray(prediction["intervals"]),
            np.asarray(prediction["pitches_midi"]),
            gt.intervals,
            gt.pitches_midi,
        )
        pred_key = prediction.get("key")
        k = (
            key_mirex_score(pred_key, gt.key)  # type: ignore[arg-type]
            if pred_key and gt.key
            else None
        )
        pred_tab = prediction.get("tab")
        tab = (
            tab_string_accuracy(pred_tab, gt.tab)  # type: ignore[arg-type]
            if pred_tab and gt.tab
            else None
        )
        rows.append(
            FixtureMetrics(
                name=fx.name,
                source=fx.source,
                note_f_measure=f_note,
                onset_f_measure=f_onset,
                pitch_class_accuracy=pc,
                key_mirex_score=k,
                tab_string_accuracy=tab,
            )
        )
    report = EvaluationReport(
        timestamp=datetime.now(UTC).isoformat(),
        per_fixture=rows,
    )
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(
                {
                    "timestamp": report.timestamp,
                    "per_fixture": [asdict(r) for r in rows],
                },
                indent=2,
            )
        )
    return report
