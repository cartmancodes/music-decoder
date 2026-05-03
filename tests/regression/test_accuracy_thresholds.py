"""
Regression test: the placeholder pipeline returns identity predictions, so
every metric is 1.0 — well above thresholds. Once Phase 4 lands, this test
swaps in the real pipeline and gates accuracy regressions.
"""
from pathlib import Path

import numpy as np
import pytest

from music_decoder.evaluation.fixtures.base import Fixture, GroundTruth
from music_decoder.evaluation.regression import (
    check_against_baseline,
    check_against_thresholds,
    load_baseline,
    load_thresholds,
)
from music_decoder.evaluation.runner import run_evaluation


def _identity_pipeline(audio_path, fixture):
    gt = fixture.ground_truth
    return {
        "intervals": gt.intervals.copy(),
        "pitches_midi": gt.pitches_midi.copy(),
        "key": gt.key,
        "tab": list(gt.tab) if gt.tab is not None else None,
    }


@pytest.fixture
def trivial_fixture(tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")
    return Fixture(
        name="trivial", source="manual", audio_path=audio,
        ground_truth=GroundTruth(
            intervals=np.array([[0.0, 0.5]]), pitches_midi=np.array([60]),
            key=("C", "major"), tempo_bpm=120.0, tab=[(60, 4, 1)],
        ),
    )


@pytest.mark.regression
def test_thresholds_loadable():
    thresholds = load_thresholds(Path("config/eval_thresholds.yaml"))
    assert "note_f_measure" in thresholds
    assert "regression_tolerance" in thresholds


@pytest.mark.regression
def test_identity_pipeline_passes_thresholds(trivial_fixture):
    report = run_evaluation(_identity_pipeline, [trivial_fixture])
    thresholds = load_thresholds(Path("config/eval_thresholds.yaml"))
    failures = check_against_thresholds(report, thresholds)
    assert failures == []


@pytest.mark.regression
def test_baseline_comparison_no_regression(trivial_fixture):
    report = run_evaluation(_identity_pipeline, [trivial_fixture])
    baseline = load_baseline(Path("evaluation_reports/baseline.json"))
    failures = check_against_baseline(report, baseline, tolerance=0.02)
    assert failures == []
