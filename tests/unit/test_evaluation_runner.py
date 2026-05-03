from pathlib import Path

import numpy as np

from music_decoder.evaluation.fixtures.base import Fixture, GroundTruth
from music_decoder.evaluation.runner import EvaluationReport, run_evaluation


def _identity_pipeline(audio_path: Path, fixture: Fixture):
    """Echoes the ground truth back as a perfect prediction."""
    gt = fixture.ground_truth
    return {
        "intervals": gt.intervals.copy(),
        "pitches_midi": gt.pitches_midi.copy(),
        "key": gt.key,
        "tab": list(gt.tab) if gt.tab is not None else None,
    }


def _make_fixture(tmp_path: Path) -> Fixture:
    audio = tmp_path / "x.wav"
    audio.write_bytes(b"fake")
    return Fixture(
        name="x", source="manual", audio_path=audio,
        ground_truth=GroundTruth(
            intervals=np.array([[0.0, 0.5], [0.5, 1.0]]),
            pitches_midi=np.array([60, 62]),
            key=("C", "major"),
            tempo_bpm=120.0,
            tab=[(60, 4, 1), (62, 4, 3)],
        ),
    )


def test_runner_returns_perfect_report_for_identity_pipeline(tmp_path):
    fx = _make_fixture(tmp_path)
    report = run_evaluation(_identity_pipeline, [fx])
    assert isinstance(report, EvaluationReport)
    assert report.per_fixture[0].note_f_measure == 1.0
    assert report.per_fixture[0].key_mirex_score == 1.0
    assert report.per_fixture[0].tab_string_accuracy == 1.0


def test_runner_writes_json_report(tmp_path):
    fx = _make_fixture(tmp_path)
    out = tmp_path / "report.json"
    run_evaluation(_identity_pipeline, [fx], report_path=out)
    assert out.exists()
    import json
    parsed = json.loads(out.read_text())
    assert parsed["per_fixture"][0]["note_f_measure"] == 1.0
