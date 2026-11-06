import json
from pathlib import Path

import pytest

from music_decoder.evaluation import compose_musicality as ev
from music_decoder.types import ChordSymbol, Scale

BASELINE = Path(__file__).parents[1] / "fixtures" / "compose" / "musicality_baseline.json"


@pytest.mark.parametrize(
    ("key", "mode", "numeral", "label"),
    [
        ("C", "major", "I", "C"),
        ("C", "major", "vi7", "Am7"),
        ("C", "major", "V7", "G7"),
        ("G", "major", "Imaj7", "Gmaj7"),
        ("A", "minor", "V7", "E7"),
        ("A", "minor", "VI", "F"),
        ("A", "minor", "iidim", "Bdim"),
        ("F#", "minor", "v", "C#m"),
    ],
)
def test_realize(key, mode, numeral, label):
    assert ev.realize(Scale(key, mode), numeral) == ChordSymbol.parse(label)


def test_matrices_have_the_documented_size_and_held_out_seeds():
    validation = ev.validation_inputs()
    assert len(validation) == 400
    assert len(ev.edge_inputs()) == 110
    assert min(seed for *_, seed in validation) >= 1000


def test_case_metrics_cover_every_style():
    (case,) = ev.build_cases(ev.edge_inputs()[:1])
    row = ev.case_metrics(case)
    for style in ev.STYLES:
        for prefix in ("A1", "A2", "A3", "A4", "H1", "H2"):
            assert f"{prefix}.{style}" in row
    assert {"M1", "M7", "M11a", "M12", "M13", "T1", "T2", "V1", "S1.missing"} <= set(row)


def test_summarize_groups_and_pools_counts():
    summary = ev.summarize(ev.build_cases(ev.validation_inputs()[:10]))
    assert {"overall", "major", "bars1"} <= set(summary)
    assert set(summary["overall"]["M5"]) == {"pooled", "n"}
    assert set(summary["overall"]["M2"]) == {"mean", "min", "max"}


def test_committed_baseline_has_both_matrices():
    baseline = json.loads(BASELINE.read_text())
    assert {"source", "validation", "edge"} <= set(baseline)
    assert baseline["validation"]["overall"]["M9"]["mean"] == 1.0  # uniform quarters
