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


def test_tunings_and_holdout_matrices():
    inputs = ev.tunings_inputs()
    assert len(inputs) == 15 and {seed for *_, seed in inputs} == set(ev.VALIDATION_SEEDS)
    assert {bars for _, _, bars, _ in inputs} == {1}
    labels = {tuple(c.to_label() for c in prog) for _, prog, _, _ in inputs}
    assert ("G#", "C7", "Fm", "C#", "D#7", "G#") in labels
    holdout = ev.holdout_inputs()
    assert len(holdout) == 50 and {seed for *_, seed in holdout} == set(range(3000, 3005))
    used = {seed for *_, seed in ev.validation_inputs() + ev.edge_inputs()}
    assert not used & {seed for *_, seed in holdout}


def test_case_metrics_use_the_case_tuning_and_report_leaps():
    from music_decoder.tabs.tuning import EB_HALF_STEP_DOWN

    (case,) = ev.build_cases(ev.tunings_inputs()[5:6], EB_HALF_STEP_DOWN)
    assert case.tuning == EB_HALF_STEP_DOWN
    row = ev.case_metrics(case)
    assert {
        "L0.large",
        "L0.intervals",
        "L0.any",
        "L1",
        "L2.leaps",
        "L2.recovered",
        "L3",
        "L4",
        "L4max",
        "L5",
        "S2",
    } <= set(row)


def test_summarize_pools_l2_and_groups_by_tuning():
    from music_decoder.tabs.tuning import DROP_D

    cases = ev.build_cases(ev.tunings_inputs()[:2]) + ev.build_cases(
        ev.tunings_inputs()[:2], DROP_D
    )
    summary = ev.summarize(cases)
    assert set(summary["overall"]["L2"]) == {"pooled", "n"}
    assert {"tuning.EADGBE", "tuning.Drop_D"} <= set(summary)
    assert "tuning.EADGBE" not in ev.summarize(cases[:2])


def test_l0_pools_intervals_while_m4_averages_pieces():  # review r1 F6
    rows = [
        {"M4": 0.002, "L0.large": 2, "L0.intervals": 1000, "L0.any": True},
        {"M4": 0.0, "L0.large": 0, "L0.intervals": 10, "L0.any": False},
    ]
    out = ev._aggregate(rows)
    assert out["M4"]["mean"] == 0.001  # mean of per-piece rates
    assert out["L0"] == {"pooled": 0.002, "n": 1010.0}  # 2 / 1010, rounded to 4 places
    assert out["L0.any"]["mean"] == 0.5


def test_development_evaluation_never_builds_acceptance_inputs(monkeypatch):
    def forbidden():
        raise AssertionError("acceptance inputs built on the development path")

    monkeypatch.setattr(ev, "holdout_inputs", forbidden)
    monkeypatch.setattr(ev, "reserve_inputs", forbidden)
    report = ev.evaluate(("edge",))  # fast: 110 cases; must not touch acceptance inputs
    assert set(report) == {"edge"}
    assert ev.DEVELOPMENT_MATRICES == ("validation", "edge", "tunings")
    assert set(ev.ALL_MATRICES) == {"validation", "edge", "tunings", "holdout", "reserve"}
    with pytest.raises(ValueError):
        ev.evaluate(("nope",))


def test_reserve_matrix_is_frozen_and_disjoint():
    reserve = ev.reserve_inputs()
    assert len(reserve) == 50 and {seed for *_, seed in reserve} == set(range(4000, 4005))
    progressions = {tuple(c.to_label() for c in prog) for _, prog, _, _ in reserve}
    holdout = {tuple(c.to_label() for c in prog) for _, prog, _, _ in ev.holdout_inputs()}
    assert not progressions & holdout


def test_format_report_labels_every_row_with_its_matrix():
    text = ev.format_report({"edge": {"overall": {"M1": {"mean": 0.98, "min": 0.9, "max": 1.0}}}})
    assert text.splitlines() == ["edge  M1  mean=0.98 min=0.9 max=1"]


V2 = Path(__file__).parents[1] / "fixtures" / "compose" / "musicality_v2.json"


def test_committed_v2_snapshot_has_every_matrix_and_the_new_metrics():
    snap = json.loads(V2.read_text())
    assert {"source", "validation", "edge", "tunings", "holdout", "reserve"} <= set(snap)
    assert not snap["source"].endswith("-dirty")
    v = snap["validation"]["overall"]
    assert v["L1"]["max"] == 1 and v["L4max"]["max"] == 12  # v2's measured defects
    assert set(v["L0"]) == {"pooled", "n"}
    assert snap["tunings"]["overall"]["S1.missing"]["mean"] > 0
