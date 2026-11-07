"""Musicality regression test: the composer against its pre-change baseline.

``music_decoder.evaluation.compose_musicality.evaluate()`` generates the
held-out validation matrix (8 keys x 5 progressions per mode x 1-2 bars per
chord x seeds 1000-1004 = 400 melodies, each accompanied in 3 styles) and an
edge matrix (odd lengths, a single chord, secondary dominants, a borrowed
chord). ``tests/fixtures/compose/musicality_baseline.json`` holds the same
summaries for the pre-change composer, written by
``scripts/compose_musicality_report.py``. Targets and their reasons:
docs/superpowers/specs/2026-11-06-compose-improvements-design.md sections 3-4.

The v3 composer (singable leaps, faithful four-finger voicings in every
tuning) is checked against ``tests/fixtures/compose/musicality_v2.json`` on
the development matrices (validation, edge, tunings) and, as the acceptance
check, on the holdout matrix: see
docs/superpowers/specs/2026-11-07-compose-voicings-leaps-design.md section 6.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from music_decoder.compose.api import arrange
from music_decoder.evaluation.compose_musicality import ACCEPTANCE_MATRICES, STYLES, evaluate
from music_decoder.tabs.tuning import PRESETS
from music_decoder.types import ChordSymbol, Scale

BASELINE_PATH = Path(__file__).parents[1] / "fixtures" / "compose" / "musicality_baseline.json"
V2_PATH = Path(__file__).parents[1] / "fixtures" / "compose" / "musicality_v2.json"
GROUPS = ("overall", "major", "minor", "bars1", "bars2")

# Higher is better / lower is better; every subgroup must beat the baseline.
IMPROVE_UP = (
    "M1",
    "M2",
    "M7",
    "M8",
    "M9",
    "M10",
    "M11b",
    "M12",
    "A1.arpeggio",
    *(f"A2.{s}" for s in STYLES),
    *(f"A4.{s}" for s in STYLES),
)
IMPROVE_DOWN = ("M3", "M13", "T2", *(f"A3.{s}" for s in STYLES))


@pytest.fixture(scope="module")
def report():
    return evaluate()


@pytest.fixture(scope="module")
def acceptance_report():
    return evaluate(ACCEPTANCE_MATRICES)


@pytest.fixture(scope="module")
def all_reports(report, acceptance_report):
    return {**report, **acceptance_report}


@pytest.fixture(scope="module")
def baseline():
    return json.loads(BASELINE_PATH.read_text())


@pytest.fixture(scope="module")
def v2():
    return json.loads(V2_PATH.read_text())


def _mean(report, metric, matrix="validation", group="overall"):
    return report[matrix][group][metric]["mean"]


# ---- absolute targets (validation matrix) -----------------------------------


@pytest.mark.parametrize(
    ("metric", "low", "high"),
    [
        ("M1", 0.95, None),  # strong-beat chord tones; baseline 0.863
        ("M2", 0.50, None),  # stepwise motion; baseline 0.352
        ("M3", None, 0.15),  # repeated notes; baseline 0.279
        ("M4", None, 0.02),  # leaps > a fifth; baseline 0.0005 (allowed rise, spec §3)
        ("M9", 3.0, None),  # distinct durations; baseline 1.0
        ("M10", 1.0, None),  # IOI entropy (bits); baseline 0.0
        ("M11a", 0.95, None),  # echo pairs that can carry the motif; baseline 1.0
        ("M11b", 0.60, None),  # motif contour agreement; baseline 0.336
        ("M12", 0.95, None),  # long phrase-final notes; baseline 0.0
        ("M13", None, 0.04),  # exposed chromatic clashes; baseline 0.094
        ("T2", None, 0.14),  # hand shifts per note; baseline 0.171
        *((f"A2.{s}", 0.97, None) for s in STYLES),  # melody on top; baseline 0.82-0.91
        *((f"A3.{s}", None, 0.10) for s in STYLES),  # exposed semitone clashes; 0.11-0.145
    ],
)
def test_validation_targets(report, metric, low, high):
    value = _mean(report, metric)
    if low is not None:
        assert value >= low, f"{metric} = {value}"
    if high is not None:
        assert value <= high, f"{metric} = {value}"


def test_validation_floors(report):
    o = report["validation"]["overall"]
    assert o["M1"]["min"] >= 0.60  # no piece ignores the harmony; baseline min 0.5
    assert o["M9"]["min"] >= 2  # no piece is rhythmically uniform; baseline 1


def test_pooled_leap_recovery_and_leading_tone_resolution(report):
    overall = report["validation"]["overall"]
    assert overall["M5"]["n"] >= 100 and overall["M5"]["pooled"] >= 0.70  # baseline 0.0 (n=5)
    assert overall["M15"]["n"] >= 20 and overall["M15"]["pooled"] >= 0.75  # baseline 0.385


@pytest.mark.parametrize("matrix", ["validation", "edge"])
def test_hard_invariants(report, matrix):
    o = report[matrix]["overall"]
    assert o["M4max"]["max"] <= 12  # no leap over an octave; baseline max 23
    assert o["M6"]["max"] <= 17  # ambitus; baseline max 24
    assert o["M7"]["min"] == 1.0  # every piece resolves; baseline 0.24
    assert o["M8"]["min"] >= 1.5  # final note >= 1.5 beats; baseline 0.95
    assert o["M12"]["min"] == 1.0  # every phrase ends long
    assert o["M14"]["min"] == 1.0  # melody on the eighth grid
    assert o["T1"]["max"] <= 15  # melody tab frets; baseline max 20
    assert o["Tcount"]["min"] == 1.0  # the tab has every melody note
    for style in STYLES:
        assert o[f"A1.{style}"]["min"] == 1.0  # baseline arpeggio 0.10
        assert o[f"A4.{style}"]["min"] == 1.0  # final chord rings
        assert o[f"H1.{style}"]["min"] == 1.0  # chord identity heard in every bar
        assert o[f"H2.{style}"]["min"] == 1.0  # full harmony under the held final note


# ---- paired comparison with the baseline ----------------------------------


@pytest.mark.parametrize("group", GROUPS)
def test_every_subgroup_improves_on_the_baseline(report, baseline, group):
    cand, base = report["validation"][group], baseline["validation"][group]
    for metric in (*IMPROVE_UP, *IMPROVE_DOWN):
        if metric not in cand or metric not in base:
            continue  # e.g. M13 is undefined over all-diatonic major progressions
        c, b = cand[metric]["mean"], base[metric]["mean"]
        assert (c > b) if metric in IMPROVE_UP else (c < b), f"{group} {metric}: {b} -> {c}"


def test_edge_matrix_improves_on_the_baseline(report, baseline):
    cand, base = report["edge"]["overall"], baseline["edge"]["overall"]
    for metric in (*IMPROVE_UP, *IMPROVE_DOWN):
        c, b = cand[metric]["mean"], base[metric]["mean"]
        assert (c > b) if metric in IMPROVE_UP else (c < b), f"edge {metric}: {b} -> {c}"


def test_relative_clash_reduction(report, baseline):
    for metric in ("M13", *(f"A3.{s}" for s in STYLES)):
        c, b = _mean(report, metric), _mean(baseline, metric)
        assert c <= 0.7 * b, f"{metric}: {b} -> {c}"


def test_guards_do_not_regress(report, baseline, v2):
    for matrix in ("validation", "edge"):
        cand, base = report[matrix]["overall"], baseline[matrix]["overall"]
        # The library shapes are repaired (spec section 6.2), so V1 (semitones per
        # voice) moves; it may rise by at most 0.1 over the v2 composer.
        assert cand["V1"]["mean"] <= v2[matrix]["overall"]["V1"]["mean"] + 0.1
        assert cand["M11a"]["mean"] >= base["M11a"]["mean"] - 0.05
        # every shape is faithful: nothing missing, nothing foreign, anywhere
        assert cand["S1.missing"]["max"] == 0 and cand["S1.foreign"]["max"] == 0
        for style in STYLES:  # cadence identity never below the old composer's
            assert cand[f"H2.{style}"]["mean"] >= base[f"H2.{style}"]["mean"]


# ---- v3 targets: leaps, shapes, tunings, holdout acceptance (spec 2026-11-07 section 6) ----

MATRICES = ("validation", "edge", "tunings", "holdout")
TOL_MEAN, TOL_POOLED = 0.02, 0.05  # spec §6.3: under half a note per piece; about one pooled event


def test_leap_regression_locks_on_development_matrices(report, v2):
    # Development data (spec §6): these are regression locks, not generalisation evidence.
    for matrix, l4_lock in (("validation", 0.87), ("edge", 0.83)):
        o, ref = report[matrix]["overall"], v2[matrix]["overall"]
        assert o["L0"]["pooled"] <= 0.001 and o["L0"]["pooled"] < ref["L0"]["pooled"]
        # M4 is a mean of per-piece rates, not a frequency (see L0)
        assert o["M4"]["mean"] <= 0.001 and o["M4"]["mean"] < ref["M4"]["mean"]
        assert o["L4"]["mean"] >= l4_lock


@pytest.mark.parametrize("matrix", MATRICES)
def test_chord_changes_are_smoother_than_v2(all_reports, v2, matrix):
    assert all_reports[matrix]["overall"]["L4"]["mean"] >= v2[matrix]["overall"]["L4"]["mean"]
    # strict on development data (measured); spec §6.4 asks >= on the holdout
    if matrix != "holdout":
        assert all_reports[matrix]["overall"]["L4"]["mean"] > v2[matrix]["overall"]["L4"]["mean"]


def test_holdout_acceptance(all_reports, v2):  # spec §6.4, frozen before evaluation
    o, ref = all_reports["holdout"]["overall"], v2["holdout"]["overall"]
    assert o["L0"]["pooled"] <= min(0.002, ref["L0"]["pooled"]) or o["L0"]["pooled"] == 0.0
    assert o["M4max"]["max"] <= 12 and o["M6"]["max"] <= 17 and o["T1"]["max"] <= 15
    assert o["M7"]["min"] == 1.0 and o["M8"]["min"] >= 1.5
    assert o["M12"]["min"] == 1.0 and o["M14"]["min"] == 1.0 and o["Tcount"]["min"] == 1.0
    for style in STYLES:
        for key in ("A1", "A4", "H1", "H2"):
            assert o[f"{key}.{style}"]["min"] == 1.0, f"{key}.{style}"


@pytest.mark.parametrize("matrix", MATRICES)
def test_leap_and_shape_invariants(all_reports, matrix):
    o = all_reports[matrix]["overall"]
    assert o["L1"]["max"] == 0  # no tritones, sevenths or leaps over an octave
    assert o["L3"]["max"] == 0  # every leap > a fifth is arpeggiation inside one chord
    assert o["L2"]["n"] == 0 or o["L2"]["pooled"] == 1.0  # ...and is recovered by a step
    assert o["L4max"]["max"] <= 7 and o["L5"]["max"] <= 7
    assert o["S1.missing"]["max"] == 0 and o["S1.foreign"]["max"] == 0 and o["S2"]["max"] == 0


def test_tunings_matrix_keeps_every_v2_hard_invariant(report):
    o = report["tunings"]["overall"]
    assert o["M4max"]["max"] <= 12 and o["M6"]["max"] <= 17
    assert o["M7"]["min"] == 1.0 and o["M8"]["min"] >= 1.5
    assert o["M12"]["min"] == 1.0 and o["M14"]["min"] == 1.0 and o["Tcount"]["min"] == 1.0
    for style in STYLES:
        for key in ("A1", "A4", "H1", "H2"):
            assert o[f"{key}.{style}"]["min"] == 1.0, f"{key}.{style}"
    for name, tuning in PRESETS.items():  # spec D15
        cap = 15 + (64 - tuning.open_pitches[-1])
        assert report["tunings"][f"tuning.{name}"]["T1"]["max"] <= cap, name


@pytest.mark.parametrize(
    ("matrix", "group"),
    [*(("validation", g) for g in GROUPS), ("edge", "overall"), ("holdout", "overall")],
)
def test_no_regression_against_v2(all_reports, v2, matrix, group):
    cand, ref = all_reports[matrix][group], v2[matrix][group]
    for metric in (*IMPROVE_UP, *IMPROVE_DOWN):
        if metric not in ref:  # e.g. M13 is undefined on validation/major in both
            continue
        assert metric in cand, f"{matrix}/{group} {metric}: missing from the candidate"
        c, r = cand[metric]["mean"], ref[metric]["mean"]
        if metric in IMPROVE_UP:
            assert c >= r - TOL_MEAN, f"{matrix}/{group} {metric}: {r} -> {c}"
        else:
            assert c <= r + TOL_MEAN, f"{matrix}/{group} {metric}: {r} -> {c}"
    for metric in ("M5", "M15"):
        if cand[metric]["n"] and ref[metric]["n"]:
            assert cand[metric]["pooled"] >= ref[metric]["pooled"] - TOL_POOLED, metric


def test_same_seed_same_music():
    progression = [ChordSymbol.parse(c) for c in ("Am", "F", "C", "G")]
    a = arrange(Scale("A", "minor"), progression, bars_per_chord=2, seed=1003)
    b = arrange(Scale("A", "minor"), progression, bars_per_chord=2, seed=1003)
    assert a == b
