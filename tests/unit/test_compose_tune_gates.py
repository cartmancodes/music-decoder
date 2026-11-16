"""Gates of the tune engine against real-melody statistics (composer v6 spec 5.5).

T1 and T2 measure candidate 0 of each dev piece (before any selection), T3
the pieces as ``compose()`` writes them (best of 12); T4 checks determinism
and T5 that every key, mode, edge input and tuning composes.

``KNOWN_MISSES`` holds the controller's rulings on bands that cannot be met
together with a guitar guarantee; only the controller adds to it.
``PENDING_TUNING`` holds the dev-matrix misses left after the tuning of Task 7
part B, awaiting a ruling, each value naming the gate it misses; such a gate
is an expected failure, and passes normally once met.
"""

import pytest

from music_decoder.compose.api import arrange
from music_decoder.compose.tune_metrics import FEATURES
from music_decoder.compose.tune_stats import load_tune_stats
from music_decoder.evaluation import compose_musicality as cm
from music_decoder.evaluation.compose_tune import (
    build_tune_case,
    describe,
    guarantee_failures,
    triad_identity,
    tune_cases,
    tune_inputs,
    tune_report,
)
from music_decoder.tabs.tuning import PRESETS, STANDARD_EADGBE
from music_decoder.types import VALID_TONICS, Scale

STATS = load_tune_stats()
SPREAD_FEATURES = ("repeat", "notes_per_second", "rhythm_variety", "figure_recurrence")
KNOWN_MISSES: dict[tuple[str, str], str] = {}
PENDING_REASON = "pending tuning (Task 7 part B)"
# Dev-matrix misses awaiting a ruling (none since round 2 of Task 7 part B); the value
# starts with the gate it misses.
PENDING_TUNING: dict[tuple[str, str], str] = {}
MIN_TRIAD_IDENTITY = 0.99


def _check(ok: bool, gate: str, mode: str, feature: str, detail: str) -> None:
    pending = PENDING_TUNING.get((mode, feature), "")
    if not ok and gate in pending.split(":")[0].split("+"):
        pytest.xfail(PENDING_REASON)
    assert ok, detail


@pytest.fixture(scope="module")
def dev_cases():
    return tune_cases("dev")


@pytest.fixture(scope="module")
def dev_report(dev_cases):
    return tune_report(dev_cases)


@pytest.mark.parametrize("mode", ["major", "minor"])
@pytest.mark.parametrize("feature", FEATURES)
def test_t1_medians_sit_in_the_real_melody_bands(dev_report, mode, feature):
    if (mode, feature) in KNOWN_MISSES:
        pytest.xfail(KNOWN_MISSES[(mode, feature)])
    band = STATS.bands[mode][feature]
    median = dev_report[mode][feature]["median"]
    _check(
        band.low <= median <= band.high,
        "T1",
        mode,
        feature,
        f"{mode} {feature}: median {median:.3f} outside {band.low:.3f}-{band.high:.3f}",
    )


@pytest.mark.parametrize("mode", ["major", "minor"])
@pytest.mark.parametrize("feature", SPREAD_FEATURES)
def test_t2_pieces_are_not_all_alike(dev_report, mode, feature):
    if (mode, feature) in KNOWN_MISSES:
        pytest.xfail(KNOWN_MISSES[(mode, feature)])
    band, got = STATS.bands[mode][feature], dev_report[mode][feature]
    spread, need = got["q75"] - got["q25"], 0.5 * (band.high - band.low)
    _check(
        spread >= need,
        "T2",
        mode,
        feature,
        f"{mode} {feature}: spread {spread:.3f} below {need:.3f}",
    )


def test_t3_guitar_guarantees(dev_cases):
    failures = [(describe(c), f) for c in dev_cases if (f := guarantee_failures(c))]
    assert not failures, failures
    triads = [met for c in dev_cases for met in triad_identity(c)]
    assert sum(triads) / len(triads) >= MIN_TRIAD_IDENTITY


def test_known_misses_stay_few():
    assert all(sum(m == mode for m, _ in KNOWN_MISSES) <= 2 for mode in ("major", "minor"))


def test_pending_tuning_names_gates_and_features():
    for (mode, feature), note in PENDING_TUNING.items():
        assert mode in ("major", "minor") and feature in FEATURES
        assert set(note.split(":")[0].split("+")) <= {"T1", "T2"}


def test_t4_same_seed_same_music(dev_cases):
    inputs = tune_inputs("dev")[:3]
    a, b = [build_tune_case(i) for i in inputs], [build_tune_case(i) for i in inputs]
    assert a == b == dev_cases[:3]


def test_t5_every_key_and_mode_composes():
    for tonic in VALID_TONICS:
        for mode, numerals in (("major", "I V vi IV"), ("minor", "i VI III VII")):
            scale = Scale(tonic, mode)
            progression = [cm.realize(scale, n) for n in numerals.split()]
            arrangement = arrange(
                scale,
                progression,
                bars_per_chord=2,
                seed=1,
                candidates=2,
                melody_engine="tune",
            )
            assert arrangement.melody, f"{tonic}:{mode}"


def test_t5_edge_and_tunings_inputs_compose():
    runs = [(inputs, STANDARD_EADGBE) for inputs in cm.edge_inputs()]
    runs += [(inputs, tuning) for inputs in cm.tunings_inputs() for tuning in PRESETS.values()]
    for (scale, progression, bars, seed), tuning in runs:
        arrangement = arrange(
            scale,
            progression,
            bars_per_chord=bars,
            tempo_bpm=cm.TEMPO_BPM,
            tuning=tuning,
            seed=seed,
            candidates=2,
            melody_engine="tune",
        )
        assert arrangement.melody, (scale, progression, seed, tuning.name)
