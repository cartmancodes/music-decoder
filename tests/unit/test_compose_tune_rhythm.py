import numpy as np
import pytest

from music_decoder.compose.tune import bar_roles, plan_rhythm
from music_decoder.compose.tune_stats import load_tune_stats

STATS = load_tune_stats()


def plan(n, tempo=96.0, seed=1):
    return plan_rhythm(n, tempo, np.random.default_rng(seed), STATS)


@pytest.mark.parametrize(
    "n,roles",
    [
        (1, ["cadence"]),
        (2, ["open", "cadence"]),
        (4, ["open", "body", "body", "cadence"]),
        (5, ["open", "body", "cadence", "open", "cadence"]),
        (8, ["open", "body", "body", "cadence"] * 2),
    ],
)
def test_bar_roles(n, roles):
    assert bar_roles(n) == roles


def test_plan_shape_and_determinism():
    a, b = plan(8), plan(8)
    assert a == b and len(a) == 8 and a[-1].pattern == (0,) and 0 in a[0].pattern
    assert all(bar.pattern in STATS.rhythm[bar.role] for bar in a[:-1])
    assert [bar.role for bar in a] == bar_roles(8)


@pytest.mark.parametrize("n", [1, 2, 3, 5, 9, 12, 16])
def test_every_length_plans(n):
    assert len(plan(n)) == n


def test_slow_tempos_are_denser():
    def mean(tempo):
        plans = [plan(8, tempo, s) for s in range(60)]
        return np.mean([len(bar.pattern) for p in plans for bar in p[:-1]])

    assert mean(66.0) > mean(120.0) + 0.5


def test_restatement_only_in_a_prime_and_about_half_the_time():
    plans = [plan(8, seed=s) for s in range(200)]
    assert all(bar.restated_from is None for p in plans for bar in p[:4])
    share = np.mean([p[k].restated_from == k - 4 for p in plans for k in (4, 5, 6)])
    assert 0.4 <= share <= 0.6
    assert all(
        p[k].pattern == p[k - 4].pattern
        for p in plans
        for k in (4, 5, 6)
        if p[k].restated_from is not None
    )


def test_rhythms_vary_like_real_tunes():
    variety = [len({bar.pattern for bar in plan(8, seed=s)}) for s in range(100)]
    assert np.median(variety) >= 4


def test_without_final_whole_note_the_last_bar_is_a_cadence_pattern():
    p = plan_rhythm(8, 96.0, np.random.default_rng(3), STATS, final_whole_note=False)
    assert p[-1].pattern in STATS.rhythm["cadence"]
