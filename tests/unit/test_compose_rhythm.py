import numpy as np
import pytest

from music_decoder.compose.rhythm import (
    BODY_RHYTHMS,
    CADENCE_RHYTHMS,
    BarPlan,
    phrase_layout,
    plan_bars,
    split_phrases,
)


def test_templates_fill_one_bar():
    for rhythm in BODY_RHYTHMS + CADENCE_RHYTHMS:
        assert sum(rhythm) == 8, rhythm
        assert all(v >= 1 for v in rhythm)


def test_cadence_templates_end_long():
    assert all(r[-1] >= 4 for r in CADENCE_RHYTHMS)


@pytest.mark.parametrize(
    ("n_bars", "expected"),
    [
        (1, [1]),
        (2, [2]),
        (3, [3]),
        (4, [4]),
        (5, [3, 2]),
        (6, [4, 2]),
        (7, [4, 3]),
        (8, [4, 4]),
        (9, [4, 3, 2]),
        (12, [4, 4, 4]),
    ],
)
def test_split_phrases(n_bars, expected):
    assert split_phrases(n_bars) == expected
    assert sum(split_phrases(n_bars)) == n_bars


def test_split_phrases_rejects_zero():
    with pytest.raises(ValueError):
        split_phrases(0)


@pytest.mark.parametrize(
    ("n_bars", "cadences", "echoes"),
    [
        (1, (0,), ()),
        (3, (2,), ()),
        (4, (3,), ((0, 2),)),
        (5, (2, 4), ((0, 3),)),  # 3 + 2: the consequent echoes antecedent bar 1
        (7, (3, 6), ((0, 2), (0, 4), (1, 5))),  # 4 + 3
        (8, (3, 7), ((0, 2), (0, 4), (1, 5), (2, 6))),
        (9, (3, 6, 8), ((0, 2), (0, 4), (1, 5))),  # 4 + 3 + 2 (third phrase is fresh)
    ],
)
def test_phrase_layout(n_bars, cadences, echoes):
    layout = phrase_layout(n_bars)
    assert layout.cadence_bars == cadences
    assert layout.echo_pairs == echoes
    assert sum(layout.phrase_lengths) == n_bars


def test_plan_bars_structure_for_eight_bars():
    plans = plan_bars(8, np.random.default_rng(3))
    assert len(plans) == 8
    assert all(isinstance(p, BarPlan) for p in plans)
    assert [p.cadence for p in plans] == [False, False, False, True] * 2
    assert [p.final for p in plans] == [False] * 7 + [True]
    assert plans[2].rhythm == plans[0].rhythm
    assert plans[2].motif_from == 0
    assert [p.rhythm for p in plans[4:7]] == [p.rhythm for p in plans[0:3]]
    assert [p.motif_from for p in plans[4:7]] == [0, 1, 2]
    assert plans[7].rhythm[-1] >= 4


def test_plan_bars_single_bar_is_a_final_cadence():
    (only,) = plan_bars(1, np.random.default_rng(0))
    assert only.cadence and only.final
    assert only.rhythm in CADENCE_RHYTHMS


@pytest.mark.parametrize("n_bars", range(1, 14))
def test_plan_bars_follows_the_layout(n_bars):
    plans = plan_bars(n_bars, np.random.default_rng(n_bars))
    layout = phrase_layout(n_bars)
    assert len(plans) == n_bars
    assert tuple(i for i, p in enumerate(plans) if p.cadence) == layout.cadence_bars
    pairs = tuple((p.motif_from, i) for i, p in enumerate(plans) if p.motif_from is not None)
    assert tuple(sorted(pairs, key=lambda x: x[1])) == tuple(
        sorted(layout.echo_pairs, key=lambda x: x[1])
    )
    for source, target in layout.echo_pairs:
        assert plans[source].rhythm == plans[target].rhythm


def test_plan_bars_is_deterministic_per_seed():
    assert plan_bars(8, np.random.default_rng(11)) == plan_bars(8, np.random.default_rng(11))
