# tests/unit/test_tab_heuristic.py
from music_decoder.pipeline.contracts import TabPosition
from music_decoder.tab_assignment.heuristic import remaining_high_fret_penalty


_W = {"w_high": 0.4}


def test_no_remaining_groups_returns_zero():
    assert remaining_high_fret_penalty([], weights=_W) == 0.0


def test_single_group_low_fret_returns_zero():
    groups = [[(TabPosition(5, 3),)]]
    assert remaining_high_fret_penalty(groups, weights=_W) == 0.0


def test_high_fret_in_group_contributes():
    groups = [
        [(TabPosition(5, 14),)],   # min fret = 14, penalty (14-12)^1.2 * 0.4
        [(TabPosition(5, 18),)],
    ]
    h = remaining_high_fret_penalty(groups, weights=_W)
    assert h > 0


def test_uses_minimum_fret_in_each_group():
    groups = [[
        (TabPosition(5, 18),),
        (TabPosition(4, 14),),  # min fret = 14
    ]]
    h = remaining_high_fret_penalty(groups, weights=_W)
    expected = _W["w_high"] * max(0, 14 - 12) ** 1.2
    assert h == expected
