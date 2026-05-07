# tests/unit/test_tab_astar.py
from music_decoder.types import TabPosition
from music_decoder.tab_assignment.astar import (
    Group,
    astar_min_cost_path,
)

_W = {
    "w_move": 1.0, "w_string": 0.3, "w_span": 0.5, "w_high": 0.4,
    "w_open": 0.2, "w_chord_intra": 0.6,
}


def test_single_group_picks_lowest_cost_candidate():
    groups: list[Group] = [
        Group([(TabPosition(5, 0),), (TabPosition(4, 5),)]),
    ]
    path, _cost = astar_min_cost_path(groups, weights=_W, hand_anchor_window=8)
    # Open string should win because of open_bonus.
    assert path[0] == (TabPosition(5, 0),)


def test_two_groups_avoids_high_cost_transition():
    groups: list[Group] = [
        Group([(TabPosition(5, 0),)]),
        Group([(TabPosition(4, 1),), (TabPosition(0, 18),)]),
    ]
    path, _cost = astar_min_cost_path(groups, weights=_W, hand_anchor_window=8)
    assert path[1] == (TabPosition(4, 1),)


def test_empty_groups_returns_empty_path():
    path, cost = astar_min_cost_path([], weights=_W, hand_anchor_window=8)
    assert path == []
    assert cost == 0.0
