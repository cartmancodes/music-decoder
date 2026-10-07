# tests/unit/test_tab_cost.py

from music_decoder.tabs.cost import (
    chord_collides,
    chord_span_penalty,
    transition_cost,
)
from music_decoder.types import TabPosition

_W = {
    "w_move": 1.0,
    "w_string": 0.3,
    "w_span": 0.5,
    "w_high": 0.4,
    "w_open": 0.2,
    "w_chord_intra": 0.6,
}


def test_transition_cost_zero_for_identical_position():
    p = TabPosition(string=4, fret=3)
    cost = transition_cost(prev=p, curr=p, weights=_W, hand_anchor=3.0)
    assert cost <= 0.0  # open-string bonus may pull negative; non-positive at minimum


def test_open_string_gets_bonus():
    # Same move, with and without the open-string bonus. (The v2 version of
    # this test compared against a free downward move, which was the bug.)
    p = TabPosition(string=5, fret=2)
    o = TabPosition(string=5, fret=0)
    no_bonus = transition_cost(prev=p, curr=o, weights={**_W, "w_open": 0.0}, hand_anchor=2.0)
    with_bonus = transition_cost(prev=p, curr=o, weights=_W, hand_anchor=2.0)
    assert with_bonus < no_bonus


def test_high_fret_penalizes():
    low = TabPosition(string=5, fret=5)
    high = TabPosition(string=5, fret=18)
    c_low = transition_cost(prev=low, curr=low, weights=_W, hand_anchor=5.0)
    c_high = transition_cost(prev=low, curr=high, weights=_W, hand_anchor=5.0)
    assert c_high > c_low


def test_movement_cost_grows_with_fret_distance():
    a = TabPosition(string=5, fret=3)
    b = TabPosition(string=5, fret=7)
    c = TabPosition(string=5, fret=15)
    near = transition_cost(prev=a, curr=b, weights=_W, hand_anchor=3.0)
    far = transition_cost(prev=a, curr=c, weights=_W, hand_anchor=3.0)
    assert far > near


def test_chord_span_penalty_zero_within_4_fret_span():
    chord = (TabPosition(0, 3), TabPosition(1, 2), TabPosition(2, 0), TabPosition(3, 0))
    assert chord_span_penalty(chord) == 0.0


def test_chord_collides_detects_string_duplicates():
    chord_ok = (TabPosition(0, 3), TabPosition(1, 2))
    chord_bad = (TabPosition(0, 3), TabPosition(0, 5))
    assert not chord_collides(chord_ok)
    assert chord_collides(chord_bad)


def test_moving_down_the_neck_costs_like_moving_up():
    a = TabPosition(string=3, fret=9)
    b = TabPosition(string=3, fret=2)
    # Isolate the hand-move term (span is deliberately one-sided: stretching
    # up from the anchor is harder than reaching back).
    w = {**_W, "w_span": 0.0}
    down = transition_cost(prev=a, curr=b, weights=w, hand_anchor=9.0)
    up = transition_cost(prev=b, curr=a, weights=w, hand_anchor=2.0)
    assert down > 0.0
    assert abs(down - up) < 1e-9


def test_jumping_to_an_open_string_still_costs_hand_movement():
    # GuitarSet players don't treat open strings as free (measured: -5 pts).
    a = TabPosition(string=3, fret=9)
    o = TabPosition(string=5, fret=0)
    near = TabPosition(string=5, fret=9)
    far_open = transition_cost(prev=a, curr=o, weights=_W, hand_anchor=9.0)
    in_position = transition_cost(prev=a, curr=near, weights=_W, hand_anchor=9.0)
    assert far_open > in_position
