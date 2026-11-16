"""v5 feature 4: the symbolic score and best of N fingerstyle candidates (spec 5.4)."""

from __future__ import annotations

import os
import statistics
import subprocess
import sys
from dataclasses import replace

import pytest

from music_decoder.compose.api import arrange
from music_decoder.compose.scoring import candidate_seeds, score_arrangement
from music_decoder.evaluation import compose_musicality as ev
from music_decoder.tabs.tuning import STANDARD_EADGBE
from music_decoder.types import ChordSymbol, Scale

PROG8 = [ChordSymbol.parse(c) for c in ("C", "Am", "F", "G", "C", "F", "G", "C")]


def test_one_candidate_equals_the_plain_arrangement():
    a = arrange(Scale("C", "major"), PROG8, tempo_bpm=120.0, seed=4)
    b = arrange(Scale("C", "major"), PROG8, tempo_bpm=120.0, seed=4, candidates=1)
    assert a == b and b.chosen_candidate == 0 and b.candidates == 1


def test_best_of_n_scores_at_least_every_candidate():
    best = arrange(Scale("C", "major"), PROG8, tempo_bpm=120.0, seed=4, candidates=12)
    assert best.candidates == 12 and 0 <= best.chosen_candidate < 12
    singles = [
        arrange(Scale("C", "major"), PROG8, tempo_bpm=120.0, seed=s, candidates=1)
        for s in candidate_seeds(4, 12)
    ]
    assert best.score == max(s.score for s in singles)
    assert best == replace(
        singles[best.chosen_candidate], candidates=12, chosen_candidate=best.chosen_candidate
    )


def test_best_of_n_raises_the_mean_score():
    cases = ev.validation_inputs()[:40]
    one = [arrange(s, p, bars_per_chord=b, tempo_bpm=120.0, seed=k).score for s, p, b, k in cases]
    many = [
        arrange(s, p, bars_per_chord=b, tempo_bpm=120.0, seed=k, candidates=12).score
        for s, p, b, k in cases
    ]
    assert statistics.fmean(many) > statistics.fmean(one)
    assert all(m >= o for m, o in zip(many, one, strict=True))


def test_ties_choose_the_first_candidate():
    a = arrange(Scale("C", "major"), [ChordSymbol.parse("C")], seed=1, candidates=12)
    assert a.score is not None  # whichever wins, re-running gives the same k
    assert arrange(Scale("C", "major"), [ChordSymbol.parse("C")], seed=1, candidates=12) == a
    singles = [
        arrange(Scale("C", "major"), [ChordSymbol.parse("C")], seed=s)
        for s in candidate_seeds(1, 12)
    ]
    scores = [s.score for s in singles]
    assert a.chosen_candidate == scores.index(max(scores))  # the first of the best


def test_seed_none_still_chooses_deterministically_within_a_call():
    a = arrange(Scale("C", "major"), PROG8, candidates=3)
    assert a.chosen_candidate in (0, 1, 2) and a.score is not None


def test_candidates_must_be_at_least_one():
    with pytest.raises(ValueError):
        arrange(Scale("C", "major"), PROG8, candidates=0)


def test_candidate_seeds_start_with_the_seed_and_are_distinct():
    seeds = candidate_seeds(4, 12)
    assert len(seeds) == 12 and seeds[0] == 4 and len(set(seeds)) == 12
    assert candidate_seeds(4, 3) == seeds[:3]


def test_candidate_seeds_do_not_depend_on_the_hash_seed():
    code = (
        "from music_decoder.compose.scoring import candidate_seeds; print(candidate_seeds(4, 12))"
    )
    outputs = [
        subprocess.run(
            [sys.executable, "-c", code],
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for hash_seed in ("0", "12345")
    ]
    assert outputs[0] == outputs[1] and outputs[0].strip()


def test_other_styles_ignore_candidates_and_have_no_score():
    a = arrange(Scale("C", "major"), PROG8, style="strum", seed=4, candidates=12)
    assert a == arrange(Scale("C", "major"), PROG8, style="strum", seed=4)
    assert a.candidates == 1 and a.chosen_candidate == 0 and a.score is None


def test_final_fallback_costs_three_points():
    arr = arrange(Scale("C", "major"), PROG8, tempo_bpm=120.0, seed=4)
    kwargs = {"bars_per_chord": 1, "tempo_bpm": 120.0, "tuning": STANDARD_EADGBE}
    plain = score_arrangement(replace(arr, final_fallback=False), PROG8, **kwargs)
    fallback = score_arrangement(replace(arr, final_fallback=True), PROG8, **kwargs)
    assert plain - fallback == pytest.approx(3.0)
    assert arr.score == score_arrangement(arr, PROG8, **kwargs)
