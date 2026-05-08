"""Regression suite: gates accuracy metrics against ``config/eval_thresholds.yaml``.

Each test runs ``analyze()`` on a committed synthetic fixture, computes one
metric, and asserts the score meets the threshold pinned in the YAML. Marked
``regression`` so it is excluded from the default ``pytest`` invocation;
``make regression`` (or ``pytest -m regression``) opts in.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest
import yaml

from music_decoder import analyze
from music_decoder.evaluation.fixtures.synthetic import iter_synthetic_fixtures
from music_decoder.evaluation.metrics import (
    chord_recognition_score,
    key_mirex_score,
    tab_string_accuracy,
)


@lru_cache(maxsize=1)
def _thresholds() -> dict[str, float]:
    p = Path("config/eval_thresholds.yaml")
    return yaml.safe_load(p.read_text())


@pytest.mark.regression
@pytest.mark.parametrize(
    "fixture",
    list(iter_synthetic_fixtures()),
    ids=lambda f: f.name,
)
def test_chord_recognition_score(fixture):
    res = analyze(
        fixture.audio_path,
        declared_kind="solo_guitar",
        use_separation=False,
    )
    score = chord_recognition_score(
        res.chord_progression,
        fixture.gt.chord_progression,
    )
    assert score >= _thresholds()["chord_recognition_score"], (
        f"chord_recognition_score={score:.3f} below {_thresholds()['chord_recognition_score']}"
    )


@pytest.mark.regression
@pytest.mark.parametrize(
    "fixture",
    list(iter_synthetic_fixtures()),
    ids=lambda f: f.name,
)
def test_key_mirex_score(fixture):
    res = analyze(
        fixture.audio_path,
        declared_kind="solo_guitar",
        use_separation=False,
    )
    score = key_mirex_score(res.key, fixture.gt.key)
    assert score >= _thresholds()["key_mirex_score"], (
        f"key_mirex_score={score:.3f} below {_thresholds()['key_mirex_score']}"
    )


@pytest.mark.regression
@pytest.mark.parametrize(
    "fixture",
    list(iter_synthetic_fixtures()),
    ids=lambda f: f.name,
)
def test_tab_string_accuracy(fixture):
    if not fixture.gt.tab:
        pytest.skip("fixture has no tab ground truth")
    res = analyze(
        fixture.audio_path,
        declared_kind="solo_guitar",
        use_separation=False,
    )
    score = tab_string_accuracy(res.tab, fixture.gt.tab)
    assert score >= _thresholds()["tab_string_accuracy"], (
        f"tab_string_accuracy={score:.3f} below {_thresholds()['tab_string_accuracy']}"
    )
