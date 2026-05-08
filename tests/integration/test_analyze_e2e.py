"""End-to-end smoke test for the public ``analyze()`` API.

Runs the full audio pipeline on each committed synthetic fixture and asserts
the basic shape of the result. Metric-gated correctness lives in
``test_regression.py``; this file only checks that the pipeline produces a
non-trivial result without raising.
"""
from __future__ import annotations

import pytest

from music_decoder import analyze
from music_decoder.evaluation.fixtures.synthetic import iter_synthetic_fixtures


@pytest.mark.parametrize(
    "fixture",
    list(iter_synthetic_fixtures()),
    ids=lambda f: f.name,
)
def test_analyze_runs_on_synthetic_fixture(fixture):
    result = analyze(
        fixture.audio_path,
        declared_kind="solo_guitar",
        use_separation=False,
    )
    assert result.duration_s > 0
    assert len(result.chord_progression) >= 1
    assert len(result.tab) >= 1
