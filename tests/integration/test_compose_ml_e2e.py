"""The composer's optional models with their real weights: the audio
re-rank (v5 spec 5.4) and the AMT melody model (v5 spec 5.5).

Opt-in: runs only with ``MUSIC_DECODER_ML_E2E=1``, and needs the ``ml``
extra (``pip install -e ".[ml]"``) and the weights
(``music-decoder fetch-models --composer-models``, about 930 MB).
"""

from __future__ import annotations

import math
import os
import time

import pytest

from music_decoder.compose import metrics
from music_decoder.compose.api import arrange, compose_with_options
from music_decoder.compose.ml import aesthetics, amt
from music_decoder.tabs.tuning import STANDARD_EADGBE
from music_decoder.types import ChordSymbol, Scale

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("MUSIC_DECODER_ML_E2E") != "1",
        reason="set MUSIC_DECODER_ML_E2E=1 (needs the ml extra and fetch-models --composer-models)",
    ),
]

PROG = [ChordSymbol.parse(c) for c in ("C", "Am", "F", "G")]
PROG8 = [ChordSymbol.parse(c) for c in ("C", "Am", "F", "G", "C", "F", "G", "C")]


def test_rank_audio_with_the_real_scorer_repeats(tmp_path):
    scorer = aesthetics.load_scorer()
    assert isinstance(scorer, aesthetics.AestheticScorer)

    runs = [
        compose_with_options(
            Scale("C", "major"), PROG, seed=1, out_dir=tmp_path / str(i), rank_audio=True
        )
        for i in range(2)
    ]
    first, second = (r.metadata for r in runs)
    assert first["rank_audio"] is True and first["candidates"] == 12
    assert set(first["audio_scores"]) == {"CE", "PQ"}
    for value in first["audio_scores"].values():
        assert math.isfinite(value) and 0.0 <= value <= 10.0
    assert second["chosen_candidate"] == first["chosen_candidate"]
    assert second["audio_scores"] == first["audio_scores"]
    assert runs[0].wav_path.read_bytes() == runs[1].wav_path.read_bytes()


def test_melody_model_amt_with_the_real_model(tmp_path):
    start = time.perf_counter()
    first = compose_with_options(
        Scale("C", "major"), PROG8, seed=1, out_dir=tmp_path / "a", melody_model="amt"
    )
    elapsed = time.perf_counter() - start
    assert elapsed <= 60.0, f"one 8-bar piece took {elapsed:.1f} s"
    assert first.metadata["melody_model"] == "amt"

    # The same piece as an Arrangement, for the guarantees on its tab.
    proposer = amt.load_proposer()
    assert isinstance(proposer, amt.AMTProposer)
    a = arrange(Scale("C", "major"), PROG8, seed=1, candidates=12, melody_proposer=proposer)
    assert a.melody == first.melody_notes
    part = [*a.melody_tab, *a.accompaniment_tab]
    assert metrics.playable_moment_ratio(part) == 1.0  # P1
    assert metrics.melody_onset_on_top_ratio(a.melody, a.accompaniment) == 1.0  # P2
    assert metrics.tab_consistent((*a.melody, *a.accompaniment), part, STANDARD_EADGBE)  # P4
    assert metrics.final_chord_rings(a.accompaniment, len(PROG8), 100.0)  # A4
    assert metrics.forbidden_interval_count(a.melody) == 0
    # The proposal was not empty: it moved the tune off the no-model one.
    assert a.melody != arrange(Scale("C", "major"), PROG8, seed=1, candidates=12).melody

    second = compose_with_options(
        Scale("C", "major"), PROG8, seed=1, out_dir=tmp_path / "b", melody_model="amt"
    )
    assert second.melody_notes == first.melody_notes
