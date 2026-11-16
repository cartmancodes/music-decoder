"""The audio re-rank with the real Audiobox Aesthetics weights (v5 spec 5.4).

Opt-in: runs only with ``MUSIC_DECODER_ML_E2E=1``, and needs the ``ml``
extra (``pip install -e ".[ml]"``) and the weights
(``music-decoder fetch-models --composer-models``, about 415 MB).
"""

from __future__ import annotations

import math
import os

import pytest

from music_decoder.compose.api import compose_with_options
from music_decoder.compose.ml import aesthetics
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
