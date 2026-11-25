"""Phrase model training on the mini corpus."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from music_decoder.compose import phrase_train as ptr
from music_decoder.compose.phrase_model import PhraseMeta, load_phrase_model

MINI = Path(__file__).parents[1] / "fixtures" / "compose" / "pop909_mini"


def test_build_on_the_mini_corpus_writes_a_loadable_model(tmp_path):
    out = tmp_path / "m.npz"
    meta = ptr.build(
        MINI,
        out,
        arch={"d": 16, "layers": 1, "heads": 2},
        steps=3,
        seed=0,
        temperature=1.0,
        nucleus=1.0,
        figure_length=4,
        device="cpu",
    )
    assert isinstance(meta, PhraseMeta)
    m = load_phrase_model(out)
    assert m.meta == meta
    assert meta.split_seed == ptr.SPLIT_SEED == 909
    assert meta.licence == "MIT" and meta.sources
    assert meta.pitch_lo <= meta.pitch_hi and m.vocab == meta.pitch_hi - meta.pitch_lo + 2
    assert np.isfinite(meta.heldout_bits) and meta.heldout_bits > 0
    assert set(meta.surprise) <= {"major", "minor"}
    assert all(iqr >= 0 for _, iqr in meta.surprise.values())
    assert meta.figure_length == 4
    assert len(m.figures) > 0


def test_split_is_by_song_and_seeded():
    names = [f"{i:03d}" for i in range(100)]
    a, b = ptr.split(names, 909), ptr.split(names, 909)
    assert a == b
    assert [len(a[k]) for k in ("train", "valid", "heldout")] == [80, 10, 10]
    assert sorted(a["train"] + a["valid"] + a["heldout"]) == names
    assert ptr.split(names, 1) != a


def test_pitch_range_is_the_narrowest_covering_window():
    vals = np.array([0] * 995 + [40] * 5)
    assert ptr.pitch_range(vals, 0.995) == (0, 0)
    assert ptr.pitch_range(np.arange(-5, 6), 1.0) == (-5, 5)
