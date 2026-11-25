"""Test helper: a tiny, untrained phrase model saved to a temporary file."""

from __future__ import annotations

from pathlib import Path

from music_decoder.compose import phrase_model as pm
from music_decoder.compose.phrase_tokens import figure_hashes

_load = (
    pm.load_phrase_model
)  # bound at import, so tests that monkeypatch the loader still build models

TINY_ARCH = {"d": 16, "layers": 1, "heads": 2}
TINY_RUN = ([0, 2, 4, 6], [60, 62, 64, 65])


def tiny_meta(**over) -> pm.PhraseMeta:
    base = dict(
        sources=["test"],
        licence="MIT",
        split_seed=909,
        pitch_lo=-12,
        pitch_hi=17,
        arch=dict(TINY_ARCH),
        heldout_bits=15.0,
        surprise={"major": (14.9, 9.1), "minor": (14.2, 7.4)},
        figure_length=4,
        temperature=1.0,
        nucleus=1.0,
    )
    base.update(over)
    return pm.PhraseMeta(**base)


def tiny_model(tmp_path: Path, name: str = "tiny_phrase_model.npz", **meta_over) -> pm.PhraseModel:
    meta = tiny_meta(**meta_over)
    path = tmp_path / name
    arrays = pm.init_arrays(TINY_ARCH, meta.pitch_hi - meta.pitch_lo + 2, seed=0)
    pm.save_phrase_model(path, arrays, meta, figure_hashes(*TINY_RUN, 4))
    return _load(path)
