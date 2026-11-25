"""Phrase model runtime: weights file, bits, sampling, copy check."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from music_decoder.compose import phrase_model as pm
from music_decoder.compose.phrase_tokens import encode_piece
from music_decoder.types import ChordSymbol, Note, Scale
from tests.compose_phrase_helpers import TINY_RUN, tiny_model


def _enc():
    scale, prog = Scale.parse("A:minor"), [ChordSymbol.parse(c) for c in ("Am", "F")]
    notes = [
        Note(
            start_s=i * 0.25,
            end_s=i * 0.25 + 0.2,
            pitch=57 + (i * 5) % 12,
            velocity=80,
            confidence=1.0,
        )
        for i in range(12)
    ]
    return encode_piece(
        notes,
        scale=scale,
        progression=prog,
        bars_per_chord=1,
        tempo_bpm=120.0,
        middle=60,
        pitch_lo=-12,
        pitch_hi=17,
    )


def test_tiny_model_is_tiny(tmp_path):
    m = tiny_model(tmp_path)
    assert m.n_params < 20_000
    assert len(m.weights_hash) == 12


def test_saved_model_loads_without_pickle(tmp_path):
    m = tiny_model(tmp_path)
    with np.load(m.path, allow_pickle=False) as z:
        for key in z.files:
            assert z[key] is not None
        assert "meta" in z.files and "figures" in z.files
        assert z["figures"].dtype == np.uint32
        assert all(z[k].dtype == np.float16 for k in z.files if k.startswith("w."))


def test_slot_bits_match_the_sampler(tmp_path):
    m, enc = tiny_model(tmp_path), _enc()
    bits = m.slot_bits(enc)
    assert bits.shape == enc.tokens.shape
    s = m.sampler(enc.cond)
    for t, tok in enumerate(enc.tokens):
        assert abs(-np.log2(s.probs()[tok]) - bits[t]) < 1e-6
        s.push(int(tok))


def test_probs_sum_to_one(tmp_path):
    m, enc = tiny_model(tmp_path), _enc()
    s = m.sampler(enc.cond)
    for tok in enc.tokens[:5]:
        p = s.probs()
        assert p.dtype == np.float64 and p.shape == (m.vocab,)
        assert abs(p.sum() - 1.0) < 1e-9
        s.push(int(tok))


def test_same_inputs_same_bits(tmp_path):
    m, enc = tiny_model(tmp_path), _enc()
    assert (m.slot_bits(enc) == m.slot_bits(enc)).all()


def test_has_figure_finds_a_stored_run(tmp_path):
    m = tiny_model(tmp_path)
    ons, ps = TINY_RUN
    assert m.has_figure(ons, ps)
    assert m.has_figure([o + 32 for o in ons], [p - 3 for p in ps])
    assert not m.has_figure(ons, [60, 62, 64, 66])
    assert not m.has_figure(ons[:3], ps[:3])


def test_missing_weights_raise_with_the_path(tmp_path):
    missing = tmp_path / "nope.npz"
    with pytest.raises(pm.PhraseModelError, match=str(missing)):
        pm.load_phrase_model(missing)
    junk = tmp_path / "junk.npz"
    junk.write_bytes(b"not a zip")
    with pytest.raises(pm.PhraseModelError, match=str(junk)):
        pm.load_phrase_model(junk)


def test_compose_api_import_does_not_import_torch():
    code = (
        "import sys, music_decoder.compose.api, music_decoder.cli.main, "
        "music_decoder.compose.phrase_model; print('torch' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_shipped_weights_score_a_fixed_melody():
    assert pm.WEIGHTS_PATH.stat().st_size <= 6_000_000
    m = pm.load_phrase_model()
    assert m.n_params <= 2_000_000 and m.meta.figure_length == 11
    assert (m.meta.pitch_lo, m.meta.pitch_hi, m.meta.split_seed) == (-12, 17, 909)
    scale = Scale.parse("C:major")
    prog = [ChordSymbol.parse(c) for c in "C Am F G C Am Dm G".split()]
    line = [64, 62, 60, 62, 64, 64, 64, 62, 62, 62, 64, 67, 67, 64, 62, 60]
    notes = [
        Note(start_s=i * 0.6, end_s=i * 0.6 + 0.55, pitch=p, velocity=80, confidence=1.0)
        for i, p in enumerate(line)
    ]
    enc = encode_piece(
        notes,
        scale=scale,
        progression=prog,
        bars_per_chord=1,
        tempo_bpm=100.0,
        middle=62,
        pitch_lo=m.meta.pitch_lo,
        pitch_hi=m.meta.pitch_hi,
    )
    assert round(float(m.slot_bits(enc).sum()), 3) == 75.526


def test_batched_bits_match_one_at_a_time(tmp_path):
    m, enc = tiny_model(tmp_path), _enc()
    half = type(enc)(enc.tokens[:16], {k: v[:16] for k, v in enc.cond.items()}, frozenset())
    many = m.slot_bits_many([enc, half, enc])
    assert [len(b) for b in many] == [32, 16, 32]
    for got, one in zip(many, [enc, half, enc], strict=True):
        assert np.allclose(got, m.slot_bits(one), atol=1e-9)
    assert m.slot_bits_many([]) == []
