"""The phrase-model pick rule: --melody-pick phrase (phrase-model spec 5.4, 6.1)."""

from __future__ import annotations

import hashlib
from unittest import mock

import numpy as np
import pytest
from click.testing import CliRunner

from music_decoder.cli.main import main
from music_decoder.compose import api, phrase_model, scoring
from music_decoder.compose.api import arrange_candidates, compose_with_options
from music_decoder.compose.ml import aesthetics
from music_decoder.compose.phrase_tokens import SLOTS_PER_BAR, encode_piece
from music_decoder.tabs.tuning import STANDARD_EADGBE
from music_decoder.types import ChordSymbol, Note, Scale
from tests.compose_phrase_helpers import tiny_model

SCALE = Scale.parse("C:major")
PROG = [ChordSymbol.parse(c) for c in ("C", "G", "Am", "F")]


@pytest.fixture
def fake_wav():
    with mock.patch("music_decoder.compose.api.render_wav") as render:
        render.side_effect = lambda midi_path, out_path: out_path.write_bytes(b"RIFF")
        yield render


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    m = tiny_model(tmp_path)
    monkeypatch.setattr(phrase_model, "load_phrase_model", lambda path=None: m)
    return m


def _no_model(path=None):
    raise AssertionError("the phrase model must not load")


def _tag(result) -> str:
    return result.midi_path.parent.name.rsplit("-", 1)[1]


def test_phrase_surprise_averages_cold_eight_bar_windows(tiny):
    prog = PROG * 4  # 16 bars
    notes = [
        Note(
            start_s=b * 2.4 + k * 0.6,
            end_s=b * 2.4 + k * 0.6 + 0.5,
            pitch=60 + (b + k) % 7,
            velocity=80,
            confidence=1.0,
        )
        for b in range(16)
        for k in range(4)
    ]
    s = scoring.phrase_surprise(
        notes, scale=SCALE, progression=prog, bars_per_chord=1, tempo_bpm=100.0
    )
    middle = int(np.median([n.pitch for n in notes]))
    enc = encode_piece(
        notes,
        scale=SCALE,
        progression=prog,
        bars_per_chord=1,
        tempo_bpm=100.0,
        middle=middle,
        pitch_lo=-12,
        pitch_hi=17,
    )
    want = []
    for b0 in (0, 8):
        sl = slice(b0 * SLOTS_PER_BAR, (b0 + 8) * SLOTS_PER_BAR)
        win = type(enc)(enc.tokens[sl], {k: v[sl] for k, v in enc.cond.items()}, frozenset())
        bars = tiny.slot_bits(win).reshape(8, SLOTS_PER_BAR).sum(1)
        want.append(bars[:-1].mean() if b0 == 8 else bars.mean())
    assert s == pytest.approx(float(np.mean(want)), abs=1e-9)


def test_out_of_range_melody_scores_the_fixed_distance(tiny):
    notes = [
        Note(start_s=0.0, end_s=0.5, pitch=60, velocity=80, confidence=1.0),
        Note(start_s=0.6, end_s=1.0, pitch=110, velocity=80, confidence=1.0),
    ]
    kw = dict(scale=SCALE, progression=PROG, bars_per_chord=1, tempo_bpm=100.0)
    assert scoring.phrase_surprise(notes, **kw) is None
    assert scoring.phrase_distance(notes, **kw) == scoring.PHRASE_OUT_OF_RANGE == 10.0
    assert scoring.phrase_distance([], **kw) == scoring.PHRASE_OUT_OF_RANGE


def test_phrase_pick_reorders_candidates(tmp_path, monkeypatch):
    kw = dict(bars_per_chord=2, tempo_bpm=100.0, seed=5, candidates=6, melody_engine="tune")
    bands = arrange_candidates(SCALE, PROG, **kw)
    target = bands[-1]  # the band score's last choice
    probe = tiny_model(tmp_path, "probe.npz")
    monkeypatch.setattr(phrase_model, "load_phrase_model", lambda path=None: probe)
    s = scoring.phrase_surprise(
        target.melody, scale=SCALE, progression=PROG, bars_per_chord=2, tempo_bpm=100.0
    )
    sharp = tiny_model(tmp_path, "sharp.npz", surprise={"major": (s, 1e-6), "minor": (s, 1e-6)})
    monkeypatch.setattr(phrase_model, "load_phrase_model", lambda path=None: sharp)
    picked = arrange_candidates(SCALE, PROG, melody_pick="phrase", **kw)
    assert picked[0].chosen_candidate == target.chosen_candidate
    assert picked[0].melody == target.melody


def test_phrase_score_swaps_the_band_term(tiny):
    arr = arrange_candidates(SCALE, PROG, seed=2, candidates=1, melody_engine="tune")[0]
    kw = dict(
        bars_per_chord=1, tempo_bpm=100.0, tuning=STANDARD_EADGBE, scale=SCALE, melody_engine="tune"
    )
    bands = scoring.score_arrangement(arr, PROG, **kw)
    phrase = scoring.score_arrangement(arr, PROG, melody_pick="phrase", **kw)
    from music_decoder.compose.tune_metrics import band_distance, tune_features
    from music_decoder.compose.tune_stats import load_tune_stats

    band = band_distance(
        tune_features(arr.melody, SCALE, PROG, 1, 100.0), load_tune_stats().bands["major"]
    )
    dist = scoring.phrase_distance(
        arr.melody, scale=SCALE, progression=PROG, bars_per_chord=1, tempo_bpm=100.0
    )
    assert phrase == pytest.approx(bands + scoring.W_CORPUS * band - scoring.W_PHRASE * dist)
    assert scoring.W_PHRASE == 2.0


def test_phrase_pick_with_rules_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(phrase_model, "load_phrase_model", _no_model)
    with pytest.raises(ValueError, match="melody_pick"):
        arrange_candidates(SCALE, PROG, seed=1, melody_engine="rules", melody_pick="phrase")
    with pytest.raises(ValueError, match="melody_pick"):
        compose_with_options(
            SCALE, PROG, seed=1, out_dir=tmp_path, melody_engine="rules", melody_pick="phrase"
        )
    with pytest.raises(ValueError, match="melody_pick"):
        compose_with_options(SCALE, PROG, seed=1, out_dir=tmp_path, melody_pick="best")  # type: ignore[arg-type]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("style", ["strum", "arpeggio"])
def test_phrase_pick_is_ignored_by_strum_and_arpeggio(tmp_path, fake_wav, monkeypatch, style):
    monkeypatch.setattr(phrase_model, "load_phrase_model", _no_model)
    c = compose_with_options(
        SCALE, PROG, seed=4, style=style, out_dir=tmp_path / "a", melody_pick="phrase"
    )
    plain = compose_with_options(
        SCALE, PROG, seed=4, style=style, out_dir=tmp_path / "b", melody_pick="bands"
    )
    assert c.metadata["melody_pick"] == "bands" and c.metadata["phrase_model"] is None
    assert c.melody_notes == plain.melody_notes
    assert _tag(c) == _tag(plain)


def test_bands_take_folder_hash_is_the_v7_formula(tmp_path, fake_wav, monkeypatch):
    monkeypatch.setattr(phrase_model, "load_phrase_model", _no_model)
    c = compose_with_options(SCALE, PROG, seed=3, out_dir=tmp_path, melody_pick="bands")
    old = hashlib.sha1(
        (
            f"{SCALE}|{[x.to_label() for x in PROG]}|100.0|fingerstyle|3|tune"
            f"|{STANDARD_EADGBE.name}"
            "|12|soundfont|1|False|None"
        ).encode()
    ).hexdigest()[:8]
    assert _tag(c) == old


def test_compose_defaults_to_the_phrase_pick(tmp_path, fake_wav, tiny):
    c = api.compose(SCALE, PROG, seed=3, out_dir=tmp_path / "a")
    assert c.metadata["melody_pick"] == "phrase" and c.metadata["phrase_model"] == tiny.weights_hash
    assert c.metadata["candidates"] == api.PHRASE_PICK_CANDIDATES
    assert c.metadata["composer_version"] == api.COMPOSER_VERSION == 8
    d = compose_with_options(SCALE, PROG, seed=3, out_dir=tmp_path / "b", candidates=4)
    assert d.metadata["melody_pick"] == "phrase" and d.metadata["candidates"] == 4


def test_rules_engine_defaults_to_the_band_pick(tmp_path, fake_wav, monkeypatch):
    monkeypatch.setattr(phrase_model, "load_phrase_model", _no_model)
    c = compose_with_options(SCALE, PROG, seed=3, out_dir=tmp_path, melody_engine="rules")
    assert c.metadata["melody_pick"] == "bands" and c.metadata["phrase_model"] is None
    assert c.metadata["candidates"] == 12


def test_metadata_records_the_pick_and_the_weights_hash(tmp_path, fake_wav, tiny):
    plain = compose_with_options(
        SCALE, PROG, seed=3, out_dir=tmp_path / "a", candidates=3, melody_pick="bands"
    )
    assert plain.metadata["melody_pick"] == "bands" and plain.metadata["phrase_model"] is None
    c = compose_with_options(
        SCALE, PROG, seed=3, out_dir=tmp_path / "b", candidates=3, melody_pick="phrase"
    )
    assert c.metadata["melody_pick"] == "phrase"
    assert c.metadata["phrase_model"] == tiny.weights_hash
    assert c.metadata["candidates"] == 3
    assert _tag(c) != _tag(plain)
    manifest = (c.midi_path.parent / api.MANIFEST_NAME).read_text()
    assert '"melody_pick": "phrase"' in manifest and tiny.weights_hash in manifest


def test_phrase_pick_defaults_to_its_own_candidate_count(tmp_path, fake_wav, tiny):
    c = compose_with_options(SCALE, PROG, seed=3, out_dir=tmp_path, melody_pick="phrase")
    assert c.metadata["candidates"] == api.PHRASE_PICK_CANDIDATES == 200


def test_cli_melody_pick_reaches_the_options(tmp_path):
    with mock.patch("music_decoder.cli.main.compose_with_options") as cwo:
        cwo.side_effect = RuntimeError("stop")
        CliRunner().invoke(
            main,
            [
                "compose",
                "--scale",
                "C:major",
                "--progression",
                "C G",
                "--melody-pick",
                "phrase",
                "--out",
                str(tmp_path),
            ],
        )
        assert cwo.call_args.kwargs["melody_pick"] == "phrase"
        assert cwo.call_args.kwargs["candidates"] is None
        CliRunner().invoke(
            main, ["compose", "--scale", "C:major", "--progression", "C G", "--out", str(tmp_path)]
        )
        assert (
            cwo.call_args.kwargs["melody_pick"] is None
        )  # the API picks: phrase for the tune engine, bands for rules
    bad = CliRunner().invoke(
        main,
        [
            "compose",
            "--scale",
            "C:major",
            "--progression",
            "C G",
            "--melody-pick",
            "best",
            "--out",
            str(tmp_path),
        ],
    )
    assert bad.exit_code != 0


def test_rank_audio_with_phrase_pick_runs(tmp_path, fake_wav, tiny, monkeypatch):
    class _Scorer:
        def score(self, wavs):
            return [{"CE": 5.0, "CU": 5.0, "PC": 5.0, "PQ": 5.0} for _ in wavs]

    monkeypatch.setattr(aesthetics, "load_scorer", lambda: _Scorer())
    seen = {}
    real = aesthetics.choose_by_audio

    def spy(top, rows, *, symbolic_scale):
        seen["scale"] = symbolic_scale
        return real(top, rows, symbolic_scale=symbolic_scale)

    monkeypatch.setattr(aesthetics, "choose_by_audio", spy)
    c = compose_with_options(
        SCALE,
        PROG,
        seed=3,
        out_dir=tmp_path,
        candidates=4,
        rank_audio=True,
        melody_pick="phrase",
        sound="plucked",
    )
    assert c.metadata["melody_pick"] == "phrase" and c.metadata["audio_scores"] is not None
    assert seen["scale"] == aesthetics.SYMBOLIC_SCALE


def test_candidate_scores_equal_the_single_score(tiny):
    kw = dict(bars_per_chord=2, tempo_bpm=100.0, seed=7, candidates=5, melody_engine="tune")
    for arr in arrange_candidates(SCALE, PROG, melody_pick="phrase", **kw):
        one = scoring.score_arrangement(
            arr,
            PROG,
            bars_per_chord=2,
            tempo_bpm=100.0,
            tuning=STANDARD_EADGBE,
            scale=SCALE,
            melody_engine="tune",
            melody_pick="phrase",
        )
        assert arr.score == pytest.approx(one, abs=1e-9)


def test_phrase_distances_batch_matches_singles(tiny):
    arrs = arrange_candidates(SCALE, PROG, seed=8, candidates=4, melody_engine="tune")
    kw = dict(scale=SCALE, progression=PROG, bars_per_chord=1, tempo_bpm=100.0)
    bad = [
        Note(start_s=0.0, end_s=0.5, pitch=60, velocity=80, confidence=1.0),
        Note(start_s=0.6, end_s=1.0, pitch=110, velocity=80, confidence=1.0),
    ]  # 110 is far above the anchor under the median, 85
    melodies = [a.melody for a in arrs] + [bad, []]
    got = scoring.phrase_distances(melodies, **kw)
    assert got == pytest.approx([scoring.phrase_distance(m, **kw) for m in melodies], abs=1e-9)
    assert got[-2:] == [10.0, 10.0]


def test_candidates_are_scored_in_one_model_call(tiny, monkeypatch):
    calls = []
    real = tiny.slot_bits_many
    monkeypatch.setattr(
        tiny, "slot_bits_many", lambda encs, **kw: calls.append(len(encs)) or real(encs)
    )
    arrange_candidates(
        SCALE,
        PROG,
        bars_per_chord=4,
        seed=9,
        candidates=6,
        melody_engine="tune",
        melody_pick="phrase",
    )
    assert calls == [12]  # 16 bars: two windows for each of six candidates
