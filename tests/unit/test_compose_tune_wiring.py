"""Composer v6 wiring: the tune engine as the fingerstyle default (spec 5.3, 5.4)."""

import json
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pytest
from click.testing import CliRunner

from music_decoder.cli.main import main
from music_decoder.compose import metrics
from music_decoder.compose.api import (
    COMPOSER_VERSION,
    arrange,
    arrange_candidates,
    compose,
    compose_with_options,
)
from music_decoder.compose.ml import amt
from music_decoder.compose.scoring import W_CORPUS, score_arrangement
from music_decoder.compose.tune_metrics import band_distance, tune_features
from music_decoder.compose.tune_stats import load_tune_stats
from music_decoder.evaluation.compose_musicality import realize
from music_decoder.tabs.tuning import PRESETS, STANDARD_EADGBE
from music_decoder.types import ChordSymbol, Note, Scale

PROG = [ChordSymbol.parse(c) for c in ("C", "G", "Am", "F")]
SCALE = Scale.parse("C:major")
FIXTURE = Path(__file__).parents[1] / "fixtures" / "compose" / "rules_engine_melodies.json"


@pytest.fixture
def rules_fixture() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(FIXTURE.read_text())
    return data


@pytest.fixture
def fake_wav():
    with mock.patch("music_decoder.compose.api.render_wav") as render:
        render.side_effect = lambda midi_path, out_path: out_path.write_bytes(b"RIFF")
        yield render


def test_rules_engine_is_unchanged(rules_fixture):
    assert sorted(rules_fixture) == [str(s) for s in range(11, 17)]
    for seed, want in rules_fixture.items():
        arr = arrange(
            SCALE,
            PROG,
            bars_per_chord=2,
            tempo_bpm=96.0,
            seed=int(seed),
            candidates=12,
            melody_engine="rules",
        )
        assert [[round(n.start_s, 6), round(n.end_s, 6), n.pitch] for n in arr.melody] == want[
            "melody"
        ]
        assert arr.chosen_candidate == want["chosen_candidate"]


def test_arrange_defaults_to_rules_and_compose_to_tune(tmp_path, fake_wav):
    assert arrange(SCALE, PROG, seed=3) == arrange(SCALE, PROG, seed=3, melody_engine="rules")
    c = compose(SCALE, PROG, seed=3, out_dir=tmp_path / "a")
    assert c.metadata["melody_engine"] == "tune" and c.metadata["composer_version"] == 6
    assert COMPOSER_VERSION == 6
    r = compose_with_options(SCALE, PROG, seed=3, out_dir=tmp_path / "b", melody_engine="rules")
    assert r.metadata["melody_engine"] == "rules"
    assert [n.pitch for n in c.melody_notes] != [n.pitch for n in r.melody_notes]


def test_output_folder_hash_depends_on_the_melody_engine_and_other_options(tmp_path, fake_wav):
    # The folder is {timestamp}-{hash}; compare the hash alone, so a second tick proves nothing.
    def tag(sub, **kw):
        r = compose_with_options(SCALE, PROG, seed=3, out_dir=tmp_path / sub, **kw)
        return r.midi_path.parent.name.rsplit("-", 1)[1]

    base = tag("a", melody_engine="tune")
    assert tag("b", melody_engine="tune") == base
    assert tag("c", melody_engine="rules") != base
    assert tag("d", melody_engine="tune", candidates=3) != base
    assert tag("e", melody_engine="tune", sound="plucked") != base


def test_compose_with_options_tune_is_arrange_with_tune(tmp_path, fake_wav):
    c = compose_with_options(SCALE, PROG, seed=3, out_dir=tmp_path, candidates=3)
    arr = arrange(SCALE, PROG, seed=3, candidates=3, melody_engine="tune")
    assert c.melody_notes == arr.melody
    assert c.metadata["chosen_candidate"] == arr.chosen_candidate
    assert c.metadata["score"] == arr.score


class _Fixed:  # proposes a fixed line regardless of context
    def __init__(self, notes: list[Note]) -> None:
        self.notes = notes

    def propose(self, context, *, end_s, tempo_bpm, seed):
        return list(self.notes)


def test_melody_model_works_with_the_tune_engine(tmp_path, fake_wav, monkeypatch):
    line = [
        Note(start_s=k * 0.3, end_s=(k + 1) * 0.3, pitch=79, velocity=80, confidence=1.0)
        for k in range(32)
    ]  # G5 throughout, high in the reach of every shape
    prop = _Fixed(line)
    monkeypatch.setattr(amt, "load_proposer", lambda: prop)
    r = compose_with_options(
        SCALE, PROG, seed=1, out_dir=tmp_path, candidates=2, melody_model="amt"
    )
    assert r.metadata["melody_engine"] == "tune" and r.metadata["melody_model"] == "amt"
    want = arrange(SCALE, PROG, seed=1, candidates=2, melody_proposer=prop, melody_engine="tune")
    assert r.melody_notes == want.melody
    assert want.melody != arrange(SCALE, PROG, seed=1, candidates=2, melody_engine="tune").melody


@pytest.mark.parametrize("style", ["strum", "arpeggio"])
def test_strum_and_arpeggio_ignore_the_engine_and_record_rules(tmp_path, fake_wav, style):
    tune = arrange(SCALE, PROG, style=style, seed=4, melody_engine="tune")
    assert tune == arrange(SCALE, PROG, style=style, seed=4, melody_engine="rules")
    c = compose(SCALE, PROG, style=style, seed=4, out_dir=tmp_path)
    assert c.metadata["melody_engine"] == "rules"
    assert c.melody_notes == tune.melody


@pytest.mark.parametrize("tuning", sorted(PRESETS))
@pytest.mark.parametrize(
    "key,mode", [("C", "major"), ("A", "minor"), ("F#", "minor"), ("G#", "major")]
)
def test_guitar_guarantees_hold_with_the_tune_engine(tuning, key, mode):
    scale = Scale(key, mode)
    numerals = ("I", "V", "vi", "IV") if mode == "major" else ("i", "VI", "III", "VII")
    prog = [realize(scale, n) for n in numerals]
    for seed in (1, 2, 3):
        arr = arrange(
            scale,
            prog,
            bars_per_chord=2,
            tempo_bpm=96.0,
            tuning=PRESETS[tuning],
            seed=seed,
            candidates=4,
            melody_engine="tune",
        )
        assert metrics.playable_moment_ratio((*arr.melody_tab, *arr.accompaniment_tab)) == 1.0
        assert metrics.melody_onset_on_top_ratio(arr.melody, arr.accompaniment) == 1.0
        assert metrics.tab_consistent(
            (*arr.melody, *arr.accompaniment),
            (*arr.melody_tab, *arr.accompaniment_tab),
            PRESETS[tuning],
        )
        assert metrics.forbidden_interval_count(arr.melody) == 0
        assert metrics.max_interval(arr.melody) <= 7


@pytest.mark.parametrize("tempo", [40.0, 200.0])
@pytest.mark.parametrize("chords", ["Cmaj7 Am7 Dm7 G7", "C Csus4 Caug Bdim"])
def test_extreme_tempos_and_chord_qualities(tempo, chords):
    prog = [ChordSymbol.parse(c) for c in chords.split()]
    arr = arrange(SCALE, prog, tempo_bpm=tempo, seed=None, candidates=2, melody_engine="tune")
    assert arr.melody


def test_tune_score_prefers_real_melody_statistics():
    cands = arrange_candidates(
        SCALE, PROG, bars_per_chord=2, tempo_bpm=96.0, seed=5, candidates=12, melody_engine="tune"
    )
    stats = load_tune_stats()
    dist = [
        band_distance(tune_features(a.melody, SCALE, PROG, 2, 96.0), stats.bands["major"])
        for a in cands
    ]
    assert dist[0] <= float(np.median(dist))
    assert [a.score for a in cands] == sorted((a.score for a in cands), reverse=True)


def test_tune_score_formula():
    """The tune score keeps v5's identity, above, shape-move and fallback terms and
    replaces the melody terms by -W_CORPUS x band distance (spec 5.4)."""
    assert W_CORPUS == 2.0
    arr = arrange(SCALE, PROG, bars_per_chord=2, tempo_bpm=96.0, seed=8, melody_engine="tune")
    kw: dict[str, Any] = {"bars_per_chord": 2, "tempo_bpm": 96.0, "tuning": STANDARD_EADGBE}
    identity = metrics.bar_identity(arr.accompaniment, PROG, 2, 96.0)
    above = metrics.melody_above_accompaniment_ratio(arr.melody, arr.accompaniment)
    moves = metrics.shape_moves(arr.bar_voicings, STANDARD_EADGBE)
    dist = band_distance(
        tune_features(arr.melody, SCALE, PROG, 2, 96.0), load_tune_stats().bands["major"]
    )
    want = (
        3.0 * sum(identity) / len(identity)
        + 1.0 * (1.0 if above is None else above)
        - 0.5 * (0.0 if moves is None else moves)
        - 3.0 * float(arr.final_fallback)
        - 2.0 * dist
    )
    got = score_arrangement(arr, PROG, scale=SCALE, melody_engine="tune", **kw)
    assert got == pytest.approx(want)
    assert arr.score == pytest.approx(want)
    # "rules" stays the default and does not need the scale.
    assert score_arrangement(arr, PROG, **kw) == score_arrangement(
        arr, PROG, scale=SCALE, melody_engine="rules", **kw
    )
    with pytest.raises(ValueError):
        score_arrangement(arr, PROG, melody_engine="tune", **kw)


def _cli_compose(tmp_path: Path, *extra: str) -> dict[str, Any]:
    r = CliRunner().invoke(
        main,
        [
            "compose",
            "--scale",
            "C:major",
            "--progression",
            "C G Am F",
            "--seed",
            "3",
            "--candidates",
            "2",
            "--format",
            "json",
            "--out",
            str(tmp_path),
            *extra,
        ],
    )
    assert r.exit_code == 0, r.output
    out: dict[str, Any] = json.loads(r.output)
    return out


def test_cli_flag(tmp_path, fake_wav):
    rules = _cli_compose(tmp_path / "r", "--melody-engine", "rules")
    assert rules["metadata"]["melody_engine"] == "rules"
    assert _cli_compose(tmp_path / "t")["metadata"]["melody_engine"] == "tune"
    tune = _cli_compose(tmp_path / "t2", "--melody-engine", "tune")
    assert tune["metadata"]["melody_engine"] == "tune"


def test_cli_rejects_unknown_engine(tmp_path):
    r = CliRunner().invoke(
        main,
        [
            "compose",
            "--scale",
            "C:major",
            "--progression",
            "C G",
            "--melody-engine",
            "jazz",
            "--out",
            str(tmp_path),
        ],
    )
    assert r.exit_code != 0


def test_unknown_engine_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="melody_engine"):
        arrange(SCALE, PROG, seed=1, melody_engine="jazz")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="melody_engine"):
        compose_with_options(SCALE, PROG, seed=1, out_dir=tmp_path, melody_engine="jazz")  # type: ignore[arg-type]
    assert list(tmp_path.iterdir()) == []
