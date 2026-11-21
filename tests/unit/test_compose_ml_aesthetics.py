"""v5 feature 4: the opt-in audio re-rank with Audiobox Aesthetics (spec 5.4).

Every test here uses a fake scorer: no network, no ``ml`` extra, no weights.
"""

from __future__ import annotations

import random
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from unittest import mock

import pytest
from click.testing import CliRunner

from music_decoder import assets
from music_decoder.cli.main import main
from music_decoder.compose.api import arrange, arrange_candidates, compose_with_options
from music_decoder.compose.ml import MLUnavailableError, aesthetics
from music_decoder.compose.ml.aesthetics import rank_by_audio
from music_decoder.types import ChordSymbol, Scale

PROG8 = [ChordSymbol.parse(c) for c in ("C", "Am", "F", "G", "C", "F", "G", "C")]
PROG4 = [ChordSymbol.parse(c) for c in ("C", "Am", "F", "G")]

base_arr = arrange(Scale("C", "major"), PROG4, tempo_bpm=120.0, seed=1)


class _FakeScorer:
    def __init__(self, by_name):
        self.by_name = by_name

    def score(self, wav_paths):
        return [self.by_name[p.name] for p in wav_paths]


class _PositionalScorer:
    """Scores the i-th WAV with ``rows[i]``; remembers the paths it was given."""

    def __init__(self, rows):
        self.rows = rows
        self.seen: list[Path] = []

    def score(self, wav_paths):
        self.seen = list(wav_paths)
        assert all(p.exists() and p.stat().st_size > 0 for p in self.seen)
        return self.rows[: len(self.seen)]


def _raise_unavailable():
    raise MLUnavailableError("audiobox_aesthetics is not installed")


def _fail_if_called():
    raise AssertionError("no model may load for strum or arpeggio")


def _row(ce, pq=6.0):
    return {"CE": ce, "CU": 5.0, "PC": 5.0, "PQ": pq}


def test_rank_by_audio_blends_audio_and_symbolic(tmp_path):
    arrs = [replace(base_arr, score=s) for s in (5.0, 4.0, 3.0, 2.0)]
    wavs = [tmp_path / f"{k}.wav" for k in range(4)]
    scorer = _FakeScorer(
        {w.name: _row(ce) for w, ce in zip(wavs, (5.0, 5.0, 8.0, 5.0), strict=True)}
    )
    # x (5.5, 5.5, 7.0, 5.5) / 0.3 + 0.5 * s / 0.995: 20.8, 20.3, 24.8, 19.8
    assert rank_by_audio(arrs, wavs, scorer) == 2


def test_rank_by_audio_an_audio_gap_over_the_floor_decides_against_the_symbolic_order(tmp_path):
    arrs = [replace(base_arr, score=s) for s in (4.0, 3.0, 2.0, 1.0)]
    wavs = [tmp_path / f"{k}.wav" for k in range(4)]
    means = (6.00, 6.05, 6.40, 6.02)  # span 0.4, over the 0.3 floor
    scorer = _FakeScorer({w.name: _row(m, pq=m) for w, m in zip(wavs, means, strict=True)})
    # x / 0.3 (20.0, 20.2, 21.3, 20.1) + 0.5 * s / 0.995 (2.0, 1.5, 1.0, 0.5):
    # totals 22.0, 21.7, 22.3, 20.6
    assert rank_by_audio(arrs, wavs, scorer) == 2
    assert aesthetics.SYMBOLIC_WEIGHT == 0.5


def test_rank_by_audio_a_gap_under_the_floor_keeps_the_symbolic_order(tmp_path):
    arrs = [replace(base_arr, score=s) for s in (4.0, 3.0, 2.0, 1.0)]
    wavs = [tmp_path / f"{k}.wav" for k in range(4)]
    means = (6.00, 6.02, 6.08, 6.01)  # span 0.08: about the model's noise
    scorer = _FakeScorer({w.name: _row(m, pq=m) for w, m in zip(wavs, means, strict=True)})
    # x / 0.3 (20.0, 20.07, 20.27, 20.03) + 0.5 * s / 0.995 (2.0, 1.5, 1.0, 0.5):
    # totals 22.0, 21.6, 21.3, 20.5
    assert rank_by_audio(arrs, wavs, scorer) == 0
    assert aesthetics.AUDIO_SPAN_FLOOR == 0.3


def test_choose_by_audio_a_gap_under_the_floor_keeps_the_symbolic_lead():
    arrs = [replace(base_arr, score=s) for s in (aesthetics.SYMBOLIC_SCALE, 0.0)]
    # a 0.1 audio gap is a third of the floor: 0.33 loses to the symbolic lead of one
    # SYMBOLIC_SCALE, which is worth SYMBOLIC_WEIGHT = 0.5
    assert aesthetics.AUDIO_SCALE >= aesthetics.AUDIO_SPAN_FLOOR
    assert aesthetics.choose_by_audio(arrs, [_row(6.0, pq=6.0), _row(6.1, pq=6.1)]) == 0
    # a 0.2 gap, 0.67, beats it
    assert aesthetics.choose_by_audio(arrs, [_row(6.0, pq=6.0), _row(6.2, pq=6.2)]) == 1


def test_a_dominated_candidate_does_not_flip_the_order():
    arrs = [replace(base_arr, score=s) for s in (1.0, 0.0)]
    rows = [_row(6.1, pq=6.1), _row(6.2, pq=6.2)]
    assert aesthetics.choose_by_audio(arrs, rows) == 0
    worse = [*arrs, replace(base_arr, score=-10.0)]
    assert aesthetics.choose_by_audio(worse, [*rows, _row(6.0, pq=6.0)]) == 0


def test_appending_a_dominated_candidate_never_reorders():
    rng = random.Random(0)
    for _ in range(200):
        n = rng.randint(2, 4)
        xs = [rng.uniform(4, 8) for _ in range(n)]
        ss = [rng.uniform(-2, 2) for _ in range(n)]
        arrs = [replace(base_arr, score=s) for s in ss]
        rows = [_row(x, pq=x) for x in xs]
        before = aesthetics.choose_by_audio(arrs, rows)
        # strictly below every existing candidate on both scores
        bad_x = min(xs) - rng.uniform(0.01, 3)
        bad_s = min(ss) - rng.uniform(0.01, 20)
        after = aesthetics.choose_by_audio(
            [*arrs, replace(base_arr, score=bad_s)], [*rows, _row(bad_x, pq=bad_x)]
        )
        assert after == before


def test_equal_scores_pick_the_first():
    arrs = [replace(base_arr, score=3.0) for _ in range(3)]
    assert aesthetics.choose_by_audio(arrs, [_row(5.0, pq=5.0)] * 3) == 0


def test_rank_by_audio_a_larger_symbolic_gap_keeps_the_symbolic_best(tmp_path):
    arrs = [replace(base_arr, score=s) for s in (9.0, 1.0, 0.0)]
    wavs = [tmp_path / f"{k}.wav" for k in range(3)]
    means = (6.0, 6.1, 5.0)
    scorer = _FakeScorer({w.name: _row(m, pq=m) for w, m in zip(wavs, means, strict=True)})
    # z_audio (0.91, 1, 0), z_symbolic * 0.5 (0.5, 0.06, 0): 1.41, 1.06, 0
    assert rank_by_audio(arrs, wavs, scorer) == 0


def test_rank_by_audio_symbolic_breaks_an_audio_tie(tmp_path):
    arrs = [replace(base_arr, score=s) for s in (2.0, 5.0, 3.0)]
    wavs = [tmp_path / f"{k}.wav" for k in range(3)]
    scorer = _FakeScorer({w.name: _row(5.0) for w in wavs})
    assert rank_by_audio(arrs, wavs, scorer) == 1


def test_rank_by_audio_ties_go_to_the_lowest_index(tmp_path):
    arrs = [replace(base_arr, score=3.0) for _ in range(3)]  # all equal: z is 0
    wavs = [tmp_path / f"{k}.wav" for k in range(3)]
    scorer = _FakeScorer({w.name: _row(4.0 if w.name == "0.wav" else 6.0) for w in wavs})
    assert rank_by_audio(arrs, wavs, scorer) == 1


def test_ml_unavailable_error_names_both_steps():
    err = MLUnavailableError("weights missing")
    assert isinstance(err, RuntimeError)
    assert 'pip install -e ".[ml]"' in str(err)
    assert "music-decoder fetch-models --composer-models" in str(err)
    assert "weights missing" in str(err)


def test_load_scorer_without_the_package_raises(monkeypatch):
    for name in [m for m in sys.modules if m.startswith("audiobox_aesthetics")] + [
        "audiobox_aesthetics"
    ]:
        monkeypatch.setitem(sys.modules, name, None)  # importing it now fails
    with pytest.raises(MLUnavailableError, match=r"fetch-models --composer-models"):
        aesthetics.load_scorer()


def test_load_scorer_without_the_weights_raises(monkeypatch):
    pytest.importorskip("audiobox_aesthetics")
    monkeypatch.setattr(assets, "audiobox_dir", lambda: None)
    with pytest.raises(MLUnavailableError, match=r"pip install -e \"\.\[ml\]\""):
        aesthetics.load_scorer()


def test_rank_audio_without_the_extra_explains_how_to_install(monkeypatch, tmp_path):
    monkeypatch.setattr(aesthetics, "load_scorer", _raise_unavailable)
    with pytest.raises(
        MLUnavailableError, match=r"pip install -e \"\.\[ml\]\".*fetch-models --composer-models"
    ):
        compose_with_options(Scale("C", "major"), PROG8, seed=1, out_dir=tmp_path, rank_audio=True)
    assert list(tmp_path.iterdir()) == []  # fails before writing anything


def test_rank_audio_is_fingerstyle_only(monkeypatch, tmp_path):
    monkeypatch.setattr(aesthetics, "load_scorer", _fail_if_called)
    r = compose_with_options(
        Scale("C", "major"), PROG8, style="strum", seed=1, out_dir=tmp_path, rank_audio=True
    )
    assert r.metadata["rank_audio"] is False
    assert r.metadata["audio_scores"] is None
    assert r.metadata["candidates"] == 1


def test_default_compose_does_not_rank_audio(monkeypatch, tmp_path):
    monkeypatch.setattr(aesthetics, "load_scorer", _fail_if_called)
    r = compose_with_options(Scale("C", "major"), PROG4, seed=1, out_dir=tmp_path, candidates=2)
    assert r.metadata["rank_audio"] is False and r.metadata["audio_scores"] is None


def test_rank_audio_writes_only_the_chosen_candidate(monkeypatch, tmp_path):
    scorer = _PositionalScorer([_row(0.0), _row(0.0), _row(10.0, pq=9.0), _row(0.0)])
    monkeypatch.setattr(aesthetics, "load_scorer", lambda: scorer)
    r = compose_with_options(
        Scale("C", "major"), PROG8, seed=1, out_dir=tmp_path, rank_audio=True, melody_engine="rules"
    )

    cands = arrange_candidates(Scale("C", "major"), PROG8, seed=1, candidates=12)
    third = cands[2]
    assert len(scorer.seen) == 4  # the top K = 4 by symbolic score
    assert r.metadata["rank_audio"] is True
    assert r.metadata["chosen_candidate"] == third.chosen_candidate
    assert r.metadata["score"] == third.score
    assert r.metadata["candidates"] == 12
    assert r.metadata["audio_scores"] == {"CE": 10.0, "PQ": 9.0}
    assert r.melody_notes == third.melody

    assert sorted(p.name for p in r.midi_path.parent.iterdir()) == [
        "composition.json",
        "composition.mid",
        "composition.wav",
    ]
    assert not any(p.exists() for p in scorer.seen)  # no temporary WAV is left behind
    assert not any(p.parent.exists() for p in scorer.seen)


def test_rank_audio_with_fewer_candidates_than_k(monkeypatch, tmp_path):
    scorer = _PositionalScorer([_row(1.0), _row(9.0)])
    monkeypatch.setattr(aesthetics, "load_scorer", lambda: scorer)
    r = compose_with_options(
        Scale("C", "major"),
        PROG4,
        seed=1,
        out_dir=tmp_path,
        rank_audio=True,
        candidates=2,
        melody_engine="rules",
    )
    cands = arrange_candidates(Scale("C", "major"), PROG4, seed=1, candidates=2)
    assert len(scorer.seen) == 2
    assert r.metadata["chosen_candidate"] == cands[1].chosen_candidate


def test_rank_audio_plucked_renders_candidates_with_the_plucked_sound(monkeypatch, tmp_path):
    scorer = _PositionalScorer([_row(9.0), _row(1.0)])
    monkeypatch.setattr(aesthetics, "load_scorer", lambda: scorer)
    with mock.patch("music_decoder.compose.api.render_wav") as soundfont:
        r = compose_with_options(
            Scale("C", "major"),
            PROG4,
            seed=1,
            out_dir=tmp_path,
            rank_audio=True,
            candidates=2,
            sound="plucked",
        )
    soundfont.assert_not_called()
    assert r.wav_path.stat().st_size > 0


def test_arrange_candidates_are_sorted_and_first_is_arrange():
    cands = arrange_candidates(Scale("C", "major"), PROG8, tempo_bpm=120.0, seed=4, candidates=12)
    assert len(cands) == 12
    assert [c.score for c in cands] == sorted((c.score for c in cands), reverse=True)
    assert cands[0] == arrange(Scale("C", "major"), PROG8, tempo_bpm=120.0, seed=4, candidates=12)
    assert sorted(c.chosen_candidate for c in cands) == list(range(12))
    assert all(c.candidates == 12 for c in cands)


def test_arrange_candidates_for_strum_is_the_single_arrangement():
    cands = arrange_candidates(Scale("C", "major"), PROG4, style="strum", seed=4, candidates=12)
    assert cands == (arrange(Scale("C", "major"), PROG4, style="strum", seed=4),)


def test_importing_compose_loads_no_ml_package():
    code = (
        "import sys, music_decoder.compose.api, music_decoder.cli.main, "
        "music_decoder.compose.ml.aesthetics; "
        "print(sorted(m for m in ('audiobox_aesthetics', 'transformers', 'huggingface_hub') "
        "if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


# --- assets ---------------------------------------------------------------


def test_fetch_audiobox_downloads_the_snapshot(tmp_path):
    pytest.importorskip("huggingface_hub")
    with mock.patch("huggingface_hub.snapshot_download", return_value=str(tmp_path)) as dl:
        assert assets.fetch_audiobox() == tmp_path
    assert dl.call_args.args == (assets.AUDIOBOX_REPO,)
    assert assets.AUDIOBOX_REPO == "facebook/audiobox-aesthetics"
    assert "model.safetensors" in dl.call_args.kwargs["allow_patterns"]


def test_audiobox_cached_is_false_when_the_snapshot_is_missing():
    pytest.importorskip("huggingface_hub")
    from huggingface_hub.errors import LocalEntryNotFoundError

    with mock.patch(
        "huggingface_hub.snapshot_download", side_effect=LocalEntryNotFoundError("missing")
    ) as dl:
        assert assets.audiobox_cached() is False
    assert dl.call_args.kwargs["local_files_only"] is True


def test_audiobox_cached_is_false_without_huggingface_hub(monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    assert assets.audiobox_cached() is False


def test_audiobox_cached_is_false_when_a_file_is_missing(tmp_path):
    pytest.importorskip("huggingface_hub")
    (tmp_path / "config.json").write_text("{}")
    with mock.patch("huggingface_hub.snapshot_download", return_value=str(tmp_path)):
        assert assets.audiobox_cached() is False


def test_audiobox_cached_is_true_when_the_weights_are_there(tmp_path):
    pytest.importorskip("huggingface_hub")
    (tmp_path / "model.safetensors").write_bytes(b"x")
    (tmp_path / "config.json").write_text("{}")
    with mock.patch("huggingface_hub.snapshot_download", return_value=str(tmp_path)):
        assert assets.audiobox_cached() is True
        assert assets.audiobox_dir() == tmp_path


# --- CLI ------------------------------------------------------------------


def test_fetch_models_composer_models_fetches_audiobox(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"x" * 2_000_000)
    with (
        mock.patch("music_decoder.assets.fetch_demucs"),
        mock.patch("music_decoder.assets.fetch_soundfont", return_value=tmp_path / "x.sf2"),
        mock.patch("music_decoder.assets.fetch_audiobox", return_value=tmp_path) as audiobox,
        mock.patch("music_decoder.assets.fetch_amt", return_value=tmp_path),
    ):
        r = CliRunner().invoke(main, ["fetch-models", "--composer-models"])
    assert r.exit_code == 0, r.output
    audiobox.assert_called_once_with()
    assert "Audiobox Aesthetics" in r.output and "2.0 MB" in r.output


def test_fetch_models_skips_composer_models_by_default(tmp_path):
    with (
        mock.patch("music_decoder.assets.fetch_demucs"),
        mock.patch("music_decoder.assets.fetch_soundfont", return_value=tmp_path / "x.sf2"),
        mock.patch("music_decoder.assets.fetch_audiobox") as audiobox,
        mock.patch("music_decoder.assets.fetch_amt") as fetch_amt,
    ):
        r = CliRunner().invoke(main, ["fetch-models"])
    assert r.exit_code == 0, r.output
    audiobox.assert_not_called()
    fetch_amt.assert_not_called()


def test_fetch_models_composer_models_failure_exits_nonzero(tmp_path):
    with (
        mock.patch("music_decoder.assets.fetch_audiobox", side_effect=OSError("offline")),
        mock.patch("music_decoder.assets.fetch_amt", return_value=tmp_path),
    ):
        r = CliRunner().invoke(
            main, ["fetch-models", "--no-demucs", "--no-soundfont", "--composer-models"]
        )
    assert r.exit_code == 1 and "offline" in r.output


def test_fetch_models_composer_models_without_the_extra_names_it():
    missing = ImportError("No module named 'huggingface_hub'")
    with (
        mock.patch("music_decoder.assets.fetch_audiobox", side_effect=missing),
        mock.patch("music_decoder.assets.fetch_amt", side_effect=missing),
    ):
        r = CliRunner().invoke(
            main, ["fetch-models", "--no-demucs", "--no-soundfont", "--composer-models"]
        )
    assert r.exit_code == 1 and 'pip install -e ".[ml]"' in r.output


def _compose_args(tmp_path, *extra):
    return [
        "compose",
        "--scale",
        "C:major",
        "--progression",
        "C G",
        "--out",
        str(tmp_path),
        *extra,
    ]


def test_cli_rank_audio_is_passed_through(tmp_path):
    with mock.patch("music_decoder.cli.main.compose_with_options") as c:
        c.return_value = mock.Mock(
            midi_path=tmp_path / "a.mid",
            wav_path=tmp_path / "a.wav",
            ascii_tab="e|---",
            melody_notes=(),
            chord_voicings=(),
            metadata={},
        )
        r = CliRunner().invoke(main, _compose_args(tmp_path, "--rank-audio"))
        assert r.exit_code == 0, r.output
        assert c.call_args.kwargs["rank_audio"] is True
        r = CliRunner().invoke(main, _compose_args(tmp_path))
        assert c.call_args.kwargs["rank_audio"] is False


def test_cli_rank_audio_without_the_extra_is_a_clear_error(tmp_path):
    with mock.patch(
        "music_decoder.cli.main.compose_with_options",
        side_effect=MLUnavailableError("audiobox_aesthetics is not installed"),
    ):
        r = CliRunner().invoke(main, _compose_args(tmp_path, "--rank-audio"))
    assert r.exit_code == 1
    assert 'pip install -e ".[ml]"' in r.output
    assert "fetch-models --composer-models" in r.output
