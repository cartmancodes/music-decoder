"""v5 feature 5: the soft pitch prior and the melody-proposer hook (spec 5.5).

Every test here uses a fake proposer: no network, no ``ml`` extra, no weights.
"""

from __future__ import annotations

from unittest import mock

import pytest
from click.testing import CliRunner

from music_decoder.cli.main import main
from music_decoder.compose import metrics
from music_decoder.compose.api import _thumb_context, arrange, compose_with_options
from music_decoder.compose.fingerstyle import hand_shape
from music_decoder.compose.melody import generate_melody
from music_decoder.compose.ml import MelodyProposer, MLUnavailableError, amt
from music_decoder.compose.shapes import choose_bar_shapes
from music_decoder.tabs.tuning import STANDARD_EADGBE
from music_decoder.types import ChordSymbol, Note, Scale

PROG8 = [ChordSymbol.parse(c) for c in ("C", "Am", "F", "G", "C", "F", "G", "C")]


class _Fixed:  # proposes a fixed line regardless of context
    def __init__(self, notes):
        self.notes = notes

    def propose(self, context, *, end_s, tempo_bpm, seed):
        return list(self.notes)


def _line(pitches, step=0.5):
    return [
        Note(start_s=k * step, end_s=(k + 1) * step, pitch=p, velocity=80, confidence=1.0)
        for k, p in enumerate(pitches)
    ]


def _fail_if_called():
    raise AssertionError("no model may load for strum or arpeggio")


def test_prior_off_is_unchanged():
    kw = dict(scale=Scale("C", "major"), progression=PROG8, tempo_bpm=120.0, seed=3)
    assert generate_melody(**kw) == generate_melody(**kw, pitch_prior=None)


def test_prior_pulls_pitch_classes_toward_the_proposal():
    target = _line([67] * 32)  # G4 for 16 s
    hits_with = hits_without = 0
    for seed in range(20):
        a = generate_melody(
            scale=Scale("C", "major"), progression=PROG8, tempo_bpm=120.0, seed=seed
        )
        b = generate_melody(
            scale=Scale("C", "major"),
            progression=PROG8,
            tempo_bpm=120.0,
            seed=seed,
            pitch_prior=target,
        )
        hits_without += sum(n.pitch % 12 == 7 for n in a)
        hits_with += sum(n.pitch % 12 == 7 for n in b)
        assert metrics.forbidden_interval_count(b) == 0
    assert hits_with > 1.3 * hits_without


def test_prior_is_octave_aware():
    far = _line([67 + 24] * 32)  # G6, far above any reach set
    near = _line([67] * 32)
    kw = dict(scale=Scale("C", "major"), progression=PROG8, tempo_bpm=120.0, seed=1)
    a = generate_melody(**kw, pitch_prior=far)
    b = generate_melody(**kw, pitch_prior=near)
    assert [n.pitch % 12 for n in a] == [n.pitch % 12 for n in b]


def test_an_empty_or_unreachable_prior_changes_nothing_or_little():
    a = arrange(Scale("C", "major"), PROG8, tempo_bpm=120.0, seed=2)
    b = arrange(Scale("C", "major"), PROG8, tempo_bpm=120.0, seed=2, melody_proposer=_Fixed([]))
    assert a.melody == b.melody


def test_a_proposal_changes_the_fingerstyle_melody():
    prop = _Fixed(_line([64, 65, 67, 69, 71, 72, 71, 69] * 4))
    a = arrange(Scale("C", "major"), PROG8, tempo_bpm=120.0, seed=2)
    b = arrange(Scale("C", "major"), PROG8, tempo_bpm=120.0, seed=2, melody_proposer=prop)
    assert a.melody != b.melody


def test_fingerstyle_with_a_proposer_keeps_every_guarantee():
    prop = _Fixed(_line([64, 65, 67, 69, 71, 72, 71, 69] * 4))
    for seed in range(10):
        a = arrange(
            Scale("C", "major"),
            PROG8,
            tempo_bpm=120.0,
            seed=seed,
            candidates=4,
            melody_proposer=prop,
        )
        part = [*a.melody_tab, *a.accompaniment_tab]
        assert metrics.playable_moment_ratio(part) == 1.0
        assert metrics.melody_onset_on_top_ratio(a.melody, a.accompaniment) == 1.0
        assert metrics.final_chord_rings(a.accompaniment, 8, 120.0)
        assert metrics.forbidden_interval_count(a.melody) == 0


def test_the_proposal_is_made_once_per_piece():
    calls = []

    class _Count(_Fixed):
        def propose(self, context, **kw):
            calls.append(kw["seed"])
            return super().propose(context, **kw)

    arrange(Scale("C", "major"), PROG8, seed=5, candidates=12, melody_proposer=_Count([]))
    assert calls == [5]


def test_the_proposer_sees_the_thumb_pattern_and_the_piece_length():
    seen = {}

    class _Spy(_Fixed):
        def propose(self, context, *, end_s, tempo_bpm, seed):
            seen.update(context=tuple(context), end_s=end_s, tempo_bpm=tempo_bpm)
            return super().propose(context, end_s=end_s, tempo_bpm=tempo_bpm, seed=seed)

    arrange(Scale("C", "major"), PROG8, tempo_bpm=120.0, seed=5, melody_proposer=_Spy([]))
    bars = choose_bar_shapes(PROG8, bars_per_chord=1, tuning=STANDARD_EADGBE)
    assert seen["context"] == _thumb_context(bars, STANDARD_EADGBE, tempo_bpm=120.0)
    assert seen["end_s"] == pytest.approx(16.0) and seen["tempo_bpm"] == 120.0


def test_without_a_seed_the_proposal_is_still_made_once():
    calls = []

    class _Count(_Fixed):
        def propose(self, context, **kw):
            calls.append(kw["seed"])
            return super().propose(context, **kw)

    arrange(Scale("C", "major"), PROG8, candidates=2, melody_proposer=_Count([]))
    assert len(calls) == 1 and isinstance(calls[0], int)


def test_thumb_context_is_root_on_beats_1_and_3_alternate_on_2_and_4():
    bars = choose_bar_shapes(PROG8, bars_per_chord=1, tuning=STANDARD_EADGBE)
    ctx = _thumb_context(bars, STANDARD_EADGBE, tempo_bpm=120.0)
    assert len(ctx) == 4 * len(bars)
    open_pitches = STANDARD_EADGBE.open_pitches
    for b, voicing in enumerate(bars):
        hand = hand_shape(voicing, STANDARD_EADGBE)
        root = open_pitches[hand.root.string] + hand.root.fret
        alt = open_pitches[hand.alternate.string] + hand.alternate.fret
        bar = ctx[4 * b : 4 * b + 4]
        assert [n.pitch for n in bar] == [root, alt, root, alt]
        assert [n.start_s for n in bar] == pytest.approx([2.0 * b + 0.5 * k for k in range(4)])
        assert all(n.end_s > n.start_s for n in bar)
    assert list(ctx) == sorted(ctx, key=lambda n: n.start_s)


def test_a_proposer_is_a_melody_proposer():
    assert isinstance(_Fixed([]), MelodyProposer)


def test_the_amt_stub_is_unavailable():
    with pytest.raises(MLUnavailableError):
        amt.load_proposer()


def test_melody_model_amt_loads_the_proposer(monkeypatch, tmp_path):
    prop = _Fixed(_line([64, 65, 67, 69, 71, 72, 71, 69] * 4))
    monkeypatch.setattr(amt, "load_proposer", lambda: prop)
    r = compose_with_options(
        Scale("C", "major"), PROG8, seed=1, out_dir=tmp_path, candidates=2, melody_model="amt"
    )
    assert r.metadata["melody_model"] == "amt"
    expected = arrange(Scale("C", "major"), PROG8, seed=1, candidates=2, melody_proposer=prop)
    assert r.melody_notes == expected.melody


def test_melody_model_unavailable_fails_before_writing(monkeypatch, tmp_path):
    with pytest.raises(MLUnavailableError):
        compose_with_options(
            Scale("C", "major"), PROG8, seed=1, out_dir=tmp_path, melody_model="amt"
        )
    assert list(tmp_path.iterdir()) == []


def test_melody_model_is_fingerstyle_only(tmp_path, monkeypatch):
    monkeypatch.setattr(amt, "load_proposer", _fail_if_called)
    r = compose_with_options(
        Scale("C", "major"), PROG8, style="strum", seed=1, out_dir=tmp_path, melody_model="amt"
    )
    assert r.metadata["melody_model"] is None


def test_default_compose_has_no_melody_model(tmp_path, monkeypatch):
    monkeypatch.setattr(amt, "load_proposer", _fail_if_called)
    r = compose_with_options(Scale("C", "major"), PROG8, seed=1, out_dir=tmp_path, candidates=2)
    assert r.metadata["melody_model"] is None


def _compose_args(tmp_path, *extra):
    return ["compose", "--scale", "C:major", "--progression", "C G", "--out", str(tmp_path), *extra]


def test_cli_accepts_melody_model_none(tmp_path):
    with mock.patch("music_decoder.cli.main.compose_with_options") as c:
        c.return_value = mock.Mock(
            midi_path=tmp_path / "a.mid",
            wav_path=tmp_path / "a.wav",
            ascii_tab="e|---",
            melody_notes=(),
            chord_voicings=(),
            metadata={},
        )
        r = CliRunner().invoke(main, _compose_args(tmp_path, "--melody-model", "none"))
        assert r.exit_code == 0, r.output
        assert c.call_args.kwargs["melody_model"] is None
        r = CliRunner().invoke(main, _compose_args(tmp_path))
        assert c.call_args.kwargs["melody_model"] is None
        r = CliRunner().invoke(main, _compose_args(tmp_path, "--melody-model", "amt"))
        assert r.exit_code == 0, r.output
        assert c.call_args.kwargs["melody_model"] == "amt"
    r = CliRunner().invoke(main, _compose_args(tmp_path, "--melody-model", "gpt"))
    assert r.exit_code != 0


def test_cli_melody_model_unavailable_is_a_clear_error(tmp_path):
    r = CliRunner().invoke(main, _compose_args(tmp_path, "--melody-model", "amt"))
    assert r.exit_code == 1
    assert 'pip install -e ".[ml]"' in r.output
