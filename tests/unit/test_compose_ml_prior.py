"""v5 feature 5: the soft pitch prior and the melody-proposer hook (spec 5.5).

Every test here uses a fake proposer: no network, no ``ml`` extra, no weights.
"""

from __future__ import annotations

import subprocess
import sys
from unittest import mock

import pytest
from click.testing import CliRunner

from music_decoder import assets
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


def _no_amt_weights(monkeypatch):
    monkeypatch.setattr(assets, "amt_dir", lambda: None)


def test_amt_without_weights_explains_how_to_fetch(monkeypatch):
    pytest.importorskip("anticipation")
    pytest.importorskip("transformers")
    _no_amt_weights(monkeypatch)
    with pytest.raises(MLUnavailableError, match=r"fetch-models --composer-models"):
        amt.load_proposer()


def test_amt_without_the_package_explains_how_to_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "anticipation", None)  # importing it now fails
    with pytest.raises(MLUnavailableError, match=r"pip install -e \"\.\[ml\]\""):
        amt.load_proposer()


def _note(start, pitch, dur=0.25):
    return Note(start_s=start, end_s=start + dur, pitch=pitch, velocity=72, confidence=1.0)


def test_amt_top_line_keeps_the_highest_note_per_onset():
    decoded = [
        (0, _note(0.00, 60)),
        (0, _note(0.02, 67)),  # within 30 ms of 0.00: same onset, higher
        (0, _note(0.01, 64)),
        (128, _note(0.50, 80)),  # drums never count
        (25, _note(0.50, 62)),
        (0, _note(0.51, 90)),  # above the range
        (0, _note(1.00, 40)),  # below the range
        (33, _note(1.50, 84)),
        (0, _note(1.52, 55)),
    ]
    line = amt.top_line(decoded)
    assert [(n.start_s, n.pitch) for n in line] == [(0.02, 67), (0.50, 62), (1.50, 84)]
    assert amt.top_line([]) == []


def test_amt_top_line_groups_onsets_by_the_first_of_the_group():
    decoded = [(0, _note(0.00, 60)), (0, _note(0.025, 62)), (0, _note(0.05, 64))]
    line = amt.top_line(decoded)  # 0.05 is 50 ms after the group's first onset
    assert [n.pitch for n in line] == [62, 64]
    assert line[0].end_s == 0.05  # cut short at the next onset: one note at a time
    assert line[1].end_s == 0.30


def test_amt_controls_encode_the_context_as_steel_guitar():
    pytest.importorskip("anticipation")
    from anticipation import convert
    from anticipation.vocab import CONTROL_OFFSET

    context = [_note(0.5, 48, dur=0.475), _note(1.0, 55, dur=0.475)]
    controls = amt.encode_controls(context)
    assert len(controls) == 6 and all(tok >= CONTROL_OFFSET for tok in controls)
    compound = convert.events_to_compound([tok - CONTROL_OFFSET for tok in controls])
    assert compound[0::5] == [50, 100]  # 10 ms ticks
    assert compound[1::5] == [48, 48]
    assert compound[2::5] == [48, 55]
    assert compound[3::5] == [amt.INSTRUMENT, amt.INSTRUMENT] == [25, 25]


def test_amt_decode_round_trips_events():
    pytest.importorskip("anticipation")
    from anticipation.vocab import CONTROL_OFFSET

    controls = amt.encode_controls([_note(0.5, 67, dur=0.5), _note(1.0, 72, dur=0.25)])
    decoded = amt.decode_events([tok - CONTROL_OFFSET for tok in controls])
    assert [(i, n.start_s, n.end_s, n.pitch) for i, n in decoded] == [
        (25, 0.5, 1.0, 67),
        (25, 1.0, 1.25, 72),
    ]
    assert amt.decode_events([]) == []


class _FlatLM:  # every next token equally likely
    def __init__(self, torch, vocab_size):
        self.torch, self.vocab_size = torch, vocab_size
        self.device = torch.device("cpu")

    def __call__(self, tokens):
        logits = self.torch.zeros(1, tokens.shape[1], self.vocab_size)
        return type("Out", (), {"logits": logits})()


def _constrained_logits(tokens, *, max_events=10):
    torch = pytest.importorskip("torch")
    pytest.importorskip("anticipation")
    from anticipation.vocab import VOCAB_SIZE

    model = amt._Constrained(_FlatLM(torch, VOCAB_SIZE), max_events=max_events)
    assert model.device == torch.device("cpu")
    return model, lambda: model(torch.tensor([tokens])).logits[0, -1]


def test_amt_samples_only_pitched_notes_and_none_twice_at_one_onset():
    pytest.importorskip("anticipation")
    from anticipation.config import MAX_PITCH
    from anticipation.vocab import (
        ANTICIPATE,
        ATIME_OFFSET,
        DUR_OFFSET,
        NOTE_OFFSET,
        REST,
        TIME_OFFSET,
    )

    guitar_g4 = NOTE_OFFSET + 25 * MAX_PITCH + 67
    flute_c5 = NOTE_OFFSET + 73 * MAX_PITCH + 72
    control = [ATIME_OFFSET + 30, ATIME_OFFSET + 5, ATIME_OFFSET + 1]
    tokens = [
        ANTICIPATE,
        *[TIME_OFFSET + 30, DUR_OFFSET + 10, guitar_g4],
        *control,  # skipped over
        *[TIME_OFFSET + 30, DUR_OFFSET + 20, flute_c5],
        *[TIME_OFFSET + 30, DUR_OFFSET + 10],  # this event's note comes next
    ]
    _, logits = _constrained_logits(tokens)
    out = logits()
    drums = NOTE_OFFSET + 128 * MAX_PITCH
    assert torch_isinf(out[REST]) and torch_isinf(out[drums : drums + MAX_PITCH]).all()
    assert torch_isinf(out[guitar_g4]) and torch_isinf(out[flute_c5])
    assert int(torch_isinf(out).sum()) == MAX_PITCH + 3
    # A later onset: nothing generated there yet, so only rests and drums are out.
    tokens[-2] = TIME_OFFSET + 40
    assert int(torch_isinf(logits()).sum()) == MAX_PITCH + 1


def test_amt_stops_generating_after_its_event_budget():
    pytest.importorskip("anticipation")
    from anticipation.config import MAX_TIME
    from anticipation.vocab import ANTICIPATE, TIME_OFFSET

    _, logits = _constrained_logits([ANTICIPATE], max_events=2)
    for _ in range(2):  # within the budget: the time slot is left alone
        assert not torch_isinf(logits()).any()
    out = logits()  # spent: the only onset left is the latest there is
    assert not torch_isinf(out[TIME_OFFSET + MAX_TIME - 1])
    assert int(torch_isinf(out).sum()) == out.shape[0] - 1


def torch_isinf(x):
    import torch

    return torch.isinf(x)


def _long_context():  # 48 bars at 60 bpm: 192 s, past AMT's 100 s of onsets
    prog = [ChordSymbol.parse(c) for c in ("C", "Am", "F", "G")] * 12
    bars = choose_bar_shapes(prog, bars_per_chord=1, tuning=STANDARD_EADGBE)
    return _thumb_context(bars, STANDARD_EADGBE, tempo_bpm=60.0)


def _generate_call(context, *, end_s):
    """What ``AMTProposer.propose`` hands ``sample.generate``, with a flat fake model."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("anticipation")
    from anticipation import sample
    from anticipation.vocab import VOCAB_SIZE

    seen = {}

    def generate(model, *, start_time, end_time, controls, top_p):
        seen.update(end_time=end_time, controls=list(controls), max_events=model._events_left)
        return []

    proposer = amt.AMTProposer(_FlatLM(torch, VOCAB_SIZE))
    with mock.patch.object(sample, "generate", generate):
        assert proposer.propose(context, end_s=end_s, tempo_bpm=60.0, seed=1) == []
    return seen


def test_amt_controls_of_a_long_piece_stay_in_the_control_ranges():
    pytest.importorskip("anticipation")
    from anticipation.vocab import ADUR_OFFSET, ANOTE_OFFSET, ATIME_OFFSET, SPECIAL_OFFSET

    context = _long_context()
    assert context[-1].start_s > 100
    controls = _generate_call(context, end_s=192.0)["controls"]
    assert controls and len(controls) % 3 == 0
    assert all(ATIME_OFFSET <= tok < ADUR_OFFSET for tok in controls[0::3])
    assert all(ADUR_OFFSET <= tok < ANOTE_OFFSET for tok in controls[1::3])
    assert all(ANOTE_OFFSET <= tok < SPECIAL_OFFSET for tok in controls[2::3])


def test_amt_proposes_for_the_first_40_s_and_sees_no_context_past_it():
    pytest.importorskip("anticipation")
    from anticipation.vocab import CONTROL_OFFSET

    context = _long_context()
    seen = _generate_call(context, end_s=192.0)
    assert seen["end_time"] == amt.AMT_MAX_SECONDS == 40.0
    assert seen["max_events"] == 4 * 40  # the event budget follows the capped length
    decoded = amt.decode_events([tok - CONTROL_OFFSET for tok in seen["controls"]])
    assert len(decoded) == sum(1 for n in context if n.start_s < 40.0)
    assert max(n.end_s for _, n in decoded) <= 40.0


def test_amt_a_short_piece_is_proposed_whole():
    context = [n for n in _long_context() if n.end_s <= 16.0]  # within the first 16 s
    seen = _generate_call(context, end_s=16.0)
    assert seen["end_time"] == 16.0
    assert seen["controls"] == amt.encode_controls(context)  # nothing clipped


def test_amt_clip_context_cuts_notes_at_the_end():
    notes = [_note(0.0, 48, dur=1.0), _note(39.5, 50, dur=1.0), _note(40.0, 52), _note(50, 55)]
    clipped = amt.clip_context(notes, 40.0)
    assert [(n.start_s, n.end_s, n.pitch) for n in clipped] == [(0.0, 1.0, 48), (39.5, 40.0, 50)]
    assert amt.proposal_seconds(192.0) == 40.0 and amt.proposal_seconds(19.2) == 19.2


def test_importing_amt_loads_no_ml_package():
    code = (
        "import sys, music_decoder.compose.ml.amt; "
        "print(sorted(m for m in ('anticipation', 'transformers', 'huggingface_hub', 'tqdm') "
        "if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_fetch_amt_downloads_the_pinned_snapshot(tmp_path):
    pytest.importorskip("huggingface_hub")
    with mock.patch("huggingface_hub.snapshot_download", return_value=str(tmp_path)) as dl:
        assert assets.fetch_amt() == tmp_path
    assert dl.call_args.args == ("stanford-crfm/music-small-800k",)
    assert dl.call_args.kwargs["revision"] == assets.AMT_REVISION
    assert len(assets.AMT_REVISION) == 40
    assert dl.call_args.kwargs["token"] is False
    assert set(dl.call_args.kwargs["allow_patterns"]) == {"config.json", "pytorch_model.bin"}


def test_amt_dir_is_none_when_a_file_is_missing(tmp_path):
    pytest.importorskip("huggingface_hub")
    (tmp_path / "config.json").write_text("{}")
    with mock.patch("huggingface_hub.snapshot_download", return_value=str(tmp_path)) as dl:
        assert assets.amt_dir() is None
        (tmp_path / "pytorch_model.bin").write_bytes(b"x")
        assert assets.amt_dir() == tmp_path
    assert dl.call_args.kwargs["local_files_only"] is True


def test_fetch_models_composer_models_fetches_amt(tmp_path):
    (tmp_path / "pytorch_model.bin").write_bytes(b"x" * 3_000_000)
    with (
        mock.patch("music_decoder.assets.fetch_audiobox", return_value=tmp_path),
        mock.patch("music_decoder.assets.fetch_amt", return_value=tmp_path) as fetch,
    ):
        r = CliRunner().invoke(
            main, ["fetch-models", "--no-demucs", "--no-soundfont", "--composer-models"]
        )
    assert r.exit_code == 0, r.output
    fetch.assert_called_once_with()
    assert "stanford-crfm/music-small-800k" in r.output and "3.0 MB" in r.output


def test_fetch_models_amt_failure_exits_nonzero(tmp_path):
    with (
        mock.patch("music_decoder.assets.fetch_audiobox", return_value=tmp_path),
        mock.patch("music_decoder.assets.fetch_amt", side_effect=OSError("offline")),
    ):
        r = CliRunner().invoke(
            main, ["fetch-models", "--no-demucs", "--no-soundfont", "--composer-models"]
        )
    assert r.exit_code == 1 and "offline" in r.output


def test_melody_model_amt_loads_the_proposer(monkeypatch, tmp_path):
    prop = _Fixed(_line([64, 65, 67, 69, 71, 72, 71, 69] * 4))
    monkeypatch.setattr(amt, "load_proposer", lambda: prop)
    r = compose_with_options(
        Scale("C", "major"), PROG8, seed=1, out_dir=tmp_path, candidates=2, melody_model="amt"
    )
    assert r.metadata["melody_model"] == "amt"
    expected = arrange(Scale("C", "major"), PROG8, seed=1, candidates=2, melody_proposer=prop)
    assert r.melody_notes == expected.melody


def test_melody_model_records_the_seconds_it_covers(monkeypatch, tmp_path):
    prop = _Fixed(_line([64, 65, 67]))
    monkeypatch.setattr(amt, "load_proposer", lambda: prop)
    r = compose_with_options(
        Scale("C", "major"), PROG8, seed=1, out_dir=tmp_path, candidates=1, melody_model="amt"
    )
    assert r.metadata["melody_model_seconds"] == pytest.approx(8 * 2.4)  # 8 bars at 100 bpm
    prog4 = PROG8[:4]
    r = compose_with_options(
        Scale("C", "major"),
        prog4,
        bars_per_chord=4,
        tempo_bpm=60.0,
        seed=1,
        out_dir=tmp_path,
        candidates=1,
        melody_model="amt",
    )
    assert r.metadata["melody_model_seconds"] == 40.0  # 16 bars at 60 bpm is 64 s
    r = compose_with_options(Scale("C", "major"), PROG8, seed=1, out_dir=tmp_path, candidates=1)
    assert r.metadata["melody_model_seconds"] is None


def test_melody_model_unavailable_fails_before_writing(monkeypatch, tmp_path):
    _no_amt_weights(monkeypatch)
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


def _cli_result(tmp_path, metadata):
    return mock.Mock(
        midi_path=tmp_path / "a.mid",
        wav_path=tmp_path / "a.wav",
        ascii_tab="e|---",
        melody_notes=(),
        chord_voicings=(),
        metadata=metadata,
    )


def test_cli_says_when_the_melody_model_covers_only_the_start(tmp_path):
    note = "note: the melody model guides the first 40 s"
    with mock.patch("music_decoder.cli.main.compose_with_options") as c:
        c.return_value = _cli_result(tmp_path, {"melody_model_seconds": 40.0})
        args = _compose_args(tmp_path, "--melody-model", "amt", "--tempo", "60", "--bars", "8")
        r = CliRunner().invoke(main, args)  # 16 bars at 60 bpm: 64 s
        assert r.exit_code == 0, r.output
        assert note in r.stderr and note not in r.stdout
        c.return_value = _cli_result(tmp_path, {"melody_model_seconds": 4.8})
        r = CliRunner().invoke(main, _compose_args(tmp_path, "--melody-model", "amt"))
        assert r.exit_code == 0 and "melody model" not in r.stderr  # 2 bars: covered whole


def test_cli_melody_model_unavailable_is_a_clear_error(tmp_path, monkeypatch):
    _no_amt_weights(monkeypatch)
    r = CliRunner().invoke(main, _compose_args(tmp_path, "--melody-model", "amt"))
    assert r.exit_code == 1
    assert 'pip install -e ".[ml]"' in r.output
