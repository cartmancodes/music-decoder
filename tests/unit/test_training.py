"""Tests for the basic-pitch fine-tuning harness scaffold (Phase C-3)."""
from __future__ import annotations

from pathlib import Path

import pytest

from music_decoder.training.config import (
    DatasetSpec,
    FineTuneConfig,
    load_finetune_config,
)
from music_decoder.training.dataset import enumerate_pairs
from music_decoder.training.evaluate import aggregate_metrics
from music_decoder.training.finetune import dry_run, execute_training


def _write_config(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(content)
    return p


def test_load_finetune_config_minimal(tmp_path: Path):
    p = _write_config(tmp_path, """
id: test
output_checkpoint_dir: /tmp/checkpoints
train:
  name: guitarset
  cache_dir: /data/guitarset
  track_ids: [a, b]
val:
  name: guitarset
  cache_dir: /data/guitarset
  track_ids: [c]
""")
    cfg = load_finetune_config(p)
    assert isinstance(cfg, FineTuneConfig)
    assert cfg.id == "test"
    assert cfg.train.track_ids == ["a", "b"]
    assert cfg.val.track_ids == ["c"]
    assert cfg.epochs == 5  # default
    assert cfg.batch_size == 4
    assert cfg.learning_rate == 1e-4


def test_load_finetune_config_overrides(tmp_path: Path):
    p = _write_config(tmp_path, """
id: deep
output_checkpoint_dir: /tmp/c
base_checkpoint: /existing/checkpoint.npz
train:
  name: guitarset
  cache_dir: /data
  track_ids: [a]
val:
  name: guitarset
  cache_dir: /data
  track_ids: [b]
epochs: 12
batch_size: 8
learning_rate: 0.0005
sample_rate_hz: 44100
""")
    cfg = load_finetune_config(p)
    assert cfg.epochs == 12
    assert cfg.batch_size == 8
    assert cfg.learning_rate == 0.0005
    assert cfg.sample_rate_hz == 44100
    assert cfg.base_checkpoint == Path("/existing/checkpoint.npz")


def test_load_finetune_config_missing_id_rejected(tmp_path: Path):
    p = _write_config(tmp_path, "epochs: 5\n")
    with pytest.raises(ValueError, match="must have a top-level 'id'"):
        load_finetune_config(p)


def test_load_finetune_config_missing_train_rejected(tmp_path: Path):
    p = _write_config(tmp_path, """
id: x
output_checkpoint_dir: /tmp
val:
  name: guitarset
  cache_dir: /data
  track_ids: [b]
""")
    with pytest.raises(ValueError, match="'train' and 'val'"):
        load_finetune_config(p)


def test_enumerate_pairs_finds_existing_guitarset_tracks():
    """When the GuitarSet cache exists, the enumerator yields real pairs."""
    cache = Path("tests/fixtures/guitarset")
    if not (cache / "audio_mono-mic").exists():
        pytest.skip("GuitarSet cache not present")
    spec = DatasetSpec(
        name="guitarset", cache_dir=cache,
        track_ids=["00_BN1-129-Eb_comp"],
    )
    pairs = list(enumerate_pairs(spec))
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair.audio_path.exists()
    assert pair.jams_path.exists()


def test_enumerate_pairs_skips_missing_tracks(tmp_path: Path):
    spec = DatasetSpec(
        name="guitarset", cache_dir=tmp_path,
        track_ids=["nonexistent_track"],
    )
    assert list(enumerate_pairs(spec)) == []


def test_enumerate_pairs_unknown_dataset_raises(tmp_path: Path):
    spec = DatasetSpec(name="alien", cache_dir=tmp_path, track_ids=["x"])
    with pytest.raises(ValueError, match="unknown dataset"):
        list(enumerate_pairs(spec))


def test_dry_run_reports_estimate_and_pairs():
    """Dry-run validates the config and reports estimated wall-clock."""
    cache = Path("tests/fixtures/guitarset")
    if not (cache / "audio_mono-mic").exists():
        pytest.skip("GuitarSet cache not present")
    cfg = FineTuneConfig(
        id="test-dryrun",
        base_checkpoint=None,
        output_checkpoint_dir=Path("/tmp/checkpoints"),
        train=DatasetSpec(
            name="guitarset", cache_dir=cache,
            track_ids=["00_BN1-129-Eb_comp", "00_BN1-129-Eb_solo"],
        ),
        val=DatasetSpec(
            name="guitarset", cache_dir=cache,
            track_ids=["00_Funk1-114-Ab_comp"],
        ),
        epochs=3, batch_size=4, learning_rate=1e-4, sample_rate_hz=22050,
    )
    report = dry_run(cfg)
    assert len(report.train_pairs) == 2
    assert len(report.val_pairs) == 1
    # 2 tracks * 3 epochs * 10 sec/track-epoch = 60s = 1.0 min
    assert report.estimated_wall_clock_min == pytest.approx(1.0, abs=0.01)
    assert report.output_checkpoint_path == Path("/tmp/checkpoints/test-dryrun.npz")


def test_dry_run_warns_when_train_dataset_empty(tmp_path: Path):
    cfg = FineTuneConfig(
        id="empty",
        base_checkpoint=None,
        output_checkpoint_dir=tmp_path,
        train=DatasetSpec(name="guitarset", cache_dir=tmp_path, track_ids=["x"]),
        val=DatasetSpec(name="guitarset", cache_dir=tmp_path, track_ids=["y"]),
        epochs=1, batch_size=4, learning_rate=1e-4, sample_rate_hz=22050,
    )
    report = dry_run(cfg)
    assert any("0 pairs" in n for n in report.notes)


def test_execute_training_requires_legacy_keras_env(tmp_path: Path, monkeypatch):
    """Without TF_USE_LEGACY_KERAS=1, execute_training should refuse cleanly.

    basic-pitch's training code uses Keras-2 idioms that don't work under
    Keras 3 (TF 2.16+); the wrapper enforces this prerequisite.
    """
    monkeypatch.delenv("TF_USE_LEGACY_KERAS", raising=False)
    cfg = FineTuneConfig(
        id="x",
        base_checkpoint=None,
        output_checkpoint_dir=tmp_path,
        train=DatasetSpec(name="guitarset", cache_dir=tmp_path, track_ids=[]),
        val=DatasetSpec(name="guitarset", cache_dir=tmp_path, track_ids=[]),
        epochs=1, batch_size=4, learning_rate=1e-4, sample_rate_hz=22050,
    )
    with pytest.raises(RuntimeError, match=r"TF_USE_LEGACY_KERAS=1"):
        execute_training(cfg, tfrecord_source=tmp_path)


def test_basic_pitch_compat_apply_is_idempotent():
    """Applying the shim twice should not error or duplicate work."""
    from music_decoder.training import basic_pitch_compat
    basic_pitch_compat.apply_basic_pitch_shims()
    basic_pitch_compat.apply_basic_pitch_shims()  # second call is a no-op


def test_aggregate_metrics_handles_partial_nones():
    """If some fixtures have null metrics, the aggregator skips them."""
    from music_decoder.evaluation.runner import EvaluationReport, FixtureMetrics

    rows = [
        FixtureMetrics(
            name="a", source="manual",
            note_f_measure=0.5, onset_f_measure=None,
            pitch_class_accuracy=0.6, key_mirex_score=None,
            tab_string_accuracy=None, chord_recognition_score=0.7,
        ),
        FixtureMetrics(
            name="b", source="manual",
            note_f_measure=0.7, onset_f_measure=0.8,
            pitch_class_accuracy=None, key_mirex_score=None,
            tab_string_accuracy=None, chord_recognition_score=0.9,
        ),
    ]
    report = EvaluationReport(timestamp="t", per_fixture=rows)
    summary = aggregate_metrics(report)
    assert summary.note_f_measure == pytest.approx(0.6)
    assert summary.onset_f_measure == 0.8        # only one non-None
    assert summary.pitch_class_accuracy == 0.6   # only one non-None
    assert summary.key_mirex_score is None
    assert summary.tab_string_accuracy is None
    assert summary.chord_recognition_score == pytest.approx(0.8)
