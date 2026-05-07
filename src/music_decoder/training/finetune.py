"""Fine-tuning harness wrapper around basic-pitch's training package.

This module is the v1 scaffolding for Phase C-3 of the accuracy roadmap. It
DOES NOT actually invoke training in this commit — the spec defers that to
the user, since training takes hours of CPU time and may need GPU. Instead,
the harness:

1. Validates a ``FineTuneConfig``.
2. Enumerates the train + val datasets and confirms files exist.
3. Reports an estimated wall-clock budget based on the dataset size.
4. Optionally launches basic-pitch's training entry point (gated behind
   --execute so a stray invocation doesn't accidentally start a multi-hour
   training run).

When the user runs ``--execute``, the wrapper invokes basic-pitch's
``basic_pitch.training.train()`` function (or its current equivalent) with
the dataset paths converted to basic-pitch's expected format. We don't
re-implement the training loop — basic-pitch owns it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import FineTuneConfig
from .dataset import TrainingPair, enumerate_pairs


@dataclass(frozen=True)
class DryRunReport:
    """What ``finetune`` would do if executed."""

    train_pairs: list[TrainingPair]
    val_pairs: list[TrainingPair]
    estimated_wall_clock_min: float
    output_checkpoint_path: Path
    notes: list[str]


# Empirical estimate: basic-pitch fine-tuning on CPU runs roughly 10
# seconds per (track, epoch) pair on a 30-second GuitarSet track at the
# default batch size. GPU is much faster but we're conservative.
_CPU_SECONDS_PER_TRACK_EPOCH = 10.0


def dry_run(config: FineTuneConfig) -> DryRunReport:
    """Validate the config and report what would happen at full execution."""
    train_pairs = list(enumerate_pairs(config.train))
    val_pairs = list(enumerate_pairs(config.val))
    notes: list[str] = []
    if not train_pairs:
        notes.append(
            "WARNING: train dataset enumerated 0 pairs; check cache_dir + track_ids"
        )
    if not val_pairs:
        notes.append(
            "WARNING: val dataset enumerated 0 pairs; held-out evaluation will be skipped"
        )
    if config.train.name == "manual":
        notes.append(
            "NOTE: manual fixtures use our JSON schema, not JAMS; basic-pitch "
            "cannot train on them directly. Use them only for evaluation."
        )

    estimated_seconds = (
        len(train_pairs) * config.epochs * _CPU_SECONDS_PER_TRACK_EPOCH
    )

    output_path = (
        config.output_checkpoint_dir / f"{config.id}.npz"
    )

    return DryRunReport(
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        estimated_wall_clock_min=estimated_seconds / 60.0,
        output_checkpoint_path=output_path,
        notes=notes,
    )


def execute_training(
    config: FineTuneConfig,
    *,
    tfrecord_source: Path,
    steps_per_epoch: int = 100,
    validation_steps: int = 5,
    shuffle_size: int = 100,
) -> Path:
    """Invoke basic-pitch's training loop with the configured dataset.

    Prerequisites the caller must satisfy:

    1. ``basic_pitch`` and ``tensorflow`` are installed.
    2. ``tf-keras`` is installed and ``TF_USE_LEGACY_KERAS=1`` is set in the
       environment. basic-pitch's training code uses Keras-2-style symbolic
       tensor ops that don't work under Keras 3 (TF 2.16+).
    3. ``tfrecord_source`` is the parent directory of
       ``<dataset_name>/splits/{train,validation,test}/*.tfrecord``. Generate
       it with ``scripts/convert_guitarset_to_tfrecord.py``.

    Returns the path to the ``model.best`` checkpoint directory written by
    the trainer.
    """
    import os

    if os.environ.get("TF_USE_LEGACY_KERAS") != "1":
        raise RuntimeError(
            "TF_USE_LEGACY_KERAS=1 must be set in the environment before "
            "importing tensorflow. Re-launch with: "
            "TF_USE_LEGACY_KERAS=1 python scripts/finetune_basic_pitch.py ..."
        )

    # Apply our shims BEFORE importing basic_pitch.train (which imports the
    # affected layers).
    from . import basic_pitch_compat  # noqa: F401  side effect: apply shims

    try:
        import basic_pitch.train as bp_train
    except ImportError as e:
        raise NotImplementedError(
            "basic_pitch.train not importable. Install with "
            "`pip install basic-pitch[tf]`. Original error: " + str(e)
        ) from e

    config.output_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    bp_train.main(
        source=str(tfrecord_source),
        output=str(config.output_checkpoint_dir),
        batch_size=config.batch_size,
        shuffle_size=shuffle_size,
        learning_rate=config.learning_rate,
        epochs=config.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        size_evaluation_callback_datasets=2,
        datasets_to_use=[config.train.name],
        dataset_sampling_frequency=[1.0],
        no_sonify=True,
        no_contours=False,
        weighted_onset_loss=False,
        positive_onset_weight=1.0,
    )

    # basic_pitch.train writes <output>/<timestamp>/model.best (a directory of
    # TF SavedModel files). The latest timestamp is the run we just kicked off.
    runs = sorted(
        d for d in config.output_checkpoint_dir.iterdir()
        if d.is_dir() and (d / "model.best").exists()
    )
    if not runs:
        raise RuntimeError(
            f"basic_pitch.train completed but produced no model.best under "
            f"{config.output_checkpoint_dir}; check the trainer logs."
        )
    return runs[-1] / "model.best"
