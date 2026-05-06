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


def execute_training(config: FineTuneConfig) -> Path:
    """Invoke basic-pitch's training loop with the configured dataset.

    This function attempts to import basic-pitch's training package and
    invoke its ``train`` entry point. If the package isn't available (the
    pip-installed basic-pitch may ship inference-only), the function raises
    ``NotImplementedError`` with a clear message pointing at the workflow
    doc.

    NOTE: this is the deferred path. The C-3 spec ships the harness; users
    actually run training out-of-band when they have time.
    """
    raise NotImplementedError(
        "basic-pitch training is not wired up in this scaffold. See "
        "docs/training.md for the manual workflow: clone spotify/basic-pitch, "
        "run their training package directly with the dataset spec, then "
        "drop the resulting checkpoint into the model_cache_dir and use "
        "evaluate_checkpoint() to measure the delta.\n\n"
        f"Config id: {config.id}\n"
        f"Train pairs target: {len(list(enumerate_pairs(config.train)))}\n"
        f"Val pairs target: {len(list(enumerate_pairs(config.val)))}"
    )
