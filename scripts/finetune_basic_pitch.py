#!/usr/bin/env python3
"""Fine-tune basic-pitch on a configured dataset (Phase C-3 scaffold).

Usage:
    python scripts/finetune_basic_pitch.py --config configs/finetune_guitarset.yaml --dry-run
    python scripts/finetune_basic_pitch.py --config configs/finetune_guitarset.yaml --execute

The default mode is --dry-run: validates the config, enumerates the
dataset, reports the estimated wall-clock, and exits without touching any
model. Use --execute (consciously) to attempt the actual training run.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from music_decoder.logging_setup import configure_logging, get_logger
from music_decoder.training.config import load_finetune_config
from music_decoder.training.finetune import dry_run, execute_training


_log = get_logger("scripts.finetune")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True,
        help="Path to fine-tune YAML config (e.g. configs/finetune_guitarset.yaml)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=True,
        help="Validate config + enumerate dataset (default).",
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="Actually run training. Refuses if --dry-run also set.",
    )
    parser.add_argument(
        "--tfrecord-source", default="/tmp/bp_tfrecords",
        help=(
            "Parent dir of <dataset_name>/splits/{train,validation,test}/*.tfrecord. "
            "Generate with scripts/convert_guitarset_to_tfrecord.py. Required "
            "for --execute."
        ),
    )
    parser.add_argument(
        "--steps-per-epoch", type=int, default=100,
        help="Number of training batches per epoch.",
    )
    parser.add_argument(
        "--validation-steps", type=int, default=5,
        help="Number of validation batches per epoch.",
    )
    args = parser.parse_args(argv)
    configure_logging("INFO")

    config = load_finetune_config(Path(args.config))
    print(f"Loaded fine-tune config: id={config.id}")
    print(f"  output: {config.output_checkpoint_dir}")
    print(
        f"  train: {config.train.name} {len(config.train.track_ids)} tracks @ "
        f"{config.train.cache_dir}"
    )
    print(
        f"  val:   {config.val.name} {len(config.val.track_ids)} tracks @ "
        f"{config.val.cache_dir}"
    )
    print(
        f"  epochs={config.epochs}, batch_size={config.batch_size}, "
        f"lr={config.learning_rate}"
    )

    report = dry_run(config)
    print()
    print(f"Train pairs found: {len(report.train_pairs)}")
    print(f"Val pairs found:   {len(report.val_pairs)}")
    print(
        f"Estimated wall-clock (CPU): "
        f"{report.estimated_wall_clock_min:.1f} minutes"
    )
    print(f"Would write checkpoint to: {report.output_checkpoint_path}")
    if report.notes:
        print("Notes:")
        for n in report.notes:
            print(f"  - {n}")

    if args.execute:
        from pathlib import Path as _P
        try:
            ckpt = execute_training(
                config,
                tfrecord_source=_P(args.tfrecord_source),
                steps_per_epoch=args.steps_per_epoch,
                validation_steps=args.validation_steps,
            )
        except RuntimeError as e:
            print()
            print("Training prerequisite missing:")
            print(str(e))
            return 1
        except NotImplementedError as e:
            print()
            print("Training execution is not yet wired up:")
            print(str(e))
            return 1
        print(f"\nWrote checkpoint: {ckpt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
