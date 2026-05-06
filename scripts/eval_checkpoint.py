#!/usr/bin/env python3
"""Evaluate a candidate basic-pitch checkpoint against the regression corpus.

Usage:
    python scripts/eval_checkpoint.py [--checkpoint path/to/checkpoint.npz]
                                       [--hp config/hyperparameters.yaml]
                                       [--report-out evaluation_reports/eval_<id>.json]

If no --checkpoint is given, evaluates the stock basic-pitch checkpoint
that ships with the package (i.e. the v1 baseline).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from music_decoder.logging_setup import configure_logging
from music_decoder.training.evaluate import (
    aggregate_metrics,
    default_fixtures,
    evaluate_checkpoint,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", default=None,
        help="Optional candidate basic-pitch checkpoint path. Defaults to stock.",
    )
    parser.add_argument(
        "--hp", default="config/hyperparameters.yaml",
        help="Hyperparameter YAML path",
    )
    parser.add_argument(
        "--report-out", default=None,
        help="Where to write the JSON report (default: evaluation_reports/eval_<TS>.json)",
    )
    args = parser.parse_args(argv)
    configure_logging("INFO")

    fixtures = default_fixtures()
    if not fixtures:
        print("No fixtures available — run `make fixtures` and ensure synthetic dir is populated.")
        return 1

    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    print(f"Evaluating {'stock checkpoint' if checkpoint is None else checkpoint}")
    print(f"Fixtures: {len(fixtures)} ({', '.join(f.name for f in fixtures)})")

    report = evaluate_checkpoint(
        checkpoint_path=checkpoint, hp_path=Path(args.hp), fixtures=fixtures,
    )
    summary = aggregate_metrics(report)

    print("\nPer-fixture metrics:")
    for r in report.per_fixture:
        print(
            f"  {r.name:30s} note_F={r.note_f_measure} "
            f"chord={r.chord_recognition_score}"
        )
    print("\nAggregate:")
    print(f"  note_f_measure        = {summary.note_f_measure}")
    print(f"  onset_f_measure       = {summary.onset_f_measure}")
    print(f"  pitch_class_accuracy  = {summary.pitch_class_accuracy}")
    print(f"  tab_string_accuracy   = {summary.tab_string_accuracy}")
    print(f"  chord_recognition     = {summary.chord_recognition_score}")

    out_path = (
        Path(args.report_out)
        if args.report_out
        else Path("evaluation_reports")
        / f"eval_{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "timestamp": datetime.now(UTC).isoformat(),
        "checkpoint": str(checkpoint) if checkpoint else "stock",
        "per_fixture": [asdict(r) for r in report.per_fixture],
        "aggregate": asdict(summary),
    }, indent=2))
    print(f"\nReport written to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
