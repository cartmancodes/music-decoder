"""Print (and optionally save) composer musicality metrics.

    python scripts/compose_musicality_report.py
    python scripts/compose_musicality_report.py \
        --write tests/fixtures/compose/musicality_baseline.json

The committed baseline was written with ``--write`` on the pre-change
composer (commit recorded in the JSON's ``"source"``); re-running it now
measures the current composer on the same inputs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from music_decoder.evaluation.compose_musicality import evaluate


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", type=Path, help="write the summary JSON here")
    args = parser.parse_args()
    report = evaluate()
    for matrix, groups in report.items():
        print(f"== {matrix}")
        for key, stats in groups["overall"].items():
            print(f"  {key:16s} " + "  ".join(f"{k}={v:g}" for k, v in stats.items()))
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        payload = {"source": _git_head(), **report}
        args.write.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
        print(f"wrote {args.write}")


if __name__ == "__main__":
    main()
