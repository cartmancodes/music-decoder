"""Print (and optionally save) composer musicality metrics.

    python scripts/compose_musicality_report.py
    python scripts/compose_musicality_report.py \
        --write tests/fixtures/compose/musicality_baseline.json
    python scripts/compose_musicality_report.py --matrices all \
        --write tests/fixtures/compose/musicality_v2.json

The committed baseline was written with ``--write`` on the pre-change
composer (commit recorded in the JSON's ``"source"``: ``69c2fa7``, the
last commit before the composer changed); re-running it now measures the
current composer on the same inputs. ``musicality_v2.json`` is the same
report for the v2 composer, written with ``--matrices all`` (it also holds the
holdout and reserve numbers); the default, ``development``, builds only the
validation, edge and tunings matrices. The recorded label comes from
``git describe --always --dirty``, so a run from a modified tree is marked
``-dirty``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from music_decoder.evaluation.compose_musicality import (
    ACCEPTANCE_MATRICES,
    ALL_MATRICES,
    DEVELOPMENT_MATRICES,
    RESERVE_MATRICES,
    evaluate,
    format_report,
)

MATRICES = {
    "development": DEVELOPMENT_MATRICES,
    "acceptance": ACCEPTANCE_MATRICES,
    "reserve": RESERVE_MATRICES,
    "all": ALL_MATRICES,
}


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "describe", "--always", "--dirty"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", type=Path, help="write the summary JSON here")
    parser.add_argument(
        "--matrices", choices=sorted(MATRICES), default="development", help="default: development"
    )
    args = parser.parse_args()
    report = evaluate(MATRICES[args.matrices])
    print(format_report(report))
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        payload = {"source": _git_head(), **report}
        args.write.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
        print(f"wrote {args.write}")


if __name__ == "__main__":
    main()
