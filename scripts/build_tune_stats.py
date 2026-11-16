"""Build the composer's melody statistics from POP909.

    python scripts/build_tune_stats.py \
        --pop909 ~/.music-decoder/datasets/pop909-hsa/POP909 \
        --out src/music_decoder/compose/data/tune_stats.json [--pilot FILE]

``--pop909`` is the ``POP909`` folder of the hierarchical-structure release
(one folder per song). Only the derived statistics are written; ``--pilot``
is a JSON file stored under ``"pilot"`` as it is.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from music_decoder.compose.tune_corpus import Song, build_stats, read_song
from music_decoder.compose.tune_metrics import FEATURES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pop909", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pilot", type=Path)
    args = parser.parse_args()

    folders = sorted(p for p in args.pop909.expanduser().iterdir() if p.is_dir())
    songs: list[Song] = []
    skipped: list[str] = []
    for folder in folders:
        song = read_song(folder)
        if song is None:
            skipped.append(folder.name)
        else:
            songs.append(song)
    pilot = json.loads(args.pilot.read_text()) if args.pilot else None
    stats = build_stats(songs, pilot)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(stats, indent=1) + "\n", encoding="utf-8")

    meta = stats["meta"]
    print(f"songs: {meta['songs']} read, {len(skipped)} skipped {skipped}")
    print(f"windows: {meta['windows']}")
    n_patterns = len({p for counts in stats["rhythm"].values() for p in counts})
    print(f"patterns kept: {n_patterns}")
    print("chord_tone: " + ", ".join(f"{c} {v:.3f}" for c, v in stats["chord_tone"].items()))
    nps = stats["notes_per_second"]
    print(f"notes_per_second: {nps['low']:.3f} {nps['mid']:.3f} {nps['high']:.3f}")
    for mode, bands in stats["bands"].items():
        print(f"bands[{mode}]  (25th / 50th / 75th)")
        for feature in FEATURES:
            if feature in bands:
                b = bands[feature]
                print(f"  {feature:<18} {b['low']:8.3f} {b['mid']:8.3f} {b['high']:8.3f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
