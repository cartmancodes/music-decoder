"""Train the composer's phrase model on POP909 and write its weights file.

    python scripts/train_phrase_model.py ~/.music-decoder/datasets/pop909-hsa/POP909 \
        [--out src/music_decoder/compose/data/phrase_model.npz] [--steps 1500]

The settings below come from the 2026-11-25 spike (spikes/phrase-model/findings.json):
the network size with the best validation bits, 1,500 steps, copy-check length 11 and
sampling temperature and nucleus 1.0. Takes about 13 minutes on an Apple-silicon GPU.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from music_decoder.compose.phrase_model import WEIGHTS_PATH
from music_decoder.compose.phrase_train import build

ARCH = {"d": 128, "layers": 6, "heads": 4}
STEPS = 1500
SEED = 0
FIGURE_LENGTH = 11
TEMPERATURE = 1.0
NUCLEUS = 1.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("pop909", type=Path)
    ap.add_argument("--out", type=Path, default=WEIGHTS_PATH)
    ap.add_argument("--steps", type=int, default=STEPS)
    args = ap.parse_args()
    meta = build(
        args.pop909.expanduser(),
        args.out,
        arch=ARCH,
        steps=args.steps,
        seed=SEED,
        temperature=TEMPERATURE,
        nucleus=NUCLEUS,
        figure_length=FIGURE_LENGTH,
    )
    print(json.dumps(asdict(meta), indent=1))
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
