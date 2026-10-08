#!/usr/bin/env python
"""Download a General MIDI soundfont for composition synthesis.

Thin wrapper around :func:`music_decoder.assets.fetch_soundfont`; prefer
``music-decoder fetch-models``, which also pre-downloads the Demucs weights.

By default installs GeneralUser GS (~30 MB) at the path configured in
``config/runtime.yaml`` (``~/.music-decoder/GeneralUser-GS.sf2``). Without
it, synthesis uses pretty_midi's bundled TimGM6mb.sf2.

    python scripts/download_soundfont.py
    python scripts/download_soundfont.py --url https://example.com/your.sf2 --dest /path/x.sf2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    from music_decoder import assets

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default=assets.GENERALUSER_GS_URL, help="soundfont URL")
    ap.add_argument("--dest", type=Path, default=None, help="target path (default: runtime config)")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args(argv)

    try:
        path = assets.fetch_soundfont(args.dest, args.url, force=args.force)
    except Exception as e:
        print(f"soundfont download failed: {e}", file=sys.stderr)
        return 1
    print(f"OK: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
