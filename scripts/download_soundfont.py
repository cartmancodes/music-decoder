#!/usr/bin/env python3
"""Download a small public-domain soundfont for synthetic-fixture rendering.

Once installed, the synthetic-fixture loader (Phase B-5) renders MIDI files
through fluidsynth + the soundfont, producing realistic instrument timbres
that are representative of what basic-pitch will see on real audio. Without
a soundfont the loader falls back to deterministic sine-wave synthesis.

The default target is TimGM6mb.sf2 — a 6MB MIT-licensed General MIDI bank.
The download mirror is the FluidSynth project's GitHub which hosts a few
classic public-domain SF2s for testing.

Usage:
    python scripts/download_soundfont.py
    python scripts/download_soundfont.py --output tests/fixtures/synthetic/soundfont/Custom.sf2
    python scripts/download_soundfont.py --url https://example.com/your.sf2

The script verifies the download is a real RIFF SF2 file (4-byte magic) and
not an HTML error page.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path


# Multiple mirrors in case one is down. Each entry is (URL, expected min size MB).
_DEFAULT_MIRRORS: tuple[tuple[str, int], ...] = (
    (
        "https://musical-artifacts.com/artifacts/2744/TimGM6mb.sf2",
        5,
    ),
    (
        # GitHub fallback: a known TimGM6mb mirror used by the music21 corpus.
        "https://github.com/musescore/MuseScore/raw/master/share/sound/Default.sf2",
        4,
    ),
)


def _download(url: str, target: Path, *, min_size_mb: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}\n  -> {target}")
    with urllib.request.urlopen(url) as resp:  # noqa: S310 — known mirror URLs
        data = resp.read()
    size_mb = len(data) / (1024 * 1024)
    if size_mb < min_size_mb:
        raise RuntimeError(
            f"download too small ({size_mb:.2f} MB); expected at least "
            f"{min_size_mb} MB. Got an error page instead of the SF2?"
        )
    if not data.startswith(b"RIFF") or b"sfbk" not in data[:64]:
        raise RuntimeError(
            f"download is not a valid SF2 file (no RIFF/sfbk header). "
            f"First 64 bytes: {data[:64]!r}"
        )
    target.write_bytes(data)
    print(f"OK: wrote {size_mb:.2f} MB to {target}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default="tests/fixtures/synthetic/soundfont/TimGM6mb.sf2",
        help="Where to write the SF2 file",
    )
    parser.add_argument(
        "--url", default=None,
        help="Custom soundfont URL (overrides built-in mirrors)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite if the target file exists",
    )
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists() and not args.force:
        print(f"Already exists: {output} (use --force to overwrite)")
        return 0

    mirrors = (
        ((args.url, 1),)
        if args.url else _DEFAULT_MIRRORS
    )
    last_error: Exception | None = None
    for url, min_size in mirrors:
        try:
            _download(url, output, min_size_mb=min_size)
            return 0
        except Exception as e:
            print(f"  failed: {e}")
            last_error = e
    print()
    print("All mirrors failed. Last error:", last_error)
    print(
        "You can supply your own soundfont URL via --url, or download a "
        "public-domain SF2 manually and place it at:"
    )
    print(f"  {output}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
