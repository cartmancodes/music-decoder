# src/music_decoder/audio_io/ffmpeg.py
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def find_ffmpeg(explicit: Path | None = None) -> Path:
    if explicit and explicit.exists():
        return explicit
    found = shutil.which("ffmpeg")
    if found is None:
        raise FileNotFoundError("ffmpeg binary not found on PATH")
    return Path(found)


def decode_to_wav(
    src: Path, dst: Path, *, sr: int, ffmpeg: Path | None = None,
) -> None:
    """Decode any audio file to a mono PCM WAV at the requested sample rate."""
    binary = find_ffmpeg(ffmpeg)
    cmd = [
        str(binary), "-y", "-loglevel", "error",
        "-i", str(src), "-ac", "1", "-ar", str(sr),
        "-acodec", "pcm_s16le", str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed ({proc.returncode}): {proc.stderr.decode(errors='replace')}"
        )
