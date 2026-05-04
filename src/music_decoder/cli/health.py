from __future__ import annotations

import shutil
from pathlib import Path


class FfmpegMissing(RuntimeError):
    pass


def check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise FfmpegMissing(
            "ffmpeg is required but not on PATH. Install it:\n"
            "  macOS:    brew install ffmpeg\n"
            "  Ubuntu:   apt-get install ffmpeg\n"
            "  Windows:  https://ffmpeg.org/download.html"
        )


def ensure_data_dirs(
    *, db_path: Path, artifact_dir: Path, model_cache: Path,
) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_cache.mkdir(parents=True, exist_ok=True)
