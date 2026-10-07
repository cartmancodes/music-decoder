"""Materialize in-memory samples as a temporary 16-bit WAV for file-based models (madmom)."""

from __future__ import annotations

import contextlib
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io.wavfile as wavfile


@contextlib.contextmanager
def temp_wav(samples: np.ndarray[Any, np.dtype[Any]], sr: int) -> Iterator[Path]:
    """Yield the path of a temp WAV holding *samples*; deleted on exit."""
    fd, name = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    path = Path(name)
    try:
        pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
        wavfile.write(str(path), sr, pcm)
        yield path
    finally:
        with contextlib.suppress(OSError):
            path.unlink()
