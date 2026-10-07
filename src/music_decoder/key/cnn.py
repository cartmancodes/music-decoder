"""madmom CNN key classifier (Korzeniowski & Widmer, "Genre-Agnostic Key
Classification with Convolutional Neural Networks", ISMIR 2018)."""

from __future__ import annotations

from typing import Any, Literal, cast

import numpy as np

from music_decoder.dsp.tempwav import temp_wav
from music_decoder.types import KeyEstimate

_TONICS = ("A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#")
# madmom.features.key.KEY_LABELS order (A major .. G# minor), sharp spelling.
_LABELS: tuple[tuple[str, str], ...] = tuple((t, "major") for t in _TONICS) + tuple(
    (t, "minor") for t in _TONICS
)

_processor: Any = None


def probs_to_key_estimate(probs: np.ndarray[Any, np.dtype[Any]]) -> KeyEstimate:
    """24-way key probabilities (madmom order) → :class:`KeyEstimate`."""
    p = np.asarray(probs, dtype=float).reshape(-1)
    order = np.argsort(p)[::-1]
    tonic, mode = _LABELS[int(order[0])]
    return KeyEstimate(
        tonic=tonic,
        mode=cast(Literal["major", "minor"], mode),
        profile="cnn",
        correlation=float(p[order[0]]),
        margin=float(p[order[0]] - p[order[1]]),
    )


def _apply_madmom_compat() -> None:
    # Must run before any madmom import; kept in its own function so import
    # sorting can't reorder it below the madmom import.
    from music_decoder.chords import madmom_compat  # noqa: F401


def estimate_key_cnn(samples: np.ndarray[Any, np.dtype[Any]], sr: int) -> KeyEstimate:
    """Run the madmom CNN key ensemble on *samples*."""
    global _processor
    if _processor is None:
        _apply_madmom_compat()
        from madmom.features.key import CNNKeyRecognitionProcessor

        _processor = CNNKeyRecognitionProcessor()
    with temp_wav(samples, sr) as path:
        probs = _processor(str(path))
    return probs_to_key_estimate(probs)
