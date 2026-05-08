"""Source separation adapters for the public ``analyze()`` pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np

from music_decoder.errors import SeparationError


def run_separation(
    samples: np.ndarray[Any, np.dtype[np.float32]],
    sr: int,
) -> tuple[np.ndarray[Any, np.dtype[np.float32]], int]:
    """Apply Demucs guitar isolation to raw samples.

    Adapter that wraps :func:`_apply_demucs` and returns ``(guitar_samples, sr)``.
    On any backend error, raises :class:`SeparationError` so callers can
    fall back to the original mix.
    """
    # Lazy import: demucs/torch are heavy, keep ``import
    # music_decoder.separation`` itself cheap.
    from music_decoder.separation.demucs import _apply_demucs

    try:
        guitar = _apply_demucs(samples, sr)
    except Exception as e:  # pragma: no cover — exercised in integration tests
        raise SeparationError(f"demucs_failed: {e}") from e
    return guitar, sr


__all__ = ["run_separation"]
