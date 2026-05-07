"""Public chord-detection entry point.

The actual algorithm lives in a backend (see ``backends/``). This module is a
thin dispatcher that picks the requested backend and validates the chord-
quality vocabulary.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from music_decoder.config.hyperparameters import ChordDetectionParams
from music_decoder.logging_setup import get_logger
from music_decoder.pipeline.contracts import (
    BeatGrid,
    ChordRecognitionResult,
)

from .backends.template_hmm import TemplateHmmBackend
from .templates import QUALITIES

_log = get_logger("chord_detection.api")


def detect_chords(
    *,
    chroma: np.ndarray[Any, np.dtype[Any]],
    sr: int,
    hop_length: int,
    beat_grid: BeatGrid,
    params: ChordDetectionParams,
    audio_path: Path | None = None,
) -> ChordRecognitionResult:
    # Guard against config drift: the hardcoded QUALITIES tuple in templates.py
    # is what actually drives the 48 chord templates. If hyperparameters.yaml
    # is edited to a different list, fail loudly rather than silently ignoring.
    if tuple(params.qualities) != QUALITIES:
        raise ValueError(
            f"chord_detection.qualities {params.qualities!r} does not match "
            f"the templated vocabulary {list(QUALITIES)!r}; v1 supports only "
            "the templated qualities. Edit templates.py to extend the vocabulary."
        )

    from .backends.base import ChordBackend

    backend_name = params.backend
    backend: ChordBackend
    if backend_name == "madmom_deep_chroma":
        try:
            from .backends.madmom_deep_chroma import MadmomDeepChromaBackend
            backend = MadmomDeepChromaBackend()
        except ImportError as e:
            _log.warning(
                "madmom_unavailable_falling_back",
                extra={"error": str(e)},
            )
            backend = TemplateHmmBackend()
    else:
        backend = TemplateHmmBackend()

    result: ChordRecognitionResult = backend.detect(
        chroma=chroma,
        sr=sr,
        hop_length=hop_length,
        beat_grid=beat_grid,
        params=params,
        audio_path=audio_path,
    )
    return result
