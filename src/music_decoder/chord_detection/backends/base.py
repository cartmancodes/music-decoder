"""Backend protocol for chord recognition.

The chord-detection module supports multiple recognition backends behind a
common interface. v1 ships two:

- ``template_hmm`` — chroma + 49 templates + Viterbi (pure NumPy).
- ``madmom_deep_chroma`` — madmom's pre-trained DeepChromaProcessor
  + DeepChromaChordRecognitionProcessor.

Backends consume different inputs:

- ``template_hmm`` takes the existing chroma matrix and beat grid.
- ``madmom_deep_chroma`` reads the audio file directly through madmom's
  pipeline and ignores ``chroma`` / ``beat_grid``.

The shared ``ChordBackend`` Protocol accepts both. Backend implementations
must accept any combination of arguments and ignore what they don't need.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np

from music_decoder.config.hyperparameters import ChordDetectionParams
from music_decoder.pipeline.contracts import BeatGrid, ChordRecognitionResult


class ChordBackend(Protocol):
    """Common interface for chord-recognition backends.

    Implementations should be cheap to construct (the pipeline creates one
    per job). Heavy resources (model weights, processor pipelines) should be
    lazy-loaded the first time ``detect`` is called.
    """

    def detect(
        self,
        *,
        chroma: np.ndarray[Any, np.dtype[Any]],
        sr: int,
        hop_length: int,
        beat_grid: BeatGrid,
        params: ChordDetectionParams,
        audio_path: Path | None,
    ) -> ChordRecognitionResult: ...
