"""Transcription adapters for the public ``analyze()`` pipeline."""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from music_decoder.tabs.tuning import STANDARD_EADGBE
from music_decoder.types import AudioSource, LoadedAudio, Note


def transcribe(
    samples: np.ndarray[Any, np.dtype[np.float32]],
    sr: int,
) -> Sequence[Note]:
    """Run basic-pitch transcription on raw samples.

    Adapter that wraps the existing :func:`transcribe_basic_pitch`, which
    expects a fully-formed :class:`LoadedAudio` and a `BasicPitchParams` /
    output dir. Builds a stub ``LoadedAudio`` around the raw samples and
    returns just the ``Note`` sequence.
    """
    # Lazy import: basic-pitch pulls in TF/CoreML; keep
    # ``import music_decoder.transcription`` cheap.
    from music_decoder.config.hyperparameters import BasicPitchParams
    from music_decoder.transcription.basic_pitch_wrapper import transcribe_basic_pitch

    stub_path = Path(tempfile.gettempdir()) / "music_decoder_transcribe_input.wav"
    source = AudioSource(
        path=stub_path,
        declared_kind="solo_guitar",
        requested_quality="standard",
        requested_tuning=STANDARD_EADGBE,
    )
    audio = LoadedAudio(
        samples=samples.astype(np.float32, copy=False),
        sr=sr,
        duration_s=samples.size / sr,
        sha256="",
        source=source,
    )
    params = BasicPitchParams(
        onset_threshold=0.5,
        frame_threshold=0.3,
        minimum_note_length_ms=58,
        minimum_frequency_hz=65.0,
        maximum_frequency_hz=2093.0,
    )
    with tempfile.TemporaryDirectory() as out:
        result = transcribe_basic_pitch(audio, params, output_dir=Path(out))
    return tuple(result.notes)


__all__ = ["transcribe"]
