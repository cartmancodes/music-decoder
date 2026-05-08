"""Transcription adapters for the public ``analyze()`` pipeline."""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from music_decoder.logging_setup import get_logger

# 65 Hz ~= low E2; 2093 Hz ~= C7. The YAML's 32.7 Hz / 2000 Hz are
# general-purpose; tune via YAML if you want piano-style range.
from music_decoder.tabs.tuning import STANDARD_EADGBE
from music_decoder.types import AudioSource, LoadedAudio, Note

_log = get_logger("transcription.adapter")

_GUITAR_DEFAULT_MIN_HZ = 65.0
_GUITAR_DEFAULT_MAX_HZ = 2093.0
# Heuristic guitar-friendly bounds; values outside this range get a warning
# but are still respected (the YAML is authoritative).
_REASONABLE_MIN_HZ = 50.0
_REASONABLE_MAX_HZ = 3000.0


def _resolve_basic_pitch_bounds() -> tuple[float, float]:
    """Return (min_hz, max_hz) from the YAML, with a guitar-default fallback.

    Logs a warning when the YAML values look unreasonable for guitar
    (e.g. < 50 Hz min or > 3000 Hz max) but still respects them.
    """
    try:
        from music_decoder.config.hyperparameters import load_hyperparameters

        hp = load_hyperparameters()
    except Exception as e:  # pragma: no cover - defensive
        _log.warning("hyperparameters_load_failed", extra={"error": str(e)})
        return _GUITAR_DEFAULT_MIN_HZ, _GUITAR_DEFAULT_MAX_HZ
    min_hz = float(hp.basic_pitch.minimum_frequency_hz)
    max_hz = float(hp.basic_pitch.maximum_frequency_hz)
    if min_hz < _REASONABLE_MIN_HZ or max_hz > _REASONABLE_MAX_HZ:
        _log.warning(
            "basic_pitch_bounds_outside_guitar_range",
            extra={"min_hz": min_hz, "max_hz": max_hz},
        )
    return min_hz, max_hz


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
    min_hz, max_hz = _resolve_basic_pitch_bounds()
    params = BasicPitchParams(
        onset_threshold=0.5,
        frame_threshold=0.3,
        minimum_note_length_ms=58,
        minimum_frequency_hz=min_hz,
        maximum_frequency_hz=max_hz,
    )
    with tempfile.TemporaryDirectory() as out:
        result = transcribe_basic_pitch(audio, params, output_dir=Path(out))
    return tuple(result.notes)


__all__ = ["transcribe"]
