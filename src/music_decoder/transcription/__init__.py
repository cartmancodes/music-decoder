"""Transcription adapters for the public ``analyze()`` pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from music_decoder.dsp.tempwav import temp_wav
from music_decoder.logging_setup import get_logger
from music_decoder.types import Note

if TYPE_CHECKING:
    from music_decoder.config.hyperparameters import BasicPitchParams

_log = get_logger("transcription.adapter")

# 65 Hz ~= low E2; 2093 Hz ~= C7. The YAML's 32.7 Hz / 2000 Hz are
# general-purpose; tune via YAML if you want piano-style range.
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


def resolve_params() -> BasicPitchParams:
    """basic-pitch decoding parameters used by the public pipeline."""
    from music_decoder.config.hyperparameters import BasicPitchParams

    min_hz, max_hz = _resolve_basic_pitch_bounds()
    return BasicPitchParams(
        onset_threshold=0.5,
        frame_threshold=0.3,
        minimum_note_length_ms=58,
        minimum_frequency_hz=min_hz,
        maximum_frequency_hz=max_hz,
    )


def decode_model_output(model_output: dict[str, Any]) -> tuple[Note, ...]:
    """Decode a raw basic-pitch output into notes with the pipeline's parameters."""
    from basic_pitch import note_creation
    from basic_pitch.constants import AUDIO_SAMPLE_RATE, FFT_HOP

    from music_decoder.transcription.basic_pitch_wrapper import notes_from_events

    params = resolve_params()
    min_note_len = int(
        np.round(params.minimum_note_length_ms / 1000 * (AUDIO_SAMPLE_RATE / FFT_HOP))
    )
    _midi, events = note_creation.model_output_to_notes(
        model_output,
        onset_thresh=params.onset_threshold,
        frame_thresh=params.frame_threshold,
        min_note_len=min_note_len,
        min_freq=params.minimum_frequency_hz,
        max_freq=params.maximum_frequency_hz,
    )
    return tuple(notes_from_events(events))


def transcribe(
    samples: np.ndarray[Any, np.dtype[np.float32]],
    sr: int,
) -> Sequence[Note]:
    """Run basic-pitch on raw samples: network inference, then decoding."""
    # Lazy import: basic-pitch pulls in TF/CoreML; keep
    # ``import music_decoder.transcription`` cheap.
    from music_decoder.transcription import basic_pitch_wrapper

    with temp_wav(samples, sr) as wav:
        model_output = basic_pitch_wrapper.run_model(wav)
    return decode_model_output(model_output)


__all__ = ["decode_model_output", "resolve_params", "transcribe"]
