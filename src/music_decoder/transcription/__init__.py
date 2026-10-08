"""Transcription adapters for the public ``analyze()`` pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from music_decoder.dsp.tempwav import temp_wav
from music_decoder.logging_setup import get_logger
from music_decoder.types import Note, TranscribedNote

if TYPE_CHECKING:
    from music_decoder.config.hyperparameters import BasicPitchParams, HyperparameterSet

_log = get_logger("transcription.adapter")

# Fallback bounds when the YAML can't be loaded: 65 Hz ~= C2 (lowest note in
# drop-C; standard-tuning low E2 is 82 Hz) to 2093 Hz ~= C7. The YAML
# (`basic_pitch.minimum_frequency_hz` / `maximum_frequency_hz`) is authoritative.
_GUITAR_DEFAULT_MIN_HZ = 65.0
_GUITAR_DEFAULT_MAX_HZ = 2093.0
# Heuristic guitar-friendly bounds; values outside this range get a warning
# but are still respected (the YAML is authoritative).
_REASONABLE_MIN_HZ = 50.0
_REASONABLE_MAX_HZ = 3000.0


def _check_bounds(min_hz: float, max_hz: float) -> None:
    """Warn (but respect the YAML) when bounds look unreasonable for guitar."""
    if min_hz < _REASONABLE_MIN_HZ or max_hz > _REASONABLE_MAX_HZ:
        _log.warning(
            "basic_pitch_bounds_outside_guitar_range",
            extra={"min_hz": min_hz, "max_hz": max_hz},
        )


def _load_hp() -> HyperparameterSet | None:
    try:
        from music_decoder.config.hyperparameters import load_hyperparameters

        return load_hyperparameters()
    except Exception as e:  # pragma: no cover - defensive
        _log.warning("hyperparameters_load_failed", extra={"error": str(e)})
        return None


def resolve_params() -> BasicPitchParams:
    """basic-pitch decoding parameters from ``config/hyperparameters.yaml``.

    Falls back to basic-pitch's own thresholds with guitar-range frequency
    bounds when the YAML cannot be loaded.
    """
    from music_decoder.config.hyperparameters import BasicPitchParams

    hp = _load_hp()
    if hp is None:
        return BasicPitchParams(
            onset_threshold=0.5,
            frame_threshold=0.3,
            minimum_note_length_ms=58,
            minimum_frequency_hz=_GUITAR_DEFAULT_MIN_HZ,
            maximum_frequency_hz=_GUITAR_DEFAULT_MAX_HZ,
        )
    params = hp.basic_pitch
    _check_bounds(float(params.minimum_frequency_hz), float(params.maximum_frequency_hz))
    return params


def _post_process(notes: list[TranscribedNote]) -> list[TranscribedNote]:
    """Merge same-pitch fragments then drop very short notes, when enabled in YAML."""
    from music_decoder.transcription.post_processing import drop_short_notes, merge_same_pitch

    hp = _load_hp()
    if hp is None or not hp.post_processing.enabled:
        return notes
    pp = hp.post_processing
    merged = merge_same_pitch(notes, gap_s=pp.same_pitch_merge_gap_s)
    kept = drop_short_notes(merged, min_duration_s=pp.min_note_duration_s)
    return sorted(kept, key=lambda n: (n.start_s, n.pitch))


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
        melodia_trick=params.melodia_trick,
    )
    return tuple(_post_process(notes_from_events(events)))


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
