"""Chord recognition adapters for the public ``analyze()`` pipeline."""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from music_decoder.dsp.tempwav import temp_wav
from music_decoder.logging_setup import get_logger
from music_decoder.types import BeatGrid, ChordSegment

# Minimum beats the template-HMM backend needs (matches its internal threshold).
_MIN_BEATS = 4

_log = get_logger("chord_detection.adapter")


def _resolve_backend() -> str:
    """Return the chord-detection backend chosen for this run.

    Reads ``config/hyperparameters.yaml`` at call time so a single edit of
    that file can switch the public ``analyze()`` pipeline between the
    DSP-only ``template_hmm`` and madmom's ``madmom_deep_chroma`` without
    a code change. Falls back to ``madmom_deep_chroma`` if the YAML cannot
    be loaded (downstream the dispatcher in ``chords/api.py`` will further
    fall back to ``template_hmm`` when madmom itself is unimportable).
    """
    try:
        from music_decoder.config.hyperparameters import load_hyperparameters

        hp = load_hyperparameters()
    except Exception as e:  # pragma: no cover - defensive
        _log.warning("hyperparameters_load_failed", extra={"error": str(e)})
        return "madmom_deep_chroma"
    return hp.chord_detection.backend


def _synthetic_beat_grid(duration_s: float) -> BeatGrid:
    """Build a uniform 4-beat grid spanning *duration_s*.

    Used when the real beat tracker degenerates (silence, sustained chords,
    very short clips) so the chord recognizer can still emit at least one
    segment instead of bailing out with ``degenerate_beat_grid``.
    """
    duration_s = max(float(duration_s), 0.25)
    # Five edges = four equal-width beats spanning [0, duration_s].
    edges = np.linspace(0.0, duration_s, _MIN_BEATS + 1, dtype=float)
    return BeatGrid(
        tempo_bpm=60.0 * _MIN_BEATS / duration_s,
        beat_times_s=edges,
        downbeat_times_s=np.array([edges[0]], dtype=float),
        ts_numerator=4,
        ts_denominator=4,
        ts_confidence=0.0,
        ts_assumed=True,
    )


def recognize_chords(
    samples: np.ndarray[Any, np.dtype[np.float32]],
    sr: int,
    *,
    beat_grid: BeatGrid,
) -> Sequence[ChordSegment]:
    """Run beat-synchronous chord recognition on raw samples.

    Adapter that:
      1. Computes HPSS chroma at the default hop length (512).
      2. Reads the requested backend from ``config/hyperparameters.yaml``.
      3. Calls the chord-detection API; the dispatcher falls back to
         ``template_hmm`` when ``madmom_deep_chroma`` is requested but
         madmom is unimportable.
      4. Unwraps :class:`ChordRecognitionResult` to the bare segment tuple.

    Falls back to a synthetic uniform beat grid spanning the audio when the
    supplied grid is degenerate (fewer than the backend-required beats); this
    keeps short / static / sustained-chord clips from silently producing zero
    segments.
    """
    # Lazy imports: ``music_decoder.chords.api`` pulls in the template-HMM
    # backend on import; keep ``import music_decoder.chords`` itself cheap.
    from music_decoder.chords.api import detect_chords
    from music_decoder.chords.templates import QUALITIES
    from music_decoder.config.hyperparameters import ChordDetectionParams
    from music_decoder.dsp.chroma import compute_chroma_with_hpss

    backend = _resolve_backend()
    chroma = compute_chroma_with_hpss(samples, sr=sr, hpss_margin=8.0)
    params = ChordDetectionParams(
        qualities=list(QUALITIES),
        hmm_self_transition_prob=0.9,
        no_chord_threshold=0.3,
        min_segment_duration_s=0.25,
        backend=backend,
    )

    # If the beat grid degenerates, swap in a synthetic 4-beat grid spanning
    # the audio. This costs us beat-synchronous accuracy on the segmentation
    # boundaries but lets the recognizer emit a label for short/static clips.
    if beat_grid.beat_times_s.size < _MIN_BEATS:
        duration_s = float(samples.shape[-1]) / float(sr) if samples.size else 1.0
        beat_grid = _synthetic_beat_grid(duration_s)

    # madmom backends operate on an audio file, not on pre-computed chroma.
    # Materialize a temp WAV only when a madmom backend is requested;
    # template_hmm ignores audio_path.
    with contextlib.ExitStack() as stack:
        audio_path: Path | None = None
        if backend.startswith("madmom"):
            audio_path = stack.enter_context(temp_wav(samples, sr))
        result = detect_chords(
            chroma=chroma,
            sr=sr,
            hop_length=512,
            beat_grid=beat_grid,
            params=params,
            audio_path=audio_path,
        )

    if result.skipped_reason is not None and not result.segments:
        # Final safety net: if the backend still bailed (e.g. chroma window
        # too short), emit a single no-chord segment so downstream consumers
        # always see a well-formed progression.
        duration_s = float(samples.shape[-1]) / float(sr) if samples.size else 0.0
        return (
            ChordSegment(
                start_s=0.0,
                end_s=max(duration_s, 0.25),
                root="N",
                quality="",
                confidence=0.0,
            ),
        )

    return tuple(result.segments)


__all__ = ["recognize_chords"]
