"""Madmom DeepChromaChordRecognitionProcessor backend.

Wraps madmom's pre-trained deep-chroma + chord-recognition pipeline. Madmom
operates on the audio file directly (not on our pre-computed chroma) and
returns chord segments as ``(start_s, end_s, jams_label)`` triples.

We always import ``..madmom_compat`` first so the necessary collections /
numpy shims are applied before any madmom module is loaded.
"""
from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

import numpy as np

from music_decoder.config.hyperparameters import ChordDetectionParams
from music_decoder.logging_setup import get_logger
from music_decoder.pipeline.contracts import (
    BeatGrid,
    ChordRecognitionResult,
    ChordSegment,
)

from ..labels import parse_jams_chord_label

_log = get_logger("chord_detection.madmom")


class MadmomDeepChromaBackend:
    """Madmom-based chord recognition backend.

    Lazy-imports madmom on first use so a missing-or-broken madmom install
    never breaks the rest of the pipeline at module-import time.
    """

    _processor: Any = None
    _chroma_processor: Any = None

    def detect(
        self,
        *,
        chroma: np.ndarray[Any, np.dtype[Any]],
        sr: int,
        hop_length: int,
        beat_grid: BeatGrid,
        params: ChordDetectionParams,
        audio_path: Path | None,
    ) -> ChordRecognitionResult:
        if audio_path is None or not Path(audio_path).exists():
            _log.warning(
                "madmom_backend_no_audio_path",
                extra={"audio_path": str(audio_path)},
            )
            return ChordRecognitionResult(
                segments=[],
                median_confidence=0.0,
                skipped_reason="madmom_requires_audio_path",
            )

        try:
            self._ensure_processors()
        except Exception as e:
            _log.warning("madmom_init_failed", extra={"error": str(e)})
            return ChordRecognitionResult(
                segments=[],
                median_confidence=0.0,
                skipped_reason=f"madmom_init_failed: {e}",
            )

        try:
            chroma_arr = self._chroma_processor(str(audio_path))
            raw_segments = self._processor(chroma_arr)
        except Exception as e:
            _log.warning("madmom_inference_failed", extra={"error": str(e)})
            return ChordRecognitionResult(
                segments=[],
                median_confidence=0.0,
                skipped_reason=f"madmom_inference_failed: {e}",
            )

        return self._convert(raw_segments, params)

    @classmethod
    def _ensure_processors(cls) -> None:
        if cls._processor is not None:
            return
        cls._import_madmom()
        from madmom.audio.chroma import DeepChromaProcessor
        from madmom.features.chords import DeepChromaChordRecognitionProcessor

        cls._chroma_processor = DeepChromaProcessor()
        cls._processor = DeepChromaChordRecognitionProcessor()

    @staticmethod
    def _import_madmom() -> None:
        # CRITICAL: ``madmom_compat`` MUST be imported before any module from
        # madmom. We isolate the side-effect import here so ruff's import
        # sorting in ``_ensure_processors`` doesn't reorder things.
        from .. import madmom_compat  # noqa: F401

    def _convert(
        self,
        raw_segments: Any,
        params: ChordDetectionParams,
    ) -> ChordRecognitionResult:
        """Convert madmom's segment array to our ChordSegment list.

        madmom returns a numpy structured array with fields ``start``, ``end``,
        ``label``. Iterating yields tuples whose third element is the JAMS
        label string. We parse via the shared label module; unsupported
        qualities are silently dropped (the gap is filled by the next
        supported segment).
        """
        segments: list[ChordSegment] = []
        for row in raw_segments:
            try:
                start = float(row[0])
                end = float(row[1])
                raw_label = row[2]
                if isinstance(raw_label, bytes):
                    raw_label = raw_label.decode()
            except (TypeError, IndexError, ValueError):
                continue
            parsed = parse_jams_chord_label(str(raw_label))
            if parsed is None:
                continue
            root, quality = parsed
            segments.append(
                ChordSegment(
                    start_s=start,
                    end_s=end,
                    root=root,
                    quality=quality,
                    confidence=0.85,  # madmom doesn't expose per-segment confidences
                )
            )

        # Drop sub-min-duration segments by absorbing into the predecessor.
        cleaned: list[ChordSegment] = []
        for seg in segments:
            duration = seg.end_s - seg.start_s
            if cleaned and duration < params.min_segment_duration_s:
                prev = cleaned[-1]
                cleaned[-1] = ChordSegment(
                    start_s=prev.start_s,
                    end_s=seg.end_s,
                    root=prev.root,
                    quality=prev.quality,
                    confidence=prev.confidence,
                )
            else:
                cleaned.append(seg)

        # Re-coalesce adjacent same-chord segments.
        final: list[ChordSegment] = []
        for seg in cleaned:
            if final and (final[-1].root, final[-1].quality) == (seg.root, seg.quality):
                prev = final[-1]
                final[-1] = ChordSegment(
                    start_s=prev.start_s,
                    end_s=seg.end_s,
                    root=prev.root,
                    quality=prev.quality,
                    confidence=(prev.confidence + seg.confidence) / 2,
                )
            else:
                final.append(seg)

        confidences = [s.confidence for s in final] or [0.0]
        return ChordRecognitionResult(
            segments=final,
            median_confidence=float(statistics.median(confidences)),
            skipped_reason=None if final else "madmom_no_segments",
        )
