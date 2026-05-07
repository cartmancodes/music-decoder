"""Tests for the madmom_deep_chroma backend."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from music_decoder.chord_detection.backends.madmom_deep_chroma import (
    MadmomDeepChromaBackend,
)
from music_decoder.config.hyperparameters import ChordDetectionParams
from music_decoder.types import BeatGrid


def _params() -> ChordDetectionParams:
    return ChordDetectionParams(
        qualities=["maj", "min", "7", "maj7", "min7", "dim", "sus4", "aug"],
        hmm_self_transition_prob=0.7,
        no_chord_threshold=0.15,
        min_segment_duration_s=0.0,
        backend="madmom_deep_chroma",
    )


def _beat_grid(beats: list[float]) -> BeatGrid:
    return BeatGrid(
        tempo_bpm=120.0,
        beat_times_s=np.asarray(beats, dtype=float),
        downbeat_times_s=np.asarray(beats[::4], dtype=float),
        ts_numerator=4, ts_denominator=4,
        ts_confidence=0.6, ts_assumed=False,
    )


def test_madmom_backend_returns_skipped_when_no_audio_path():
    backend = MadmomDeepChromaBackend()
    result = backend.detect(
        chroma=np.zeros((12, 100)),
        sr=22050,
        hop_length=512,
        beat_grid=_beat_grid([0.0, 1.0, 2.0, 3.0, 4.0]),
        params=_params(),
        audio_path=None,
    )
    assert result.segments == []
    assert result.skipped_reason == "madmom_requires_audio_path"


def test_madmom_backend_returns_skipped_when_audio_missing(tmp_path: Path):
    backend = MadmomDeepChromaBackend()
    result = backend.detect(
        chroma=np.zeros((12, 100)),
        sr=22050,
        hop_length=512,
        beat_grid=_beat_grid([0.0, 1.0, 2.0, 3.0, 4.0]),
        params=_params(),
        audio_path=tmp_path / "nonexistent.wav",
    )
    assert result.skipped_reason == "madmom_requires_audio_path"


def test_madmom_backend_convert_parses_jams_labels():
    """Direct test of the _convert helper with a synthetic raw_segments array.

    Avoids running madmom's actual processor, just exercises label parsing
    and segment merging logic.
    """
    backend = MadmomDeepChromaBackend()
    # Mimic madmom's structured-array output. Phase B-3 promoted sus4 to a
    # first-class quality, so the F:sus4 segment now survives as ("F", "sus4").
    # We use a still-unsupported quality (alt) to test the drop path.
    raw_segments = [
        (0.0, 4.0, "C:maj"),
        (4.0, 8.0, "A:min"),
        (8.0, 9.0, "F:alt"),    # unsupported; should be dropped
        (9.0, 12.0, "G:7"),
        (12.0, 16.0, "C:maj"),  # adjacent to a different chord; not merged with first
    ]
    result = backend._convert(raw_segments, _params())
    labels = [(s.root, s.quality) for s in result.segments]
    assert labels == [("C", "maj"), ("A", "min"), ("G", "7"), ("C", "maj")]
    assert result.skipped_reason is None
    assert result.segments[0].confidence > 0.0


def test_madmom_backend_convert_merges_adjacent_same_chord():
    backend = MadmomDeepChromaBackend()
    raw_segments = [
        (0.0, 2.0, "C:maj"),
        (2.0, 4.0, "C:maj"),  # immediately follows; should merge
        (4.0, 6.0, "G:maj"),
    ]
    result = backend._convert(raw_segments, _params())
    assert len(result.segments) == 2
    assert result.segments[0].start_s == 0.0
    assert result.segments[0].end_s == 4.0


def test_madmom_backend_convert_returns_skipped_for_empty_input():
    backend = MadmomDeepChromaBackend()
    result = backend._convert([], _params())
    assert result.segments == []
    assert result.skipped_reason == "madmom_no_segments"


@pytest.mark.slow
def test_madmom_backend_runs_on_real_audio():
    """End-to-end test on a real GuitarSet track (if cached)."""
    audio_path = Path("tests/fixtures/guitarset/audio_mono-mic/00_BN1-129-Eb_comp_mic.wav")
    if not audio_path.exists():
        pytest.skip("GuitarSet cache not available")
    backend = MadmomDeepChromaBackend()
    result = backend.detect(
        chroma=np.zeros((12, 100)),  # not used by madmom
        sr=22050,
        hop_length=512,
        beat_grid=_beat_grid([0.0, 4.0, 8.0, 12.0, 16.0, 20.0]),
        params=_params(),
        audio_path=audio_path,
    )
    assert result.skipped_reason is None
    assert len(result.segments) >= 4
    assert result.segments[0].root in (
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B", "N"
    )
    assert result.segments[0].quality in (
        "maj", "min", "7", "maj7", "min7", "dim", "sus4", "aug", "",
    )
