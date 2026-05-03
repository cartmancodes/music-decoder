# src/music_decoder/pipeline/contracts.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

from music_decoder.tab_assignment.tuning import Tuning


# ---- audio_io
@dataclass(frozen=True)
class AudioSource:
    path: Path
    declared_kind: Literal["solo_guitar", "full_mix"]
    requested_quality: Literal["standard", "high"]
    requested_tuning: Tuning


@dataclass(frozen=True)
class LoadedAudio:
    samples: np.ndarray[Any, np.dtype[np.float32]]
    sr: int
    duration_s: float
    sha256: str
    source: AudioSource


# ---- separation
@dataclass(frozen=True)
class SeparationResult:
    guitar_samples: np.ndarray[Any, np.dtype[np.float32]] | None
    sr: int
    skipped_reason: str | None
    bleed_estimate_db: float | None


# ---- transcription
@dataclass(frozen=True)
class TranscribedNote:
    start_s: float
    end_s: float
    pitch: int
    velocity: int
    confidence: float


@dataclass(frozen=True)
class TranscriptionResult:
    notes: list[TranscribedNote]
    model: Literal["basic-pitch", "crepe"]
    raw_midi_path: Path
    post_midi_path: Path
    hyperparameters: dict[str, Any]
    median_confidence: float


# ---- key_detection
@dataclass(frozen=True)
class KeyEstimate:
    tonic: str
    mode: Literal["major", "minor"]
    profile: Literal["krumhansl_kessler", "temperley"]
    correlation: float
    margin: float


@dataclass(frozen=True)
class KeyDetectionResult:
    global_top3_per_profile: dict[str, list[KeyEstimate]]
    consensus_key: KeyEstimate | None
    windowed_segments: list[tuple[float, float, KeyEstimate]]
    confidence: float


# ---- beat_tracking
@dataclass(frozen=True)
class BeatGrid:
    tempo_bpm: float
    beat_times_s: np.ndarray[Any, np.dtype[np.float64]]
    downbeat_times_s: np.ndarray[Any, np.dtype[np.float64]]
    ts_numerator: int
    ts_denominator: int
    ts_confidence: float
    ts_assumed: bool


# ---- tab_assignment
@dataclass(frozen=True)
class TabPosition:
    string: int
    fret: int


@dataclass(frozen=True)
class TabbedNote:
    note: TranscribedNote
    position: TabPosition
    cost_breakdown: dict[str, float]


@dataclass(frozen=True)
class TabAssignmentResult:
    tabbed_notes: list[TabbedNote]
    tuning: Tuning
    total_cost: float
    notes_dropped: list[tuple[TranscribedNote, str]]


# ---- tab_reference
@dataclass(frozen=True)
class TabReferenceMatch:
    source: Literal["user_pasted_url", "user_pasted_text"]
    acoustid: str | None
    raw_text: str
    parsed_positions: list[TabPosition]
    similarity_to_prediction: float | None
    disagreement_spans: list[tuple[float, float]]


# ---- orchestration
@dataclass(frozen=True)
class StageEvent:
    job_id: int
    stage: str
    started_at: datetime
    ended_at: datetime
    success: bool
    error: str | None
    summary: dict[str, Any]
