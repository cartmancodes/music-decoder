from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class BasicPitchParams:
    onset_threshold: float
    frame_threshold: float
    minimum_note_length_ms: int
    minimum_frequency_hz: float
    maximum_frequency_hz: float


@dataclass(frozen=True)
class CrepeParams:
    model_capacity: str
    step_size_ms: int
    viterbi: bool


@dataclass(frozen=True)
class PostProcessingParams:
    median_filter_window: int
    min_note_duration_s: float
    same_pitch_merge_gap_s: float
    rhythmic_snap_confidence_threshold: float


@dataclass(frozen=True)
class KeyDetectionParams:
    hpss_margin: float
    windowed_segment_length_s: float
    windowed_hop_s: float
    modulation_penalty: float


@dataclass(frozen=True)
class BeatTrackingParams:
    start_bpm: float
    tightness: float
    ts_min_confidence: float


@dataclass(frozen=True)
class TabAssignmentParams:
    weights: dict[str, float]
    max_fret: int


@dataclass(frozen=True)
class UIParams:
    confidence_thresholds: dict[str, float]


@dataclass(frozen=True)
class EvaluationParams:
    thresholds: dict[str, float]
    regression_tolerance: float


@dataclass(frozen=True)
class HyperparameterSet:
    id: str
    basic_pitch: BasicPitchParams
    crepe: CrepeParams
    post_processing: PostProcessingParams
    key_detection: KeyDetectionParams
    beat_tracking: BeatTrackingParams
    tab_assignment: TabAssignmentParams
    ui: UIParams
    evaluation: EvaluationParams


def _section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    if key not in raw or not isinstance(raw[key], dict):
        raise ValueError(f"hyperparameters missing section: {key}")
    return raw[key]  # type: ignore[no-any-return]


def load_hyperparameters(path: Path) -> HyperparameterSet:
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict) or "id" not in raw:
        raise ValueError("hyperparameters yaml must have a top-level 'id'")
    return HyperparameterSet(
        id=str(raw["id"]),
        basic_pitch=BasicPitchParams(**_section(raw, "basic_pitch")),
        crepe=CrepeParams(**_section(raw, "crepe")),
        post_processing=PostProcessingParams(**_section(raw, "post_processing")),
        key_detection=KeyDetectionParams(**_section(raw, "key_detection")),
        beat_tracking=BeatTrackingParams(**_section(raw, "beat_tracking")),
        tab_assignment=TabAssignmentParams(**_section(raw, "tab_assignment")),
        ui=UIParams(**_section(raw, "ui")),
        evaluation=EvaluationParams(**_section(raw, "evaluation")),
    )
