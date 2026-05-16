from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from music_decoder.paths import bundled_config


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
class ChordDetectionParams:
    qualities: list[str]
    hmm_self_transition_prob: float
    no_chord_threshold: float
    min_segment_duration_s: float
    # Default backend; the orchestrator may pass-through to the named backend.
    # "template_hmm" — chroma + 49 templates + Viterbi (always available).
    # "madmom_deep_chroma" — madmom's pre-trained pipeline (requires audio_path
    # and an importable madmom; falls back to template_hmm if unavailable).
    backend: str = "template_hmm"


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
    chord_detection: ChordDetectionParams
    tab_assignment: TabAssignmentParams
    ui: UIParams
    evaluation: EvaluationParams


def _section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    if key not in raw or not isinstance(raw[key], dict):
        raise ValueError(f"hyperparameters missing section: {key}")
    return raw[key]  # type: ignore[no-any-return]


# Per-section defaults applied when a key is missing in the YAML. Keeps the
# v2 YAML free of plumbing-only fields (e.g. `hmm_self_transition_prob`)
# while still letting `load_hyperparameters()` succeed against the canonical
# `config/hyperparameters.yaml`. The defaults match what each adapter pins
# in code today.
_DEFAULTS: dict[str, dict[str, Any]] = {
    "crepe": {"model_capacity": "full", "step_size_ms": 10, "viterbi": True},
    "key_detection": {"modulation_penalty": 0.3},
    "beat_tracking": {"ts_min_confidence": 0.5},
    "chord_detection": {
        "hmm_self_transition_prob": 0.9,
        "no_chord_threshold": 0.3,
    },
    "ui": {"confidence_thresholds": {"high": 0.8, "medium": 0.5}},
    "evaluation": {
        "thresholds": {
            "note_f_measure": 0.65,
            "key_mirex_score": 0.75,
            "tab_string_accuracy": 0.55,
        },
        "regression_tolerance": 0.02,
    },
}


def _build(cls: type, raw_section: dict[str, Any], defaults: dict[str, Any]) -> Any:
    """Construct a frozen-dataclass section, falling back to defaults for
    fields the YAML omits. Unknown keys are dropped silently."""
    allowed = {f.name for f in fields(cls)}
    merged = {**defaults, **{k: v for k, v in raw_section.items() if k in allowed}}
    return cls(**{k: v for k, v in merged.items() if k in allowed})


_PROJECT_ROOT_HP = bundled_config("hyperparameters.yaml")


def load_hyperparameters(path: Path | None = None) -> HyperparameterSet:
    """Load the pinned hyperparameter set.

    With no argument resolves the project-root ``config/hyperparameters.yaml``
    so call sites inside the public adapters can read the YAML at call time
    without juggling paths. Missing optional fields fall back to per-section
    defaults (see ``_DEFAULTS``); a missing top-level ``id`` is still an error.
    """
    if path is None:
        path = _PROJECT_ROOT_HP
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict) or "id" not in raw:
        raise ValueError("hyperparameters yaml must have a top-level 'id'")
    return HyperparameterSet(
        id=str(raw["id"]),
        basic_pitch=_build(BasicPitchParams, _section(raw, "basic_pitch"), {}),
        crepe=_build(CrepeParams, raw.get("crepe", {}) or {}, _DEFAULTS["crepe"]),
        post_processing=_build(
            PostProcessingParams, _section(raw, "post_processing"), {}
        ),
        key_detection=_build(
            KeyDetectionParams,
            _section(raw, "key_detection"),
            _DEFAULTS["key_detection"],
        ),
        beat_tracking=_build(
            BeatTrackingParams,
            _section(raw, "beat_tracking"),
            _DEFAULTS["beat_tracking"],
        ),
        chord_detection=_build(
            ChordDetectionParams,
            _section(raw, "chord_detection"),
            _DEFAULTS["chord_detection"],
        ),
        tab_assignment=_build(TabAssignmentParams, _section(raw, "tab_assignment"), {}),
        ui=_build(UIParams, raw.get("ui", {}) or {}, _DEFAULTS["ui"]),
        evaluation=_build(
            EvaluationParams, raw.get("evaluation", {}) or {}, _DEFAULTS["evaluation"]
        ),
    )
