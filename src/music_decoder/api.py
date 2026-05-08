"""Public synchronous library API: ``analyze()`` and ``compose()``.

This module is the top-level orchestration layer. Each pipeline stage is
imported by the simple name the public surface exposes (``track_beats``,
``estimate_key`` etc.) so that tests can mock them in isolation via
``mock.patch("music_decoder.api.<stage>")``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from music_decoder.chords import recognize_chords
from music_decoder.compose.api import compose as _compose
from music_decoder.dsp import compute_chroma, track_beats
from music_decoder.errors import SeparationError
from music_decoder.ingest import load as ingest_load
from music_decoder.key import estimate_key
from music_decoder.progress import NullProgress, ProgressCallback
from music_decoder.separation import run_separation
from music_decoder.tabs import assign_tabs
from music_decoder.tabs.tuning import STANDARD_EADGBE, Tuning
from music_decoder.transcription import transcribe
from music_decoder.types import AnalysisResult


def analyze(
    source: str | Path,
    *,
    declared_kind: Literal["solo_guitar", "full_mix"] = "full_mix",
    tuning: Tuning = STANDARD_EADGBE,
    use_separation: bool = True,
    progress: ProgressCallback | None = None,
) -> AnalysisResult:
    """Audio file or YouTube URL → chord progression + tab + key.

    Synchronous; deterministic given pinned hyperparameters.
    """
    cb: ProgressCallback = progress if progress is not None else NullProgress()

    cb("ingest", 0.0)
    loaded = ingest_load(source)
    cb("ingest", 1.0)

    samples, sr = loaded.samples, loaded.sr
    if declared_kind == "full_mix" and use_separation:
        cb("separation", 0.0)
        try:
            samples, sr = run_separation(loaded.samples, loaded.sr)
        except SeparationError:
            samples, sr = loaded.samples, loaded.sr
        cb("separation", 1.0)

    cb("beats", 0.0)
    beat_grid = track_beats(samples, sr)
    cb("beats", 1.0)

    cb("chroma", 0.0)
    chroma = compute_chroma(samples, sr)
    cb("chroma", 1.0)

    cb("key", 0.0)
    key = estimate_key(chroma)
    cb("key", 1.0)

    cb("chords", 0.0)
    chord_segments = recognize_chords(samples, sr, beat_grid=beat_grid)
    cb("chords", 1.0)

    cb("transcription", 0.0)
    notes = transcribe(samples, sr)
    cb("transcription", 1.0)

    cb("tabs", 0.0)
    tabbed = assign_tabs(notes, tuning=tuning)
    cb("tabs", 1.0)

    return AnalysisResult(
        source=str(source),
        audio_path=loaded.source.path,
        duration_s=loaded.duration_s,
        sample_rate_hz=loaded.sr,
        key=key,
        chord_progression=tuple(chord_segments),
        tab=tuple(tabbed),
        tempo_bpm=float(beat_grid.tempo_bpm),
        beat_times_s=tuple(beat_grid.beat_times_s.tolist()),
        metadata={"hyperparameter_set": _hp_id()},
    )


def _hp_id() -> str:
    """Best-effort hyperparameter-set id; ``"unknown"`` if config not present."""
    try:
        from music_decoder.config.hyperparameters import load_hyperparameters

        cfg_path = Path("config/hyperparameters.yaml")
        if not cfg_path.exists():
            return "unknown"
        return load_hyperparameters(cfg_path).id
    except Exception:
        return "unknown"


# Re-export compose so callers can `from music_decoder.api import analyze, compose`.
compose = _compose

__all__ = ["analyze", "compose"]
