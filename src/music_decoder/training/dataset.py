"""Build basic-pitch-compatible training pairs from GuitarSet (or manual fixtures).

basic-pitch's training pipeline expects (audio_path, JAMS_path) pairs. This
module enumerates pairs from a ``DatasetSpec`` and validates each entry exists
on disk before returning.

Actual tensor construction is delegated to basic-pitch's training package
when the user runs ``finetune_basic_pitch.py``. This module only does the
file-system enumeration so we can dry-run a config before touching any model.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from .config import DatasetSpec


@dataclass(frozen=True)
class TrainingPair:
    """One audio file paired with its JAMS annotation."""

    track_id: str
    audio_path: Path
    jams_path: Path


def enumerate_pairs(spec: DatasetSpec) -> Iterator[TrainingPair]:
    """Yield (audio, jams) pairs for the dataset, skipping missing tracks."""
    if spec.name == "guitarset":
        yield from _enumerate_guitarset(spec)
    elif spec.name == "manual":
        yield from _enumerate_manual(spec)
    else:
        raise ValueError(f"unknown dataset name: {spec.name!r}")


def _enumerate_guitarset(spec: DatasetSpec) -> Iterator[TrainingPair]:
    audio_dir = spec.cache_dir / "audio_mono-mic"
    annotation_dir = spec.cache_dir / "annotation"
    for track_id in spec.track_ids:
        audio = audio_dir / f"{track_id}_mic.wav"
        jams = annotation_dir / f"{track_id}.jams"
        if audio.exists() and jams.exists():
            yield TrainingPair(track_id=track_id, audio_path=audio, jams_path=jams)


def _enumerate_manual(spec: DatasetSpec) -> Iterator[TrainingPair]:
    """Manual fixtures use our JSON schema, not JAMS — but we surface them
    here for completeness. basic-pitch can't train on JSON directly, so this
    branch is informational; the harness logs that manual tracks are
    enumeration-only and won't be used for training.
    """
    for track_id in spec.track_ids:
        json_path = spec.cache_dir / f"{track_id}.json"
        if json_path.exists():
            yield TrainingPair(
                track_id=track_id,
                audio_path=spec.cache_dir / f"{track_id}.wav",
                jams_path=json_path,
            )
