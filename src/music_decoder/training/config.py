"""Training configuration for the basic-pitch fine-tuning harness.

Defines a YAML-loadable dataclass that describes the dataset spec, train/val
splits, output paths, and hyperparameters for a fine-tuning run. Actual
training execution is delegated to ``finetune.py``; this module just owns
the contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DatasetSpec:
    """Where to find audio + JAMS pairs."""

    name: str  # "guitarset" or "manual"
    cache_dir: Path  # mirdata data_home for guitarset, or fixtures dir for manual
    track_ids: list[str]


@dataclass(frozen=True)
class FineTuneConfig:
    """End-to-end training-run configuration."""

    id: str
    base_checkpoint: Path | None  # None = start from basic-pitch's stock checkpoint
    output_checkpoint_dir: Path
    train: DatasetSpec
    val: DatasetSpec
    epochs: int
    batch_size: int
    learning_rate: float
    sample_rate_hz: int


def load_finetune_config(path: Path) -> FineTuneConfig:
    raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
    if "id" not in raw:
        raise ValueError("training config must have a top-level 'id'")
    train_section = raw.get("train")
    val_section = raw.get("val")
    if not train_section or not val_section:
        raise ValueError("training config must have 'train' and 'val' sections")
    return FineTuneConfig(
        id=str(raw["id"]),
        base_checkpoint=(
            Path(raw["base_checkpoint"]) if raw.get("base_checkpoint") else None
        ),
        output_checkpoint_dir=Path(str(raw["output_checkpoint_dir"])),
        train=_dataset_spec(train_section),
        val=_dataset_spec(val_section),
        epochs=int(raw.get("epochs", 5)),
        batch_size=int(raw.get("batch_size", 4)),
        learning_rate=float(raw.get("learning_rate", 1e-4)),
        sample_rate_hz=int(raw.get("sample_rate_hz", 22050)),
    )


def _dataset_spec(section: dict[str, Any]) -> DatasetSpec:
    return DatasetSpec(
        name=str(section["name"]),
        cache_dir=Path(str(section["cache_dir"])),
        track_ids=[str(t) for t in section.get("track_ids", [])],
    )
