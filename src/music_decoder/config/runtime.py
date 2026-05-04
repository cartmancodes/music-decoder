from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

_REQUIRED = ("db_path", "artifact_dir", "log_level")
_ENV_PREFIX = "MUSIC_DECODER_"


@dataclass(frozen=True)
class RuntimeConfig:
    db_path: Path
    artifact_dir: Path
    log_level: str
    ffmpeg_path: Path | None
    model_cache_dir: Path | None
    fixture_dir: Path | None


def _coerce_path(value: str | None) -> Path | None:
    return Path(value) if value else None


def load_runtime_config(path: Path) -> RuntimeConfig:
    raw: dict[str, object] = yaml.safe_load(path.read_text()) or {}
    for key, val in os.environ.items():
        if key.startswith(_ENV_PREFIX):
            raw[key[len(_ENV_PREFIX):].lower()] = val
    missing = [k for k in _REQUIRED if k not in raw]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")
    return RuntimeConfig(
        db_path=Path(str(raw["db_path"])),
        artifact_dir=Path(str(raw["artifact_dir"])),
        log_level=str(raw["log_level"]).upper(),
        ffmpeg_path=_coerce_path(raw.get("ffmpeg_path")),  # type: ignore[arg-type]
        model_cache_dir=_coerce_path(raw.get("model_cache_dir")),  # type: ignore[arg-type]
        fixture_dir=_coerce_path(raw.get("fixture_dir")),  # type: ignore[arg-type]
    )
