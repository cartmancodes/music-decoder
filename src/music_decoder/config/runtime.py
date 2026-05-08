from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

_REQUIRED = ("youtube_cache_dir", "composition_out_dir", "log_level")
_ENV_PREFIX = "MUSIC_DECODER_"
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "runtime.yaml"


@dataclass(frozen=True)
class RuntimeConfig:
    youtube_cache_dir: Path
    composition_out_dir: Path
    log_level: str
    ffmpeg_path: Path | None
    model_cache_dir: Path | None
    fixture_dir: Path | None


def _coerce_path(value: str | None) -> Path | None:
    return Path(value) if value else None


def load_runtime_config(path: Path | None = None) -> RuntimeConfig:
    if path is None:
        path = _DEFAULT_CONFIG_PATH
    raw: dict[str, object] = yaml.safe_load(path.read_text()) or {}
    for key, val in os.environ.items():
        if key.startswith(_ENV_PREFIX):
            raw[key[len(_ENV_PREFIX) :].lower()] = val
    missing = [k for k in _REQUIRED if k not in raw]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")
    return RuntimeConfig(
        youtube_cache_dir=Path(str(raw["youtube_cache_dir"])),
        composition_out_dir=Path(str(raw["composition_out_dir"])),
        log_level=str(raw["log_level"]).upper(),
        ffmpeg_path=_coerce_path(raw.get("ffmpeg_path")),  # type: ignore[arg-type]
        model_cache_dir=_coerce_path(raw.get("model_cache_dir")),  # type: ignore[arg-type]
        fixture_dir=_coerce_path(raw.get("fixture_dir")),  # type: ignore[arg-type]
    )
