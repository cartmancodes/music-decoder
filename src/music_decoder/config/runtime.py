from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from music_decoder.paths import bundled_config

_REQUIRED = ("youtube_cache_dir", "composition_out_dir", "log_level")
_ENV_PREFIX = "MUSIC_DECODER_"
_DEFAULT_CONFIG_PATH = bundled_config("runtime.yaml")
_INTERP_RE = re.compile(r"\$\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


@dataclass(frozen=True)
class RuntimeConfig:
    youtube_cache_dir: Path
    composition_out_dir: Path
    log_level: str
    ffmpeg_path: Path | None
    model_cache_dir: Path | None
    fixture_dir: Path | None
    data_dir: Path
    fluidsynth_soundfont: str
    sample_rate_hz: int


def _coerce_path(value: object | None) -> Path | None:
    if value is None or value == "":
        return None
    return Path(str(value))


def _interpolate(raw: dict[str, object]) -> dict[str, object]:
    """Resolve ``${key}`` references in string values from sibling keys."""
    resolved: dict[str, object] = dict(raw)
    # Iterate a few times to allow chained references.
    for _ in range(4):
        changed = False
        for key, val in list(resolved.items()):
            if isinstance(val, str) and _INTERP_RE.search(val):

                def _sub(match: re.Match[str]) -> str:
                    name = match.group(1)
                    if name in resolved and isinstance(resolved[name], (str, int, float)):
                        return str(resolved[name])
                    return match.group(0)

                new_val = _INTERP_RE.sub(_sub, val)
                if new_val != val:
                    resolved[key] = new_val
                    changed = True
        if not changed:
            break
    return resolved


def load_runtime_config(path: Path | None = None) -> RuntimeConfig:
    if path is None:
        path = _DEFAULT_CONFIG_PATH
    raw: dict[str, object] = yaml.safe_load(path.read_text()) or {}
    for key, val in os.environ.items():
        if key.startswith(_ENV_PREFIX):
            raw[key[len(_ENV_PREFIX) :].lower()] = val
    raw = _interpolate(raw)
    missing = [k for k in _REQUIRED if k not in raw]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")
    data_dir_raw = raw.get("data_dir", "~/.music-decoder")
    return RuntimeConfig(
        youtube_cache_dir=Path(str(raw["youtube_cache_dir"])),
        composition_out_dir=Path(str(raw["composition_out_dir"])),
        log_level=str(raw["log_level"]).upper(),
        ffmpeg_path=_coerce_path(raw.get("ffmpeg_path")),
        model_cache_dir=_coerce_path(raw.get("model_cache_dir")),
        fixture_dir=_coerce_path(raw.get("fixture_dir")),
        data_dir=Path(str(data_dir_raw)),
        fluidsynth_soundfont=str(raw.get("fluidsynth_soundfont", "GeneralUser-GS.sf2")),
        sample_rate_hz=int(str(raw.get("sample_rate_hz", 22050))),
    )
