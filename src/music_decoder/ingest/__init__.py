"""Audio ingestion: local files and YouTube URLs."""

from __future__ import annotations

from pathlib import Path

from music_decoder.ingest.audio_file import load_audio_file
from music_decoder.ingest.youtube import fetch_audio, is_youtube_url
from music_decoder.types import LoadedAudio


def load(source: str | Path, *, yt_cache_dir: Path | None = None) -> LoadedAudio:
    """Load audio from a local path or a YouTube URL.

    For YouTube URLs, downloads and caches the audio under `yt_cache_dir`
    (defaulting to <data_dir>/yt_cache via the runtime config).
    """
    if isinstance(source, str) and is_youtube_url(source):
        if yt_cache_dir is None:
            from pathlib import Path as _P

            from music_decoder.config.runtime import load_runtime_config

            yt_cache_dir = _P(load_runtime_config().youtube_cache_dir).expanduser()
        wav_path = fetch_audio(source, cache_dir=yt_cache_dir)
        return load_audio_file(wav_path)
    return load_audio_file(Path(source))


__all__ = ["fetch_audio", "is_youtube_url", "load", "load_audio_file"]
