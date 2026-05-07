"""YouTube ingestion via yt-dlp.

Audio is downloaded as 'bestaudio', extracted to WAV via the ffmpeg
post-processor, and cached under <cache_dir>/<video_id>.wav. Cache hits
short-circuit the download.
"""
from __future__ import annotations

import re
from pathlib import Path

import yt_dlp

from music_decoder.errors import YouTubeError

_YT_RE = re.compile(
    r"^https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([\w-]{11})"
)


def is_youtube_url(s: str) -> bool:
    return bool(_YT_RE.match(s or ""))


def extract_video_id(url: str) -> str:
    m = _YT_RE.match(url)
    if not m:
        raise YouTubeError(f"Not a recognizable YouTube URL: {url!r}")
    return m.group(1)


def fetch_audio(url: str, *, cache_dir: Path) -> Path:
    """Download (or read from cache) the audio of `url` as WAV. Returns the path."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    video_id = extract_video_id(url)
    out_path = cache_dir / f"{video_id}.wav"
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "format": "bestaudio/best",
        "outtmpl": str(cache_dir / f"{video_id}.%(ext)s"),
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "0",
        }],
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except yt_dlp.utils.DownloadError as e:
        raise YouTubeError(f"YouTube download failed: {e}") from e
    if not out_path.exists():
        raise YouTubeError(f"yt-dlp did not produce {out_path}")
    return out_path
