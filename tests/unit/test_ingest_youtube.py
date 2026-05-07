from pathlib import Path
from unittest import mock

import pytest

from music_decoder.errors import YouTubeError
from music_decoder.ingest.youtube import (
    extract_video_id,
    is_youtube_url,
    fetch_audio,
)


def test_is_youtube_url_positive():
    for url in [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ",
        "http://youtu.be/dQw4w9WgXcQ",
    ]:
        assert is_youtube_url(url), url


def test_is_youtube_url_negative():
    for url in [
        "https://example.com",
        "/local/path.mp3",
        "https://www.youtube.com/",
        "",
    ]:
        assert not is_youtube_url(url), url


def test_extract_video_id():
    assert extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=10") == "dQw4w9WgXcQ"


def test_fetch_audio_uses_cache(tmp_path):
    cache_dir = tmp_path / "yt_cache"
    cache_dir.mkdir()
    cached = cache_dir / "dQw4w9WgXcQ.wav"
    cached.write_bytes(b"FAKEWAV")
    out = fetch_audio("https://youtu.be/dQw4w9WgXcQ", cache_dir=cache_dir)
    assert out == cached


def test_fetch_audio_calls_yt_dlp_when_not_cached(tmp_path):
    cache_dir = tmp_path / "yt_cache"
    cache_dir.mkdir()

    def fake_download(self, urls):
        out = cache_dir / "dQw4w9WgXcQ.wav"
        out.write_bytes(b"FAKEWAV")
        return 0

    with mock.patch("yt_dlp.YoutubeDL") as ydl_cls:
        instance = ydl_cls.return_value.__enter__.return_value
        instance.download.side_effect = lambda urls: fake_download(instance, urls)
        out = fetch_audio("https://youtu.be/dQw4w9WgXcQ", cache_dir=cache_dir)
    assert out.exists()
    assert out.name == "dQw4w9WgXcQ.wav"


def test_fetch_audio_raises_youtube_error_on_failure(tmp_path):
    import yt_dlp
    cache_dir = tmp_path / "yt_cache"
    cache_dir.mkdir()
    with mock.patch("yt_dlp.YoutubeDL") as ydl_cls:
        instance = ydl_cls.return_value.__enter__.return_value
        instance.download.side_effect = yt_dlp.utils.DownloadError("private")
        with pytest.raises(YouTubeError):
            fetch_audio("https://youtu.be/dQw4w9WgXcQ", cache_dir=cache_dir)
