from pathlib import Path
from unittest import mock

from music_decoder.ingest import load


def test_load_dispatches_to_audio_file_for_local_path(tmp_path):
    audio = tmp_path / "x.wav"
    audio.write_bytes(b"fake")
    with mock.patch("music_decoder.ingest.load_audio_file") as f:
        f.return_value = "audio_obj"
        out = load(audio)
        f.assert_called_once_with(audio)
        assert out == "audio_obj"


def test_load_dispatches_to_youtube_for_url(tmp_path):
    with mock.patch("music_decoder.ingest.fetch_audio") as fetch, \
         mock.patch("music_decoder.ingest.load_audio_file") as la:
        fetch.return_value = tmp_path / "abc.wav"
        la.return_value = "audio_obj"
        out = load("https://youtu.be/dQw4w9WgXcQ", yt_cache_dir=tmp_path)
        fetch.assert_called_once_with("https://youtu.be/dQw4w9WgXcQ", cache_dir=tmp_path)
        la.assert_called_once_with(tmp_path / "abc.wav")
        assert out == "audio_obj"
