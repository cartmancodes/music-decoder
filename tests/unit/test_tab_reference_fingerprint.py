from pathlib import Path

import pytest

from music_decoder.tab_reference.fingerprint import (
    SongIdentificationFailed,
    fingerprint_audio,
)


@pytest.mark.slow
def test_fingerprint_returns_string_for_real_audio(fixtures_dir: Path):
    fp = fingerprint_audio(fixtures_dir / "audio_samples" / "sine_440.wav")
    assert isinstance(fp, str) and len(fp) > 8


def test_fingerprint_fails_gracefully_on_missing_file(tmp_path: Path):
    with pytest.raises(SongIdentificationFailed):
        fingerprint_audio(tmp_path / "nope.wav")
