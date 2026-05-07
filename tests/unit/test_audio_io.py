# tests/unit/test_audio_io.py
from pathlib import Path

import numpy as np
import pytest

from music_decoder.audio_io.load import (
    ClipTooShortError,
    CorruptAudioError,
    SilentAudioError,
    load_audio,
)
from music_decoder.types import AudioSource
from music_decoder.tabs.tuning import get_preset


def _source(path: Path) -> AudioSource:
    return AudioSource(
        path=path, declared_kind="solo_guitar",
        requested_quality="standard", requested_tuning=get_preset("EADGBE"),
    )


def test_load_decodes_sine_wav(fixtures_dir: Path):
    src = _source(fixtures_dir / "audio_samples" / "sine_440.wav")
    audio = load_audio(src)
    assert audio.sr == 22050
    assert audio.samples.dtype == np.float32
    assert audio.samples.ndim == 1
    assert 0.95 < audio.duration_s < 1.05
    assert len(audio.sha256) == 64


def test_load_high_quality_uses_44100(fixtures_dir: Path):
    src = AudioSource(
        path=fixtures_dir / "audio_samples" / "sine_440.wav",
        declared_kind="solo_guitar", requested_quality="high",
        requested_tuning=get_preset("EADGBE"),
    )
    audio = load_audio(src)
    assert audio.sr == 44100


def test_load_rejects_silent_audio(fixtures_dir: Path):
    src = _source(fixtures_dir / "audio_samples" / "silence_2s.wav")
    with pytest.raises(SilentAudioError):
        load_audio(src)


def test_load_rejects_too_short(tmp_path: Path):
    sr = 22050
    short = (0.5 * np.sin(2 * np.pi * 440 * np.arange(int(sr * 0.5)) / sr)).astype(np.float32)
    import scipy.io.wavfile as wavfile
    p = tmp_path / "tiny.wav"
    wavfile.write(str(p), sr, (short * 32767).astype(np.int16))
    with pytest.raises(ClipTooShortError):
        load_audio(_source(p))


def test_load_corrupt_raises(tmp_path: Path):
    p = tmp_path / "broken.mp3"
    p.write_bytes(b"definitely not audio")
    with pytest.raises(CorruptAudioError):
        load_audio(_source(p))
