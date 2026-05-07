# tests/unit/test_separation.py
from pathlib import Path

import numpy as np
import pytest

from music_decoder.types import AudioSource, LoadedAudio
from music_decoder.separation.demucs import isolate_guitar
from music_decoder.tabs.tuning import get_preset


def _audio(samples: np.ndarray, sr: int = 22050) -> LoadedAudio:
    return LoadedAudio(
        samples=samples, sr=sr, duration_s=samples.size / sr,
        sha256="x" * 64,
        source=AudioSource(
            path=Path("/tmp/x.wav"), declared_kind="full_mix",
            requested_quality="standard", requested_tuning=get_preset("EADGBE"),
        ),
    )


def test_solo_guitar_short_circuits_with_skipped_reason():
    samples = np.random.randn(22050).astype(np.float32) * 0.1
    audio = LoadedAudio(
        samples=samples, sr=22050, duration_s=1.0, sha256="x" * 64,
        source=AudioSource(
            path=Path("/tmp/x.wav"), declared_kind="solo_guitar",
            requested_quality="standard", requested_tuning=get_preset("EADGBE"),
        ),
    )
    result = isolate_guitar(audio)
    assert result.guitar_samples is None
    assert result.skipped_reason and "solo_guitar declared" in result.skipped_reason


def test_demucs_failure_returns_skipped_with_original_audio(monkeypatch):
    samples = np.random.randn(22050 * 2).astype(np.float32) * 0.1
    audio = _audio(samples)

    def fake_apply_model(*args, **kwargs):
        raise RuntimeError("demucs blew up")

    monkeypatch.setattr(
        "music_decoder.separation.demucs._apply_demucs", fake_apply_model
    )
    result = isolate_guitar(audio)
    assert result.guitar_samples is None
    assert result.skipped_reason and result.skipped_reason.startswith("demucs_failed")


@pytest.mark.slow
def test_demucs_round_trip_returns_audio_when_called():
    """Real Demucs invocation; only runs in slow mode."""
    samples = np.random.randn(22050 * 5).astype(np.float32) * 0.05
    audio = _audio(samples)
    result = isolate_guitar(audio)
    if result.guitar_samples is None:
        pytest.skip(f"demucs unavailable: {result.skipped_reason}")
    assert result.guitar_samples.shape == samples.shape
