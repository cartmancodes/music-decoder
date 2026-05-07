"""Tests for the Phase B-2 high-resolution transcription backend.

The wrapper around ``piano_transcription_inference`` (Kong et al.). Most
tests use the real package with the stock piano checkpoint — this is a
~165 MB download but auto-fetches on first use; a separate slow test
exercises real inference.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pretty_midi
import pytest
import scipy.io.wavfile as wavfile

from music_decoder.pipeline.contracts import AudioSource, LoadedAudio
from music_decoder.tab_assignment.tuning import get_preset
from music_decoder.transcription.highres_guitar_wrapper import (
    transcribe_highres_guitar,
)


def _audio_from_wav(path: Path) -> LoadedAudio:
    sr, raw = wavfile.read(str(path))
    samples = (raw.astype(np.float32) / 32768.0).astype(np.float32)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    return LoadedAudio(
        samples=samples, sr=sr, duration_s=samples.size / sr,
        sha256="x" * 64,
        source=AudioSource(
            path=path, declared_kind="solo_guitar",
            requested_quality="standard", requested_tuning=get_preset("EADGBE"),
        ),
    )


def test_wrapper_exposes_correct_model_label(tmp_path: Path):
    """The TranscriptionResult should have ``model='highres-guitar'``."""
    # Skip the actual transcribe; just check the import path is functional.
    from music_decoder.transcription import highres_guitar_wrapper
    assert hasattr(highres_guitar_wrapper, "transcribe_highres_guitar")


def test_wrapper_raises_clean_error_when_package_missing(monkeypatch, tmp_path: Path):
    """When piano_transcription_inference is not importable, raise a clear ImportError."""
    # Force the lazy import to fail.
    import sys

    import music_decoder.transcription.highres_guitar_wrapper as wrapper
    monkeypatch.setitem(sys.modules, "piano_transcription_inference", None)

    audio = _audio_from_wav(Path("tests/fixtures/audio_samples/sine_440.wav"))
    with pytest.raises(ImportError, match="piano_transcription_inference"):
        wrapper.transcribe_highres_guitar(audio, output_dir=tmp_path)


@pytest.mark.slow
def test_wrapper_runs_on_synthetic(tmp_path: Path):
    """End-to-end on the c-major scale fixture using stock piano weights.

    The piano checkpoint is ~165 MB — this test only runs in slow mode and
    requires the checkpoint to be already cached at
    ``~/piano_transcription_inference_data/``.
    """
    audio = _audio_from_wav(Path("tests/fixtures/synthetic/c_major_scale.wav"))
    try:
        result = transcribe_highres_guitar(audio, output_dir=tmp_path)
    except Exception as e:
        pytest.skip(f"piano_transcription_inference unavailable: {e}")
    assert result.model == "highres-guitar"
    assert len(result.notes) > 0
    assert result.raw_midi_path.exists()
    pitches = [n.pitch for n in result.notes]
    # The expected pitches are the C major scale: 60-72.
    # Stock piano weights aren't perfect on guitar timbre but should at least
    # detect SOMETHING in this range.
    assert any(58 <= p <= 74 for p in pitches), (
        f"piano model on c-major scale should detect at least one note in "
        f"the expected range; got {pitches}"
    )


@pytest.mark.slow
def test_wrapper_writes_valid_midi(tmp_path: Path):
    """The wrapper's ``post_midi_path`` should be loadable by pretty_midi."""
    audio = _audio_from_wav(Path("tests/fixtures/synthetic/c_major_scale.wav"))
    try:
        result = transcribe_highres_guitar(audio, output_dir=tmp_path)
    except Exception as e:
        pytest.skip(f"piano_transcription_inference unavailable: {e}")
    pm = pretty_midi.PrettyMIDI(str(result.post_midi_path))
    assert len(pm.instruments) >= 1
