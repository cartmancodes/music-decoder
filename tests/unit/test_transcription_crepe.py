# tests/unit/test_transcription_crepe.py
from pathlib import Path

import numpy as np
import pytest

from music_decoder.config.hyperparameters import CrepeParams
from music_decoder.pipeline.contracts import AudioSource, LoadedAudio
from music_decoder.tab_assignment.tuning import get_preset
from music_decoder.transcription.crepe_wrapper import transcribe_crepe


def _audio_from_wav(path: Path) -> LoadedAudio:
    import scipy.io.wavfile as wavfile
    sr, raw = wavfile.read(str(path))
    samples = (raw.astype(np.float32) / 32768.0).astype(np.float32)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    return LoadedAudio(
        samples=samples, sr=sr, duration_s=samples.size / sr, sha256="x" * 64,
        source=AudioSource(
            path=path, declared_kind="solo_guitar",
            requested_quality="standard", requested_tuning=get_preset("EADGBE"),
        ),
    )


@pytest.mark.slow
def test_crepe_returns_pitches_for_sine(fixtures_dir: Path, tmp_path: Path):
    audio = _audio_from_wav(fixtures_dir / "audio_samples" / "sine_440.wav")
    params = CrepeParams(model_capacity="tiny", step_size_ms=10, viterbi=True)
    result = transcribe_crepe(audio, params, output_dir=tmp_path)
    assert result.model == "crepe"
    pitches = [n.pitch for n in result.notes]
    # 440 Hz = MIDI 69. Allow ±1 semitone tolerance for rounding.
    assert any(68 <= p <= 70 for p in pitches)
