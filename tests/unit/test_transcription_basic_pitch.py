# tests/unit/test_transcription_basic_pitch.py
from pathlib import Path

import numpy as np
import pytest

from music_decoder.config.hyperparameters import BasicPitchParams
from music_decoder.pipeline.contracts import AudioSource, LoadedAudio
from music_decoder.tab_assignment.tuning import get_preset
from music_decoder.transcription.basic_pitch_wrapper import transcribe_basic_pitch


def _hparams() -> BasicPitchParams:
    return BasicPitchParams(
        onset_threshold=0.5, frame_threshold=0.3,
        minimum_note_length_ms=58,
        minimum_frequency_hz=32.7, maximum_frequency_hz=2000.0,
    )


@pytest.mark.slow
def test_basic_pitch_returns_some_notes_for_synthetic(fixtures_dir: Path, tmp_path: Path):
    """Render the committed C major scale to WAV and ensure basic-pitch finds notes."""
    from music_decoder.evaluation.fixtures.synthetic import SyntheticFixtures
    fx = next(f for f in SyntheticFixtures(root=fixtures_dir / "synthetic").load()
              if f.name == "c_major_scale")

    sr = 22050
    import scipy.io.wavfile as wavfile
    rate, raw = wavfile.read(str(fx.audio_path))
    samples = (raw.astype(np.float32) / 32768.0).astype(np.float32)
    if rate != sr:
        from librosa import resample
        samples = resample(samples, orig_sr=rate, target_sr=sr)
    audio = LoadedAudio(
        samples=samples, sr=sr, duration_s=samples.size / sr, sha256="x" * 64,
        source=AudioSource(
            path=fx.audio_path, declared_kind="solo_guitar",
            requested_quality="standard", requested_tuning=get_preset("EADGBE"),
        ),
    )
    result = transcribe_basic_pitch(audio, _hparams(), output_dir=tmp_path)
    assert result.model == "basic-pitch"
    assert len(result.notes) > 0
    assert all(0 <= n.confidence <= 1 for n in result.notes)
    assert all(n.start_s < n.end_s for n in result.notes)
    assert result.raw_midi_path.exists()
    assert result.post_midi_path.exists()
