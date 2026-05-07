"""Asserts basic-pitch hits a minimum F-measure on the deterministic synthetic fixture."""
from pathlib import Path

import numpy as np
import pytest

from music_decoder.audio_io.load import load_audio
from music_decoder.config.hyperparameters import BasicPitchParams
from music_decoder.evaluation.fixtures.synthetic import SyntheticFixtures
from music_decoder.evaluation.metrics import note_f_measure
from music_decoder.types import AudioSource
from music_decoder.tab_assignment.tuning import get_preset
from music_decoder.transcription.basic_pitch_wrapper import transcribe_basic_pitch


@pytest.mark.regression
@pytest.mark.slow
def test_basic_pitch_meets_f_measure_on_c_major_scale(tmp_path: Path):
    fx = next(f for f in SyntheticFixtures(
        root=Path("tests/fixtures/synthetic")
    ).load() if f.name == "c_major_scale")
    src = AudioSource(
        path=fx.audio_path, declared_kind="solo_guitar",
        requested_quality="standard", requested_tuning=get_preset("EADGBE"),
    )
    audio = load_audio(src)
    params = BasicPitchParams(
        onset_threshold=0.5, frame_threshold=0.3,
        minimum_note_length_ms=58,
        minimum_frequency_hz=32.7, maximum_frequency_hz=2000.0,
    )
    result = transcribe_basic_pitch(audio, params, output_dir=tmp_path)
    pred_iv = np.array([(n.start_s, n.end_s) for n in result.notes], dtype=float)
    pred_p = np.array([n.pitch for n in result.notes], dtype=float)
    f = note_f_measure(pred_iv, pred_p, fx.ground_truth.intervals,
                       fx.ground_truth.pitches_midi)
    assert f >= 0.50, (
        f"basic-pitch F-measure {f:.3f} below 0.50 on synthetic C major scale; "
        "the model is unusable as currently configured."
    )
