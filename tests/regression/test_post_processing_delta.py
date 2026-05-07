"""Asserts that applying the post-processing chain does not regress F-measure."""
from pathlib import Path

import numpy as np
import pytest

from music_decoder.audio_io.load import load_audio
from music_decoder.config.hyperparameters import (
    BasicPitchParams,
    PostProcessingParams,
)
from music_decoder.evaluation.fixtures.synthetic import SyntheticFixtures
from music_decoder.evaluation.metrics import note_f_measure
from music_decoder.types import AudioSource
from music_decoder.tab_assignment.tuning import get_preset
from music_decoder.transcription.basic_pitch_wrapper import transcribe_basic_pitch
from music_decoder.transcription.post_processing import apply_post_processing


def _intervals_pitches(notes):
    iv = np.array([(n.start_s, n.end_s) for n in notes], dtype=float)
    p = np.array([n.pitch for n in notes], dtype=float)
    return iv, p


@pytest.mark.regression
@pytest.mark.slow
def test_post_processing_does_not_decrease_f_measure(tmp_path: Path):
    fx = next(f for f in SyntheticFixtures(
        root=Path("tests/fixtures/synthetic")
    ).load() if f.name == "c_major_scale")
    src = AudioSource(
        path=fx.audio_path, declared_kind="solo_guitar",
        requested_quality="standard", requested_tuning=get_preset("EADGBE"),
    )
    audio = load_audio(src)
    bp_params = BasicPitchParams(
        onset_threshold=0.5, frame_threshold=0.3, minimum_note_length_ms=58,
        minimum_frequency_hz=32.7, maximum_frequency_hz=2000.0,
    )
    raw = transcribe_basic_pitch(audio, bp_params, output_dir=tmp_path)
    iv, p = _intervals_pitches(raw.notes)
    f_raw = note_f_measure(iv, p, fx.ground_truth.intervals,
                           fx.ground_truth.pitches_midi)
    pp_params = PostProcessingParams(
        median_filter_window=5, min_note_duration_s=0.05,
        same_pitch_merge_gap_s=0.05,
        rhythmic_snap_confidence_threshold=0.7,
    )
    cleaned = apply_post_processing(raw.notes, params=pp_params, beats=None)
    iv2, p2 = _intervals_pitches(cleaned)
    f_post = note_f_measure(iv2, p2, fx.ground_truth.intervals,
                            fx.ground_truth.pitches_midi)
    assert f_post >= f_raw - 0.02, (
        f"post-processing regressed F-measure: raw={f_raw:.3f}, post={f_post:.3f}"
    )
