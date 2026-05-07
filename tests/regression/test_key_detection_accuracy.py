"""On synthetic C-major fixture, both profile estimators should pick C major."""
from pathlib import Path

import pytest

from music_decoder.audio_io.load import load_audio
from music_decoder.evaluation.fixtures.synthetic import SyntheticFixtures
from music_decoder.key_detection.api import detect_key
from music_decoder.types import AudioSource
from music_decoder.tab_assignment.tuning import get_preset


@pytest.mark.regression
@pytest.mark.slow
def test_synthetic_c_major_scale_returns_c_major():
    fx = next(f for f in SyntheticFixtures(
        root=Path("tests/fixtures/synthetic")
    ).load() if f.name == "c_major_scale")
    src = AudioSource(
        path=fx.audio_path, declared_kind="solo_guitar",
        requested_quality="standard", requested_tuning=get_preset("EADGBE"),
    )
    audio = load_audio(src)
    result = detect_key(
        audio.samples, sr=audio.sr,
        hpss_margin=1.0, segment_length_s=8.0, hop_s=2.0,
    )
    top_kk = result.global_top3_per_profile["krumhansl_kessler"][0]
    top_t = result.global_top3_per_profile["temperley"][0]
    assert top_kk.tonic == "C" and top_kk.mode == "major"
    assert top_t.tonic == "C" and top_t.mode == "major"
    assert result.consensus_key is not None
