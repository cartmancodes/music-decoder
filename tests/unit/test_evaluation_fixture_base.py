import numpy as np

from music_decoder.evaluation.fixtures.base import Fixture, GroundTruth


def test_groundtruth_holds_intervals_and_key():
    gt = GroundTruth(
        intervals=np.array([[0.0, 0.5], [0.5, 1.0]]),
        pitches_midi=np.array([60, 62]),
        key=("C", "major"),
        tempo_bpm=120.0,
        tab=[(60, 4, 1), (62, 4, 3)],
    )
    assert gt.key == ("C", "major")
    assert gt.tempo_bpm == 120.0
    assert gt.tab == [(60, 4, 1), (62, 4, 3)]


def test_fixture_has_audio_path_and_truth(tmp_path):
    audio = tmp_path / "x.wav"
    audio.write_bytes(b"x")
    gt = GroundTruth(
        intervals=np.zeros((0, 2)), pitches_midi=np.zeros(0),
        key=None, tempo_bpm=None, tab=None,
    )
    fx = Fixture(name="x", source="manual", audio_path=audio, ground_truth=gt)
    assert fx.audio_path == audio
    assert fx.source == "manual"
