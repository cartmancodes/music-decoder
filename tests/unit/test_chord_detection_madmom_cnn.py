from pathlib import Path
from unittest import mock

import numpy as np

from music_decoder.chords.api import detect_chords
from music_decoder.chords.backends import madmom_deep_chroma as m
from music_decoder.chords.templates import QUALITIES
from music_decoder.config.hyperparameters import ChordDetectionParams
from music_decoder.types import BeatGrid, ChordRecognitionResult


def _params(backend: str) -> ChordDetectionParams:
    return ChordDetectionParams(
        qualities=list(QUALITIES),
        hmm_self_transition_prob=0.9,
        no_chord_threshold=0.3,
        min_segment_duration_s=0.25,
        backend=backend,
    )


def _grid() -> BeatGrid:
    return BeatGrid(120.0, np.arange(0, 4, 0.5), np.array([0.0]), 4, 4, 0.5, True)


def _c_major_chroma() -> np.ndarray:
    chroma = np.zeros((12, 200), dtype=np.float32)
    chroma[[0, 4, 7], :] = 1.0
    return chroma


def test_madmom_cnn_backend_dispatch(tmp_path: Path) -> None:
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"x")
    rows = [(0.0, 2.0, "A:min"), (2.0, 4.0, "C:maj")]
    feats = mock.MagicMock(return_value=np.zeros((40, 128)))
    crf = mock.MagicMock(return_value=rows)
    with (
        mock.patch.object(m.MadmomCNNBackend, "_ensure_processors"),
        mock.patch.object(m.MadmomCNNBackend, "_chroma_processor", feats),
        mock.patch.object(m.MadmomCNNBackend, "_processor", crf),
    ):
        res = detect_chords(
            chroma=np.zeros((12, 100)),
            sr=22050,
            hop_length=512,
            beat_grid=_grid(),
            params=_params("madmom_cnn"),
            audio_path=wav,
        )
    feats.assert_called_once_with(str(wav))
    assert [(s.root, s.quality) for s in res.segments] == [("A", "min"), ("C", "maj")]


def test_madmom_runtime_failure_falls_back_to_template_hmm(tmp_path: Path) -> None:
    failed = ChordRecognitionResult(
        segments=[], median_confidence=0.0, skipped_reason="madmom_inference_failed: x"
    )
    with mock.patch.object(m.MadmomCNNBackend, "detect", return_value=failed):
        res = detect_chords(
            chroma=_c_major_chroma(),
            sr=22050,
            hop_length=512,
            beat_grid=_grid(),
            params=_params("madmom_cnn"),
            audio_path=tmp_path / "a.wav",
        )
    assert res.segments
    assert (res.segments[0].root, res.segments[0].quality) == ("C", "maj")
