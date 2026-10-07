from unittest import mock

import numpy as np

from music_decoder.key import estimate_key
from music_decoder.key.cnn import probs_to_key_estimate


def _c_major_chroma() -> np.ndarray:
    chroma = np.zeros((12, 4))
    chroma[[0, 4, 7], :] = 1.0
    return chroma


def test_probs_to_key_estimate_maps_madmom_order_to_sharps() -> None:
    probs = np.zeros(24)
    probs[6] = 0.7  # madmom KEY_LABELS[6] == "Eb major"
    probs[18] = 0.2  # "D# minor"
    k = probs_to_key_estimate(probs)
    assert (k.tonic, k.mode, k.profile) == ("D#", "major", "cnn")
    assert abs(k.correlation - 0.7) < 1e-9
    assert abs(k.margin - 0.5) < 1e-9


def test_probs_to_key_estimate_accepts_madmom_2d_output() -> None:
    probs = np.zeros((1, 24))
    probs[0, 15] = 1.0  # "C minor"
    k = probs_to_key_estimate(probs)
    assert (k.tonic, k.mode) == ("C", "minor")


def test_estimate_key_uses_cnn_when_configured() -> None:
    fake = probs_to_key_estimate(np.eye(24)[15])  # "C minor"
    with (
        mock.patch("music_decoder.key._resolve_backend", return_value="cnn"),
        mock.patch("music_decoder.key.cnn.estimate_key_cnn", return_value=fake),
    ):
        k = estimate_key(_c_major_chroma(), samples=np.zeros(22050, np.float32), sr=22050)
    assert (k.tonic, k.mode) == ("C", "minor")


def test_estimate_key_falls_back_to_profiles_when_cnn_raises() -> None:
    with (
        mock.patch("music_decoder.key._resolve_backend", return_value="cnn"),
        mock.patch("music_decoder.key.cnn.estimate_key_cnn", side_effect=RuntimeError("x")),
    ):
        k = estimate_key(_c_major_chroma(), samples=np.zeros(22050, np.float32), sr=22050)
    assert k.profile in ("krumhansl_kessler", "temperley")
    assert (k.tonic, k.mode) == ("C", "major")


def test_estimate_key_without_samples_uses_profiles() -> None:
    with mock.patch("music_decoder.key._resolve_backend", return_value="cnn"):
        k = estimate_key(_c_major_chroma())
    assert (k.tonic, k.mode) == ("C", "major")
