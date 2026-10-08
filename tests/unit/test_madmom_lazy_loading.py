"""Lazy madmom model loading must build each model once under concurrent first use."""

import threading
import time
from typing import Any
from unittest import mock

import numpy as np

import music_decoder.chords.madmom_compat  # noqa: F401  (must precede madmom)


class _SlowCounter:
    """Stand-in processor class: slow constructor, counts constructions."""

    built = 0
    lock = threading.Lock()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        time.sleep(0.05)
        with _SlowCounter.lock:
            _SlowCounter.built += 1

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return np.full((1, 24), 1 / 24)


def _hammer(fn: Any, n: int = 8) -> None:
    threads = [threading.Thread(target=fn) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def test_cnn_key_model_built_once_under_concurrency() -> None:
    from music_decoder.key import cnn

    _SlowCounter.built = 0
    with (
        mock.patch.object(cnn, "_processor", None),
        mock.patch("madmom.features.key.CNNKeyRecognitionProcessor", _SlowCounter),
    ):
        _hammer(lambda: cnn.cnn_probabilities(np.zeros(2205, np.float32), 22050))
    assert _SlowCounter.built == 1


def test_downbeat_models_built_once_under_concurrency() -> None:
    from music_decoder.dsp import beats_madmom

    _SlowCounter.built = 0
    with (
        mock.patch.object(beats_madmom, "_rnn", None),
        mock.patch.object(beats_madmom, "_dbn", None),
        mock.patch("madmom.features.downbeats.RNNDownBeatProcessor", _SlowCounter),
        mock.patch("madmom.features.downbeats.DBNDownBeatTrackingProcessor", _SlowCounter),
    ):
        _hammer(beats_madmom._load_processors)
    assert _SlowCounter.built == 2  # one RNN + one DBN


def test_chord_models_built_once_under_concurrency() -> None:
    from music_decoder.chords.backends import madmom_deep_chroma as m

    _SlowCounter.built = 0
    with (
        mock.patch.object(m.MadmomCNNBackend, "_processor", None),
        mock.patch.object(m.MadmomCNNBackend, "_chroma_processor", None),
        mock.patch("madmom.features.chords.CNNChordFeatureProcessor", _SlowCounter),
        mock.patch("madmom.features.chords.CRFChordRecognitionProcessor", _SlowCounter),
    ):
        _hammer(m.MadmomCNNBackend._ensure_processors)
    assert _SlowCounter.built == 2  # one feature extractor + one CRF
