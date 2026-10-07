from typing import Any
from unittest import mock

import numpy as np

from music_decoder import transcription
from music_decoder.types import TranscribedNote


def _fake_m2n(output: Any, **kwargs: Any) -> tuple[None, list[tuple[Any, ...]]]:
    _fake_m2n.kwargs = kwargs  # type: ignore[attr-defined]
    return None, [(0.5, 1.0, 64, 0.8, None), (0.0, 0.4, 60, 0.5, None)]


def test_decode_model_output_uses_adapter_params_and_sorts() -> None:
    with mock.patch("basic_pitch.note_creation.model_output_to_notes", _fake_m2n):
        notes = transcription.decode_model_output({"note": np.zeros((1, 88))})
    params = transcription.resolve_params()
    kw = _fake_m2n.kwargs  # type: ignore[attr-defined]
    assert kw["onset_thresh"] == params.onset_threshold
    assert kw["frame_thresh"] == params.frame_threshold
    assert kw["min_freq"] == params.minimum_frequency_hz
    assert notes == (
        TranscribedNote(0.0, 0.4, 60, 64, 0.5),
        TranscribedNote(0.5, 1.0, 64, 102, 0.8),
    )


def test_transcribe_is_run_model_then_decode() -> None:
    sentinel = {"note": np.zeros((1, 88))}
    with (
        mock.patch(
            "music_decoder.transcription.basic_pitch_wrapper.run_model", return_value=sentinel
        ) as rm,
        mock.patch(
            "music_decoder.transcription.decode_model_output", return_value=()
        ) as dec,
    ):
        out = transcription.transcribe(np.zeros(22050, np.float32), 22050)
    rm.assert_called_once()
    dec.assert_called_once_with(sentinel)
    assert out == ()
