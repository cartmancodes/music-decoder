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


_HP = """
id: t
basic_pitch: {onset_threshold: 0.61, frame_threshold: 0.27, minimum_note_length_ms: 90,
              minimum_frequency_hz: 70, maximum_frequency_hz: 1500, melodia_trick: false}
post_processing: {median_filter_window: 5, min_note_duration_s: 0.05,
                  same_pitch_merge_gap_s: 0.05, rhythmic_snap_confidence_threshold: 0.7,
                  enabled: ENABLED}
key_detection: {hpss_margin: 1.0, windowed_segment_length_s: 8, windowed_hop_s: 2}
beat_tracking: {start_bpm: 120, tightness: 100}
chord_detection: {qualities: [maj], min_segment_duration_s: 0.4}
tab_assignment: {weights: {w_move: 1.0}, max_fret: 22}
"""


def _use_hp(tmp_path: Any, monkeypatch: Any, enabled: bool) -> None:
    p = tmp_path / "hp.yaml"
    p.write_text(_HP.replace("ENABLED", "true" if enabled else "false"))
    monkeypatch.setenv("MUSIC_DECODER_HYPERPARAMETERS", str(p))


def test_resolve_params_reads_yaml(tmp_path: Any, monkeypatch: Any) -> None:
    _use_hp(tmp_path, monkeypatch, enabled=False)
    p = transcription.resolve_params()
    assert (p.onset_threshold, p.frame_threshold, p.minimum_note_length_ms) == (0.61, 0.27, 90)
    assert (p.minimum_frequency_hz, p.maximum_frequency_hz) == (70, 1500)
    assert p.melodia_trick is False


def test_decode_passes_melodia_and_min_len_frames(tmp_path: Any, monkeypatch: Any) -> None:
    _use_hp(tmp_path, monkeypatch, enabled=False)
    with mock.patch("basic_pitch.note_creation.model_output_to_notes", _fake_m2n):
        transcription.decode_model_output({"note": np.zeros((1, 88))})
    kw = _fake_m2n.kwargs  # type: ignore[attr-defined]
    assert kw["melodia_trick"] is False
    assert kw["min_note_len"] == round(0.090 * 22050 / 256)


def _events(*args: Any, **kwargs: Any) -> tuple[None, list[tuple[Any, ...]]]:
    return None, [
        (0.0, 0.5, 60, 0.9, None),
        (0.52, 1.0, 60, 0.8, None),  # same pitch, 20 ms gap → merged
        (2.0, 2.02, 64, 0.4, None),  # 20 ms long → dropped
    ]


def test_post_processing_applied_when_enabled(tmp_path: Any, monkeypatch: Any) -> None:
    _use_hp(tmp_path, monkeypatch, enabled=True)
    with mock.patch("basic_pitch.note_creation.model_output_to_notes", _events):
        notes = transcription.decode_model_output({})
    assert [(n.start_s, n.end_s, n.pitch) for n in notes] == [(0.0, 1.0, 60)]


def test_post_processing_skipped_when_disabled(tmp_path: Any, monkeypatch: Any) -> None:
    _use_hp(tmp_path, monkeypatch, enabled=False)
    with mock.patch("basic_pitch.note_creation.model_output_to_notes", _events):
        notes = transcription.decode_model_output({})
    assert len(notes) == 3
