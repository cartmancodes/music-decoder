from pathlib import Path
from typing import Any
from unittest import mock

from music_decoder.tabs import assign_tabs
from music_decoder.tabs.tuning import STANDARD_EADGBE
from music_decoder.types import TabAssignmentResult, TranscribedNote

_HP = """
id: t
basic_pitch: {onset_threshold: 0.5, frame_threshold: 0.3, minimum_note_length_ms: 58,
              minimum_frequency_hz: 60, maximum_frequency_hz: 2000}
post_processing: {median_filter_window: 5, min_note_duration_s: 0.05,
                  same_pitch_merge_gap_s: 0.05, rhythmic_snap_confidence_threshold: 0.7}
key_detection: {hpss_margin: 1.0, windowed_segment_length_s: 8, windowed_hop_s: 2}
beat_tracking: {start_bpm: 120, tightness: 100}
chord_detection: {qualities: [maj], min_segment_duration_s: 0.4}
tab_assignment:
  weights: {w_move: 2.5, w_string: 0.1, w_span: 0.25, w_open: 0.0, w_high: 0.8, w_chord_intra: 0.6}
  max_fret: 19
"""


def _capture() -> Any:
    empty = TabAssignmentResult(
        tabbed_notes=[], tuning=STANDARD_EADGBE, total_cost=0.0, notes_dropped=[]
    )
    return mock.patch("music_decoder.tabs._assign_tab_impl", return_value=empty)


def test_assign_tabs_reads_weights_and_max_fret_from_yaml(tmp_path: Path, monkeypatch: Any) -> None:
    p = tmp_path / "hp.yaml"
    p.write_text(_HP)
    monkeypatch.setenv("MUSIC_DECODER_HYPERPARAMETERS", str(p))
    with _capture() as impl:
        assign_tabs([TranscribedNote(0.0, 1.0, 60, 80, 1.0)])
    kw = impl.call_args.kwargs
    assert kw["weights"]["w_move"] == 2.5
    assert kw["max_fret"] == 19


def test_assign_tabs_explicit_weights_override_yaml() -> None:
    w = {
        "w_move": 9.0,
        "w_string": 0.0,
        "w_span": 0.0,
        "w_open": 0.0,
        "w_high": 0.0,
        "w_chord_intra": 0.0,
    }
    with _capture() as impl:
        assign_tabs([TranscribedNote(0.0, 1.0, 60, 80, 1.0)], weights=w, max_fret=12)
    assert impl.call_args.kwargs["weights"] == w
    assert impl.call_args.kwargs["max_fret"] == 12
