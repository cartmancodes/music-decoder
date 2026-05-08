# tests/unit/test_config_hyperparameters_chord.py
from pathlib import Path

from music_decoder.config.hyperparameters import (
    ChordDetectionParams,
    load_hyperparameters,
)


def test_chord_detection_section_loads(tmp_path: Path) -> None:
    p = tmp_path / "h.yaml"
    p.write_text("""
id: 2026-05-05-test
basic_pitch:
  onset_threshold: 0.5
  frame_threshold: 0.3
  minimum_note_length_ms: 58
  minimum_frequency_hz: 32.7
  maximum_frequency_hz: 2000
crepe:
  model_capacity: full
  step_size_ms: 10
  viterbi: true
post_processing:
  median_filter_window: 5
  min_note_duration_s: 0.05
  same_pitch_merge_gap_s: 0.05
  rhythmic_snap_confidence_threshold: 0.7
key_detection:
  hpss_margin: 1.0
  windowed_segment_length_s: 8.0
  windowed_hop_s: 2.0
  modulation_penalty: 0.3
beat_tracking:
  start_bpm: 120
  tightness: 100
  ts_min_confidence: 0.5
chord_detection:
  qualities: [maj, min, "7", maj7, min7, dim, sus4, aug]
  hmm_self_transition_prob: 0.7
  no_chord_threshold: 0.15
  min_segment_duration_s: 0.25
tab_assignment:
  weights: {w_move: 1.0, w_string: 0.3, w_span: 0.5, w_open: 0.2, w_high: 0.4, w_chord_intra: 0.6}
  max_fret: 22
ui:
  confidence_thresholds: {high: 0.8, medium: 0.5}
evaluation:
  thresholds: {note_f_measure: 0.65}
  regression_tolerance: 0.02
""")
    hp = load_hyperparameters(p)
    assert isinstance(hp.chord_detection, ChordDetectionParams)
    assert hp.chord_detection.qualities == [
        "maj",
        "min",
        "7",
        "maj7",
        "min7",
        "dim",
        "sus4",
        "aug",
    ]
    assert hp.chord_detection.hmm_self_transition_prob == 0.7
    assert hp.chord_detection.no_chord_threshold == 0.15
    assert hp.chord_detection.min_segment_duration_s == 0.25
