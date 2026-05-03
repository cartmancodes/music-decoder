# tests/unit/test_post_processing.py
import numpy as np
import pytest

from music_decoder.pipeline.contracts import TranscribedNote
from music_decoder.transcription.post_processing import (
    drop_short_notes,
    median_filter_pitch_contour,
    merge_same_pitch,
)


def _n(start, end, pitch=60, conf=0.9):
    return TranscribedNote(start_s=start, end_s=end, pitch=pitch, velocity=80, confidence=conf)


def test_drop_short_notes_removes_below_threshold():
    notes = [_n(0.0, 0.01), _n(0.5, 0.6), _n(1.0, 1.04)]
    out = drop_short_notes(notes, min_duration_s=0.05)
    assert len(out) == 1
    assert out[0].start_s == 0.5


def test_drop_short_notes_keeps_exact_threshold():
    notes = [_n(0.0, 0.05)]
    assert len(drop_short_notes(notes, min_duration_s=0.05)) == 1


def test_merge_same_pitch_combines_close_notes():
    notes = [
        _n(0.0, 0.5, pitch=60, conf=0.9),
        _n(0.52, 1.0, pitch=60, conf=0.7),   # gap 0.02s, same pitch
        _n(1.5, 2.0, pitch=60, conf=0.8),    # too far apart to merge
    ]
    out = merge_same_pitch(notes, gap_s=0.05)
    assert len(out) == 2
    assert out[0].start_s == 0.0
    assert out[0].end_s == 1.0
    assert out[0].confidence == pytest.approx(0.8, abs=0.01)


def test_merge_same_pitch_does_not_cross_pitches():
    notes = [_n(0.0, 0.5, pitch=60), _n(0.52, 1.0, pitch=62)]
    out = merge_same_pitch(notes, gap_s=0.05)
    assert len(out) == 2


def test_median_filter_removes_single_outlier():
    # A 9-frame contour with one wildly wrong frame
    contour = np.array([60, 60, 60, 60, 99, 60, 60, 60, 60], dtype=float)
    out = median_filter_pitch_contour(contour, window=3)
    assert out[4] == 60.0


def test_median_filter_window_must_be_odd():
    with pytest.raises(ValueError):
        median_filter_pitch_contour(np.array([60.0, 60.0]), window=4)
