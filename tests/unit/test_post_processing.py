# tests/unit/test_post_processing.py
from music_decoder.pipeline.contracts import TranscribedNote
from music_decoder.transcription.post_processing import (
    drop_short_notes,
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
