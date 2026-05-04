import pytest

from music_decoder.pipeline.contracts import TabbedNote, TabPosition, TranscribedNote
from music_decoder.tab_reference.alignment import similarity_to_prediction


def _tn(pitch=60, start=0.0, end=0.5, string=4, fret=1):
    return TabbedNote(
        note=TranscribedNote(start_s=start, end_s=end, pitch=pitch,
                             velocity=80, confidence=0.9),
        position=TabPosition(string=string, fret=fret),
        cost_breakdown={},
    )


def test_similarity_perfect_when_identical():
    pred = [_tn(60, 0, 0.5, 4, 1), _tn(62, 0.5, 1.0, 4, 3)]
    ref = [TabPosition(4, 1), TabPosition(4, 3)]
    assert similarity_to_prediction(pred, ref) == pytest.approx(1.0)


def test_similarity_zero_for_no_matches():
    pred = [_tn(60, 0, 0.5, 4, 1)]
    ref = [TabPosition(0, 5), TabPosition(1, 7)]
    assert similarity_to_prediction(pred, ref) == 0.0
