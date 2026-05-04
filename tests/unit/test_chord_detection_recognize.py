# tests/unit/test_chord_detection_recognize.py
import numpy as np
import pytest

from music_decoder.chord_detection.recognize import beat_sync_chroma


def test_beat_sync_averages_columns_in_each_window():
    # Chroma at 100 columns/sec (hop=441, sr=44100). 4 seconds → 400 columns.
    sr = 44100
    hop_length = 441
    chroma = np.zeros((12, 400), dtype=float)
    # Pretend C is dominant in seconds 0-1, F dominant in seconds 1-2, etc.
    chroma[0, 0:100] = 1.0     # C
    chroma[5, 100:200] = 1.0   # F
    chroma[7, 200:300] = 1.0   # G
    chroma[0, 300:400] = 1.0   # C
    beats = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    out = beat_sync_chroma(chroma, sr=sr, hop_length=hop_length, beat_times_s=beats)
    assert out.shape == (12, 4)
    assert out[0, 0] == pytest.approx(1.0)   # C in beat 0
    assert out[5, 1] == pytest.approx(1.0)   # F in beat 1
    assert out[7, 2] == pytest.approx(1.0)   # G in beat 2
    assert out[0, 3] == pytest.approx(1.0)   # C in beat 3


def test_beat_sync_returns_empty_when_too_few_beats():
    chroma = np.ones((12, 100), dtype=float)
    out = beat_sync_chroma(chroma, sr=22050, hop_length=512,
                           beat_times_s=np.array([0.0, 0.5]))
    assert out.shape == (12, 1)


def test_beat_sync_returns_empty_for_zero_beats():
    chroma = np.ones((12, 100), dtype=float)
    out = beat_sync_chroma(chroma, sr=22050, hop_length=512,
                           beat_times_s=np.array([]))
    assert out.shape == (12, 0)
