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


def test_score_beats_returns_per_beat_per_chord_matrix():
    from music_decoder.chord_detection.recognize import score_beats

    # 1 beat with a perfect C-major chroma (energy on C, E, G).
    beat_chroma = np.zeros((12, 1), dtype=float)
    beat_chroma[0, 0] = 1.0   # C
    beat_chroma[4, 0] = 1.0   # E
    beat_chroma[7, 0] = 1.0   # G

    scores = score_beats(beat_chroma)
    assert scores.shape == (1, 49)
    # C-major template should score highest.
    from music_decoder.chord_detection.templates import label_index

    best_idx = int(np.argmax(scores[0]))
    assert best_idx == label_index("C", "maj"), (
        f"expected C maj at idx {label_index('C', 'maj')}, got {best_idx}"
    )


def test_score_beats_handles_empty_input():
    from music_decoder.chord_detection.recognize import score_beats

    out = score_beats(np.zeros((12, 0), dtype=float))
    assert out.shape == (0, 49)


def test_score_beats_zero_chroma_column_returns_zero_row():
    from music_decoder.chord_detection.recognize import score_beats

    out = score_beats(np.zeros((12, 1), dtype=float))
    assert out.shape == (1, 49)
    # All zero similarity (the L2 norm guard returns zeros for zero columns).
    assert np.allclose(out[0], 0.0)


def test_viterbi_smooths_single_flicker():
    from music_decoder.chord_detection.recognize import viterbi_smooth

    # 5 beats. Peaks favor [C, C, F, C, C] but want smoothing to keep C the whole time
    # if F's score on beat 2 is only marginally higher than C's.
    n_states = 49
    scores = np.zeros((5, n_states), dtype=float)
    from music_decoder.chord_detection.templates import label_index
    c_idx = label_index("C", "maj")
    f_idx = label_index("F", "maj")
    scores[:, c_idx] = 0.80
    scores[2, c_idx] = 0.78        # slight dip on beat 2
    scores[2, f_idx] = 0.79        # slight overshoot for F

    path = viterbi_smooth(scores, p_self=0.7)
    assert path.tolist() == [c_idx] * 5


def test_viterbi_returns_argmax_when_self_prob_zero():
    from music_decoder.chord_detection.recognize import viterbi_smooth
    from music_decoder.chord_detection.templates import label_index

    scores = np.zeros((3, 49), dtype=float)
    scores[0, label_index("C", "maj")] = 0.9
    scores[1, label_index("F", "maj")] = 0.9
    scores[2, label_index("G", "maj")] = 0.9
    path = viterbi_smooth(scores, p_self=1.0 / 49)   # uniform -> no smoothing
    assert path.tolist() == [
        label_index("C", "maj"),
        label_index("F", "maj"),
        label_index("G", "maj"),
    ]


def test_viterbi_handles_zero_input():
    from music_decoder.chord_detection.recognize import viterbi_smooth

    path = viterbi_smooth(np.zeros((0, 49), dtype=float), p_self=0.7)
    assert path.shape == (0,)
