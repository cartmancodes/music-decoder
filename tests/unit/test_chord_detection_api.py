# tests/unit/test_chord_detection_api.py
import numpy as np

from music_decoder.chords.api import detect_chords
from music_decoder.config.hyperparameters import ChordDetectionParams
from music_decoder.types import BeatGrid


def _params() -> ChordDetectionParams:
    return ChordDetectionParams(
        qualities=["maj", "min", "7", "maj7", "min7", "dim", "sus4", "aug"],
        hmm_self_transition_prob=0.7,
        no_chord_threshold=0.15,
        min_segment_duration_s=0.0,
    )


def _beat_grid(beats: list[float]) -> BeatGrid:
    return BeatGrid(
        tempo_bpm=120.0,
        beat_times_s=np.asarray(beats, dtype=float),
        downbeat_times_s=np.asarray(beats[::4], dtype=float),
        ts_numerator=4,
        ts_denominator=4,
        ts_confidence=0.6,
        ts_assumed=False,
    )


def test_detect_chords_emits_c_major_for_uniform_c_chroma():
    sr = 22050
    hop_length = 512
    # 4 beats x 1 second = 4 seconds. At hop=512, sr=22050: ~43 columns/sec.
    n_cols = int(4 * sr / hop_length) + 1
    chroma = np.zeros((12, n_cols), dtype=float)
    chroma[0] = 1.0  # C
    chroma[4] = 1.0  # E
    chroma[7] = 1.0  # G

    grid = _beat_grid([0.0, 1.0, 2.0, 3.0, 4.0])
    result = detect_chords(
        chroma=chroma,
        sr=sr,
        hop_length=hop_length,
        beat_grid=grid,
        params=_params(),
    )
    assert result.skipped_reason is None
    assert len(result.segments) == 1
    seg = result.segments[0]
    assert seg.root == "C"
    assert seg.quality == "maj"


def test_detect_chords_skips_when_beat_grid_too_short():
    chroma = np.ones((12, 100), dtype=float)
    grid = _beat_grid([0.0, 0.5, 1.0])  # only 3 beats
    result = detect_chords(
        chroma=chroma,
        sr=22050,
        hop_length=512,
        beat_grid=grid,
        params=_params(),
    )
    assert result.skipped_reason == "degenerate_beat_grid"
    assert result.segments == []


def test_detect_chords_emits_n_for_silent_audio():
    chroma = np.zeros((12, 200), dtype=float)
    grid = _beat_grid([0.0, 1.0, 2.0, 3.0, 4.0])
    result = detect_chords(
        chroma=chroma,
        sr=22050,
        hop_length=512,
        beat_grid=grid,
        params=_params(),
    )
    assert result.skipped_reason is None
    assert all(s.root == "N" for s in result.segments)
