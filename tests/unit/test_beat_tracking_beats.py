import numpy as np

from music_decoder.dsp.beats import track_beats


def _click_track(bpm: float, duration_s: float, sr: int = 22050) -> np.ndarray:
    """Generate a synthetic click train at the requested tempo."""
    interval_s = 60.0 / bpm
    n = int(duration_s * sr)
    out = np.zeros(n, dtype=np.float32)
    click_len = int(0.005 * sr)
    t = 0.0
    while t < duration_s:
        start = int(t * sr)
        if start + click_len < n:
            out[start : start + click_len] = 1.0
        t += interval_s
    return out


def test_beats_detected_for_120_bpm_clicks():
    sr = 22050
    samples = _click_track(120, duration_s=8.0, sr=sr)
    result = track_beats(samples, sr=sr, start_bpm=120.0, tightness=100.0)
    assert 110 < result.tempo_bpm < 130
    assert len(result.beat_times_s) >= 8


def test_degenerate_audio_returns_zero_beats():
    sr = 22050
    samples = np.zeros(sr, dtype=np.float32)
    result = track_beats(samples, sr=sr, start_bpm=120.0, tightness=100.0)
    assert len(result.beat_times_s) == 0
    assert result.ts_assumed is True
