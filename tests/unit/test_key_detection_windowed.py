import numpy as np

from music_decoder.key.windowed import detect_windowed_keys


def test_windowed_segments_cover_audio():
    sr = 22050
    chroma = np.zeros((12, sr * 16 // 512))
    chroma[0] = 1.0   # uniform C presence
    segments = detect_windowed_keys(
        chroma, sr=sr, hop_length=512,
        segment_length_s=8.0, hop_s=2.0,
    )
    assert len(segments) > 0
    starts = [s for (s, _, _) in segments]
    ends = [e for (_, e, _) in segments]
    assert min(starts) == 0.0
    assert max(ends) >= 16.0 - 8.0


def test_windowed_segments_emit_key_estimate():
    sr = 22050
    chroma = np.zeros((12, sr * 8 // 512))
    chroma[0] = 1.0
    segs = detect_windowed_keys(
        chroma, sr=sr, hop_length=512,
        segment_length_s=8.0, hop_s=2.0,
    )
    assert all(seg[2].tonic in ("C", "C#", "D", "D#", "E", "F", "F#", "G",
                                 "G#", "A", "A#", "B") for seg in segs)
