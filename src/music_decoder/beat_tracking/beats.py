from __future__ import annotations

from typing import Any

import librosa
import numpy as np

from music_decoder.pipeline.contracts import BeatGrid


def track_beats(
    samples: np.ndarray[Any, np.dtype[np.float32]],
    *,
    sr: int,
    start_bpm: float,
    tightness: float,
) -> BeatGrid:
    rms = float(np.sqrt(np.mean(samples ** 2))) if samples.size else 0.0
    if rms < 1e-5:
        return BeatGrid(
            tempo_bpm=0.0, beat_times_s=np.zeros(0),
            downbeat_times_s=np.zeros(0),
            ts_numerator=4, ts_denominator=4,
            ts_confidence=0.0, ts_assumed=True,
        )
    tempo, beats = librosa.beat.beat_track(
        y=samples.astype(float), sr=sr,
        start_bpm=start_bpm, tightness=tightness, units="time",
    )
    beats = np.asarray(beats, dtype=float)
    # Normalize tempo to scalar float (newer librosa may return array of size 1)
    tempo_scalar = float(np.atleast_1d(tempo)[0])
    # Crude downbeat estimate: every 4th beat, anchored at first beat.
    downbeats = beats[::4]
    return BeatGrid(
        tempo_bpm=tempo_scalar,
        beat_times_s=beats,
        downbeat_times_s=downbeats,
        ts_numerator=4, ts_denominator=4,
        ts_confidence=0.5, ts_assumed=True,
    )
