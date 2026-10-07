"""madmom RNN + DBN joint beat / downbeat tracking (Böck, Krebs & Widmer, ISMIR 2016)."""

from __future__ import annotations

from typing import Any

import numpy as np

from music_decoder.dsp.tempwav import temp_wav
from music_decoder.types import BeatGrid

_MIN_BEATS = 4

_rnn: Any = None
_dbn: Any = None


def beat_grid_from_downbeats(rows: np.ndarray[Any, np.dtype[Any]]) -> BeatGrid | None:
    """madmom ``(time, beat_position)`` rows → :class:`BeatGrid`; ``None`` if too few beats."""
    arr = np.asarray(rows, dtype=float).reshape(-1, 2)
    if arr.shape[0] < _MIN_BEATS:
        return None
    beats = arr[:, 0]
    positions = arr[:, 1].astype(int)
    ibi = float(np.median(np.diff(beats)))
    return BeatGrid(
        tempo_bpm=60.0 / ibi if ibi > 0 else 0.0,
        beat_times_s=beats,
        downbeat_times_s=beats[positions == 1],
        ts_numerator=int(positions.max()),
        ts_denominator=4,
        ts_confidence=0.8,
        ts_assumed=False,
    )


def _apply_madmom_compat() -> None:
    # Must run before any madmom import; separate function so import sorting
    # can't reorder it below the madmom import.
    from music_decoder.chords import madmom_compat  # noqa: F401


def _load_processors() -> tuple[Any, Any]:
    """Lazily build (and cache) the RNN activation and DBN decoding processors."""
    global _rnn, _dbn
    if _rnn is None:
        _apply_madmom_compat()
        from music_decoder.chords.madmom_compat import patch_downbeats_numpy

        patch_downbeats_numpy()
        from madmom.features.downbeats import (
            DBNDownBeatTrackingProcessor,
            RNNDownBeatProcessor,
        )

        _rnn = RNNDownBeatProcessor()
        _dbn = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)
    return _rnn, _dbn


def track_beats_madmom(samples: np.ndarray[Any, np.dtype[Any]], sr: int) -> BeatGrid | None:
    """Joint beat/downbeat tracking over 3/4 and 4/4 bar hypotheses."""
    rnn, dbn = _load_processors()
    with temp_wav(samples, sr) as path:
        rows = dbn(rnn(str(path)))
    return beat_grid_from_downbeats(rows)
