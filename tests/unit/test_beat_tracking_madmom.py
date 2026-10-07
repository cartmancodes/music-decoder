from unittest import mock

import numpy as np

from music_decoder.dsp import track_beats
from music_decoder.dsp.beats_madmom import beat_grid_from_downbeats


def _clicks(sr: int = 22050, seconds: int = 6) -> np.ndarray:
    y = np.zeros(sr * seconds, np.float32)
    y[:: sr // 2] = 1.0
    return y


def test_grid_from_downbeats_three_four() -> None:
    t = np.arange(12) * 0.5
    pos = np.tile([1, 2, 3], 4)
    g = beat_grid_from_downbeats(np.stack([t, pos], axis=1))
    assert g is not None
    assert g.ts_numerator == 3
    assert g.ts_assumed is False
    np.testing.assert_allclose(g.downbeat_times_s, [0.0, 1.5, 3.0, 4.5])
    assert abs(g.tempo_bpm - 120.0) < 1e-6


def test_grid_from_downbeats_too_few_returns_none() -> None:
    assert beat_grid_from_downbeats(np.array([[0.0, 1.0]])) is None


def test_track_beats_uses_madmom_grid_when_configured() -> None:
    rows = np.stack([np.arange(8) * 0.5, np.tile([1, 2, 3, 4], 2)], axis=1)
    grid = beat_grid_from_downbeats(rows)
    with (
        mock.patch("music_decoder.dsp._resolve_backend", return_value="madmom"),
        mock.patch("music_decoder.dsp.beats_madmom.track_beats_madmom", return_value=grid),
    ):
        g = track_beats(_clicks(), 22050)
    assert g is grid


def test_track_beats_falls_back_to_librosa_when_madmom_empty() -> None:
    with (
        mock.patch("music_decoder.dsp._resolve_backend", return_value="madmom"),
        mock.patch("music_decoder.dsp.beats_madmom.track_beats_madmom", return_value=None),
    ):
        g = track_beats(_clicks(), 22050)
    assert g.ts_assumed is True  # librosa path
    assert g.beat_times_s.size > 0


def test_track_beats_falls_back_to_librosa_when_madmom_raises() -> None:
    with (
        mock.patch("music_decoder.dsp._resolve_backend", return_value="madmom"),
        mock.patch(
            "music_decoder.dsp.beats_madmom.track_beats_madmom", side_effect=RuntimeError("x")
        ),
    ):
        g = track_beats(_clicks(), 22050)
    assert g.ts_assumed is True


def test_madmom_downbeat_dbn_runs_on_modern_numpy() -> None:
    from music_decoder.dsp.beats_madmom import _load_processors

    _rnn, dbn = _load_processors()
    fps = 100
    act = np.full((fps * 12, 2), 0.01)
    act[::50, 0] = 0.9  # beat every 0.5 s
    act[::200, 1] = 0.9  # downbeat every 4 beats
    act[::200, 0] = 0.05
    rows = dbn(act)
    g = beat_grid_from_downbeats(rows)
    assert g is not None
    assert g.ts_numerator == 4
    assert abs(g.tempo_bpm - 120.0) < 3.0
