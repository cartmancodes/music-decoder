import numpy as np
import pytest

from music_decoder.productionize.config import BassConfig
from music_decoder.productionize.instruments.bass import (
    bass_events,
    bass_runs,
    kick_envelope,
    midi_to_hz,
    render_bass,
)
from music_decoder.productionize.song import hisaab_song

CFG = BassConfig()
SR = 44_100


def _render(kicks=()):
    song = hisaab_song()
    n = song.grid.samples(song.body_s + 7.0)
    return song, render_bass(song, CFG, kicks, n)


def test_midi_to_hz():
    assert midi_to_hz(69) == pytest.approx(440.0)
    assert midi_to_hz(30) == pytest.approx(46.249, abs=1e-3)  # F#1


def test_events_are_the_bar_root_on_steps_0_and_10():
    song = hisaab_song()
    events = bass_events(song, CFG)
    assert len(events) == 6 * 4 * 2
    g = song.grid
    first_bar = events[:2]
    assert [(e.time_s, e.pitch, e.accent) for e in first_bar] == [
        (pytest.approx(g.time(1)), 30, 1.0),
        (pytest.approx(g.time(1, 0, 10)), 30, 0.6),
    ]
    assert [e.pitch for e in events[:8:2]] == [30, 35, 37, 30]
    assert events[-1].time_s < g.time(7)


def test_one_run_from_hook_1_to_the_outro_downbeat():
    song = hisaab_song()
    assert bass_runs(song) == [(pytest.approx(song.grid.time(1)), pytest.approx(song.grid.time(7)))]


def test_bass_is_mono_and_silent_outside_its_run():
    song, out = _render()
    assert np.array_equal(out[:, 0], out[:, 1])
    start = song.grid.samples(song.grid.time(1))
    assert np.abs(out[: start - SR // 10]).max() < 1e-3
    release_end = song.grid.samples(song.grid.time(7) + 3.2 * song.grid.beat_s)
    assert np.abs(out[release_end + SR // 5 :]).max() < 1e-3
    assert np.abs(out[start + SR : start + 2 * SR]).max() > 0.3


def test_last_note_decays_after_the_outro_downbeat():
    song, out = _render()
    end = song.grid.samples(song.grid.time(7))

    def level(offset_s):
        a = end + int(offset_s * SR)
        return np.sqrt(np.mean(out[a : a + SR // 10] ** 2))

    assert level(0.2) > level(1.0) > level(2.0) > 0.0


def test_notes_join_without_clicks():
    """Continuous phase: no sample-to-sample jump beyond what a ~500 Hz
    band-limited signal of this amplitude can make."""
    _, out = _render()
    peak = np.abs(out).max()
    assert np.abs(np.diff(out[:, 0])).max() < peak * 2 * np.pi * 500.0 / SR * 1.5


def test_pitch_follows_the_bar_roots():
    song, out = _render()
    g = song.grid
    for bar, pitch in enumerate(song.bar_roots):
        a = g.samples(g.time(2, bar) + 0.3)
        seg = out[a : a + g.samples(1.2), 0]
        spectrum = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), 1 << 18))
        peak_hz = np.argmax(spectrum) * SR / (1 << 18)
        assert peak_hz == pytest.approx(midi_to_hz(pitch), rel=0.01)


def test_kicks_duck_the_bass_by_about_3_db():
    song = hisaab_song()
    kick = song.grid.time(2) + 1.0
    env = kick_envelope([kick], CFG, SR * 60, SR)
    at = round(kick * SR)
    assert env[at] == pytest.approx(1.0) and env[at - 1] == 0.0
    assert env[at + int(0.06 * SR)] == pytest.approx(np.exp(-1.0), rel=1e-2)
    assert env[at + int(0.21 * SR)] == 0.0
    _, dry = _render()
    _, ducked = _render([kick])
    ratio = np.abs(ducked[at + 5 : at + 200]).max() / np.abs(dry[at + 5 : at + 200]).max()
    assert 20 * np.log10(ratio) == pytest.approx(-3.0, abs=0.3)
    assert np.array_equal(ducked[at + SR :], dry[at + SR :])


def test_harmonics_are_added_for_small_speakers():
    song, out = _render()
    g = song.grid
    a = g.samples(g.time(2) + 0.3)
    seg = out[a : a + g.samples(1.2), 0]
    spectrum = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), 1 << 18))
    hz = np.fft.rfftfreq(1 << 18, 1 / SR)
    f0 = midi_to_hz(30)

    def band(k):
        return spectrum[(hz > k * f0 * 0.97) & (hz < k * f0 * 1.03)].max()

    assert band(2) > 0.05 * band(1) and band(3) > 0.05 * band(1)


def test_no_bass_sections_gives_silence():
    from dataclasses import replace

    song = hisaab_song()
    quiet = replace(song, sections=tuple(replace(s, bass=False) for s in song.sections))
    assert not render_bass(quiet, CFG, (), 1000).any()
