from dataclasses import replace

import numpy as np
import pytest

from music_decoder.productionize.config import PianoConfig
from music_decoder.productionize.instruments.bass import midi_to_hz
from music_decoder.productionize.instruments.piano import (
    nearest_sample,
    plan_voices,
    render_piano,
    reverb_feed,
    reverb_send_gain,
    velocity_layer,
)
from music_decoder.productionize.song import NoteEvent, hisaab_song
from tests.productionize_helpers import SR, fake_store, tiny_song

CFG = PianoConfig()


def _song(notes):
    return replace(tiny_song(), piano=tuple(notes))


def test_velocity_layers():
    # velocity x 1.05, clamped to 1..104, first layer whose bound covers it
    assert velocity_layer(1, CFG) == 1
    assert velocity_layer(24, CFG) == 1  # 25.2 -> 25 <= 26
    assert velocity_layer(25, CFG) == 1  # 26.25 -> 26
    assert velocity_layer(26, CFG) == 2  # 27.3 -> 27
    assert velocity_layer(61, CFG) == 8  # 64.05 -> 64
    assert velocity_layer(62, CFG) == 9
    assert velocity_layer(99, CFG) == 13  # 103.95 -> 104
    assert velocity_layer(127, CFG) == 13  # clamped: layers 14-16 are never used


def test_nearest_sample():
    assert nearest_sample(66, CFG) == (66, 0)
    assert nearest_sample(61, CFG) == (60, 1)
    assert nearest_sample(35, CFG) == (36, -1)
    assert all(abs(nearest_sample(p, CFG)[1]) <= 1 for p in range(21, 109))


def test_pedal_holds_a_note_to_the_end_of_its_bar():
    song = _song([NoteEvent(0.5, 0.7, 60, 70)])
    (voice,) = plan_voices(song, CFG)
    assert voice.hold_s == pytest.approx(song.grid.bar_s - 0.020 - 0.5)
    assert voice.tau_s == 0.22


def test_a_note_just_before_the_bar_line_belongs_to_the_next_bar():
    song = _song([NoteEvent(song_bar := tiny_song().grid.bar_s - 0.002, song_bar + 0.5, 60, 70)])
    (voice,) = plan_voices(song, CFG)
    assert voice.hold_s == pytest.approx(song.grid.bar_s - 0.020 + 0.002)


def test_tail_notes_ring_for_their_length_and_release_slowly():
    song = tiny_song()
    tail = song.tail_start_s
    (voice,) = plan_voices(_song([NoteEvent(tail + 0.05, tail + 4.0, 60, 70)]), CFG)
    assert voice.hold_s == pytest.approx(3.95)
    assert voice.tau_s == 0.8


def test_restrike_damps_the_previous_note():
    song = _song(
        [NoteEvent(0.0, 0.5, 60, 70), NoteEvent(0.8, 1.2, 60, 70), NoteEvent(0.8, 1.2, 64, 70)]
    )
    first, second, other = plan_voices(song, CFG)
    assert first.hold_s == pytest.approx(0.8 + 0.030)
    assert first.tau_s == 0.03
    assert second.tau_s == 0.22 and other.tau_s == 0.22


def test_a_note_that_already_died_is_not_cut():
    bar = tiny_song().grid.bar_s
    song = _song([NoteEvent(0.0, 0.5, 60, 70), NoteEvent(2 * bar + 1.0, 2 * bar + 1.5, 60, 70)])
    first, _ = plan_voices(song, CFG)
    assert first.tau_s == 0.22 and first.hold_s == pytest.approx(bar - 0.020)


def test_render_plays_the_shifted_pitch_on_time(tmp_path):
    song = _song([NoteEvent(1.0, 1.5, 61, 70)])  # C#4: the C4 sample, one semitone up
    store = fake_store(tmp_path, song)
    out = render_piano(song, CFG, store, 4 * SR)
    assert out.shape == (4 * SR, 2)
    assert np.abs(out[: SR - 200]).max() < 1e-3 * np.abs(out).max()  # nothing before the onset
    seg = out[SR + 2000 : SR + 2000 + SR // 2, 0]
    spectrum = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), 1 << 17))
    assert np.argmax(spectrum) * SR / (1 << 17) == pytest.approx(midi_to_hz(61), rel=0.005)


def test_release_ends_six_taus_after_the_bar(tmp_path):
    song = _song([NoteEvent(0.0, 0.5, 60, 70)])
    store = fake_store(tmp_path, song)
    out = render_piano(song, CFG, store, 5 * SR)
    release_start = song.grid.bar_s - 0.020
    before = np.abs(out[int((release_start - 0.2) * SR) : int(release_start * SR)]).max()
    after_one_tau = np.abs(
        out[int((release_start + 0.22) * SR) : int((release_start + 0.30) * SR)]
    ).max()
    assert after_one_tau < 0.5 * before
    silent_from = int((release_start + 6 * 0.22 + 0.2) * SR)
    assert np.abs(out[silent_from:]).max() < 1e-4 * np.abs(out).max()


def test_stereo_is_narrowed(tmp_path):
    song = _song([NoteEvent(0.0, 0.5, 60, 70)])
    store = fake_store(tmp_path, song)
    wide = render_piano(song, replace(CFG, side_gain=1.0), store, 2 * SR)
    narrow = render_piano(song, CFG, store, 2 * SR)

    def side(x):
        return np.abs(x[:, 0] - x[:, 1]).max() / np.abs(x[:, 0] + x[:, 1]).max()

    assert side(narrow) == pytest.approx(0.8 * side(wide), rel=0.05)


def test_no_notes_renders_silence(tmp_path):
    from music_decoder.productionize.samples import SampleStore

    assert not render_piano(_song([]), CFG, SampleStore(tmp_path, download=False), 1000).any()


def test_reverb_send_is_louder_in_boosted_sections_with_a_one_second_crossfade():
    song = hisaab_song()
    n = song.grid.samples(song.body_s + 7.0)
    gain = reverb_send_gain(song, CFG, n)
    base = 10 ** (-13 / 20)
    assert gain[song.grid.samples(song.grid.time(3))] == pytest.approx(base)
    assert gain[song.grid.samples(song.grid.time(0, 2))] == pytest.approx(1.5 * base)
    assert gain[song.grid.samples(song.grid.time(7, 2))] == pytest.approx(1.5 * base)
    boundary = song.grid.samples(song.grid.time(1))
    assert gain[boundary] == pytest.approx(1.25 * base, rel=1e-3)  # halfway through the crossfade
    assert gain[boundary + SR] == pytest.approx(base)


def test_reverb_feed_is_band_limited():
    song = hisaab_song()
    t = np.arange(2 * SR) / SR
    low = np.repeat(np.sin(2 * np.pi * 60.0 * t)[:, None], 2, axis=1)
    mid = np.repeat(np.sin(2 * np.pi * 1000.0 * t)[:, None], 2, axis=1)
    quiet = np.sqrt(np.mean(reverb_feed(low, song, CFG)[SR:] ** 2))
    loud = np.sqrt(np.mean(reverb_feed(mid, song, CFG)[SR:] ** 2))
    assert 20 * np.log10(quiet / loud) < -20.0
