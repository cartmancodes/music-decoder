from collections import Counter

import numpy as np
import pytest

from music_decoder.productionize import dsp
from music_decoder.productionize.config import PLATE, DrumsConfig, ProductionConfig
from music_decoder.productionize.instruments.drums import (
    drum_pattern,
    render_drums,
    shape,
    swell_times,
    thump,
)
from music_decoder.productionize.qc import off_grid
from music_decoder.productionize.song import hisaab_song
from tests.productionize_helpers import SR, fake_store, tiny_song

CFG = DrumsConfig()


def _steps(hits, song, section, piece):
    """(bar, 16th step) of each hit of ``piece`` in ``section``."""
    g = song.grid
    return [
        divmod(round((h.time_s - g.time(section)) / g.step_s), g.steps_per_bar)
        for h in hits
        if h.section == section and h.piece == piece
    ]


def test_every_hit_is_on_an_8th_note():
    song = hisaab_song()
    hits = drum_pattern(song, CFG)
    assert off_grid([h.time_s for h in hits], song.grid.beat_s / 2, 1e-6) == []


def test_drums_play_in_sections_1_to_6_and_mark_the_outro():
    song = hisaab_song()
    hits = drum_pattern(song, CFG)
    assert {h.section for h in hits} == {1, 2, 3, 4, 5, 6, 7}
    outro = [(h.piece, h.gain_db, h.layers) for h in hits if h.section == 7]
    assert sorted(outro) == [("crash", -1.5, (1,)), ("kick", -2.0, (2,))]


def test_kick_pattern():
    song = hisaab_song()
    hits = drum_pattern(song, CFG)
    kicks = _steps(hits, song, 2, "kick")
    assert kicks == [
        (0, 0),
        (0, 10),
        (1, 0),
        (1, 6),
        (1, 10),
        (2, 0),
        (2, 10),
        (3, 0),
        (3, 6),
        (3, 10),
    ]
    verse = [h for h in hits if h.section == 2 and h.piece == "kick"]
    assert [(h.layers, h.gain_db) for h in verse[:2]] == [((3,), 0.0), ((2,), -3.0)]
    hook = [h for h in hits if h.section == 1 and h.piece == "kick"]
    assert {h.layers for h in hook} == {(2,)}
    assert [h.gain_db for h in hook[:2]] == [-2.0, -5.0]


def test_snare_backbeat_plays_through_bar_4():
    song = hisaab_song()
    hits = drum_pattern(song, CFG)
    assert _steps(hits, song, 3, "snare") == [(b, s) for b in range(4) for s in (4, 12)]
    assert {h.layers for h in hits if h.section == 3 and h.piece == "snare"} == {
        (18, 19, 20, 21, 22, 23, 24)
    }
    light = [h for h in hits if h.section == 1 and h.piece == "snare"]
    assert {h.layers for h in light} == {(12, 13, 14, 15, 16)} and {h.gain_db for h in light} == {
        -2.5
    }


def test_hats_and_the_open_hat_transition():
    song = hisaab_song()
    hits = drum_pattern(song, CFG)
    closed = [h for h in hits if h.section == 4 and h.piece == "hat"]
    assert len(closed) == 4 * 8 - 1
    assert [h.gain_db for h in closed[:8]] == [0.0, -6.0, -2.5, -7.0, -0.5, -6.0, -2.5, -5.0]
    opens = [h for h in hits if h.piece == "open_hat"]
    assert len(opens) == 6  # one per groove section, nowhere else
    for h in opens:
        assert _steps([h], song, h.section, "open_hat") == [(3, 14)]
        assert h.choke_s == pytest.approx(song.grid.time(h.section + 1) + 0.030)
    assert {h.gain_db for h in hits if h.section == 1 and h.piece == "hat"} == {
        g - 1.5 for g in (0.0, -6.0, -2.5, -7.0, -0.5, -6.0, -5.0)
    }


def test_crashes_and_swells():
    song = hisaab_song()
    hits = drum_pattern(song, CFG)
    crashes = {h.section: (h.layers, h.gain_db) for h in hits if h.piece == "crash"}
    assert crashes == {
        1: ((2,), -1.0),
        2: ((1,), -1.5),
        3: ((1,), -1.5),
        4: ((1,), -1.5),
        5: ((1,), -1.5),
        6: ((2,), 0.0),
        7: ((1,), -1.5),
    }
    assert swell_times(song) == [song.grid.time(1), song.grid.time(6)]


def test_no_rejected_devices():
    """No in-between 16ths, hat rolls, claps, toms or fills."""
    song = hisaab_song()
    hits = drum_pattern(song, CFG)
    assert Counter(h.piece for h in hits).keys() == {"kick", "snare", "hat", "open_hat", "crash"}
    g = song.grid
    for h in hits:
        step = (h.time_s % g.bar_s) / g.step_s
        assert round(step) % 2 == 0 and abs(step - round(step)) < 1e-6


def test_shape_holds_then_decays_then_cuts():
    hit = np.ones((SR, 2))
    out = shape(hit, CFG.kick, SR)
    assert len(out) == int(0.8 * SR)
    assert np.all(out[: int(0.07 * SR)] == 1.0)
    at = int((0.07 + 0.16) * SR)
    assert out[at, 0] == pytest.approx(np.exp(-1.0), rel=1e-2)
    assert out[-1, 0] == pytest.approx(0.0, abs=1e-6)
    assert shape(hit, CFG.hat_open, SR) is hit  # unshaped


def test_thump_sweeps_down_to_50_hz():
    sub = thump(CFG.thump, SR)
    assert sub.shape == (int(0.220 * SR), 2)
    assert np.array_equal(sub[:, 0], sub[:, 1])
    assert 0.5 < np.abs(sub).max() <= 1.0
    late = sub[int(0.08 * SR) :, 0]
    crossings = np.nonzero(np.diff(np.signbit(late)))[0]
    assert SR / (2 * np.mean(np.diff(crossings))) == pytest.approx(50.0, abs=3.0)


def test_render_humanizes_within_two_ms_and_stays_on_the_grid(tmp_path):
    song = tiny_song()
    store = fake_store(tmp_path)
    n = song.grid.samples(song.body_s + 2.0)
    out = render_drums(song, CFG, PLATE, store, n)
    planned = sorted(h.time_s for h in drum_pattern(song, CFG))
    played = sorted(out.onsets_s)
    assert len(played) == len(planned)
    assert np.max(np.abs(np.array(played) - np.array(planned))) <= 0.002 + 1e-9
    assert off_grid(out.onsets_s, song.grid.beat_s / 2, 0.003) == []
    assert len(out.kick_times_s) == sum(1 for h in drum_pattern(song, CFG) if h.piece == "kick")


def test_render_levels_and_silence(tmp_path):
    song = tiny_song()
    store = fake_store(tmp_path)
    n = song.grid.samples(song.body_s + 2.0)
    out = render_drums(song, CFG, PLATE, store, n)
    for stem in (out.bus, out.kick, out.snare, out.hats, out.crash):
        assert stem.shape == (n, 2) and np.all(np.isfinite(stem))
    intro_end = song.section_span(0)[1]
    assert np.all(out.kick[: intro_end - 200] == 0.0)  # no drums in the intro
    assert dsp.percentile_peak(out.kick, 99.9) == pytest.approx(dsp.db_to_gain(-11.5), rel=1e-6)
    verse = [song.section_span(1)]
    kick, hats, crash = (dsp.rms_over(x, verse) for x in (out.kick, out.hats, out.crash))
    assert dsp.gain_to_db(crash / kick) == pytest.approx(-13.0, abs=0.01)
    # Hats: -6 dB, then the left channel is turned down (slightly right).
    assert np.abs(out.hats[:, 0]).max() == pytest.approx(
        0.85 * np.abs(out.hats[:, 1]).max(), rel=0.2
    )
    assert dsp.gain_to_db(hats / kick) < -6.0
    assert np.abs(out.bus).max() < 1 / 1.5  # tanh(1.5x)/1.5 bounds the bus


def test_render_is_deterministic(tmp_path):
    song = tiny_song()
    store = fake_store(tmp_path)
    n = song.grid.samples(song.body_s + 1.0)
    a = render_drums(song, CFG, PLATE, store, n)
    b = render_drums(song, CFG, PLATE, store, n)
    assert np.array_equal(a.bus, b.bus) and a.onsets_s == b.onsets_s


def test_a_song_without_drums_renders_silence_and_needs_no_samples(tmp_path):
    from dataclasses import replace

    song = tiny_song()
    quiet = replace(song, sections=tuple(replace(s, drums="none") for s in song.sections))
    from music_decoder.productionize.samples import SampleStore

    out = render_drums(
        quiet, ProductionConfig().drums, PLATE, SampleStore(tmp_path, download=False), 1000
    )
    assert not out.bus.any() and out.onsets_s == ()
