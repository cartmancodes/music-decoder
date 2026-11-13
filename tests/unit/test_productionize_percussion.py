import numpy as np
import pytest

from music_decoder.productionize import dsp
from music_decoder.productionize.config import (
    CONGAS,
    HALL,
    SHAKER,
    DrumsConfig,
    PercussionConfig,
    PercussionLayer,
)
from music_decoder.productionize.instruments.percussion import prepare, render_percussion
from music_decoder.productionize.samples import SampleStore, percussion_refs
from tests.productionize_helpers import SR, tiny_song, write_drum_samples

CFG = PercussionConfig()


def test_hand_drums_are_aligned_at_their_onset():
    raw = np.zeros(SR)
    raw[1000:1400] = 0.5
    clip, anchor = prepare(raw, CONGAS.voices[0], CFG, SR)
    assert anchor == 0
    pre_roll = int(0.0015 * SR)
    assert clip[pre_roll] == pytest.approx(1.0) and not clip[: pre_roll - 1].any()
    assert len(clip) == int(0.7 * SR)


def test_shakes_are_aligned_at_their_loudest_point():
    t = np.arange(SR // 2) / SR
    swell = np.exp(-((t - 0.1) ** 2) / (2 * 0.02**2)) * np.random.default_rng(0).standard_normal(
        len(t)
    )
    clip, anchor = prepare(swell, SHAKER.voices[0], CFG, SR)
    assert anchor == pytest.approx(0.1 * SR, abs=0.01 * SR)
    assert len(clip) == int(0.25 * SR)
    assert np.abs(clip).max() == pytest.approx(1.0)
    assert clip[-1] == pytest.approx(0.0, abs=1e-9)  # 20 ms fade at the cut


def test_off_by_default():
    assert not render_percussion(
        tiny_song(), CFG, DrumsConfig(), HALL, SampleStore(None, download=False), 100
    ).any()


def test_layer_is_levelled_and_panned(tmp_path):
    write_drum_samples(tmp_path, percussion_refs(["shaker"]))
    store = SampleStore(tmp_path, download=False)
    song = tiny_song()
    n = song.grid.samples(song.body_s + 2.0)
    cfg = PercussionConfig(layers=(PercussionLayer("shaker", (1,)),))
    out = render_percussion(song, cfg, DrumsConfig(), HALL, store, n)
    start, stop = song.section_span(1)
    assert np.abs(out[: start - 2000]).max() < 1e-12  # only in its sections
    # Level is the dry RMS over its sections (-38.5 dBFS); the hall adds a little.
    assert dsp.rms_db(out[start:stop]) == pytest.approx(-38.5, abs=1.0)
    assert dsp.rms(out[start:stop, :1]) > dsp.rms(out[start:stop, 1:])  # panned left


def test_layers_outside_the_song_are_ignored(tmp_path):
    write_drum_samples(tmp_path, percussion_refs(["tambourine"]))
    store = SampleStore(tmp_path, download=False)
    cfg = PercussionConfig(layers=(PercussionLayer("tambourine", (9,)),))
    assert not render_percussion(tiny_song(), cfg, DrumsConfig(), HALL, store, 1000).any()
