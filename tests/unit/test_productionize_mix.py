import numpy as np
import pytest

from music_decoder.productionize import dsp
from music_decoder.productionize.config import MixConfig, ProductionConfig
from music_decoder.productionize.mix import (
    STEM_NAMES,
    fade_out,
    finish,
    mix_length,
    mono_low_end,
    render_mix,
)
from music_decoder.productionize.qc import low_band_correlation
from music_decoder.productionize.song import hisaab_song
from tests.productionize_helpers import SR, fake_store, tiny_song

CFG = MixConfig()


def test_buffer_is_the_sections_plus_seven_seconds():
    song = hisaab_song()
    assert mix_length(song, CFG) / SR == pytest.approx(8 * song.grid.section_s + 7.0, abs=1e-4)


def test_low_end_is_made_mono_and_the_rest_is_left_alone():
    t = np.arange(2 * SR) / SR
    x = np.stack([np.sin(2 * np.pi * 50 * t), np.sin(2 * np.pi * 50 * t + 2.0)], axis=1)
    x += np.stack([np.sin(2 * np.pi * 3000 * t), np.zeros_like(t)], axis=1)
    out = mono_low_end(x, CFG, SR)
    assert low_band_correlation(x, SR, 80.0) < 0.5
    assert low_band_correlation(out, SR, 80.0) > 0.999
    high = dsp.butter(out, "highpass", 1000.0, SR, order=4)[SR:]
    assert dsp.rms(high[:, 1:]) < 0.01 * dsp.rms(high[:, :1])  # the 3 kHz tone stays left
    mono = np.repeat(np.sin(2 * np.pi * 50 * t)[:, None], 2, axis=1)
    assert np.allclose(mono_low_end(mono, CFG, SR), mono)


def test_fade_out_covers_the_last_two_and_a_half_seconds():
    x = np.ones((5 * SR, 2))
    out = fade_out(x, CFG, SR)
    n = int(2.5 * SR)
    assert np.all(out[:-n] == 1.0)
    assert out[-n // 2, 0] == pytest.approx(0.5**1.5, rel=1e-3)
    assert out[-1, 0] == pytest.approx(0.0, abs=1e-6)
    assert np.all(x == 1.0)  # the input is not modified


def test_finish_adds_air():
    rng = np.random.default_rng(0)
    noise = rng.standard_normal((4 * SR, 2)) * 0.1
    out = finish(noise, CFG, SR)[:SR]
    top_in = dsp.rms(dsp.butter(noise[:SR], "highpass", 14_000.0, SR, order=4))
    top_out = dsp.rms(dsp.butter(out, "highpass", 14_000.0, SR, order=4))
    assert dsp.gain_to_db(top_out / top_in) == pytest.approx(1.5, abs=0.3)


def test_each_bus_lands_on_its_level(tmp_path):
    song = tiny_song()
    cfg = ProductionConfig()
    result = render_mix(song, cfg, fake_store(tmp_path, song, cfg), tmp_path / "work")
    assert set(result.stems) == set(STEM_NAMES)
    n = mix_length(song, cfg.mix)
    assert result.mix.shape == (n, 2)
    verse = [song.section_span(1)]
    assert dsp.gain_to_db(dsp.rms_over(result.stems["drums"], verse)) == pytest.approx(
        -28.4, abs=0.01
    )
    assert dsp.gain_to_db(dsp.rms_over(result.stems["bass"], verse)) == pytest.approx(
        -15.0, abs=0.01
    )
    # The piano stem carries its hall return on top of the -22.8 dBFS dry level.
    piano = dsp.gain_to_db(dsp.rms_over(result.stems["piano"], verse))
    assert piano == pytest.approx(-22.8, abs=1.5)
    assert not result.stems["strings"].any()
    assert result.mix[-1].tolist() == pytest.approx([0.0, 0.0], abs=1e-6)
    assert low_band_correlation(result.mix, SR, 110.0) > 0.99
    assert len(result.drum_onsets_s) > 0
