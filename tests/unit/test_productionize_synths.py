import shutil
from dataclasses import replace

import numpy as np
import pytest

from music_decoder.productionize.config import (
    GuitarLeadConfig,
    PadConfig,
    PluckConfig,
    SynthLeadConfig,
)
from music_decoder.productionize.instruments import synths
from music_decoder.productionize.instruments.synths import (
    guitar_lead,
    pad_chord,
    pluck,
    saw,
    synth_lead,
)
from music_decoder.productionize.song import NoteEvent, ProductionError
from music_decoder.synth import _resolve_soundfont_path
from tests.productionize_helpers import SR

HAS_FLUIDSYNTH = shutil.which("fluidsynth") is not None


def rms(x: np.ndarray, start_s: float, stop_s: float) -> float:
    return float(np.sqrt(np.mean(x[int(start_s * SR) : int(stop_s * SR)] ** 2)))


def dominant_hz(x: np.ndarray, start_s: float, stop_s: float) -> float:
    seg = x[int(start_s * SR) : int(stop_s * SR)]
    spectrum = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    return float(np.fft.rfftfreq(len(seg), 1 / SR)[np.argmax(spectrum)])


def test_saw_is_band_limited():
    f0, n = 3000.0, 2 * SR
    x = saw(np.full(n, f0), SR)
    power = np.abs(np.fft.rfft(x * np.hanning(n))) ** 2
    freqs = np.fft.rfftfreq(n, 1 / SR)
    fundamental = power[np.abs(freqs - f0) < 100].sum()
    assert freqs[np.argmax(power)] == pytest.approx(f0, rel=0.01)
    # Above 20 kHz the saw's own 7th harmonic (21 kHz) is genuine; what must stay
    # out is aliasing: energy there that is not on a harmonic of f0.
    off_harmonic = np.abs(freqs - f0 * np.round(freqs / f0)) > 150
    aliased = power[(freqs > 20_000) & off_harmonic].sum()
    assert 10 * np.log10(aliased / fundamental) < -25


def test_saw_glides_with_its_frequency():
    n = SR
    x = saw(np.linspace(200.0, 400.0, n), SR)
    assert dominant_hz(x, 0.0, 0.2) == pytest.approx(220.0, rel=0.1)
    assert dominant_hz(x, 0.8, 1.0) == pytest.approx(380.0, rel=0.1)


def test_pad_chord_holds_then_releases():
    clip = pad_chord((61, 64, 68, 49), 2.0, PadConfig(), SR, np.random.default_rng(0))
    assert len(clip) == int(2.8 * SR)
    # The release is exp(-t / 0.6 s), so 0.7-0.8 s into it the level is about 0.28 of the
    # hold; the detune phases move the measured ratio between roughly 2.8 and 4.9.
    assert rms(clip, 1.0, 1.9) > 2 * rms(clip, 2.7, 2.8)


def test_pad_chord_is_reproducible_per_generator():
    a = pad_chord((61, 64), 0.5, PadConfig(), SR, np.random.default_rng(3))
    b = pad_chord((61, 64), 0.5, PadConfig(), SR, np.random.default_rng(3))
    assert np.array_equal(a, b)


def test_pluck_decays():
    clip = pluck(61, 1.0, 1.0, PluckConfig(), SR)
    assert len(clip) == int((1.0 + 0.4) * SR)
    assert rms(clip, 0.0, 0.2) > 4 * rms(clip, 0.8, 1.0)


def test_pluck_length_is_capped_and_velocity_scales_it():
    cfg = PluckConfig()
    assert len(pluck(61, 5.0, 1.0, cfg, SR)) == int((cfg.max_length_s + cfg.tail_s) * SR)
    assert np.allclose(pluck(61, 1.0, 0.5, cfg, SR) * 2, pluck(61, 1.0, 1.0, cfg, SR))


def test_synth_lead_plays_an_octave_up_and_rests():
    clip = synth_lead([NoteEvent(0.5, 1.5, 57, 100)], 2.5, SynthLeadConfig(), SR)
    assert len(clip) == int(2.5 * SR)
    assert dominant_hz(clip, 0.8, 1.4) == pytest.approx(440.0, rel=0.03)
    assert rms(clip, 2.0, 2.5) < 1e-3 * rms(clip, 0.8, 1.4)


def test_synth_lead_with_no_notes_is_silent():
    assert not synth_lead([], 1.0, SynthLeadConfig(), SR).any()


NOTES = [NoteEvent(0.2, 0.7, 62, 100), NoteEvent(0.8, 1.4, 66, 100)]


def test_guitar_lead_needs_fluidsynth(tmp_path, monkeypatch):
    monkeypatch.setattr(synths.shutil, "which", lambda name: None)
    with pytest.raises(ProductionError, match="fluidsynth"):
        guitar_lead(NOTES, 2.0, GuitarLeadConfig(), 120.0, tmp_path, SR)


@pytest.mark.skipif(not HAS_FLUIDSYNTH, reason="fluidsynth not installed")
def test_guitar_lead_needs_a_soundfont(tmp_path, monkeypatch):
    monkeypatch.setattr("music_decoder.synth._resolve_soundfont_path", lambda: None)
    cfg = replace(GuitarLeadConfig(), soundfont=str(tmp_path / "missing.sf2"))
    with pytest.raises(ProductionError, match="soundfont"):
        guitar_lead(NOTES, 2.0, cfg, 120.0, tmp_path, SR)


@pytest.mark.skipif(
    not HAS_FLUIDSYNTH or _resolve_soundfont_path() is None,
    reason="needs fluidsynth and a soundfont",
)
def test_guitar_lead_renders(tmp_path):
    clip = guitar_lead(NOTES, 2.0, GuitarLeadConfig(), 120.0, tmp_path, SR)
    assert clip.shape == (int(2.0 * SR),)
    assert np.isfinite(clip).all()
    assert np.abs(clip).max() > 0.0
    assert rms(clip, 0.3, 0.6) > 10 * rms(clip, 1.8, 2.0)
