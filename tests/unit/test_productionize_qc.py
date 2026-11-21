from dataclasses import replace

import numpy as np
import pytest

from music_decoder.compose.api import arrange
from music_decoder.compose.arrangement import build_midi
from music_decoder.productionize import dsp
from music_decoder.productionize.config import (
    MasterConfig,
    PopRapConfig,
    PopRapFormConfig,
    QcConfig,
)
from music_decoder.productionize.master import master, with_lead_in, write_wav
from music_decoder.productionize.poprap import poprap_song
from music_decoder.productionize.poprap_mix import render_poprap_mix
from music_decoder.productionize.qc import (
    Check,
    QcReport,
    first_sound_s,
    full_scale_samples,
    low_band_correlation,
    mono_fold_down_db,
    octave_band_levels,
    off_grid,
    run_qc,
    stereo_correlation,
    tail_peak_db,
)
from music_decoder.types import ChordSymbol, Scale

SR = 44_100


def _tone(freq_hz, seconds=2.0, amp=0.5):
    t = np.arange(int(seconds * SR)) / SR
    return dsp.to_stereo(amp * np.sin(2 * np.pi * freq_hz * t))


def test_first_sound():
    x = np.zeros((SR, 2))
    x[5292:] = 0.01  # 120 ms
    assert first_sound_s(x, SR, -60.0) == pytest.approx(0.120, abs=1e-4)
    assert first_sound_s(x * 0.01, SR, -60.0) is None  # -80 dBFS never crosses -60


def test_tail_peak():
    x = np.ones((SR, 2))
    x[-int(0.05 * SR) :] = 1e-5
    assert tail_peak_db(x, SR, 0.050) == pytest.approx(-100.0, abs=0.1)


def test_stereo_measures():
    mono = _tone(440.0)
    assert stereo_correlation(mono) == pytest.approx(1.0)
    assert mono_fold_down_db(mono) == pytest.approx(0.0, abs=1e-6)
    rng = np.random.default_rng(0)
    wide = rng.standard_normal((SR, 2))
    assert abs(stereo_correlation(wide)) < 0.05
    assert mono_fold_down_db(wide) == pytest.approx(-3.0, abs=0.2)
    inverted = mono * np.array([1.0, -1.0])
    assert stereo_correlation(inverted) == pytest.approx(-1.0)
    assert stereo_correlation(np.zeros((100, 2))) == 1.0


def test_low_band_correlation_ignores_the_highs():
    low = _tone(60.0)
    highs = np.random.default_rng(0).standard_normal((len(low), 2)) * 0.2
    highs = dsp.butter(highs, "highpass", 2000.0, SR, order=4)
    assert low_band_correlation(low + highs, SR, 110.0) > 0.99
    assert stereo_correlation(low + highs) < 0.99


def test_full_scale_samples():
    x = np.array([[0.5, -0.99], [1.0, 0.2], [-1.2, 0.99999]])
    assert full_scale_samples(x) == 3
    assert full_scale_samples(x * 0.5) == 0


def test_off_grid():
    eighth = 0.25
    onsets = [0.0, 0.2515, 0.5035, 0.7499, 1.1]
    assert off_grid(onsets, eighth, 0.003) == [0.5035, 1.1]
    assert off_grid([], eighth, 0.003) == []


def test_octave_bands_put_a_tone_in_its_band():
    centres = (62.5, 125.0, 250.0, 500.0, 1000.0, 2000.0)
    x = _tone(1000.0, seconds=4.0)
    levels = octave_band_levels(x, SR, centres, -10.0, 2048)
    assert int(np.argmax(levels)) == 4
    assert levels[4] - np.delete(levels, 4).max() > 30.0
    # Loudness-matched: the same tone at another level reads the same.
    quieter = octave_band_levels(x * 0.1, SR, centres, -10.0, 2048)
    assert np.allclose(levels[4], quieter[4], atol=0.05)


def test_octave_bands_measure_the_mono_sum():
    centres = (500.0, 1000.0, 2000.0)
    x = _tone(1000.0, seconds=4.0)
    side = _tone(1000.0, seconds=4.0, amp=0.4) * np.array([1.0, -1.0])  # cancels in mono
    assert np.allclose(
        octave_band_levels(x + side, SR, centres, -10.0, 2048),
        octave_band_levels(x, SR, centres, -10.0, 2048),
    )


def test_report():
    report = QcReport(
        (
            Check("loudness", True, "-10.5 LUFS", "-10.5 +-0.3"),
            Check("section RMS", None, "-17.0", "no reference master"),
            Check("true peak", False, "-0.5 dBTP", "at or below -1.93"),
        )
    )
    assert not report.passed
    assert [c.name for c in report.failures] == ["true peak"]
    lines = report.format().splitlines()
    assert (
        lines[0].startswith("PASS") and lines[1].startswith("info") and lines[2].startswith("FAIL")
    )
    as_dict = report.to_dict()
    assert as_dict["passed"] is False and len(as_dict["checks"]) == 3
    assert QcReport((Check("a", True, "", ""), Check("b", None, "", ""))).passed


def test_run_qc_accepts_a_poprap_song_and_a_16th_grid(tmp_path):
    progression = [ChordSymbol.parse(c) for c in ("Am", "F", "C", "G")]
    form = replace(PopRapFormConfig(), intro_bars=2, section_target_bars=8, outro_bars=1)
    cfg = replace(PopRapConfig(lead="synth"), form=form)
    arr = arrange(Scale("A", "minor"), progression, bars_per_chord=2, tempo_bpm=120.0, seed=7)
    midi = tmp_path / "composition.mid"
    build_midi(
        melody=arr.melody,
        voicings=arr.voicings,
        bars_per_chord=2,
        tempo_bpm=120.0,
        style="fingerstyle",
        accompaniment=arr.accompaniment,
        out_path=midi,
    )
    song = poprap_song(midi, cfg=cfg, progression=progression, scale=Scale("A", "minor"), key="Am")
    mix = render_poprap_mix(song, cfg, tmp_path)
    master_cfg = MasterConfig()
    mastered, _ = master(mix.mix, master_cfg, SR)
    wav = write_wav(
        tmp_path / "song.wav", with_lead_in(mastered, master_cfg, SR), SR, subtype="PCM_24"
    )

    report = run_qc(
        song,
        wav,
        None,
        mix.drum_onsets_s,
        QcConfig(),
        master_cfg,
        tail_s=cfg.form.tail_s,
        drum_grid="16th",
    )
    checks = {c.name: c for c in report.checks}
    for name in ("format", "drum grid", "DC offset", "loudness (WAV)"):
        assert checks[name].passed, checks[name]
    assert checks["drum grid"].value.endswith("on a 16th note")
    assert checks["section RMS (dBFS)"].passed is None
    assert checks["beat jitter"].passed is not None  # checked by default

    unchecked = run_qc(
        song,
        wav,
        None,
        mix.drum_onsets_s,
        QcConfig(beat_jitter_max_s=0.0),  # no jitter could pass this
        master_cfg,
        tail_s=cfg.form.tail_s,
        drum_grid="16th",
        check_beat_jitter=False,
    )
    jitter = {c.name: c for c in unchecked.checks}["beat jitter"]
    assert jitter.passed is None
    assert jitter.value.endswith(" ms")
    assert jitter.expected == "not checked: half-time trap (see drum grid)"
    assert [c.name for c in unchecked.failures] == [
        c.name for c in report.failures if c.name != "beat jitter"
    ]  # the jitter is the only difference; it does not fail the report
    assert len(unchecked.checks) == len(report.checks)
