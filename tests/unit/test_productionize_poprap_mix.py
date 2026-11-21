import shutil
from dataclasses import replace

import numpy as np
import pytest

from music_decoder.compose.api import arrange
from music_decoder.compose.arrangement import build_midi
from music_decoder.productionize.config import PopRapConfig, PopRapFormConfig
from music_decoder.productionize.poprap import drum_hits, poprap_song
from music_decoder.productionize.poprap_mix import (
    POPRAP_STEMS,
    poprap_length,
    render_poprap_mix,
)
from music_decoder.synth import _resolve_soundfont_path
from music_decoder.types import ChordSymbol, Scale

PROGRESSION = [ChordSymbol.parse(c) for c in ("Am", "F", "C", "G")]
SHORT_FORM = replace(PopRapFormConfig(), intro_bars=2, section_target_bars=8, outro_bars=1)
CFG = replace(PopRapConfig(lead="synth"), form=SHORT_FORM)
SR = 44_100
HAS_GUITAR = shutil.which("fluidsynth") is not None and _resolve_soundfont_path() is not None


def _song(tmp_path, cfg=CFG, tempo_bpm=120.0):
    midi = tmp_path / "composition.mid"
    arr = arrange(Scale("A", "minor"), PROGRESSION, bars_per_chord=2, tempo_bpm=tempo_bpm, seed=7)
    build_midi(
        melody=arr.melody,
        voicings=arr.voicings,
        bars_per_chord=2,
        tempo_bpm=tempo_bpm,
        style="fingerstyle",
        accompaniment=arr.accompaniment,
        out_path=midi,
    )
    return poprap_song(
        midi,
        cfg=cfg,
        progression=PROGRESSION,
        scale=Scale("A", "minor"),
        key="Am",
    )


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("poprap_mix")
    song = _song(tmp)
    return song, render_poprap_mix(song, CFG, tmp)


def _energy(x, span):
    a, b = span
    return float(np.sum(x[a:b] ** 2))


def test_stems_and_length(rendered):
    song, result = rendered
    n = poprap_length(song, CFG.form)
    assert n == song.grid.samples(song.body_s + CFG.form.tail_s)
    assert tuple(result.stems) == POPRAP_STEMS
    for audio in (result.mix, *result.stems.values()):
        assert audio.shape == (n, 2)
        assert np.isfinite(audio).all()
    assert np.abs(result.mix).max() > 0.0


def _hook_energy(song, result):
    hooks = [song.section_span(i) for i, s in enumerate(song.sections) if s.role == "hook"]
    return {k: sum(_energy(v, span) for span in hooks) for k, v in result.stems.items()}


def test_drums_lead_the_mix_and_the_808_sits_under_them(rendered):
    # The synth lead (-16.5 dBFS RMS) is as loud as the drums in the hooks, by the
    # spike's own levels; the guitar test below holds the drums over the lead too.
    song, result = rendered
    energy = _hook_energy(song, result)
    others = [v for k, v in energy.items() if k not in ("drums", "lead")]
    assert energy["drums"] > max(others)
    assert energy["808"] < energy["drums"]


def test_drum_onsets_are_kicks_and_claps(rendered):
    song, result = rendered
    hits = drum_hits(song, CFG.drums, riser_bars=CFG.form.riser_bars)
    expected = sorted(h.time_s for h in hits if h.kind in ("kick", "clap"))
    assert len(result.drum_onsets_s) == len(expected)
    assert list(result.drum_onsets_s) == pytest.approx(expected)


def test_render_is_deterministic(rendered, tmp_path):
    song, result = rendered
    again = render_poprap_mix(song, CFG, tmp_path)
    assert np.array_equal(again.mix, result.mix)


def test_the_mix_fades_to_silence(rendered):
    _, result = rendered
    tail = result.mix[-int(0.05 * SR) :]
    assert 20 * np.log10(np.abs(tail).max() + 1e-12) < -80.0


def test_progress_reports_each_stage(rendered, tmp_path):
    song, _ = rendered
    seen: list[str] = []
    render_poprap_mix(song, CFG, tmp_path, lambda stage, fraction: seen.append(stage))
    assert seen == ["drums", "808", "pad", "pluck", "lead", "mix"]


def test_slow_piece_renders(tmp_path):
    song = _song(tmp_path, tempo_bpm=66.0)
    assert song.sections[1].bars == 8
    result = render_poprap_mix(song, CFG, tmp_path)
    assert np.isfinite(result.mix).all()
    assert result.mix.shape == (poprap_length(song, CFG.form), 2)
    assert np.abs(result.stems["lead"]).max() > 0.0


@pytest.mark.skipif(not HAS_GUITAR, reason="needs fluidsynth and a soundfont")
def test_guitar_mix_renders(tmp_path):
    cfg = replace(CFG, lead="guitar")
    song = _song(tmp_path, cfg)
    result = render_poprap_mix(song, cfg, tmp_path)
    assert np.isfinite(result.mix).all()
    assert np.abs(result.stems["lead"]).max() > 0.0
    energy = _hook_energy(song, result)
    assert energy["drums"] == max(energy.values())


def _bad_length_form(song):
    """A form whose buffer length n has ``int((n / sr) * sr) != n``."""
    for k in range(3000):
        form = replace(SHORT_FORM, tail_s=SHORT_FORM.tail_s + k * 0.01)
        n = poprap_length(song, form)
        if int((n / SR) * SR) != n:
            return form, n
    raise AssertionError("no bad buffer length found")


@pytest.mark.parametrize(
    "lead",
    [
        "synth",
        pytest.param(
            "guitar",
            marks=pytest.mark.skipif(not HAS_GUITAR, reason="needs fluidsynth and a soundfont"),
        ),
    ],
)
def test_lead_fits_buffer_lengths_that_round_down(tmp_path, lead):
    song = _song(tmp_path, replace(CFG, lead=lead))
    form, n = _bad_length_form(song)
    assert int((n / SR) * SR) != n
    cfg = replace(CFG, lead=lead, form=form)
    result = render_poprap_mix(song, cfg, tmp_path)
    assert result.mix.shape == (n, 2)
    assert result.stems["lead"].shape == (n, 2)
    assert np.abs(result.stems["lead"]).max() > 0.0


def _energy_above(x, hz):
    spectrum = np.abs(np.fft.rfft(x.mean(axis=1))) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / SR)
    return float(spectrum[freqs > hz].sum())


def test_drums_and_fx_are_rolled_off_above_the_bright_lowpass(tmp_path):
    song = _song(tmp_path)
    bright = replace(CFG, mix=replace(CFG.mix, bright_lowpass_hz=21_000.0))
    (tmp_path / "bright").mkdir()
    open_ = render_poprap_mix(song, bright, tmp_path / "bright")
    rolled = render_poprap_mix(song, CFG, tmp_path)
    assert CFG.mix.bright_lowpass_hz == 16_000.0
    for name in ("drums", "fx"):
        assert _energy_above(rolled.stems[name], 17_000) < 0.5 * _energy_above(
            open_.stems[name], 17_000
        )
    # The other stems are untouched.
    for name in ("808", "pad", "pluck", "lead"):
        assert np.array_equal(rolled.stems[name], open_.stems[name])
