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
from music_decoder.types import ChordSymbol, Scale
from tests.productionize_helpers import fake_lead_store, needs_guitar

PROGRESSION = [ChordSymbol.parse(c) for c in ("Am", "F", "C", "G")]
SHORT_FORM = replace(PopRapFormConfig(), intro_bars=2, section_target_bars=8, outro_bars=1)
CFG = replace(PopRapConfig(lead="synth"), form=SHORT_FORM)
SR = 44_100
# Every mix here plays the guitar in hook 2 (the lead, or the synth lead's double).
pytestmark = needs_guitar


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
    ["synth", "guitar", "piano"],
)
def test_lead_fits_buffer_lengths_that_round_down(tmp_path, lead):
    song = _song(tmp_path, replace(CFG, lead=lead))
    form, n = _bad_length_form(song)
    assert int((n / SR) * SR) != n
    cfg = replace(CFG, lead=lead, form=form)
    store = fake_lead_store(tmp_path / "samples", song) if lead == "piano" else None
    result = render_poprap_mix(song, cfg, tmp_path, store=store)
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


# ---- The piano lead ------------------------------------------------------------


@pytest.fixture(scope="module")
def piano_and_guitar(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("poprap_piano")
    song = _song(tmp)
    store = fake_lead_store(tmp / "samples", song)
    piano = render_poprap_mix(song, replace(CFG, lead="piano"), tmp, store=store)
    guitar = render_poprap_mix(song, replace(CFG, lead="guitar"), tmp)
    return song, piano, guitar


def _rms(x):
    return float(np.sqrt(np.mean(x**2))) if len(x) else 0.0


def test_piano_lead_plays_the_melody_where_the_guitar_does(piano_and_guitar):
    song, piano, guitar = piano_and_guitar
    for i, section in enumerate(song.sections):
        a, b = song.section_span(i)
        sounding = {
            k: _rms(r.stems["lead"][a:b]) > 1e-4 for k, r in (("piano", piano), ("guitar", guitar))
        }
        assert sounding["piano"] == sounding["guitar"], section.name
    hooks = [song.section_span(i) for i, s in enumerate(song.sections) if s.role == "hook"]
    (verse,) = [song.section_span(i) for i, s in enumerate(song.sections) if s.role == "verse"]
    hook_energy = sum(_energy(piano.stems["lead"], span) for span in hooks)
    # The verse holds only the last hook-1 note's release and the delay's echo.
    first_bar = (verse[0] + song.grid.samples(song.grid.bar_s), verse[1])
    assert _energy(piano.stems["lead"], first_bar) < 1e-6 * hook_energy
    for name in ("drums", "808", "pad", "pluck", "fx"):
        assert np.array_equal(piano.stems[name], guitar.stems[name]), name


def test_piano_lead_levels_at_minus_17(piano_and_guitar):
    from music_decoder.productionize.config import PopRapMixConfig

    _, piano, _ = piano_and_guitar
    assert PopRapMixConfig().lead_piano_dbfs == -17.0
    lead = piano.stems["lead"]
    audible = lead[np.abs(lead).sum(axis=1) > CFG.mix.silence]
    assert 20 * np.log10(_rms(audible)) == pytest.approx(-17.0, abs=0.5)


def test_piano_lead_needs_a_store(tmp_path):
    from music_decoder.productionize.song import ProductionError

    song = _song(tmp_path)
    with pytest.raises(ProductionError, match="sample store"):
        render_poprap_mix(song, replace(CFG, lead="piano"), tmp_path)


def test_piano_lead_peaks_are_held_to_its_peak_ratio(tmp_path):
    """The piano's hammer transients are soft-limited like the drums, so the master's
    limiter does not have to take them (it cost a real render its loudness target)."""
    song = _song(tmp_path)
    store = fake_lead_store(tmp_path / "samples", song)
    cfg = replace(CFG, lead="piano", mix=replace(CFG.mix, lead_piano_peak_ratio=1.5))
    lead = render_poprap_mix(song, cfg, tmp_path, store=store).stems["lead"]
    target = 10 ** (cfg.mix.lead_piano_dbfs / 20)
    assert np.abs(lead).max() <= 1.5 * target * 1.0001
    assert PopRapConfig().mix.lead_piano_peak_ratio == 4.0


def test_piano_lead_never_plays_the_guitar(tmp_path, monkeypatch):
    """The user asked for piano only: no guitar double under the piano lead."""
    from music_decoder.productionize.instruments import synths

    def no_guitar(*args, **kwargs):
        raise AssertionError("the piano lead played the guitar")

    monkeypatch.setattr(synths, "guitar_lead", no_guitar)
    song = _song(tmp_path)
    store = fake_lead_store(tmp_path / "samples", song)
    lead = render_poprap_mix(song, replace(CFG, lead="piano"), tmp_path, store=store).stems["lead"]
    assert np.abs(lead).max() > 0.0


def test_piano_lead_is_centred_like_the_other_leads(tmp_path):
    """A wide piano plus the ping-pong delay pulled a real mix under the stereo
    correlation floor, so the piano is summed to mono and centred, like the others."""
    song = _song(tmp_path)
    store = fake_lead_store(tmp_path / "samples", song)
    # No double: the hook-2 piano double is panned like the guitar's, on purpose.
    dry_bus = replace(
        CFG.lead_bus, delay_first=0.0, delay_second=0.0, plate_send=0.0, double_gain=0.0
    )
    cfg = replace(CFG, lead="piano", lead_bus=dry_bus)
    lead = render_poprap_mix(song, cfg, tmp_path, store=store).stems["lead"]
    assert np.abs(lead).max() > 0.0
    assert np.allclose(lead[:, 0], lead[:, 1])


def test_piano_lead_doubles_hook_2_on_a_second_piano(tmp_path):
    """The user wanted the guitar's hook-2 part kept, on piano: hook 2 gets louder
    against hook 1 when the double plays, and still no guitar is involved."""
    song = _song(tmp_path)
    store = fake_lead_store(tmp_path / "samples", song)
    hooks = [song.section_span(i) for i, s in enumerate(song.sections) if s.role == "hook"]

    def hook2_over_hook1(double_gain):
        bus = replace(
            CFG.lead_bus, double_gain=double_gain, delay_first=0.0, delay_second=0.0, plate_send=0.0
        )
        lead = render_poprap_mix(
            song, replace(CFG, lead="piano", lead_bus=bus), tmp_path, store=store
        ).stems["lead"]
        return _energy(lead, hooks[1]) / _energy(lead, hooks[0])

    assert hook2_over_hook1(0.35) > 1.2 * hook2_over_hook1(0.0)
