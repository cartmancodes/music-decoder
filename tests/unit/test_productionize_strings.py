import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pretty_midi
import pytest

from music_decoder.productionize.config import StringsConfig
from music_decoder.productionize.instruments import strings
from music_decoder.productionize.instruments.strings import (
    fade_in,
    fluidsynth_command,
    render_strings,
    resolve_soundfont,
    write_strings_midi,
)
from music_decoder.productionize.song import ProductionError, hisaab_song
from tests.productionize_helpers import SR, tiny_song

CFG = StringsConfig()


def test_fluidsynth_command_matches_the_build_spec():
    cmd = fluidsynth_command(CFG, Path("font.sf2"), Path("strings.mid"), Path("strings.wav"), SR)
    assert cmd == [
        "fluidsynth", "-ni", "-q", "-g", "0.6", "-r", "44100",
        "-o", "synth.reverb.active=1", "-o", "synth.chorus.active=0",
        "-F", "strings.wav", "font.sf2", "strings.mid",
    ]  # fmt: skip


def test_strings_midi_uses_string_ensemble_2(tmp_path):
    song = hisaab_song()
    path = write_strings_midi(song, CFG, tmp_path / "strings.mid")
    (inst,) = pretty_midi.PrettyMIDI(str(path)).instruments
    assert inst.program == 49 and len(inst.notes) == 34
    assert min(n.start for n in inst.notes) == pytest.approx(song.grid.time(6), abs=2e-3)


def test_fade_in_swells_from_the_first_strings_downbeat():
    song = hisaab_song()
    n = song.grid.samples(song.body_s + 7.0)
    gain = fade_in(song, CFG, n)
    start = song.grid.samples(song.grid.time(6))
    assert np.all(gain[:start] == 0.0)
    assert gain[start + int(0.75 * SR)] == pytest.approx(0.5**0.7, rel=1e-3)
    assert np.all(gain[start + int(1.5 * SR) :] == 1.0)
    assert np.all(np.diff(gain[start : start + int(1.5 * SR)]) >= 0)


def test_no_strings_part_renders_silence_without_fluidsynth(tmp_path, monkeypatch):
    monkeypatch.setattr(strings.shutil, "which", lambda name: None)
    assert not render_strings(tiny_song(), CFG, tmp_path, 1000).any()


def test_missing_fluidsynth_is_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(strings.shutil, "which", lambda name: None)
    with pytest.raises(ProductionError, match="fluidsynth"):
        render_strings(tiny_song(strings=True), CFG, tmp_path, 1000)


def test_soundfont_falls_back_to_the_installed_one(tmp_path):
    font = tmp_path / "font.sf2"
    font.write_bytes(b"RIFF")
    assert resolve_soundfont(replace(CFG, soundfont=str(font))) == font
    fallback = resolve_soundfont(replace(CFG, soundfont=str(tmp_path / "missing.sf2")))
    assert fallback.exists()  # pretty_midi's bundled font at the least


@pytest.mark.skipif(shutil.which("fluidsynth") is None, reason="fluidsynth not installed")
def test_render_strings_sounds_only_after_the_fade_starts(tmp_path):
    song = tiny_song(strings=True)
    n = song.grid.samples(song.body_s + 2.0)
    out = render_strings(song, CFG, tmp_path, n)
    start = song.section_span(1)[0]
    assert out.shape == (n, 2)
    assert not out[:start].any()
    assert np.abs(out[start + SR : start + 2 * SR]).max() > 1e-3
