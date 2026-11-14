"""Strings: the song's strings part rendered by the FluidSynth CLI.

General MIDI String Ensemble 2 from a soundfont, band-limited and faded in
so the strings swell in instead of starting hard.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pretty_midi
import soundfile as sf

from music_decoder.logging_setup import get_logger
from music_decoder.productionize import dsp
from music_decoder.productionize.config import StringsConfig
from music_decoder.productionize.dsp import Audio
from music_decoder.productionize.samples import resample
from music_decoder.productionize.song import ProductionError, Song

_log = get_logger("productionize.strings")


def resolve_soundfont(cfg: StringsConfig) -> Path:
    """The configured soundfont, else the one the rest of the app uses."""
    preferred = Path(cfg.soundfont).expanduser()
    if preferred.exists():
        return preferred
    from music_decoder.synth import _resolve_soundfont_path

    fallback = _resolve_soundfont_path()
    if fallback is None:
        raise ProductionError(
            f"no soundfont for the strings: {preferred} is missing and no other is installed "
            "(apt install fluid-soundfont-gm, or run `music-decoder fetch-models`)"
        )
    _log.warning(
        "strings_soundfont_fallback", extra={"wanted": str(preferred), "using": str(fallback)}
    )
    return fallback


def write_strings_midi(song: Song, cfg: StringsConfig, path: Path) -> Path:
    pm = pretty_midi.PrettyMIDI(initial_tempo=song.grid.tempo_bpm, resolution=960)
    inst = pretty_midi.Instrument(program=cfg.program, name="strings")
    inst.notes.extend(
        pretty_midi.Note(velocity=n.velocity, pitch=n.pitch, start=n.start_s, end=n.end_s)
        for n in song.strings
    )
    pm.instruments.append(inst)
    pm.write(str(path))
    return path


def fluidsynth_command(
    cfg: StringsConfig, soundfont: Path, midi: Path, wav: Path, sr: int
) -> list[str]:
    return [
        "fluidsynth",
        "-ni",
        "-q",
        "-g",
        str(cfg.fluidsynth_gain),
        "-r",
        str(sr),
        "-o",
        "synth.reverb.active=1",
        "-o",
        "synth.chorus.active=0",
        "-F",
        str(wav),
        str(soundfont),
        str(midi),
    ]


def fade_in(song: Song, cfg: StringsConfig, n_samples: int) -> npt.NDArray[np.float64]:
    """0 before the first strings section, then ``ramp ** curve`` up to 1."""
    sr = song.grid.sample_rate
    gain = np.ones(n_samples)
    sections = song.strings_sections
    if not sections:
        return gain
    start = min(song.section_span(sections[0])[0], n_samples)
    stop = min(start + int(cfg.fade_in_s * sr), n_samples)
    gain[:start] = 0.0
    gain[start:stop] = (np.arange(stop - start) / (cfg.fade_in_s * sr)) ** cfg.fade_in_curve
    return gain


def render_strings(song: Song, cfg: StringsConfig, work_dir: Path, n_samples: int) -> Audio:
    """The dry strings, filtered and faded in (unlevelled)."""
    sr = song.grid.sample_rate
    if not song.strings:
        return np.zeros((n_samples, 2))
    if shutil.which("fluidsynth") is None:
        raise ProductionError(
            "the strings need the fluidsynth command-line tool "
            "(brew install fluidsynth / apt install fluidsynth)"
        )
    soundfont = resolve_soundfont(cfg)
    work_dir.mkdir(parents=True, exist_ok=True)
    midi = write_strings_midi(song, cfg, work_dir / "strings.mid")
    wav = work_dir / "strings.wav"
    result = subprocess.run(
        fluidsynth_command(cfg, soundfont, midi, wav, sr), capture_output=True, text=True
    )
    if result.returncode != 0 or not wav.exists():
        raise ProductionError(f"fluidsynth failed to render the strings: {result.stderr.strip()}")
    audio, rate = sf.read(str(wav), dtype="float64", always_2d=True)
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    audio = resample(audio, rate, sr)
    out = np.zeros((n_samples, 2))
    length = min(len(audio), n_samples)
    out[:length] = audio[:length, :2]
    out = dsp.butter(out, "highpass", cfg.highpass_hz, sr)
    out = dsp.butter(out, "lowpass", cfg.lowpass_hz, sr)
    faded: Audio = out * fade_in(song, cfg, n_samples)[:, None]
    return faded
