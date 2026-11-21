"""Shared FluidSynth command-line rendering: soundfont lookup, argv and note rendering.

Used by the strings (which render a whole song's part) and by the guitar lead
(which renders a list of notes with :func:`render_notes`).
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf

from music_decoder.productionize.dsp import Audio
from music_decoder.productionize.samples import resample
from music_decoder.productionize.song import NoteEvent, ProductionError

_MIDI_RESOLUTION = 960


def find_soundfont(preferred: str, part: str) -> Path:
    """``preferred`` if it names an existing file, else the app's soundfont.

    An empty ``preferred`` means "no preferred path". ``part`` names the
    instrument in the error when no soundfont can be found.
    """
    wanted = Path(preferred).expanduser() if preferred else None
    if wanted is not None and wanted.exists():
        return wanted
    from music_decoder.synth import _resolve_soundfont_path

    fallback = _resolve_soundfont_path()
    if fallback is None:
        missing = f"{wanted} is missing and no other" if wanted is not None else "none"
        raise ProductionError(
            f"no soundfont for the {part}: {missing} is installed "
            "(apt install fluid-soundfont-gm, or run `music-decoder fetch-models`)"
        )
    return fallback


def fluidsynth_command(gain: float, soundfont: Path, midi: Path, wav: Path, sr: int) -> list[str]:
    return [
        "fluidsynth",
        "-ni",
        "-q",
        "-g",
        str(gain),
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


def render_notes(
    notes: Sequence[NoteEvent],
    program: int,
    tempo_bpm: float,
    soundfont: Path,
    gain: float,
    work_dir: Path,
    stem: str,
    sr: int,
    n_samples: int,
) -> Audio:
    """The notes on one General MIDI program, as stereo zero-padded or cut to ``n_samples``."""
    work_dir.mkdir(parents=True, exist_ok=True)
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo_bpm, resolution=_MIDI_RESOLUTION)
    inst = pretty_midi.Instrument(program=program, name=stem)
    inst.notes.extend(
        pretty_midi.Note(velocity=n.velocity, pitch=n.pitch, start=n.start_s, end=n.end_s)
        for n in notes
    )
    pm.instruments.append(inst)
    midi = work_dir / f"{stem}.mid"
    pm.write(str(midi))
    wav = work_dir / f"{stem}.wav"
    result = subprocess.run(
        fluidsynth_command(gain, soundfont, midi, wav, sr), capture_output=True, text=True
    )
    if result.returncode != 0 or not wav.exists():
        raise ProductionError(f"fluidsynth failed to render the {stem}: {result.stderr.strip()}")
    audio, rate = sf.read(str(wav), dtype="float64", always_2d=True)
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    audio = resample(audio, rate, sr)
    out = np.zeros((n_samples, 2))
    length = min(len(audio), n_samples)
    out[:length] = audio[:length, :2]
    return out
