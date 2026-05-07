from __future__ import annotations

from enum import StrEnum
from pathlib import Path

import numpy as np
import pretty_midi
import scipy.io.wavfile as wavfile


class SynthBackend(StrEnum):
    SINE = "sine"
    FLUIDSYNTH = "fluidsynth"


def synthesize_midi_to_wav(
    midi_path: Path,
    output_wav: Path,
    *,
    sr: int = 22050,
    backend: SynthBackend = SynthBackend.SINE,
    soundfont_path: Path | None = None,
) -> Path:
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    if backend == SynthBackend.FLUIDSYNTH and soundfont_path is not None:
        audio = pm.fluidsynth(fs=sr, sf2_path=str(soundfont_path))
    else:
        audio = pm.synthesize(fs=sr)
    audio = audio.astype(np.float32)
    peak = float(np.max(np.abs(audio)) or 1.0)
    audio = (audio / peak * 0.9).astype(np.float32)
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(output_wav), sr, (audio * 32767).astype(np.int16))
    return output_wav
