from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pretty_midi
import scipy.io.wavfile as wavfile

from .base import Fixture, GroundTruth

_SR = 22050


def _safe_tempo(pm: pretty_midi.PrettyMIDI) -> float:
    """Return tempo estimate, falling back to first tempo change if estimation fails."""
    try:
        return float(pm.estimate_tempo())
    except ValueError:
        _, tempos = pm.get_tempo_changes()
        return float(tempos[0]) if len(tempos) > 0 else 120.0


class SyntheticFixtures:
    """Renders committed .mid files to .wav via pretty_midi.synthesize() (sine waves)."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _ensure_wav(self, midi_path: Path) -> Path:
        wav_path = midi_path.with_suffix(".wav")
        if wav_path.exists() and wav_path.stat().st_mtime > midi_path.stat().st_mtime:
            return wav_path
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        audio = pm.synthesize(fs=_SR).astype(np.float32)
        peak = float(np.max(np.abs(audio)) or 1.0)
        audio = (audio / peak * 0.9).astype(np.float32)
        wavfile.write(str(wav_path), _SR, (audio * 32767).astype(np.int16))
        return wav_path

    def _ground_truth(self, midi_path: Path) -> GroundTruth:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        notes = [n for inst in pm.instruments for n in inst.notes]
        notes.sort(key=lambda n: (n.start, n.pitch))
        intervals = np.array([(n.start, n.end) for n in notes], dtype=float)
        pitches = np.array([n.pitch for n in notes], dtype=float)
        return GroundTruth(
            intervals=intervals if len(notes) else np.zeros((0, 2)),
            pitches_midi=pitches if len(notes) else np.zeros(0),
            key=None,
            tempo_bpm=_safe_tempo(pm) if notes else None,
            tab=None,
        )

    def load(self) -> Iterator[Fixture]:
        for midi_path in sorted(self.root.glob("*.mid")):
            wav = self._ensure_wav(midi_path)
            yield Fixture(
                name=midi_path.stem, source="synthetic",
                audio_path=wav, ground_truth=self._ground_truth(midi_path),
            )
