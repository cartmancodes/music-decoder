"""Synthetic samples and tiny songs for the productionize tests.

Nothing here touches the network: the fake store is a temp directory of
generated FLAC/WAV files with the real file names.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import soundfile as sf

from music_decoder.productionize.config import Grid, ProductionConfig, SectionSpec
from music_decoder.productionize.instruments.bass import midi_to_hz
from music_decoder.productionize.samples import (
    SampleRef,
    SampleStore,
    drum_refs,
    piano_ref,
)
from music_decoder.productionize.song import NoteEvent, Song

SR = 44_100
SOURCE_SR = 48_000
LEAD_SILENCE_S = 0.005


def burst(seconds: float, sr: int, seed: int, channels: int) -> np.ndarray:
    """A noise burst with a fast attack and an exponential decay."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    env = np.minimum(1.0, t / 0.001) * np.exp(-t / (seconds / 5.0))
    body = rng.standard_normal((n, channels)) * env[:, None] * 0.5
    return np.concatenate([np.zeros((int(LEAD_SILENCE_S * sr), channels)), body])


def write_drum_samples(root: Path, refs: Iterable[SampleRef]) -> None:
    for seed, ref in enumerate(refs):
        path = root / ref.path
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".flac":
            sf.write(str(path), burst(0.6, SOURCE_SR, seed, 2), SOURCE_SR, subtype="PCM_24")
        else:
            sf.write(str(path), burst(0.6, SR, seed, 1)[:, 0], SR, subtype="PCM_16")


def write_piano_sample(root: Path, pitch: int, layer: int, seconds: float = 4.0) -> None:
    """A decaying sine at the sampled pitch; louder for higher layers."""
    ref = piano_ref(pitch, layer)
    path = root / ref.path
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(seconds * SOURCE_SR)) / SOURCE_SR
    tone = np.sin(2.0 * np.pi * midi_to_hz(pitch) * t) * np.exp(-t / 2.0) * (0.2 + 0.05 * layer)
    lead = np.zeros(int(LEAD_SILENCE_S * SOURCE_SR))
    mono = np.concatenate([lead, tone])
    sf.write(str(path), np.stack([mono, 0.9 * mono], axis=1), SOURCE_SR, subtype="PCM_24")


def fake_store(
    root: Path, song: Song | None = None, cfg: ProductionConfig | None = None
) -> SampleStore:
    """An offline store holding every drum sample and the piano samples
    ``song`` needs."""
    from music_decoder.productionize.instruments.piano import plan_voices

    cfg = cfg or ProductionConfig()
    write_drum_samples(root, drum_refs(cfg.drums))
    if song is not None:
        for voice in plan_voices(song, cfg.piano):
            if not (root / piano_ref(voice.sample_pitch, voice.layer).path).exists():
                write_piano_sample(root, voice.sample_pitch, voice.layer)
    return SampleStore(root, download=False)


def tiny_song(*, strings: bool = False) -> Song:
    """Three two-bar sections at 120 BPM: intro, verse, outro."""
    grid = Grid(beat_s=0.5, bars_per_section=2)
    sections = (
        SectionSpec("Intro", "intro", drums="none", bass=False, reverb_boost=True),
        SectionSpec("Verse", "verse", strings=strings),
        SectionSpec("Outro", "outro", drums="outro", bass=False),
    )
    piano = []
    for section in range(3):
        for bar in range(2):
            start = grid.time(section, bar)
            for k, pitch in enumerate((45, 57, 60, 64)):
                piano.append(NoteEvent(start + 0.02 * k, start + 1.9, pitch, 70))
            piano.append(NoteEvent(start + 1.0, start + 1.5, 72, 80))
    tail = grid.time(3)
    piano.extend(NoteEvent(tail, tail + 4.0, p, 70) for p in (45, 57, 60, 64))
    string_notes = (
        tuple(
            NoteEvent(grid.time(1, b), grid.time(1, b + 1), p, 60)
            for b in range(2)
            for p in (57, 60, 64)
        )
        if strings
        else ()
    )
    return Song(
        title="Tiny",
        key="Am",
        grid=grid,
        sections=sections,
        bar_roots=(33, 33),
        piano=tuple(piano),
        strings=string_notes,
    )
