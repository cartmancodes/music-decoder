"""Synth package: MIDI → WAV rendering.

Exposes :func:`render_wav` as a thin alias over the existing
:func:`fluidsynth_wrapper.synthesize_midi_to_wav` so callers (notably
``compose.api``) can import a stable name regardless of the underlying
backend implementation.
"""
from __future__ import annotations

from pathlib import Path

from music_decoder.synth.fluidsynth_wrapper import (
    SynthBackend,
    synthesize_midi_to_wav,
)


def render_wav(midi_path: Path, out_path: Path) -> Path:
    """Render `midi_path` to `out_path` as WAV using the default backend."""
    return synthesize_midi_to_wav(midi_path, out_path)


__all__ = ["render_wav", "synthesize_midi_to_wav", "SynthBackend"]
