"""Shared frozen dataclasses for the public API."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

# ---- Music primitives ------------------------------------------------------

VALID_TONICS: tuple[str, ...] = (
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
)
_FLAT_TO_SHARP = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}

ChordQuality = Literal["maj", "min", "maj7", "min7", "7", "dim", "sus4", "aug"]

_QUALITY_PATTERNS: tuple[tuple[str, ChordQuality], ...] = (
    ("maj7", "maj7"),
    ("min7", "min7"),
    ("m7",   "min7"),
    ("dim",  "dim"),
    ("sus4", "sus4"),
    ("aug",  "aug"),
    ("m",    "min"),
    ("7",    "7"),
    ("",     "maj"),
)

_QUALITY_TO_LABEL: dict[ChordQuality, str] = {
    "maj": "", "min": "m", "maj7": "maj7", "min7": "m7",
    "7": "7", "dim": "dim", "sus4": "sus4", "aug": "aug",
}


@dataclass(frozen=True)
class Scale:
    tonic: str
    mode: Literal["major", "minor"]

    @classmethod
    def parse(cls, label: str) -> "Scale":
        if ":" not in label:
            raise ValueError(f"Scale label must be 'TONIC:mode', got {label!r}")
        tonic, mode = label.split(":", 1)
        tonic = _FLAT_TO_SHARP.get(tonic, tonic)
        if tonic not in VALID_TONICS:
            raise ValueError(f"Unknown tonic {tonic!r}")
        if mode not in ("major", "minor"):
            raise ValueError(f"Mode must be 'major' or 'minor', got {mode!r}")
        return cls(tonic=tonic, mode=mode)


@dataclass(frozen=True)
class ChordSymbol:
    root: str
    quality: ChordQuality

    @classmethod
    def parse(cls, label: str) -> "ChordSymbol":
        m = re.match(r"^([A-G][#b]?)(.*)$", label)
        if not m:
            raise ValueError(f"Cannot parse chord symbol {label!r}")
        root = _FLAT_TO_SHARP.get(m.group(1), m.group(1))
        if root not in VALID_TONICS:
            raise ValueError(f"Unknown chord root {root!r}")
        rest = m.group(2)
        for pattern, quality in _QUALITY_PATTERNS:
            if rest == pattern:
                return cls(root=root, quality=quality)
        raise ValueError(f"Unknown chord quality {rest!r} in {label!r}")

    def to_label(self) -> str:
        return f"{self.root}{_QUALITY_TO_LABEL[self.quality]}"


# ---- Analysis-result primitives -------------------------------------------

@dataclass(frozen=True)
class ChordSegment:
    start_s: float
    end_s: float
    chord: ChordSymbol
    confidence: float


@dataclass(frozen=True)
class KeyEstimate:
    tonic: str
    mode: Literal["major", "minor"]
    profile: Literal["krumhansl_kessler", "temperley"]
    correlation: float
    margin: float


@dataclass(frozen=True)
class Note:
    start_s: float
    end_s: float
    pitch: int
    velocity: int
    confidence: float


@dataclass(frozen=True)
class TabPosition:
    string: int          # 0 = lowest
    fret: int            # 0 = open; -1 = muted (in voicings)


@dataclass(frozen=True)
class TabbedNote:
    note: Note
    position: TabPosition


@dataclass(frozen=True)
class Tuning:
    name: str
    open_pitches: tuple[int, ...]   # MIDI numbers low → high


@dataclass(frozen=True)
class VoicedChord:
    chord: ChordSymbol
    positions: tuple[TabPosition, ...]   # one per string; fret -1 = muted


# ---- Top-level results ----------------------------------------------------

@dataclass(frozen=True)
class AnalysisResult:
    source: str
    audio_path: Path
    duration_s: float
    sample_rate_hz: int
    key: KeyEstimate
    chord_progression: tuple[ChordSegment, ...]
    tab: tuple[TabbedNote, ...]
    tempo_bpm: float
    beat_times_s: tuple[float, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Composition:
    midi_path: Path
    wav_path: Path
    ascii_tab: str
    melody_notes: tuple[Note, ...]
    chord_voicings: tuple[VoicedChord, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


# ---- Internal pipeline values (kept for module re-use; not public) -------

@dataclass(frozen=True)
class LoadedAudio:
    samples: np.ndarray
    sr: int
    duration_s: float
    sha256: str
    source_path: Path


@dataclass(frozen=True)
class BeatGrid:
    tempo_bpm: float
    beat_times_s: np.ndarray
    downbeat_times_s: np.ndarray


@dataclass(frozen=True)
class TranscribedNote:
    """Used internally by transcription/. Identical shape to public Note."""
    start_s: float
    end_s: float
    pitch: int
    velocity: int
    confidence: float


@dataclass(frozen=True)
class ChordRecognitionResult:
    segments: tuple[ChordSegment, ...]
    backend: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class KeyDetectionResult:
    global_top3_per_profile: dict[str, list[KeyEstimate]]
    consensus_key: KeyEstimate | None
    windowed_segments: list[tuple[float, float, KeyEstimate]]
    confidence: float
