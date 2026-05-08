"""Shared frozen dataclasses for the public API and surviving internals.

This module is the single source of truth for shapes flowing between
ingest → dsp → key → chords → transcription → tabs and through the public
``analyze()`` / ``compose()`` entry points. It deliberately mirrors every
shape that the legacy ``pipeline.contracts`` module exported so that the
import-migration phase is purely mechanical.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

# ---- Music primitives ------------------------------------------------------

VALID_TONICS: tuple[str, ...] = (
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B",
)
_FLAT_TO_SHARP = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}

ChordQuality = Literal["maj", "min", "maj7", "min7", "7", "dim", "sus4", "aug"]

_QUALITY_PATTERNS: tuple[tuple[str, ChordQuality], ...] = (
    ("maj7", "maj7"),
    ("min7", "min7"),
    ("m7", "min7"),
    ("dim", "dim"),
    ("sus4", "sus4"),
    ("aug", "aug"),
    ("m", "min"),
    ("7", "7"),
    ("", "maj"),
)

_QUALITY_TO_LABEL: dict[ChordQuality, str] = {
    "maj": "",
    "min": "m",
    "maj7": "maj7",
    "min7": "m7",
    "7": "7",
    "dim": "dim",
    "sus4": "sus4",
    "aug": "aug",
}


@dataclass(frozen=True)
class Scale:
    tonic: str
    mode: Literal["major", "minor"]

    @classmethod
    def parse(cls, label: str) -> Scale:
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
    def parse(cls, label: str) -> ChordSymbol:
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


# ---- Tuning ----------------------------------------------------------------


@dataclass(frozen=True)
class Tuning:
    """Open-string MIDI pitches, ordered low to high (string 0 = lowest)."""

    name: str
    open_pitches: tuple[int, ...]


# ---- Audio ingestion -------------------------------------------------------


@dataclass(frozen=True)
class AudioSource:
    path: Path
    declared_kind: Literal["solo_guitar", "full_mix"]
    requested_quality: Literal["standard", "high"]
    requested_tuning: Tuning


@dataclass(frozen=True)
class LoadedAudio:
    samples: np.ndarray[Any, np.dtype[np.float32]]
    sr: int
    duration_s: float
    sha256: str
    source: AudioSource


# ---- Separation ------------------------------------------------------------


@dataclass(frozen=True)
class SeparationResult:
    guitar_samples: np.ndarray[Any, np.dtype[np.float32]] | None
    sr: int
    skipped_reason: str | None
    bleed_estimate_db: float | None


# ---- Transcription ---------------------------------------------------------


@dataclass(frozen=True)
class TranscribedNote:
    start_s: float
    end_s: float
    pitch: int
    velocity: int
    confidence: float


@dataclass(frozen=True)
class TranscriptionResult:
    notes: list[TranscribedNote]
    model: Literal["basic-pitch", "crepe", "highres-guitar"]
    raw_midi_path: Path
    post_midi_path: Path
    hyperparameters: dict[str, Any]
    median_confidence: float


# Public Note = same shape as TranscribedNote (kept distinct for clarity in API).
Note = TranscribedNote


# ---- Key detection ---------------------------------------------------------


@dataclass(frozen=True)
class KeyEstimate:
    tonic: str
    mode: Literal["major", "minor"]
    profile: Literal["krumhansl_kessler", "temperley"]
    correlation: float
    margin: float


@dataclass(frozen=True)
class KeyDetectionResult:
    global_top3_per_profile: dict[str, list[KeyEstimate]]
    consensus_key: KeyEstimate | None
    windowed_segments: list[tuple[float, float, KeyEstimate]]
    confidence: float


# ---- Beat tracking ---------------------------------------------------------


@dataclass(frozen=True)
class BeatGrid:
    tempo_bpm: float
    beat_times_s: np.ndarray[Any, np.dtype[np.float64]]
    downbeat_times_s: np.ndarray[Any, np.dtype[np.float64]]
    ts_numerator: int
    ts_denominator: int
    ts_confidence: float
    ts_assumed: bool


# ---- Tab assignment --------------------------------------------------------


@dataclass(frozen=True)
class TabPosition:
    """Fingering position. fret=-1 = string muted (used in chord voicings)."""

    string: int
    fret: int


@dataclass(frozen=True)
class TabbedNote:
    note: TranscribedNote
    position: TabPosition
    cost_breakdown: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class TabAssignmentResult:
    tabbed_notes: list[TabbedNote]
    tuning: Tuning
    total_cost: float
    notes_dropped: list[tuple[TranscribedNote, str]]


# ---- Chord recognition ----------------------------------------------------


@dataclass(frozen=True)
class ChordSegment:
    """One chord-segment from the chord recognizer.

    Carries legacy ``root`` / ``quality`` strings (so the surviving chord-
    detection backends can construct it without change) and exposes a
    ``chord`` property that coerces to the public :class:`ChordSymbol`.
    """

    start_s: float
    end_s: float
    root: str
    quality: str
    confidence: float

    @property
    def chord(self) -> ChordSymbol:
        return ChordSymbol(root=self.root, quality=cast(ChordQuality, self.quality))


@dataclass(frozen=True)
class ChordRecognitionResult:
    segments: list[ChordSegment]
    median_confidence: float
    skipped_reason: str | None


# ---- Voicings (composition) -----------------------------------------------


@dataclass(frozen=True)
class VoicedChord:
    chord: ChordSymbol
    positions: tuple[TabPosition, ...]  # one per string; fret -1 = muted


# ---- Top-level public results ---------------------------------------------


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
    melody_notes: tuple[TranscribedNote, ...]
    chord_voicings: tuple[VoicedChord, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


# ---- Legacy orchestration values (kept for surviving call sites) ----------


@dataclass(frozen=True)
class StageEvent:
    job_id: int
    stage: str
    started_at: datetime
    ended_at: datetime
    success: bool
    error: str | None
    summary: dict[str, Any]
