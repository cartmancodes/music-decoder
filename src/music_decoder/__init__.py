"""Music Decoder: chord progressions, guitar tabs, composition suggestions."""

from music_decoder.api import analyze, compose
from music_decoder.tabs.tuning import (
    D_STANDARD,
    DADGAD,
    DROP_C,
    DROP_D,
    EB_HALF_STEP_DOWN,
    STANDARD_EADGBE,
)
from music_decoder.types import (
    AnalysisResult,
    ChordSegment,
    ChordSymbol,
    Composition,
    KeyEstimate,
    Note,
    Scale,
    TabbedNote,
    TabPosition,
    Tuning,
    VoicedChord,
)

__all__ = [
    "DADGAD",
    "DROP_C",
    "DROP_D",
    "D_STANDARD",
    "EB_HALF_STEP_DOWN",
    "STANDARD_EADGBE",
    "AnalysisResult",
    "ChordSegment",
    "ChordSymbol",
    "Composition",
    "KeyEstimate",
    "Note",
    "Scale",
    "TabPosition",
    "TabbedNote",
    "Tuning",
    "VoicedChord",
    "analyze",
    "compose",
]

__version__ = "0.2.0"
