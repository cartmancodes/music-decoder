"""Music Decoder: chord progressions, guitar tabs, composition suggestions."""
from music_decoder.api import analyze, compose
from music_decoder.tabs.tuning import (
    DADGAD,
    DROP_C,
    DROP_D,
    D_STANDARD,
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
    TabPosition,
    TabbedNote,
    Tuning,
    VoicedChord,
)

__all__ = [
    "analyze",
    "compose",
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
    "STANDARD_EADGBE",
    "DROP_D",
    "DROP_C",
    "EB_HALF_STEP_DOWN",
    "D_STANDARD",
    "DADGAD",
]

__version__ = "0.2.0"
