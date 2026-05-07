# tests/unit/test_package_imports.py
"""Smoke test: top-level package exposes the documented public API."""
import music_decoder


def test_top_level_exports():
    expected = {
        "analyze", "compose",
        "AnalysisResult", "Composition",
        "Scale", "ChordSymbol", "ChordSegment", "KeyEstimate",
        "Note", "TabPosition", "TabbedNote", "Tuning", "VoicedChord",
        "STANDARD_EADGBE", "DROP_D", "DROP_C", "EB_HALF_STEP_DOWN",
        "D_STANDARD", "DADGAD",
    }
    assert expected.issubset(set(dir(music_decoder)))


def test_version_set():
    assert music_decoder.__version__
