"""Strum and arpeggio output is frozen at composer v3 (global constraint)."""

from __future__ import annotations

import json
from pathlib import Path

from music_decoder.evaluation.compose_frozen import strum_arpeggio_digests
from music_decoder.tabs.tuning import PRESETS

FIXTURE = Path(__file__).parents[1] / "fixtures" / "compose" / "v3_strum_arpeggio.json"


def test_strum_and_arpeggio_match_composer_v3() -> None:
    expected = json.loads(FIXTURE.read_text())
    assert len(expected) == 2 * (110 + 15 * len(PRESETS) + 40)
    assert strum_arpeggio_digests() == expected
