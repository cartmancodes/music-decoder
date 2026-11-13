"""Acceptance test: the reference song must pass every check.

Renders "Hisaab" with the real samples (downloaded on first use, about
180 MB) and runs the acceptance checks of the build spec on the mastered
WAV and MP3. Skipped unless ffmpeg and fluidsynth are installed.

Set ``HISAAB_NOTES`` to the final note file (``hisaab_final_notes.mid``) to
check the real notes; without it the notes are rebuilt from the spec.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from music_decoder.compose.api import arrange
from music_decoder.compose.arrangement import build_midi
from music_decoder.productionize import hisaab_song, productionize
from music_decoder.types import ChordSymbol, Scale

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(
        shutil.which("ffmpeg") is None or shutil.which("fluidsynth") is None,
        reason="needs ffmpeg and fluidsynth",
    ),
]


def test_reference_song_passes_every_acceptance_check(tmp_path):
    notes = os.environ.get("HISAAB_NOTES")
    result = productionize(
        hisaab_song(Path(notes) if notes else None),
        out_dir=tmp_path,
        name="hisaab_instrumental",
    )
    assert result.qc is not None
    assert result.qc.passed, "\n" + result.qc.format()
    assert all(c.passed for c in result.qc.checks), "every check has a reference for this song"
    # The limiter works as on the reference master: about 2 dB at the most.
    assert result.master.limiter_max_reduction_db == pytest.approx(2.0, abs=0.5)


def test_composer_output_is_release_ready(tmp_path):
    progression = [ChordSymbol.parse(c) for c in ("Am", "F", "C", "G")]
    arr = arrange(Scale("A", "minor"), progression, tempo_bpm=92.0, seed=7)
    midi = build_midi(
        melody=arr.melody,
        voicings=arr.voicings,
        bars_per_chord=1,
        tempo_bpm=92.0,
        style="fingerstyle",
        out_path=tmp_path / "composition.mid",
    )
    result = productionize(midi, out_dir=tmp_path / "out", progression=progression, key="Am")
    assert result.qc is not None
    assert result.qc.passed, "\n" + result.qc.format()
