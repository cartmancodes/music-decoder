"""Acceptance test: the reference song must pass every check.

Renders "Hisaab" with the real samples (downloaded on first use, about
180 MB) and runs the acceptance checks of the build spec on the mastered
WAV and MP3. Opt-in: runs only with ``MUSIC_DECODER_PRODUCTION_E2E=1``
(``make test-production``), and needs ffmpeg and fluidsynth.

Set ``HISAAB_NOTES`` to the final note file (``hisaab_final_notes.mid``) to
check the real notes; without it the notes are rebuilt from the spec.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from music_decoder.compose.api import arrange
from music_decoder.compose.arrangement import build_midi
from music_decoder.productionize import (
    PopRapConfig,
    ProductionConfig,
    hisaab_song,
    productionize,
)
from music_decoder.productionize.poprap import poprap_song
from music_decoder.types import ChordSymbol, Scale

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    # Opt-in: each render downloads ~180 MB on first use and needs ~2.7 GB of
    # memory, so `make test` skips them; `make test-production` runs them.
    pytest.mark.skipif(
        os.environ.get("MUSIC_DECODER_PRODUCTION_E2E") != "1",
        reason="set MUSIC_DECODER_PRODUCTION_E2E=1 (or run `make test-production`)",
    ),
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
        accompaniment=arr.accompaniment,
        out_path=tmp_path / "composition.mid",
    )
    result = productionize(midi, out_dir=tmp_path / "out", progression=progression, key="Am")
    assert result.qc is not None
    assert result.qc.passed, "\n" + result.qc.format()


@pytest.mark.parametrize("lead", ["guitar", "synth", "piano"])
def test_poprap_render_passes_every_check(tmp_path, lead):
    progression = [ChordSymbol.parse(c) for c in ("Am", "F", "C", "G")]
    arr = arrange(Scale("A", "minor"), progression, tempo_bpm=120.0, seed=7)
    midi = build_midi(
        melody=arr.melody,
        voicings=arr.voicings,
        bars_per_chord=2,
        tempo_bpm=120.0,
        style="fingerstyle",
        accompaniment=arr.accompaniment,
        out_path=tmp_path / "composition.mid",
    )
    cfg = ProductionConfig(style="pop-rap", poprap=PopRapConfig(lead=lead))  # type: ignore[arg-type]
    result = productionize(
        midi, out_dir=tmp_path / "out", progression=progression, key="Am", config=cfg
    )
    assert result.qc is not None
    assert result.qc.passed, "\n" + result.qc.format()
    assert result.master.limiter_max_reduction_db <= 3.0

    # The stems have no lead-in, so the song's hook spans index them directly.
    song = poprap_song(midi, cfg=cfg.poprap, progression=progression, key="Am")
    hooks = [song.section_span(i) for i, s in enumerate(song.sections) if s.role == "hook"]
    assert len(hooks) == 2
    energy = {}
    for name, path in result.stem_paths.items():
        audio, _ = sf.read(str(path), dtype="float64")
        energy[name] = sum(float(np.sum(audio[a:b] ** 2)) for a, b in hooks)
    total = sum(energy.values())
    drums, bass = energy["drums"] / total, energy["808"] / total
    # The synth lead is mixed forward (-16.5 vs -18 dBFS), so it takes more of the energy.
    drums_floor = 0.35 if lead == "synth" else 0.40
    assert drums_floor <= drums <= 0.60, f"drums share {drums:.3f}"
    assert 0.10 <= bass <= 0.25, f"808 share {bass:.3f}"
