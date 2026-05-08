"""Tests for the Phase B-5 fluidsynth path in the synthetic-fixture loader.

The fluidsynth path is opt-in: it activates when a .sf2 file is present
under ``<root>/soundfont/``. Without an SF2 the loader falls back to the
deterministic sine synthesis.

These tests don't require an actual SF2 — they verify the detection logic,
the cache-marker round-trip, and the fallback behavior when fluidsynth
raises.
"""

from __future__ import annotations

from pathlib import Path

import pretty_midi

from music_decoder.evaluation.fixtures.synthetic import (
    SyntheticFixtures,
    _find_soundfont,
)


def test_find_soundfont_returns_none_when_dir_absent(tmp_path: Path):
    assert _find_soundfont(tmp_path) is None


def test_find_soundfont_returns_none_when_dir_empty(tmp_path: Path):
    (tmp_path / "soundfont").mkdir()
    assert _find_soundfont(tmp_path) is None


def test_find_soundfont_returns_first_sf2(tmp_path: Path):
    (tmp_path / "soundfont").mkdir()
    sf = tmp_path / "soundfont" / "Test.sf2"
    sf.write_bytes(b"RIFF...sfbk fake")
    assert _find_soundfont(tmp_path) == sf


def test_find_soundfont_picks_alphabetically_first(tmp_path: Path):
    sf_dir = tmp_path / "soundfont"
    sf_dir.mkdir()
    (sf_dir / "Z.sf2").write_bytes(b"x")
    (sf_dir / "A.sf2").write_bytes(b"x")
    assert _find_soundfont(tmp_path) == sf_dir / "A.sf2"


def test_synthetic_loader_uses_sine_when_no_sf2(tmp_path: Path):
    """Without an SF2 the loader should fall back to sine (current behavior)."""
    # Build a one-note MIDI fixture
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=24)
    inst.notes.append(pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=0.5))
    pm.instruments.append(inst)
    midi_path = tmp_path / "test.mid"
    pm.write(str(midi_path))

    loader = SyntheticFixtures(root=tmp_path)
    assert loader._sf2_path is None  # detected nothing

    fixtures = list(loader.load())
    assert len(fixtures) == 1
    assert fixtures[0].audio_path.exists()
    assert fixtures[0].audio_path.suffix == ".wav"


def test_synthesize_falls_back_when_fluidsynth_raises(tmp_path: Path, monkeypatch):
    """When SF2 is configured but fluidsynth raises, fall back to sine."""
    sf_dir = tmp_path / "soundfont"
    sf_dir.mkdir()
    sf = sf_dir / "Bad.sf2"
    sf.write_bytes(b"not a real sf2")

    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=24)
    inst.notes.append(pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=0.3))
    pm.instruments.append(inst)
    midi_path = tmp_path / "test.mid"
    pm.write(str(midi_path))

    loader = SyntheticFixtures(root=tmp_path)
    assert loader._sf2_path == sf

    # Even though loader thinks SF2 is configured, calling fluidsynth on a
    # corrupt SF2 raises; the loader catches this and falls back to sine.
    fixtures = list(loader.load())
    assert len(fixtures) == 1
    assert fixtures[0].audio_path.exists()


def test_cache_marker_invalidates_on_sf2_change(tmp_path: Path):
    """Re-render after switching from sine to SF2 (or vice versa)."""
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=24)
    inst.notes.append(pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=0.3))
    pm.instruments.append(inst)
    midi_path = tmp_path / "test.mid"
    pm.write(str(midi_path))

    # First pass: no SF2 -> sine
    list(SyntheticFixtures(root=tmp_path).load())
    marker = tmp_path / ".test.synth_marker"
    assert marker.exists()
    assert marker.read_text() == "sine"

    # Second pass: drop in an SF2 -> marker should change to sf2= path
    sf_dir = tmp_path / "soundfont"
    sf_dir.mkdir()
    (sf_dir / "Test.sf2").write_bytes(b"corrupt sf2")
    list(SyntheticFixtures(root=tmp_path).load())
    # The fluidsynth render fails on corrupt data and falls back to sine,
    # but the cache marker reflects the configured SF2 path so a future
    # pass with a working SF2 would not be skipped.
    assert marker.read_text().startswith("sf2=")
