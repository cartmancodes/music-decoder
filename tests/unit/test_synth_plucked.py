"""The string-aware plucked-guitar renderer (composer v5, spec 5.1)."""

from __future__ import annotations

import librosa
import numpy as np
import pytest
import soundfile

from music_decoder.synth.plucked import render_plucked, tab_for_midi_notes
from music_decoder.tabs.tuning import DROP_C, STANDARD_EADGBE
from music_decoder.types import Note, TabbedNote, TabPosition


def _one(pitch_string, fret, dur=1.5, vel=80, tuning=STANDARD_EADGBE):
    pitch = tuning.open_pitches[pitch_string] + fret
    return TabbedNote(
        note=Note(start_s=0.0, end_s=dur, pitch=pitch, velocity=vel, confidence=1.0),
        position=TabPosition(string=pitch_string, fret=fret),
    )


@pytest.mark.parametrize(("tuning", "string", "fret"), [(DROP_C, 0, 0), (STANDARD_EADGBE, 5, 0)])
def test_plucked_pitch_is_in_tune(tmp_path, tuning, string, fret):  # C2 and E4
    out = render_plucked([_one(string, fret, tuning=tuning)], tuning, tmp_path / "n.wav")
    y, sr = soundfile.read(out)
    f0 = float(np.median(librosa.yin(y[:sr], fmin=50, fmax=1000, sr=sr)))
    target = 440 * 2 ** ((tuning.open_pitches[string] + fret - 69) / 12)
    assert abs(1200 * np.log2(f0 / target)) <= 10


def test_plucked_is_deterministic_and_does_not_clip(tmp_path):
    notes = [_one(1, 3), _one(5, 0)]
    a = render_plucked(notes, STANDARD_EADGBE, tmp_path / "a.wav", seed=3).read_bytes()
    b = render_plucked(notes, STANDARD_EADGBE, tmp_path / "b.wav", seed=3).read_bytes()
    assert a == b
    y, _ = soundfile.read(tmp_path / "a.wav", dtype="int16")
    assert np.abs(y).max() <= int(0.9 * 32767) + 1


def test_plucked_render_length(tmp_path):
    out = render_plucked([_one(0, 0, dur=6.0)], STANDARD_EADGBE, tmp_path / "s.wav")
    y, sr = soundfile.read(out)
    assert 6.0 <= len(y) / sr <= 7.5


def test_tab_for_midi_notes_uses_the_lowest_fret_string():
    notes = [
        Note(start_s=0.0, end_s=0.5, pitch=p, velocity=70, confidence=1.0) for p in (40, 52, 64)
    ]
    tab = tab_for_midi_notes(notes, STANDARD_EADGBE)
    assert [t.note for t in tab] == notes
    assert [(t.position.string, t.position.fret) for t in tab] == [(0, 0), (2, 2), (5, 0)]
