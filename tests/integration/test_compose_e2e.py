"""End-to-end test for the public ``compose()`` API.

Generates a short progression in C major, validates that the MIDI/WAV
artifacts are written, and confirms that every melody pitch falls inside the
diatonic C-major pitch-class set.
"""
from __future__ import annotations

import pretty_midi

from music_decoder import ChordSymbol, Scale, compose


def test_compose_e2e(tmp_path):
    res = compose(
        scale=Scale("C", "major"),
        progression=[ChordSymbol.parse(c) for c in ("Cmaj7", "Am7", "Dm7", "G7")],
        bars_per_chord=1,
        tempo_bpm=120.0,
        style="strum",
        seed=11,
        out_dir=tmp_path,
    )
    assert res.midi_path.exists()
    assert res.wav_path.exists()
    pm = pretty_midi.PrettyMIDI(str(res.midi_path))
    assert len(pm.instruments) == 2
    assert all(0 <= n.pitch < 128 for inst in pm.instruments for n in inst.notes)
    # Melody-only check: every melody pitch class must be in C major.
    c_major = {0, 2, 4, 5, 7, 9, 11}
    melody_inst = next(i for i in pm.instruments if i.name == "melody")
    assert all((n.pitch % 12) in c_major for n in melody_inst.notes)
