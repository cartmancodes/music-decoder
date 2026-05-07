from pathlib import Path

import pretty_midi
import pytest

from music_decoder.synth.fluidsynth_wrapper import (
    SynthBackend,
    synthesize_midi_to_wav,
)


@pytest.fixture
def midi_file(tmp_path: Path) -> Path:
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=24)
    inst.notes.append(pretty_midi.Note(
        velocity=80, pitch=60, start=0.0, end=1.0,
    ))
    pm.instruments.append(inst)
    p = tmp_path / "x.mid"
    pm.write(str(p))
    return p


def test_sine_backend_renders_audio(midi_file: Path, tmp_path: Path):
    out = tmp_path / "out.wav"
    synthesize_midi_to_wav(midi_file, out, sr=22050, backend=SynthBackend.SINE)
    assert out.exists()
    import scipy.io.wavfile as wavfile
    sr, raw = wavfile.read(str(out))
    assert sr == 22050
    assert len(raw) > 22050 * 0.9   # at least ~1 second
