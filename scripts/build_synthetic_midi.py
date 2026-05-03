"""Generates the synthetic .mid files committed to tests/fixtures/synthetic/."""
from pathlib import Path
import pretty_midi


OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "synthetic"


def c_major_scale() -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    inst = pretty_midi.Instrument(program=24)
    pitches = [60, 62, 64, 65, 67, 69, 71, 72]
    for i, p in enumerate(pitches):
        inst.notes.append(pretty_midi.Note(
            velocity=80, pitch=p, start=i * 0.5, end=(i + 1) * 0.5,
        ))
    pm.instruments.append(inst)
    return pm


def g_major_chord() -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    inst = pretty_midi.Instrument(program=24)
    pitches = [43, 47, 50, 55, 59, 67]
    for p in pitches:
        inst.notes.append(pretty_midi.Note(velocity=80, pitch=p, start=0.0, end=2.0))
    pm.instruments.append(inst)
    return pm


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    c_major_scale().write(str(OUT / "c_major_scale.mid"))
    g_major_chord().write(str(OUT / "g_major_chord.mid"))


if __name__ == "__main__":
    main()
