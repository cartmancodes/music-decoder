from music_decoder.compose.melody import generate_melody
from music_decoder.types import ChordSymbol, Scale


C_MAJOR_PCS = {0, 2, 4, 5, 7, 9, 11}


def test_all_notes_in_scale():
    progression = [ChordSymbol.parse(c) for c in ("C", "Am", "F", "G")]
    notes = generate_melody(
        scale=Scale("C", "major"),
        progression=progression,
        bars_per_chord=1,
        tempo_bpm=120.0,
        seed=42,
    )
    assert notes
    for n in notes:
        assert n.pitch % 12 in C_MAJOR_PCS


def test_deterministic_with_seed():
    progression = [ChordSymbol.parse(c) for c in ("C", "Am", "F", "G")]
    a = generate_melody(scale=Scale("C", "major"), progression=progression,
                        bars_per_chord=1, tempo_bpm=120.0, seed=42)
    b = generate_melody(scale=Scale("C", "major"), progression=progression,
                        bars_per_chord=1, tempo_bpm=120.0, seed=42)
    assert [(n.pitch, n.start_s) for n in a] == [(n.pitch, n.start_s) for n in b]


def test_different_seeds_produce_different_melodies():
    progression = [ChordSymbol.parse(c) for c in ("C", "Am", "F", "G")]
    a = generate_melody(scale=Scale("C", "major"), progression=progression,
                        bars_per_chord=1, tempo_bpm=120.0, seed=1)
    b = generate_melody(scale=Scale("C", "major"), progression=progression,
                        bars_per_chord=1, tempo_bpm=120.0, seed=2)
    assert [n.pitch for n in a] != [n.pitch for n in b]


def test_strong_beats_favor_chord_tones_statistically():
    # Cmaj7 = {0,4,7,11}; 4 bars all Cmaj7, 4 notes per bar; strong beats are
    # beat 1 and 3 (i.e. notes 0 and 2 of every bar).
    progression = [ChordSymbol.parse("Cmaj7")] * 8
    notes = generate_melody(scale=Scale("C", "major"), progression=progression,
                            bars_per_chord=1, tempo_bpm=120.0, seed=0,
                            notes_per_bar=4)
    chord_tones = {0, 4, 7, 11}
    strong_idxs = [i for i in range(len(notes)) if i % 4 in (0, 2)]
    strong_pcs = [notes[i].pitch % 12 for i in strong_idxs]
    fraction_chord = sum(p in chord_tones for p in strong_pcs) / len(strong_pcs)
    assert fraction_chord >= 0.55, f"strong-beat chord-tone rate {fraction_chord}"
