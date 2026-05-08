import pretty_midi

from music_decoder.compose.arrangement import build_midi
from music_decoder.compose.melody import generate_melody
from music_decoder.compose.voicings import voicings_for
from music_decoder.tabs.tuning import STANDARD_EADGBE
from music_decoder.types import ChordSymbol, Scale


def test_build_midi_produces_two_tracks(tmp_path):
    progression = [ChordSymbol.parse(c) for c in ("Cmaj7", "Am7", "Dm7", "G7")]
    voicings = [voicings_for(c, STANDARD_EADGBE)[0] for c in progression]
    melody = generate_melody(
        scale=Scale("C", "major"),
        progression=progression,
        bars_per_chord=1,
        tempo_bpm=120.0,
        seed=42,
    )
    out = tmp_path / "x.mid"
    build_midi(
        melody=melody,
        voicings=voicings,
        bars_per_chord=1,
        tempo_bpm=120.0,
        style="strum",
        out_path=out,
    )
    assert out.exists()
    pm = pretty_midi.PrettyMIDI(str(out))
    assert len(pm.instruments) == 2
    assert sum(len(i.notes) for i in pm.instruments) > 0


def test_build_midi_styles_run(tmp_path):
    progression = [ChordSymbol.parse(c) for c in ("C", "G")]
    voicings = [voicings_for(c, STANDARD_EADGBE)[0] for c in progression]
    melody = generate_melody(
        scale=Scale("C", "major"),
        progression=progression,
        bars_per_chord=1,
        tempo_bpm=120.0,
        seed=42,
    )
    for style in ("strum", "arpeggio", "fingerstyle"):
        out = tmp_path / f"{style}.mid"
        build_midi(
            melody=melody,
            voicings=voicings,
            bars_per_chord=1,
            tempo_bpm=120.0,
            style=style,
            out_path=out,
        )
        assert out.exists()
