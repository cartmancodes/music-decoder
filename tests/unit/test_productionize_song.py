from collections import Counter

import pretty_midi
import pytest

from music_decoder.compose.api import arrange
from music_decoder.compose.arrangement import build_midi
from music_decoder.productionize.config import HISAAB_GRID, PianoConfig
from music_decoder.productionize.instruments.piano import nearest_sample
from music_decoder.productionize.song import (
    NoteEvent,
    ProductionError,
    Song,
    hisaab_song,
    read_note_file,
    song_from_composition,
    song_from_midi,
)
from music_decoder.types import ChordSymbol, Composition, Scale

PROGRESSION = [ChordSymbol.parse(c) for c in ("Am", "F", "C", "G")]


@pytest.fixture
def composer_midi(tmp_path):
    arr = arrange(Scale("A", "minor"), PROGRESSION, tempo_bpm=92.0, style="fingerstyle", seed=7)
    return build_midi(
        melody=arr.melody,
        voicings=arr.voicings,
        bars_per_chord=1,
        tempo_bpm=92.0,
        style="fingerstyle",
        out_path=tmp_path / "composition.mid",
    )


def _section(song: Song, index: int, notes) -> list[NoteEvent]:
    start, end = song.grid.time(index), song.grid.time(index + 1)
    snap = song.grid.bar_s / 64
    return [n for n in notes if start - snap <= n.start_s < end - snap]


# ---- The reference song ----------------------------------------------------


def test_hisaab_has_the_documented_note_counts():
    song = hisaab_song()
    assert len(song.piano) == 263
    assert len(song.strings) == 34
    assert song.key == "F#m" and song.title == "Hisaab (Instrumental)"
    assert song.reference is not None


def test_hisaab_notes_per_section():
    song = hisaab_song()
    counts = [len(_section(song, i, song.piano)) for i in range(8)]
    assert counts == [38, 41, 27, 27, 27, 27, 41, 27]
    assert len([n for n in song.piano if n.start_s >= song.tail_start_s - 0.01]) == 8


def test_hisaab_pitches_sit_within_a_semitone_of_the_14_sampled_pitches():
    used = (30, 36, 42, 48, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81)
    cfg = PianoConfig()
    for note in hisaab_song().piano:
        sample, shift = nearest_sample(note.pitch, cfg)
        assert sample in used and abs(shift) <= 1


def test_hisaab_verses_are_identical_and_the_outro_is_12_softer():
    song = hisaab_song()
    g = song.grid
    verse = [(round(n.start_s - g.time(2), 6), n.pitch) for n in _section(song, 2, song.piano)]
    for index in (3, 4, 5, 7):
        other = [
            (round(n.start_s - g.time(index), 6), n.pitch)
            for n in _section(song, index, song.piano)
        ]
        assert other == verse
    melody = [
        n
        for n in _section(song, 2, song.piano)
        if n.pitch in (73, 74, 77) and n.end_s - n.start_s < 2
    ]
    assert [round((n.start_s - g.time(2)) % g.bar_s / g.step_s) for n in melody] == [10, 10, 10]
    held = [n.velocity for n in _section(song, 2, song.piano)[:5]]
    assert all(56 <= v <= 67 for v in held)
    assert all(
        44 <= n.velocity <= 55
        for n in _section(song, 7, song.piano)
        if n.start_s < g.time(7, 3) - 0.01
    )


def test_hisaab_strings_play_only_from_hook_2():
    song = hisaab_song()
    assert min(n.start_s for n in song.strings) == pytest.approx(song.grid.time(6))
    assert song.strings_sections == (6, 7)
    assert all(46 <= n.velocity <= 66 for n in song.strings)


def test_song_properties():
    song = hisaab_song()
    assert song.verse_sections == (2, 3, 4, 5)
    assert song.body_s == pytest.approx(8 * HISAAB_GRID.section_s)
    start, stop = song.section_span(1)
    assert (stop - start) == pytest.approx(HISAAB_GRID.section_s * 44_100, abs=1)


def test_a_song_needs_sections_and_one_root_per_bar():
    song = hisaab_song()
    with pytest.raises(ProductionError):
        Song("x", "", song.grid, (), song.bar_roots, song.piano, ())
    with pytest.raises(ProductionError):
        Song("x", "", song.grid, song.sections, (30, 35), song.piano, ())


def test_note_file_round_trip(tmp_path):
    pm = pretty_midi.PrettyMIDI(initial_tempo=80.135)
    piano = pretty_midi.Instrument(program=0, name="piano")
    piano.notes.append(pretty_midi.Note(velocity=70, pitch=66, start=0.0, end=1.0))
    piano.notes.append(pretty_midi.Note(velocity=70, pitch=61, start=0.5, end=0.5))  # zero length
    strings = pretty_midi.Instrument(program=49, name="strings")
    strings.notes.append(pretty_midi.Note(velocity=50, pitch=57, start=2.0, end=4.0))
    pm.instruments.extend([piano, strings])
    path = tmp_path / "notes.mid"
    pm.write(str(path))
    piano_notes, string_notes = read_note_file(path)
    assert [(n.pitch, n.velocity) for n in piano_notes] == [(66, 70)]
    assert [(n.pitch, n.velocity) for n in string_notes] == [(57, 50)]
    song = hisaab_song(path)
    assert len(song.piano) == 1 and "notes.mid" in song.source


def test_unreadable_note_file(tmp_path):
    bad = tmp_path / "bad.mid"
    bad.write_bytes(b"not midi")
    with pytest.raises(ProductionError, match="could not read"):
        read_note_file(bad)


# ---- Composer output -------------------------------------------------------


def test_composer_file_sets_grid_and_bass_roots(composer_midi):
    song = song_from_midi(composer_midi, progression=PROGRESSION)
    assert song.grid.tempo_bpm == pytest.approx(92.0, abs=0.01)
    assert song.grid.bars_per_section == 4
    assert song.bar_roots == (33, 29, 36, 31)  # A1 F1 C2 G1
    assert len(song.sections) == 8


def test_bass_roots_are_inferred_from_the_lowest_note_without_a_progression(composer_midi):
    assert song_from_midi(composer_midi).bar_roots == (33, 29, 36, 31)


def test_progression_must_divide_the_bars(composer_midi):
    with pytest.raises(ProductionError, match="do not divide"):
        song_from_midi(composer_midi, progression=PROGRESSION[:3])


def test_hooks_play_the_composition_as_written(composer_midi):
    song = song_from_midi(composer_midi)
    pm = pretty_midi.PrettyMIDI(str(composer_midi))
    written = Counter(n.pitch for inst in pm.instruments for n in inst.notes)
    hook = _section(song, 1, song.piano)
    assert Counter(n.pitch for n in hook) == written
    assert Counter(n.pitch for n in _section(song, 6, song.piano)) == written


def test_intro_is_the_accompaniment_alone_and_softer(composer_midi):
    song = song_from_midi(composer_midi)
    pm = pretty_midi.PrettyMIDI(str(composer_midi))
    accompaniment = next(i for i in pm.instruments if i.name == "accompaniment")
    intro = _section(song, 0, song.piano)
    assert len(intro) == len(accompaniment.notes)
    assert max(n.velocity for n in intro) == max(n.velocity for n in accompaniment.notes) - 14


def test_verses_hold_rolled_chords_under_the_melody(composer_midi):
    song = song_from_midi(composer_midi)
    g = song.grid
    verse = _section(song, 2, song.piano)
    for bar in range(3):
        start = g.time(2, bar)
        chord = [n for n in verse if n.end_s == pytest.approx(start + 3.9 * g.beat_s)]
        assert len(chord) >= 3
        onsets = [n.start_s for n in sorted(chord, key=lambda n: n.pitch)]
        assert onsets == sorted(onsets) and onsets[0] == pytest.approx(start)
        assert onsets[-1] - onsets[0] < g.step_s  # rolled across one 16th
    # No more than one chord attack per bar: nothing choppy.
    pm = pretty_midi.PrettyMIDI(str(composer_midi))
    melody = next(i for i in pm.instruments if i.name == "melody")
    final_stroke = [n for n in verse if n.start_s >= g.time(2, 3) - 0.01]
    held = sum(
        1 for n in verse if n.end_s - n.start_s > 3 * g.beat_s and n.start_s < g.time(2, 3) - 0.01
    )
    assert len(verse) <= held + len(melody.notes) + len(final_stroke)


def test_strings_double_the_chords_from_hook_2_and_tail_rings(composer_midi):
    song = song_from_midi(composer_midi)
    g = song.grid
    assert min(n.start_s for n in song.strings) == pytest.approx(g.time(6))
    per_bar = Counter(round(n.start_s / g.bar_s) for n in song.strings)
    assert all(count <= 4 for count in per_bar.values())
    tail = [n for n in song.piano if n.start_s >= song.tail_start_s - 1e-9]
    assert tail and all(n.end_s == pytest.approx(song.tail_start_s + 8 * g.beat_s) for n in tail)


def test_song_from_composition_uses_its_metadata(composer_midi):
    composition = Composition(
        midi_path=composer_midi,
        wav_path=composer_midi.with_suffix(".wav"),
        ascii_tab="",
        melody_notes=(),
        chord_voicings=(),
        metadata={"scale": "A:minor", "progression": ["Am", "F", "C", "G"], "tempo_bpm": 92.0},
    )
    song = song_from_composition(composition)
    assert song.key == "Am"
    assert song.title == "Composition in A minor"
    assert song.grid.beat_s == pytest.approx(60 / 92.0)
    assert song.bar_roots == (33, 29, 36, 31)


def test_a_file_without_two_parts_is_rejected(tmp_path):
    pm = pretty_midi.PrettyMIDI()
    only = pretty_midi.Instrument(program=0, name="piano")
    only.notes.append(pretty_midi.Note(velocity=70, pitch=60, start=0.0, end=1.0))
    pm.instruments.append(only)
    path = tmp_path / "one.mid"
    pm.write(str(path))
    with pytest.raises(ProductionError, match="not a composer file"):
        song_from_midi(path)
