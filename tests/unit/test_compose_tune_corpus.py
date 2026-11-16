"""v6 task 2: statistics of real melodies from the POP909 format."""

from __future__ import annotations

from pathlib import Path

import pytest

from music_decoder.compose.tune_corpus import build_stats, read_song
from music_decoder.compose.tune_metrics import FEATURES


@pytest.fixture
def mini() -> Path:
    return Path(__file__).parents[1] / "fixtures" / "compose" / "pop909_mini"


def test_read_song_parses_the_dataset_format(mini):
    song = read_song(mini / "001")
    assert (song.tonic_pc, song.mode, song.tempo) == (0, "major", 120)
    assert song.phrases == (("A", 0, 4), ("A", 4, 4))


def test_read_song_is_none_when_a_file_is_missing(mini, tmp_path):
    assert read_song(tmp_path) is None
    assert read_song(mini / "Icon") is None


def test_roles_patterns_and_pairs(mini):
    stats = build_stats([read_song(mini / "001")], min_pattern_count=1)
    assert stats["rhythm"]["open"] == {"0,1,2,4": 2} and stats["rhythm"]["cadence"] == {"0": 2}
    assert stats["rhythm_pairs"]["body"]["0,1,2,4"]["0,1,2,4"] == 4


def test_rare_patterns_are_removed_everywhere(mini):
    stats = build_stats([read_song(mini / "001")], min_pattern_count=3)
    assert stats["rhythm"]["cadence"] == {}
    assert "0" not in stats["rhythm_pairs"]["open"]


def test_impure_bars_are_left_out(mini):
    stats = build_stats([read_song(mini / "002")], min_pattern_count=1)
    assert sum(sum(v.values()) for v in stats["rhythm"].values()) == 7


def test_moves_are_scale_steps(mini):
    stats = build_stats([read_song(mini / "001")], min_pattern_count=1)
    assert stats["steps"]["beat"]["0"]["1"] >= 1  # 60 -> 62 after a repeat, landing on slot 2
    assert stats["degrees"]["major"][0] > 0 and abs(sum(stats["degrees"]["major"]) - 1) < 1e-9


def test_a_chains_second_note_is_counted_after_start(mini):
    stats = build_stats([read_song(mini / "001")], min_pattern_count=1)
    # one chain over the whole song: its only START move is 60 -> 60 on slot 1
    assert stats["steps"]["off"]["99"] == {"0": 1}
    assert sum(sum(m.values()) for c in stats["steps"].values() for m in c.values()) == 25
    assert sum(sum(c.values()) for c in stats["steps_base"].values()) == 25


def test_chord_tones_and_final_degrees(mini):
    stats = build_stats([read_song(mini / "001")], min_pattern_count=1)
    # strong notes: 60 and 64 under C, F, C (x2), 67 under G (x2): only 64 under F misses
    assert stats["chord_tone"]["strong"] == pytest.approx(12 / 14)
    assert stats["final_degrees"]["major"][7] == 1.0


def test_bands_use_grid_features(mini):
    stats = build_stats([read_song(mini / "001")], min_pattern_count=1)
    assert set(stats["bands"]["major"]) == set(FEATURES)
    assert stats["meta"]["songs"] == 1 and stats["meta"]["windows"] == {"major": 1, "minor": 0}


def test_pilot_is_stored(mini):
    stats = build_stats([read_song(mini / "001")], {"x": 1}, min_pattern_count=1)
    assert stats["pilot"] == {"x": 1}


def test_a_song_with_two_keys_takes_the_better_fit_per_phrase(tmp_path):
    song_dir = tmp_path / "003"
    song_dir.mkdir()
    (song_dir / "analyzed_key.txt").write_text("C:maj\nA:maj\n\n")
    (song_dir / "tempo.txt").write_text("100")
    (song_dir / "human_label1.txt").write_text("A1B1")
    (song_dir / "melody.txt").write_text("60 4\n62 4\n64 4\n65 4\n69 4\n73 4\n66 4\n68 4\n")
    (song_dir / "finalized_chord.txt").write_text("C:maj [0, 4, 7] 0 4\nA:maj [1, 4, 9] 9 4\n")
    song = read_song(song_dir)
    assert (song.tonic_pc, song.mode) == (0, "major")
    assert song.bar_keys == ((0, "major"), (9, "major"))
