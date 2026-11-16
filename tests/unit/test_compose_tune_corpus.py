"""v6 task 2: statistics of real melodies from the POP909 format."""

from __future__ import annotations

from pathlib import Path

import pytest

from music_decoder.compose.tune_corpus import build_stats, pilot_stats, read_song
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


def _line(notes, mode="major", tonic_pc=0):
    return {"id": "x", "tonic_pc": tonic_pc, "mode": mode, "bpm": 100, "notes": notes}


def _pilot(songs=150, repeat=0.9, pent=0.5, range_=(5.0, 9.0, 30.0)):
    return {
        "songs": songs,
        "interval_classes": {"repeat": repeat, "step": 0.0, "third": 0.0, "leap": 0.0},
        "pentatonic": {"major": pent, "minor": pent},
        "degrees": {"major": [1.0] + [0.0] * 11, "minor": [1.0] + [0.0] * 11},
        "range": list(range_),
    }


def test_pilot_stats_from_lines():
    line = _line([[0, 60], [1, 60], [2, 62], [3, 67]])
    p = pilot_stats([line] * 3)
    assert p["songs"] == 3
    assert p["interval_classes"] == {"repeat": 1 / 3, "step": 1 / 3, "third": 0.0, "leap": 1 / 3}
    assert p["pentatonic"]["major"] == 1.0 and p["degrees"]["major"][0] == 0.5
    assert p["pentatonic"]["minor"] == 0.0 and p["degrees"]["minor"] == [0.0] * 12


def test_pilot_stats_are_relative_to_the_tonic_and_use_the_median_over_songs():
    minor = _line([[0, 57], [1, 60], [2, 62]], mode="minor", tonic_pc=9)  # A C D: all pentatonic
    chromatic = _line([[0, 60], [1, 61], [2, 62], [3, 63]], mode="minor", tonic_pc=9)
    p = pilot_stats([minor, minor, chromatic])
    assert p["degrees"]["minor"][0] == pytest.approx((1 / 3 + 1 / 3 + 0.0) / 3)
    assert p["pentatonic"]["minor"] == 1.0  # median of 1.0, 1.0 and 0.5
    assert p["interval_classes"]["step"] == 0.5  # median of 1/2, 1/2 and 3/3


def test_pilot_range_is_per_window_and_quartiles_over_songs():
    songs = [_line([[0, 60], [8, 60 + r]]) for r in (4, 8, 12)]
    assert pilot_stats(songs)["range"] == [6.0, 8.0, 10.0]


def test_merge_widens_pitch_bands_and_averages_degrees(mini):
    base = build_stats([read_song(mini / "001")], min_pattern_count=1)
    merged = build_stats([read_song(mini / "001")], _pilot(), min_pattern_count=1)
    assert merged["bands"]["minor"] == base["bands"]["minor"] == {}  # no minor windows in the mini
    assert merged["bands"]["major"]["repeat"]["high"] == 0.9
    assert merged["bands"]["major"]["repeat"]["low"] == min(
        base["bands"]["major"]["repeat"]["low"], 0.9
    )
    assert merged["bands"]["major"]["range"]["high"] == 30.0
    assert merged["bands"]["major"]["range"]["low"] == min(
        5.0, base["bands"]["major"]["range"]["low"]
    )
    assert (
        merged["bands"]["major"]["pentatonic"]["low"]
        <= 0.5
        <= merged["bands"]["major"]["pentatonic"]["high"]
    )
    assert merged["degrees"]["major"][0] == pytest.approx((base["degrees"]["major"][0] + 1.0) / 2)
    assert sum(merged["degrees"]["major"]) == pytest.approx(1.0)
    assert merged["rhythm"] == base["rhythm"] and merged["steps"] == base["steps"]
    assert (
        merged["bands"]["major"]["notes_per_second"] == base["bands"]["major"]["notes_per_second"]
    )


def test_merge_never_narrows_a_band(mini):
    base = build_stats([read_song(mini / "001")], min_pattern_count=1)
    inside = _pilot(repeat=0.0, pent=0.5, range_=(5.0, 6.0, 7.0))
    merged = build_stats([read_song(mini / "001")], inside, min_pattern_count=1)
    for feature in ("repeat", "pentatonic", "range"):
        old, new = base["bands"]["major"][feature], merged["bands"]["major"][feature]
        assert new["low"] <= old["low"] and new["high"] >= old["high"] and new["mid"] == old["mid"]


def test_small_pilot_is_ignored(mini):
    base = build_stats([read_song(mini / "001")], min_pattern_count=1)
    small = build_stats([read_song(mini / "001")], _pilot(songs=20), min_pattern_count=1)
    assert small["bands"] == base["bands"] and small["degrees"] == base["degrees"]
    assert small["meta"]["sources"] == base["meta"]["sources"]


def test_sources_carry_their_licence(mini):
    base = build_stats([read_song(mini / "001")], min_pattern_count=1)
    assert len(base["meta"]["sources"]) == 1 and "licence: MIT" in base["meta"]["sources"][0]
    merged = build_stats([read_song(mini / "001")], _pilot(), min_pattern_count=1)
    assert len(merged["meta"]["sources"]) == 2
    assert "own YouTube pilot, statistics only" in merged["meta"]["sources"][1]
    assert "150 songs" in merged["meta"]["sources"][1]
