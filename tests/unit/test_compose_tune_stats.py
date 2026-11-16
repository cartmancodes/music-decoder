"""v6 task 2: the shipped statistics file and its typed loader."""

from __future__ import annotations

from importlib import resources

from music_decoder.compose.tune_metrics import FEATURES, Band
from music_decoder.compose.tune_stats import START, load_tune_stats, metric_class


def test_metric_classes():
    assert [metric_class(s) for s in range(8)] == [
        "strong",
        "off",
        "beat",
        "off",
        "strong",
        "off",
        "beat",
        "off",
    ]


def test_the_json_ships_inside_the_package():
    path = resources.files("music_decoder.compose") / "data" / "tune_stats.json"
    assert path.is_file()


def test_shipped_stats_load_and_are_sane():
    s = load_tune_stats()
    assert s.meta["songs"] >= 800 and min(s.meta["windows"].values()) >= 100
    assert sum(len(v) for v in s.rhythm.values()) >= 60
    assert all(
        p and p == tuple(sorted(p)) and 0 <= p[0] and p[-1] <= 7
        for r in s.rhythm.values()
        for p in r
    )
    assert 0.6 <= s.chord_tone["strong"] <= 0.95 and s.chord_tone["strong"] > s.chord_tone["off"]
    assert 0.10 <= s.bands["major"]["repeat"].mid <= 0.35
    assert load_tune_stats() is s


def test_loader_converts_keys_and_bands():
    s = load_tune_stats()
    assert set(s.rhythm) == {"open", "body", "cadence"}
    assert all(isinstance(m, int) for c in s.steps.values() for row in c.values() for m in row)
    assert START in s.steps["strong"] and START not in s.steps_base["strong"]
    assert all(len(d) == 12 and abs(sum(d) - 1) < 1e-6 for d in s.degrees.values())
    assert all(len(d) == 12 and abs(sum(d) - 1) < 1e-6 for d in s.final_degrees.values())
    for mode in ("major", "minor"):
        assert set(s.bands[mode]) == set(FEATURES)
        assert all(b.low <= b.mid <= b.high for b in s.bands[mode].values())
    assert isinstance(s.notes_per_second, Band)
