from pathlib import Path

import pytest

from music_decoder.evaluation.fixtures.guitarset import GuitarSetFixtures


@pytest.mark.slow
def test_chord_segments_parsed_when_cache_present():
    cache = Path("tests/fixtures/guitarset")
    loader = GuitarSetFixtures(cache_dir=cache, track_ids=["00_BN1-129-Eb_comp"])
    if not loader.is_available():
        pytest.skip("GuitarSet cache absent")
    fixtures = list(loader.load())
    assert len(fixtures) == 1
    gt = fixtures[0].ground_truth
    # Bossa nova comping has chord annotations; expect at least a handful.
    assert gt.chord_segments is not None
    assert len(gt.chord_segments) > 0
    # Every entry should be (start, end, root, quality) with valid values.
    for start, end, root, quality in gt.chord_segments:
        assert end > start
        assert root in {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B", "N"}
        assert quality in {
            "maj", "min", "7", "maj7", "min7", "dim", "sus4", "aug", "",
        }
