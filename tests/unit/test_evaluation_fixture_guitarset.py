import pytest

from music_decoder.evaluation.fixtures.guitarset import GuitarSetFixtures


@pytest.mark.slow
def test_guitarset_yields_fixtures_when_data_present(tmp_path):
    """If the GuitarSet cache is present (CI restores it), this returns fixtures."""
    loader = GuitarSetFixtures(
        cache_dir=tmp_path / "guitarset",
        track_ids=["00_BN1-129-Eb_comp", "00_BN1-129-Eb_solo"],
    )
    if not loader.is_available():
        pytest.skip("GuitarSet cache unavailable; run `make fixtures` to download.")
    fixtures = list(loader.load())
    assert len(fixtures) >= 1
    fx = fixtures[0]
    assert fx.source == "guitarset"
    assert fx.ground_truth.tab is not None
    assert all(0 <= s < 6 for (_, s, _) in fx.ground_truth.tab)


def test_guitarset_is_available_returns_false_for_missing_cache(tmp_path):
    loader = GuitarSetFixtures(cache_dir=tmp_path / "missing", track_ids=["x"])
    assert loader.is_available() is False
