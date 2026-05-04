import shutil
from pathlib import Path

import pytest

from music_decoder.evaluation.fixtures.synthetic import SyntheticFixtures


@pytest.fixture
def synthetic_dir(tmp_path: Path) -> Path:
    src = Path("tests/fixtures/synthetic")
    dst = tmp_path / "synthetic"
    dst.mkdir()
    shutil.copy(src / "c_major_scale.mid", dst)
    shutil.copy(src / "g_major_chord.mid", dst)
    return dst


def test_loader_renders_wav_from_midi(synthetic_dir: Path):
    loader = SyntheticFixtures(root=synthetic_dir)
    fixtures = list(loader.load())
    names = {f.name for f in fixtures}
    assert {"c_major_scale", "g_major_chord"}.issubset(names)
    for f in fixtures:
        assert f.audio_path.exists()
        assert f.audio_path.suffix == ".wav"
        assert f.ground_truth.pitches_midi.size > 0


def test_c_major_scale_has_eight_notes(synthetic_dir: Path):
    loader = SyntheticFixtures(root=synthetic_dir)
    fx = next(f for f in loader.load() if f.name == "c_major_scale")
    assert len(fx.ground_truth.pitches_midi) == 8


def test_g_major_chord_is_simultaneous(synthetic_dir: Path):
    loader = SyntheticFixtures(root=synthetic_dir)
    fx = next(f for f in loader.load() if f.name == "g_major_chord")
    starts = fx.ground_truth.intervals[:, 0]
    assert (starts == starts[0]).all()
