import json
from pathlib import Path

import numpy as np
import pytest
import scipy.io.wavfile as wavfile

from music_decoder.evaluation.fixtures.manual import ManualFixtures


def _write_dummy_wav(path: Path) -> None:
    sr = 22050
    samples = (np.sin(2 * np.pi * 440 * np.arange(sr) / sr) * 0.5).astype(np.float32)
    wavfile.write(str(path), sr, (samples * 32767).astype(np.int16))


@pytest.fixture
def manual_dir(tmp_path: Path) -> Path:
    d = tmp_path / "manual"
    d.mkdir()
    _write_dummy_wav(d / "clip01.wav")
    (d / "clip01.json").write_text(json.dumps({
        "audio": "clip01.wav",
        "tuning": "EADGBE",
        "key": {"tonic": "G", "mode": "major"},
        "tempo_bpm": 120,
        "tab": [
            {"start_s": 0.0, "end_s": 0.5, "pitch": 67, "string": 3, "fret": 12}
        ],
    }))
    return d


def test_manual_loader_reads_drop_in_pair(manual_dir: Path):
    loader = ManualFixtures(root=manual_dir)
    fixtures = list(loader.load())
    assert len(fixtures) == 1
    fx = fixtures[0]
    assert fx.name == "clip01"
    assert fx.source == "manual"
    assert fx.ground_truth.key == ("G", "major")
    assert fx.ground_truth.tempo_bpm == 120.0
    assert fx.ground_truth.tab == [(67, 3, 12)]


def test_manual_loader_skips_orphan_json(manual_dir: Path):
    (manual_dir / "broken.json").write_text(json.dumps({"audio": "missing.wav"}))
    fixtures = list(ManualFixtures(root=manual_dir).load())
    assert len(fixtures) == 1
