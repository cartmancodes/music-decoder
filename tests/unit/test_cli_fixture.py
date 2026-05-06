"""Tests for the fixture-init and fixture-validate CLI subcommands."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from music_decoder.cli.fixture import (
    ValidationError,
    init_fixture,
    validate_fixture,
)


def _real_audio_path() -> Path:
    """Use the synthetic sine fixture as a known-good audio file."""
    return Path("tests/fixtures/audio_samples/sine_440.wav")


def test_init_fixture_creates_stub_with_audio_metadata(tmp_path: Path):
    out = init_fixture(
        _real_audio_path(),
        name="my_clip",
        output_dir=tmp_path,
    )
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["audio"] == "sine_440.wav"
    assert data["tuning"] == "EADGBE"
    assert "duration_s" in data["_audio_metadata"]
    assert 0.5 < data["_audio_metadata"]["duration_s"] < 1.5  # synthetic is ~1s


def test_init_fixture_refuses_to_overwrite_without_force(tmp_path: Path):
    init_fixture(_real_audio_path(), name="x", output_dir=tmp_path)
    with pytest.raises(FileExistsError):
        init_fixture(_real_audio_path(), name="x", output_dir=tmp_path)


def test_init_fixture_force_overwrites(tmp_path: Path):
    out1 = init_fixture(_real_audio_path(), name="x", output_dir=tmp_path)
    out2 = init_fixture(
        _real_audio_path(), name="x", output_dir=tmp_path, force=True,
    )
    assert out1 == out2  # same path, overwritten


def test_init_fixture_rejects_unknown_tuning(tmp_path: Path):
    with pytest.raises(ValueError, match="unknown tuning"):
        init_fixture(
            _real_audio_path(), name="x", output_dir=tmp_path,
            tuning="not_a_real_tuning",
        )


def test_init_fixture_missing_audio_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        init_fixture(
            tmp_path / "nope.wav", name="x", output_dir=tmp_path,
        )


def _write_fixture(tmp_path: Path, data: dict) -> Path:
    """Helper: write a fixture JSON with an accompanying real audio file."""
    audio = tmp_path / "test_audio.wav"
    src = _real_audio_path()
    audio.write_bytes(src.read_bytes())
    data.setdefault("audio", audio.name)
    fp = tmp_path / "fixture.json"
    fp.write_text(json.dumps(data))
    return fp


def test_validate_passes_for_valid_fixture(tmp_path: Path):
    fp = _write_fixture(tmp_path, {
        "tuning": "EADGBE",
        "tempo_bpm": 120,
        "key": {"tonic": "C", "mode": "major"},
        "tab": [
            {"start_s": 0.0, "end_s": 0.5, "pitch": 67, "string": 5, "fret": 3},
        ],
        "chords": [
            {"start_s": 0.0, "end_s": 0.5, "root": "C", "quality": "maj"},
        ],
    })
    errors = validate_fixture(fp)
    assert errors == []


def test_validate_reports_missing_audio_file(tmp_path: Path):
    fp = tmp_path / "fixture.json"
    fp.write_text(json.dumps({
        "audio": "missing.wav", "tuning": "EADGBE",
        "tab": [], "chords": [],
    }))
    errors = validate_fixture(fp)
    assert any("does not exist" in str(e) for e in errors)


def test_validate_rejects_unknown_tuning(tmp_path: Path):
    fp = _write_fixture(tmp_path, {
        "tuning": "AlienTuning", "tab": [], "chords": [],
    })
    errors = validate_fixture(fp)
    assert any("not a known preset" in str(e) for e in errors)


def test_validate_catches_string_fret_pitch_mismatch(tmp_path: Path):
    # Pitch 60 (C4) on string 5 (high E open=64) fret 0 doesn't match.
    fp = _write_fixture(tmp_path, {
        "tuning": "EADGBE",
        "tab": [
            {"start_s": 0.0, "end_s": 0.5, "pitch": 60, "string": 5, "fret": 0},
        ],
        "chords": [],
    })
    errors = validate_fixture(fp)
    assert any("yields MIDI" in str(e) for e in errors)


def test_validate_catches_pitch_out_of_range(tmp_path: Path):
    fp = _write_fixture(tmp_path, {
        "tuning": "EADGBE",
        "tab": [
            {"start_s": 0.0, "end_s": 0.5, "pitch": 200},
        ],
        "chords": [],
    })
    errors = validate_fixture(fp)
    assert any("out of MIDI range" in str(e) for e in errors)


def test_validate_catches_end_before_start(tmp_path: Path):
    fp = _write_fixture(tmp_path, {
        "tuning": "EADGBE",
        "tab": [
            {"start_s": 1.0, "end_s": 0.5, "pitch": 60},
        ],
        "chords": [],
    })
    errors = validate_fixture(fp)
    assert any("must be > start_s" in str(e) for e in errors)


def test_validate_catches_invalid_chord_root(tmp_path: Path):
    fp = _write_fixture(tmp_path, {
        "tuning": "EADGBE",
        "tab": [],
        "chords": [
            {"start_s": 0.0, "end_s": 1.0, "root": "Z", "quality": "maj"},
        ],
    })
    errors = validate_fixture(fp)
    assert any("not a valid pitch class" in str(e) for e in errors)


def test_validate_catches_invalid_chord_quality(tmp_path: Path):
    fp = _write_fixture(tmp_path, {
        "tuning": "EADGBE",
        "tab": [],
        "chords": [
            {"start_s": 0.0, "end_s": 1.0, "root": "C", "quality": "sus4"},
        ],
    })
    errors = validate_fixture(fp)
    assert any("not in supported set" in str(e) for e in errors)


def test_validate_handles_invalid_json(tmp_path: Path):
    fp = tmp_path / "bad.json"
    fp.write_text("{ this isn't json")
    errors = validate_fixture(fp)
    assert any("invalid JSON" in str(e) for e in errors)


def test_validate_handles_missing_file(tmp_path: Path):
    errors = validate_fixture(tmp_path / "nope.json")
    assert any("not found" in str(e) for e in errors)


def test_validation_error_str_format():
    e = ValidationError("foo", "bar")
    assert str(e) == "foo: bar"
