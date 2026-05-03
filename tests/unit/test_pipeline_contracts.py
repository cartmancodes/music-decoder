# tests/unit/test_pipeline_contracts.py
import datetime

import numpy as np
import pytest

from music_decoder.pipeline.contracts import (  # noqa: F401
    AudioSource,
    BeatGrid,
    KeyDetectionResult,
    KeyEstimate,
    LoadedAudio,
    SeparationResult,
    StageEvent,
    TabAssignmentResult,
    TabbedNote,
    TabPosition,
    TabReferenceMatch,
    TranscribedNote,
    TranscriptionResult,
)
from music_decoder.tab_assignment.tuning import get_preset


def test_loaded_audio_holds_basic_fields(tmp_path):
    src = AudioSource(
        path=tmp_path / "a.wav", declared_kind="solo_guitar",
        requested_quality="standard", requested_tuning=get_preset("EADGBE"),
    )
    a = LoadedAudio(
        samples=np.zeros(22050, dtype=np.float32), sr=22050,
        duration_s=1.0, sha256="abc", source=src,
    )
    assert a.duration_s == 1.0
    assert a.source.requested_tuning.name == "EADGBE"


def test_transcribed_note_is_immutable():
    n = TranscribedNote(start_s=0.0, end_s=1.0, pitch=60, velocity=80, confidence=0.9)
    with pytest.raises(Exception):  # noqa: B017
        n.pitch = 61  # type: ignore[misc]


def test_key_estimate_records_profile_and_margin():
    k = KeyEstimate(
        tonic="C", mode="major", profile="krumhansl_kessler",
        correlation=0.82, margin=0.15,
    )
    assert k.tonic == "C"
    assert k.profile == "krumhansl_kessler"


def test_tab_position_indexing():
    p = TabPosition(string=5, fret=0)
    assert p.string == 5 and p.fret == 0


def test_stage_event_serializable():
    e = StageEvent(
        job_id=1, stage="transcription",
        started_at=datetime.datetime.now(datetime.UTC),
        ended_at=datetime.datetime.now(datetime.UTC),
        success=True, error=None, summary={"notes": 42},
    )
    assert e.summary["notes"] == 42
