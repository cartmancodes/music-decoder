from datetime import UTC
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from music_decoder.artifacts.filesystem import FilesystemArtifactStore
from music_decoder.persistence.models import Base, Job, Upload
from music_decoder.ui.services import enqueue_upload


def test_enqueue_upload_writes_upload_and_job(tmp_path: Path):
    engine = create_engine(f"sqlite:///{tmp_path / 'app.sqlite3'}")
    Base.metadata.create_all(engine)
    artifacts = FilesystemArtifactStore(root=tmp_path / "artifacts")
    audio_bytes = (Path("tests/fixtures/audio_samples/sine_440.wav")).read_bytes()
    job_id = enqueue_upload(
        engine=engine, artifacts=artifacts,
        original_filename="sine.wav", mime_type="audio/wav",
        content=audio_bytes,
        declared_kind="solo_guitar",
        transcription_model="basic-pitch",
        requested_tuning="EADGBE", requested_quality="standard",
        use_demucs=False,
        hyperparameter_set="2026-05-04-baseline",
    )
    assert job_id is not None
    with Session(engine) as s:
        job = s.get(Job, job_id)
        assert job is not None
        assert job.status == "queued"
        upload = s.get(Upload, job.upload_id)
        assert upload.original_filename == "sine.wav"
        assert artifacts.exists(upload.artifact_path)


def test_job_status_returns_none_for_missing_job(tmp_path: Path):
    from music_decoder.ui.services import job_status
    engine = create_engine(f"sqlite:///{tmp_path / 'app.sqlite3'}")
    Base.metadata.create_all(engine)
    assert job_status(engine, 999) is None


def test_job_status_returns_progress_rows(tmp_path: Path):
    from datetime import datetime

    from music_decoder.persistence.repositories import JobProgressRepo
    from music_decoder.ui.services import job_status

    engine = create_engine(f"sqlite:///{tmp_path / 'app.sqlite3'}")
    Base.metadata.create_all(engine)
    audio_bytes = (Path("tests/fixtures/audio_samples/sine_440.wav")).read_bytes()
    artifacts = FilesystemArtifactStore(root=tmp_path / "artifacts")
    job_id = enqueue_upload(
        engine=engine, artifacts=artifacts,
        original_filename="x.wav", mime_type="audio/wav", content=audio_bytes,
        declared_kind="solo_guitar", transcription_model="basic-pitch",
        requested_tuning="EADGBE", requested_quality="standard",
        use_demucs=False, hyperparameter_set="v1",
    )
    with Session(engine) as s:
        JobProgressRepo(s).record(
            job_id=job_id, stage="audio_io",
            started_at=datetime.now(UTC),
            ended_at=datetime.now(UTC),
            success=True, error=None, summary_json="{}",
        )
        s.commit()
    status = job_status(engine, job_id)
    assert status is not None
    assert status["status"] == "queued"
    assert any(p["stage"] == "audio_io" for p in status["progress"])


def test_enqueue_upload_deduplicates_by_sha256(tmp_path: Path):
    """Uploading the same file twice creates only one Upload row but two Jobs."""
    from sqlalchemy import func
    from sqlalchemy import select as sa_select

    engine = create_engine(f"sqlite:///{tmp_path / 'app.sqlite3'}")
    Base.metadata.create_all(engine)
    artifacts = FilesystemArtifactStore(root=tmp_path / "artifacts")
    audio_bytes = (Path("tests/fixtures/audio_samples/sine_440.wav")).read_bytes()

    job_id_1 = enqueue_upload(
        engine=engine, artifacts=artifacts,
        original_filename="sine.wav", mime_type="audio/wav",
        content=audio_bytes,
        declared_kind="solo_guitar",
        transcription_model="basic-pitch",
        requested_tuning="EADGBE", requested_quality="standard",
        use_demucs=False,
        hyperparameter_set="2026-05-04-baseline",
    )
    job_id_2 = enqueue_upload(
        engine=engine, artifacts=artifacts,
        original_filename="sine_copy.wav", mime_type="audio/wav",
        content=audio_bytes,
        declared_kind="solo_guitar",
        transcription_model="basic-pitch",
        requested_tuning="EADGBE", requested_quality="standard",
        use_demucs=False,
        hyperparameter_set="2026-05-04-baseline",
    )

    assert job_id_1 != job_id_2

    with Session(engine) as s:
        upload_count = s.execute(
            sa_select(func.count()).select_from(Upload)
        ).scalar_one()
        assert upload_count == 1, "expected exactly one Upload row for identical content"

        job1 = s.get(Job, job_id_1)
        job2 = s.get(Job, job_id_2)
        assert job1 is not None and job2 is not None
        assert job1.upload_id == job2.upload_id, "both jobs should reference the same upload"


def test_record_tab_reference_persists_row(tmp_path: Path):
    """record_tab_reference inserts a TabReference row with correct source and similarity."""
    from music_decoder.persistence.models import TabReference
    from music_decoder.ui.services import record_tab_reference

    engine = create_engine(f"sqlite:///{tmp_path / 'app.sqlite3'}")
    Base.metadata.create_all(engine)
    audio_bytes = (Path("tests/fixtures/audio_samples/sine_440.wav")).read_bytes()
    artifacts = FilesystemArtifactStore(root=tmp_path / "artifacts")
    job_id = enqueue_upload(
        engine=engine, artifacts=artifacts,
        original_filename="x.wav", mime_type="audio/wav", content=audio_bytes,
        declared_kind="solo_guitar", transcription_model="basic-pitch",
        requested_tuning="EADGBE", requested_quality="standard",
        use_demucs=False, hyperparameter_set="v1",
    )

    raw_tab_text = "e|--0--|\nB|--1--|\nG|--2--|\nD|--3--|\nA|--4--|\nE|--5--|"
    predicted_notes: list[dict[str, object]] = [
        {
            "start_s": 0.0, "end_s": 0.5, "pitch": 60, "velocity": 80,
            "confidence": 0.9, "string": 2, "fret": 3,
            "dropped_reason": None,
        }
    ]

    fetched, sim = record_tab_reference(engine, job_id, raw_tab_text, predicted_notes)

    assert fetched is not None
    assert 0.0 <= sim <= 1.0

    with Session(engine) as s:
        rows = list(s.execute(
            __import__("sqlalchemy").select(TabReference).where(TabReference.job_id == job_id)
        ).scalars())
        assert len(rows) == 1
        row = rows[0]
        assert row.source in ("user_pasted_text", "user_pasted_url")
        assert row.similarity_to_prediction is not None
        assert abs(float(row.similarity_to_prediction) - sim) < 1e-9


def test_load_results_returns_assembled_payload(tmp_path: Path):
    """After a successful job, load_results returns key, tempo, notes, and tab refs."""
    import json

    from music_decoder.persistence.repositories import (
        KeyEstimateRepo,
        NoteRepo,
        TempoEstimateRepo,
    )
    from music_decoder.ui.services import load_results

    engine = create_engine(f"sqlite:///{tmp_path / 'app.sqlite3'}")
    Base.metadata.create_all(engine)
    audio_bytes = (Path("tests/fixtures/audio_samples/sine_440.wav")).read_bytes()
    artifacts = FilesystemArtifactStore(root=tmp_path / "artifacts")
    job_id = enqueue_upload(
        engine=engine, artifacts=artifacts,
        original_filename="x.wav", mime_type="audio/wav", content=audio_bytes,
        declared_kind="solo_guitar", transcription_model="basic-pitch",
        requested_tuning="EADGBE", requested_quality="standard",
        use_demucs=False, hyperparameter_set="v1",
    )
    with Session(engine) as s:
        KeyEstimateRepo(s).bulk_insert(job_id, [{
            "scope": "global", "window_start_s": None, "window_end_s": None,
            "profile": "krumhansl_kessler", "rank": 1,
            "tonic": "C", "mode": "major",
            "correlation": 0.85, "margin": 0.10,
        }])
        TempoEstimateRepo(s).upsert(
            job_id=job_id, tempo_bpm=120.0,
            beat_times_s_json=json.dumps([0.0, 0.5, 1.0]),
            downbeat_times_s_json=json.dumps([0.0]),
            ts_numerator=4, ts_denominator=4,
            ts_confidence=0.6, ts_assumed=False,
        )
        NoteRepo(s).bulk_insert(job_id, [{
            "start_s": 0.0, "end_s": 0.5, "pitch": 60, "velocity": 80,
            "confidence": 0.9, "string": 4, "fret": 1,
            "cost_breakdown_json": None, "dropped_reason": None,
        }])
        s.commit()
    payload = load_results(engine, job_id)
    assert payload is not None
    assert payload["tempo"]["tempo_bpm"] == 120.0
    assert any(k["tonic"] == "C" for k in payload["keys"])
    assert payload["notes"][0]["pitch"] == 60


def test_load_results_includes_chord_segments(tmp_path: Path):
    from sqlalchemy import create_engine as _create_engine
    from sqlalchemy.orm import Session as _Session

    from music_decoder.artifacts.filesystem import FilesystemArtifactStore
    from music_decoder.persistence.models import Base as _Base
    from music_decoder.persistence.repositories import ChordSegmentRepo
    from music_decoder.ui.services import enqueue_upload, load_results

    engine = _create_engine(f"sqlite:///{tmp_path / 'app.sqlite3'}")
    _Base.metadata.create_all(engine)
    audio = (Path("tests/fixtures/audio_samples/sine_440.wav")).read_bytes()
    artifacts = FilesystemArtifactStore(root=tmp_path / "artifacts")
    job_id = enqueue_upload(
        engine=engine, artifacts=artifacts, original_filename="x.wav",
        mime_type="audio/wav", content=audio, declared_kind="solo_guitar",
        transcription_model="basic-pitch", requested_tuning="EADGBE",
        requested_quality="standard", use_demucs=False, hyperparameter_set="v1",
    )
    with _Session(engine) as s:
        ChordSegmentRepo(s).bulk_insert(job_id, [
            {"start_s": 0.0, "end_s": 4.0, "root": "C", "quality": "maj", "confidence": 0.85},
            {"start_s": 4.0, "end_s": 8.0, "root": "G", "quality": "7",   "confidence": 0.75},
        ])
        s.commit()
    payload = load_results(engine, job_id)
    assert payload is not None
    assert len(payload["chord_segments"]) == 2
    assert payload["chord_segments"][0]["root"] == "C"
    assert payload["chord_segments"][1]["quality"] == "7"
