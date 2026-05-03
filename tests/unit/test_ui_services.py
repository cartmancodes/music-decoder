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
    from datetime import datetime, timezone
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
            started_at=datetime.now(timezone.utc),
            ended_at=datetime.now(timezone.utc),
            success=True, error=None, summary_json="{}",
        )
        s.commit()
    status = job_status(engine, job_id)
    assert status is not None
    assert status["status"] == "queued"
    assert any(p["stage"] == "audio_io" for p in status["progress"])
