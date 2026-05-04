import json

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from music_decoder.persistence.models import Base
from music_decoder.persistence.repositories import (
    JobProgressRepo,
    JobRepo,
    UploadRepo,
)
from music_decoder.pipeline.events import StageEventEmitter


def test_emitter_writes_progress_rows():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        u = UploadRepo(s).create(
            sha256="x", original_filename="a.wav", mime_type="audio/wav",
            duration_s=1.0, sample_rate_hz=22050,
            declared_kind="solo_guitar",
            artifact_path="uploads/1/source.wav",
        )
        s.flush()
        j = JobRepo(s).enqueue(
            u.id, "basic-pitch", "EADGBE", "standard", False,
            "2026-05-04-baseline",
        )
        s.flush()
        emitter = StageEventEmitter(JobProgressRepo(s), job_id=j.id)
        with emitter("audio_io") as record:
            record["bytes_read"] = 1024
        s.commit()
        from music_decoder.persistence.models import JobProgress

        rows = list(s.query(JobProgress))
        assert len(rows) == 1
        assert rows[0].stage == "audio_io"
        assert rows[0].success is True
        assert json.loads(rows[0].summary_json) == {"bytes_read": 1024}


def test_emitter_records_failure():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        u = UploadRepo(s).create(
            sha256="y", original_filename="a.wav", mime_type="audio/wav",
            duration_s=1.0, sample_rate_hz=22050, declared_kind="solo_guitar",
            artifact_path="uploads/1/source.wav",
        )
        s.flush()
        j = JobRepo(s).enqueue(u.id, "basic-pitch", "EADGBE", "standard", False, "v1")
        s.flush()
        emitter = StageEventEmitter(JobProgressRepo(s), job_id=j.id)
        try:
            with emitter("transcription"):
                raise ValueError("boom")
        except ValueError:
            pass
        s.commit()
        from music_decoder.persistence.models import JobProgress
        row = s.query(JobProgress).filter_by(stage="transcription").one()
        assert row.success is False
        assert "boom" in (row.error or "")
