from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from music_decoder.persistence.models import (
    Base,
    Job,
    Note,
    Upload,
)


def _engine():
    e = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(e)
    # SQLite needs explicit FK pragma for cascade delete to fire
    with e.connect() as conn:
        from sqlalchemy import text

        conn.execute(text("PRAGMA foreign_keys=ON"))
    return e


def test_create_upload_and_job():
    engine = _engine()
    with Session(engine) as s:
        # Re-enable FK pragma for this connection
        from sqlalchemy import text

        s.execute(text("PRAGMA foreign_keys=ON"))
        u = Upload(
            sha256="abc",
            original_filename="a.wav",
            mime_type="audio/wav",
            duration_s=30.0,
            sample_rate_hz=22050,
            declared_kind="solo_guitar",
            artifact_path="uploads/1/source.wav",
            created_at=datetime.now(UTC),
        )
        s.add(u)
        s.flush()
        j = Job(
            upload_id=u.id,
            status="queued",
            transcription_model="basic-pitch",
            requested_tuning="EADGBE",
            requested_quality="standard",
            use_demucs=False,
            hyperparameter_set="2026-05-04-baseline",
            created_at=datetime.now(UTC),
        )
        s.add(j)
        s.commit()
        assert j.id is not None
        assert j.upload_id == u.id


def test_status_check_constraint_rejects_garbage():
    from sqlalchemy.exc import IntegrityError

    engine = _engine()
    with Session(engine) as s:
        from sqlalchemy import text

        s.execute(text("PRAGMA foreign_keys=ON"))
        u = Upload(
            sha256="x",
            original_filename="a.wav",
            mime_type="audio/wav",
            duration_s=1.0,
            sample_rate_hz=22050,
            declared_kind="solo_guitar",
            artifact_path="uploads/1/source.wav",
            created_at=datetime.now(UTC),
        )
        s.add(u)
        s.flush()
        s.add(
            Job(
                upload_id=u.id,
                status="WHATEVER",
                transcription_model="basic-pitch",
                requested_tuning="EADGBE",
                requested_quality="standard",
                use_demucs=False,
                hyperparameter_set="x",
                created_at=datetime.now(UTC),
            )
        )
        with pytest.raises(IntegrityError):
            s.commit()


def test_cascade_delete_removes_children():
    engine = _engine()
    # Apply the FK pragma to all connections from this engine.
    from sqlalchemy import event

    @event.listens_for(engine, "connect")
    def _fk_on(dbapi_conn, _):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    with Session(engine) as s:
        u = Upload(
            sha256="z",
            original_filename="a.wav",
            mime_type="audio/wav",
            duration_s=1.0,
            sample_rate_hz=22050,
            declared_kind="solo_guitar",
            artifact_path="uploads/1/source.wav",
            created_at=datetime.now(UTC),
        )
        s.add(u)
        s.flush()
        j = Job(
            upload_id=u.id,
            status="succeeded",
            transcription_model="basic-pitch",
            requested_tuning="EADGBE",
            requested_quality="standard",
            use_demucs=False,
            hyperparameter_set="x",
            created_at=datetime.now(UTC),
        )
        s.add(j)
        s.flush()
        s.add(Note(job_id=j.id, start_s=0.0, end_s=1.0, pitch=60, velocity=80, confidence=0.9))
        s.commit()
        s.delete(u)
        s.commit()
        assert s.query(Note).count() == 0
        assert s.query(Job).count() == 0
