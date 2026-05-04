import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from music_decoder.persistence.models import Base
from music_decoder.persistence.repositories import (
    ChordSegmentRepo,
    JobRepo,
    UploadRepo,
)


@pytest.fixture
def session() -> Session:
    from sqlalchemy import event, text
    from sqlalchemy.orm import sessionmaker

    engine = create_engine("sqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def _fk_on(dbapi_conn: object, _: object) -> None:
        import sqlite3

        assert isinstance(dbapi_conn, sqlite3.Connection)
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as s:
        s.execute(text("PRAGMA foreign_keys=ON"))
        yield s


def test_bulk_insert_and_list(session: Session) -> None:
    upload = UploadRepo(session).create(
        sha256="x", original_filename="a.wav", mime_type="audio/wav",
        duration_s=10.0, sample_rate_hz=22050, declared_kind="solo_guitar",
        artifact_path="uploads/1/source.wav",
    )
    session.flush()
    job = JobRepo(session).enqueue(
        upload.id, "basic-pitch", "EADGBE", "standard", False, "v1",
    )
    session.flush()

    repo = ChordSegmentRepo(session)
    repo.bulk_insert(job.id, [
        {"start_s": 0.0, "end_s": 1.0, "root": "C", "quality": "maj", "confidence": 0.85},
        {"start_s": 1.0, "end_s": 2.0, "root": "F", "quality": "maj", "confidence": 0.70},
        {"start_s": 2.0, "end_s": 3.0, "root": "G", "quality": "7",   "confidence": 0.65},
    ])
    session.commit()

    rows = repo.list_for_job(job.id)
    assert len(rows) == 3
    assert rows[0].root == "C" and rows[0].quality == "maj"
    assert rows[2].root == "G" and rows[2].quality == "7"


def test_cascade_delete_removes_chord_rows(session: Session) -> None:
    upload = UploadRepo(session).create(
        sha256="y", original_filename="a.wav", mime_type="audio/wav",
        duration_s=10.0, sample_rate_hz=22050, declared_kind="solo_guitar",
        artifact_path="uploads/1/source.wav",
    )
    session.flush()
    job = JobRepo(session).enqueue(
        upload.id, "basic-pitch", "EADGBE", "standard", False, "v1",
    )
    session.flush()
    ChordSegmentRepo(session).bulk_insert(job.id, [
        {"start_s": 0.0, "end_s": 1.0, "root": "C", "quality": "maj", "confidence": 0.85},
    ])
    session.commit()
    session.delete(upload)
    session.commit()
    assert ChordSegmentRepo(session).list_for_job(job.id) == []
