import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from music_decoder.persistence.models import Base
from music_decoder.persistence.repositories import (
    JobRepo,
    UploadRepo,
)


@pytest.fixture
def session():
    e = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(e)
    with Session(e) as s:
        yield s


def test_upload_create_and_lookup_by_sha(session: Session):
    repo = UploadRepo(session)
    u = repo.create(
        sha256="abc", original_filename="a.wav", mime_type="audio/wav",
        duration_s=10.0, sample_rate_hz=22050, declared_kind="solo_guitar",
        artifact_path="uploads/1/source.wav",
    )
    session.commit()
    found = repo.find_by_sha256("abc")
    assert found is not None and found.id == u.id


def test_job_lifecycle(session: Session):
    upload = UploadRepo(session).create(
        sha256="x", original_filename="a.wav", mime_type="audio/wav",
        duration_s=1.0, sample_rate_hz=22050, declared_kind="solo_guitar",
        artifact_path="uploads/1/source.wav",
    )
    session.flush()
    jrepo = JobRepo(session)
    j = jrepo.enqueue(
        upload_id=upload.id, transcription_model="basic-pitch",
        requested_tuning="EADGBE", requested_quality="standard",
        use_demucs=False, hyperparameter_set="2026-05-04-baseline",
    )
    session.commit()
    assert j.status == "queued"
    jrepo.mark_running(j.id)
    session.commit()
    assert jrepo.get(j.id).status == "running"
    jrepo.mark_succeeded(j.id)
    session.commit()
    assert jrepo.get(j.id).status == "succeeded"


def test_pop_next_queued_returns_oldest_first(session: Session):
    upload = UploadRepo(session).create(
        sha256="x", original_filename="a.wav", mime_type="audio/wav",
        duration_s=1.0, sample_rate_hz=22050, declared_kind="solo_guitar",
        artifact_path="uploads/1/source.wav",
    )
    session.flush()
    jrepo = JobRepo(session)
    j1 = jrepo.enqueue(upload.id, "basic-pitch", "EADGBE", "standard", False, "v1")
    j2 = jrepo.enqueue(upload.id, "basic-pitch", "EADGBE", "standard", False, "v1")
    session.commit()
    popped = jrepo.pop_next_queued()
    assert popped.id == j1.id
    session.commit()
    popped2 = jrepo.pop_next_queued()
    assert popped2.id == j2.id
