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
