from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from music_decoder.artifacts.filesystem import FilesystemArtifactStore
from music_decoder.config.hyperparameters import load_hyperparameters
from music_decoder.persistence.models import Base, Job
from music_decoder.persistence.repositories import JobRepo, UploadRepo
from music_decoder.worker.daemon import WorkerDaemon


@pytest.mark.integration
@pytest.mark.slow
def test_daemon_picks_up_queued_job_and_marks_it_terminal(tmp_path: Path):
    engine = create_engine(f"sqlite:///{tmp_path / 'app.sqlite3'}")
    Base.metadata.create_all(engine)
    artifacts = FilesystemArtifactStore(root=tmp_path / "artifacts")
    hp = load_hyperparameters(Path("config/hyperparameters.yaml"))

    src = Path("tests/fixtures/audio_samples/sine_440.wav")
    artifacts.put("uploads/1/source.wav", src.read_bytes())

    with Session(engine) as s:
        u = UploadRepo(s).create(
            sha256="x", original_filename="sine.wav", mime_type="audio/wav",
            duration_s=1.0, sample_rate_hz=22050, declared_kind="solo_guitar",
            artifact_path="uploads/1/source.wav",
        )
        s.flush()
        j = JobRepo(s).enqueue(
            upload_id=u.id, transcription_model="basic-pitch",
            requested_tuning="EADGBE", requested_quality="standard",
            use_demucs=False, hyperparameter_set=hp.id,
        )
        s.commit()
        job_id = j.id

    daemon = WorkerDaemon(
        engine=engine, artifacts=artifacts, hyperparameters=hp,
        poll_interval_s=0.2,
    )
    daemon.run_until_idle(idle_timeout_s=2.0)

    with Session(engine) as s:
        job = s.get(Job, job_id)
        assert job.status in ("succeeded", "failed")
