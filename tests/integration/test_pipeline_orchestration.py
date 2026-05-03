import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from music_decoder.artifacts.filesystem import FilesystemArtifactStore
from music_decoder.config.hyperparameters import HyperparameterSet, load_hyperparameters
from music_decoder.persistence.models import Base, Job
from music_decoder.persistence.repositories import UploadRepo, JobRepo
from music_decoder.pipeline.orchestrator import process_audio


@pytest.mark.integration
@pytest.mark.slow
def test_process_audio_succeeds_on_synthetic_clip(tmp_path: Path):
    engine = create_engine(f"sqlite:///{tmp_path / 'app.sqlite3'}")
    Base.metadata.create_all(engine)
    artifacts = FilesystemArtifactStore(root=tmp_path / "artifacts")
    hp = load_hyperparameters(Path("config/hyperparameters.yaml"))

    src_wav = Path("tests/fixtures/audio_samples/sine_440.wav")
    artifact_key = "uploads/1/source.wav"
    artifacts.put(artifact_key, src_wav.read_bytes())

    with Session(engine) as s:
        u = UploadRepo(s).create(
            sha256="x" * 64, original_filename="sine.wav", mime_type="audio/wav",
            duration_s=1.0, sample_rate_hz=22050, declared_kind="solo_guitar",
            artifact_path=artifact_key,
        )
        s.flush()
        j = JobRepo(s).enqueue(
            upload_id=u.id, transcription_model="basic-pitch",
            requested_tuning="EADGBE", requested_quality="standard",
            use_demucs=False, hyperparameter_set=hp.id,
        )
        s.commit()
        job_id = j.id

    process_audio(
        job_id=job_id, engine=engine, artifacts=artifacts, hyperparameters=hp,
    )

    with Session(engine) as s:
        j = s.get(Job, job_id)
        assert j.status == "succeeded"
        from music_decoder.persistence.models import JobProgress, Note, KeyEstimate, TempoEstimate
        progress = s.query(JobProgress).filter_by(job_id=job_id).all()
        stages = [p.stage for p in progress]
        assert "audio_io" in stages
        assert "transcription" in stages
        assert "key_detection" in stages
        assert "beat_tracking" in stages
        assert "tab_assignment" in stages
        assert s.query(TempoEstimate).filter_by(job_id=job_id).count() == 1
        assert s.query(KeyEstimate).filter_by(job_id=job_id).count() >= 6
