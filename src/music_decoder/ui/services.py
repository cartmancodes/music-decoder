from __future__ import annotations

import hashlib
from typing import Literal

from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from music_decoder.artifacts.base import ArtifactStore
from music_decoder.persistence.repositories import JobRepo, UploadRepo


def enqueue_upload(
    *,
    engine: Engine,
    artifacts: ArtifactStore,
    original_filename: str,
    mime_type: str,
    content: bytes,
    declared_kind: Literal["solo_guitar", "full_mix"],
    transcription_model: Literal["basic-pitch", "crepe"],
    requested_tuning: str,
    requested_quality: Literal["standard", "high"],
    use_demucs: bool,
    hyperparameter_set: str,
) -> int:
    sha = hashlib.sha256(content).hexdigest()
    with Session(engine) as s:
        upload = UploadRepo(s).create(
            sha256=sha, original_filename=original_filename, mime_type=mime_type,
            duration_s=None, sample_rate_hz=None, declared_kind=declared_kind,
            artifact_path="placeholder",
        )
        s.flush()
        ext = original_filename.rsplit(".", 1)[-1].lower() or "wav"
        key = f"uploads/{upload.id}/source.{ext}"
        artifacts.put(key, content)
        upload.artifact_path = key

        job = JobRepo(s).enqueue(
            upload_id=upload.id, transcription_model=transcription_model,
            requested_tuning=requested_tuning, requested_quality=requested_quality,
            use_demucs=use_demucs, hyperparameter_set=hyperparameter_set,
        )
        s.commit()
        return int(job.id)
