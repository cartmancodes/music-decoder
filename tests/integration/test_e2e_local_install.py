# tests/integration/test_e2e_local_install.py
"""End-to-end: simulates the local-install flow without actually launching Streamlit.

1. Bootstrap config + DB + artifacts.
2. Enqueue a job through the same service the UI uses.
3. Run the worker daemon's `run_until_idle()`.
4. Assert the job reaches a terminal state and produced notes/keys/tempo rows.
"""
from pathlib import Path

import pytest

from music_decoder.artifacts.filesystem import FilesystemArtifactStore
from music_decoder.config.hyperparameters import load_hyperparameters
from music_decoder.persistence.session import build_engine, run_migrations
from music_decoder.ui.services import enqueue_upload, load_results
from music_decoder.worker.daemon import WorkerDaemon


@pytest.mark.integration
@pytest.mark.slow
def test_full_pipeline_on_synthetic_audio(tmp_path: Path):
    db_path = tmp_path / "app.sqlite3"
    artifact_dir = tmp_path / "artifacts"
    engine = build_engine(db_path)
    run_migrations(engine)
    artifacts = FilesystemArtifactStore(root=artifact_dir)
    hp = load_hyperparameters(Path("config/hyperparameters.yaml"))

    audio_bytes = (Path("tests/fixtures/audio_samples/sine_440.wav")).read_bytes()
    job_id = enqueue_upload(
        engine=engine, artifacts=artifacts,
        original_filename="sine.wav", mime_type="audio/wav",
        content=audio_bytes, declared_kind="solo_guitar",
        transcription_model="basic-pitch", requested_tuning="EADGBE",
        requested_quality="standard", use_demucs=False,
        hyperparameter_set=hp.id,
    )

    daemon = WorkerDaemon(
        engine=engine, artifacts=artifacts, hyperparameters=hp,
        poll_interval_s=0.2,
    )
    daemon.run_until_idle(idle_timeout_s=2.0)

    payload = load_results(engine, job_id)
    assert payload is not None
    assert payload["job"]["status"] == "succeeded", (
        f"job did not succeed: {payload['job']}"
    )
    assert payload["tempo"] is not None
    assert len(payload["keys"]) >= 6   # 3 per profile minimum
    # Notes: a 1-second sine tone at 440 Hz should yield at least one detected note.
    assert len(payload["notes"]) >= 1
