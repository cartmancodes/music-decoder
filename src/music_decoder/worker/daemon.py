from __future__ import annotations

import signal
import time

from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from music_decoder.artifacts.base import ArtifactStore
from music_decoder.config.hyperparameters import HyperparameterSet
from music_decoder.logging_setup import get_logger
from music_decoder.persistence.repositories import JobRepo
from music_decoder.pipeline.orchestrator import process_audio

_log = get_logger("worker")


class WorkerDaemon:
    def __init__(
        self,
        *,
        engine: Engine,
        artifacts: ArtifactStore,
        hyperparameters: HyperparameterSet,
        poll_interval_s: float = 0.5,
    ) -> None:
        self.engine = engine
        self.artifacts = artifacts
        self.hp = hyperparameters
        self.poll_interval_s = poll_interval_s
        self._stop = False

    def warm_models(self) -> None:
        """Force load of basic-pitch and CREPE so the first job doesn't pay the cost."""
        try:
            import basic_pitch  # noqa: F401
            from basic_pitch import ICASSP_2022_MODEL_PATH  # noqa: F401
        except Exception as e:
            _log.warning("basic_pitch_warmup_failed", extra={"error": str(e)})
        try:
            import crepe  # noqa: F401
        except Exception as e:
            _log.warning("crepe_warmup_failed", extra={"error": str(e)})

    def stop(self) -> None:
        self._stop = True

    def install_signal_handlers(self) -> None:
        def handler(_signum: int, _frame: object) -> None:
            _log.info("worker_signal_received")
            self._stop = True

        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGINT, handler)

    def _claim_next(self) -> int | None:
        with Session(self.engine) as s:
            j = JobRepo(s).pop_next_queued()
            if j is None:
                return None
            s.commit()
            return j.id

    def _process_one(self, job_id: int) -> None:
        try:
            process_audio(
                job_id=job_id, engine=self.engine,
                artifacts=self.artifacts, hyperparameters=self.hp,
            )
        except Exception as e:
            _log.exception("worker_job_failed", extra={"job_id": job_id, "error": str(e)})

    def run(self) -> None:
        _log.info("worker_starting")
        self.warm_models()
        while not self._stop:
            job_id = self._claim_next()
            if job_id is None:
                time.sleep(self.poll_interval_s)
                continue
            _log.info("worker_processing", extra={"job_id": job_id})
            self._process_one(job_id)
        _log.info("worker_stopped")

    def run_until_idle(self, idle_timeout_s: float = 5.0) -> None:
        """Test helper: process anything queued, then exit when idle."""
        self.warm_models()
        idle_for = 0.0
        while idle_for < idle_timeout_s:
            job_id = self._claim_next()
            if job_id is None:
                time.sleep(self.poll_interval_s)
                idle_for += self.poll_interval_s
                continue
            idle_for = 0.0
            self._process_one(job_id)
