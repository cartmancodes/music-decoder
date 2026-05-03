from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime

from music_decoder.persistence.repositories import JobProgressRepo


class StageEventEmitter:
    def __init__(self, repo: JobProgressRepo, *, job_id: int) -> None:
        self.repo = repo
        self.job_id = job_id

    @contextmanager
    def __call__(self, stage: str) -> Iterator[dict]:  # type: ignore[type-arg]
        summary: dict = {}  # type: ignore[type-arg]
        started = datetime.now(UTC)
        try:
            yield summary
        except Exception as e:
            self.repo.record(
                job_id=self.job_id, stage=stage,
                started_at=started, ended_at=datetime.now(UTC),
                success=False, error=f"{type(e).__name__}: {e}",
                summary_json=json.dumps(summary, default=str),
            )
            raise
        else:
            self.repo.record(
                job_id=self.job_id, stage=stage,
                started_at=started, ended_at=datetime.now(UTC),
                success=True, error=None,
                summary_json=json.dumps(summary, default=str),
            )
