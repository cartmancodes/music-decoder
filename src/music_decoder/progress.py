"""Progress callback protocol used by long-running stages."""

from __future__ import annotations

from typing import Protocol


class ProgressCallback(Protocol):
    """Stage progress reporter; called many times per pipeline run."""

    def __call__(self, stage: str, fraction: float) -> None: ...


class NullProgress:
    """No-op progress callback used when the caller does not pass one."""

    def __call__(self, stage: str, fraction: float) -> None:
        return None
