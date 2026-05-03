from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np


@dataclass(frozen=True)
class GroundTruth:
    intervals: np.ndarray[Any, np.dtype[Any]]
    pitches_midi: np.ndarray[Any, np.dtype[Any]]
    key: tuple[str, str] | None
    tempo_bpm: float | None
    tab: list[tuple[int, int, int]] | None


@dataclass(frozen=True)
class Fixture:
    name: str
    source: Literal["guitarset", "synthetic", "manual"]
    audio_path: Path
    ground_truth: GroundTruth


class FixtureLoader(Protocol):
    def load(self) -> Iterable[Fixture]: ...
