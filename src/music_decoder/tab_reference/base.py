from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class RawTabInput:
    text: str


@dataclass(frozen=True)
class FetchedTab:
    source: str
    raw_text: str


class TabReferenceProvider(Protocol):
    def fetch(self, payload: RawTabInput) -> FetchedTab: ...
