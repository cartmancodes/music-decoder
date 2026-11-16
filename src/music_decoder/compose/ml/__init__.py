"""Optional pretrained models for the composer (v5 spec 5.4 and 5.5).

Everything here needs the ``ml`` extra and weights fetched ahead of time.
Importing this package loads no ML library: each module imports its
dependencies lazily, when a model is loaded.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from music_decoder.types import Note

INSTALL_HINT = (
    'install the ml extra with `pip install -e ".[ml]"` and fetch the weights with '
    "`music-decoder fetch-models --composer-models`"
)


class MLUnavailableError(RuntimeError):
    """An optional model was asked for, but its package or weights are missing."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"{reason}; {INSTALL_HINT}")


@runtime_checkable
class MelodyProposer(Protocol):
    """Proposes a melody over an accompaniment (v5 spec 5.5).

    ``context`` is the accompaniment the melody goes over, ``end_s`` where the
    piece ends. The same arguments give the same proposal. The composer uses
    it as a soft prior only (see ``generate_melody(pitch_prior=)``).
    """

    def propose(
        self, context: Sequence[Note], *, end_s: float, tempo_bpm: float, seed: int
    ) -> list[Note]: ...


__all__ = ["INSTALL_HINT", "MLUnavailableError", "MelodyProposer"]
