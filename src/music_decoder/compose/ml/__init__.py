"""Optional pretrained models for the composer (v5 spec 5.4 and 5.5).

Everything here needs the ``ml`` extra and weights fetched ahead of time.
Importing this package loads no ML library: each module imports its
dependencies lazily, when a model is loaded.
"""

from __future__ import annotations

INSTALL_HINT = (
    'install the ml extra with `pip install -e ".[ml]"` and fetch the weights with '
    "`music-decoder fetch-models --composer-models`"
)


class MLUnavailableError(RuntimeError):
    """An optional model was asked for, but its package or weights are missing."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"{reason}; {INSTALL_HINT}")


__all__ = ["INSTALL_HINT", "MLUnavailableError"]
