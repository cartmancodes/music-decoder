# src/music_decoder/paths.py
"""Single source of truth for filesystem paths the package needs.

Replaces the ``Path(__file__).resolve().parents[3] / "config" / ...``
idiom that was copy-pasted across config/runtime, config/hyperparameters
and synth, and which previously resolved wrong under a wheel install.
"""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return the repository root (the directory containing ``pyproject.toml``).

    ``paths.py`` lives at ``src/music_decoder/paths.py``; ``parents[2]``
    is therefore the repo root.
    """
    return Path(__file__).resolve().parents[2]


def bundled_config(name: str) -> Path:
    """Return the absolute path to ``config/<name>`` under the repo root."""
    return project_root() / "config" / name
