from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_KEYS = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def render_chromagram_figure(
    chroma: np.ndarray[Any, np.dtype[np.float64]], *, sr: int, hop_length: int,
) -> matplotlib.figure.Figure:
    fig, ax = plt.subplots(figsize=(8, 3))
    duration_s = chroma.shape[1] * hop_length / sr
    ax.imshow(
        chroma, aspect="auto", origin="lower",
        extent=(0, duration_s, 0, 12), cmap="magma",
    )
    ax.set_yticks(np.arange(12) + 0.5)
    ax.set_yticklabels(_KEYS)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("pitch class")
    fig.tight_layout()
    return fig
