from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_Array = np.ndarray[Any, np.dtype[np.float64]]


def render_waveform_figure(
    samples: _Array, *, sr: int, onsets_s: _Array | None = None,
) -> matplotlib.figure.Figure:
    fig, ax = plt.subplots(figsize=(8, 2))
    t = np.arange(samples.size) / sr
    ax.plot(t, samples, linewidth=0.5, color="#234")
    if onsets_s is not None:
        for o in onsets_s:
            ax.axvline(o, color="#c33", linewidth=0.7, alpha=0.6)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("amplitude")
    fig.tight_layout()
    return fig
