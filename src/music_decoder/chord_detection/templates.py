"""Chord templates and (root, quality) <-> index/label helpers.

96 chord templates (12 roots x 8 qualities) plus a no-chord template = 97 rows.
Each template is a length-12 binary vector indicating active pitch classes
(C=0, C#=1, ..., B=11). The no-chord template is a uniform 1/12 vector so its
cosine similarity against any beat-chroma is bounded; the per-beat scorer also
checks an absolute threshold to decide N vs. a real chord.

Phase B-3 expanded the vocabulary from 4 -> 8 qualities to score the wider
range of chords that real-world detectors (e.g. madmom) emit:

    maj    — major triad           (0, 4, 7)
    min    — minor triad           (0, 3, 7)
    7      — dominant 7th          (0, 4, 7, 10)
    maj7   — major 7th             (0, 4, 7, 11)
    min7   — minor 7th             (0, 3, 7, 10)   [Phase B-3]
    dim    — diminished triad      (0, 3, 6)        [Phase B-3]
    sus4   — suspended 4th         (0, 5, 7)        [Phase B-3]
    aug    — augmented triad       (0, 4, 8)        [Phase B-3]
"""
from __future__ import annotations

from typing import Any

import numpy as np

ROOTS: tuple[str, ...] = (
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
)
QUALITIES: tuple[str, ...] = (
    "maj", "min", "7", "maj7", "min7", "dim", "sus4", "aug",
)
NO_CHORD = "N"

# Pitch-class offsets (relative to root) for each quality.
_QUALITY_INTERVALS: dict[str, tuple[int, ...]] = {
    "maj": (0, 4, 7),
    "min": (0, 3, 7),
    "7": (0, 4, 7, 10),
    "maj7": (0, 4, 7, 11),
    "min7": (0, 3, 7, 10),
    "dim": (0, 3, 6),
    "sus4": (0, 5, 7),
    "aug": (0, 4, 8),
}

_ROOT_INDEX: dict[str, int] = {r: i for i, r in enumerate(ROOTS)}


def template_for(root: str, quality: str) -> np.ndarray[Any, np.dtype[np.float64]]:
    if root == NO_CHORD:
        return np.full(12, 1.0 / 12.0, dtype=float)
    if quality not in _QUALITY_INTERVALS:
        raise KeyError(f"unknown chord quality: {quality!r}")
    if root not in _ROOT_INDEX:
        raise KeyError(f"unknown chord root: {root!r}")
    template = np.zeros(12, dtype=float)
    base = _ROOT_INDEX[root]
    for offset in _QUALITY_INTERVALS[quality]:
        template[(base + offset) % 12] = 1.0
    return template


def label_index(root: str, quality: str) -> int:
    """Stable index in [0, 96]. Index 96 is the no-chord state."""
    if root == NO_CHORD:
        return len(ROOTS) * len(QUALITIES)
    return _ROOT_INDEX[root] * len(QUALITIES) + QUALITIES.index(quality)


def all_templates() -> np.ndarray[Any, np.dtype[np.float64]]:
    """Stack of 97 rows: 96 chord templates followed by the N template."""
    rows = []
    for root in ROOTS:
        for quality in QUALITIES:
            rows.append(template_for(root, quality))
    rows.append(template_for(NO_CHORD, ""))
    return np.stack(rows)


def chord_label(root: str, quality: str) -> str:
    if root == NO_CHORD:
        return "N"
    if quality == "maj":
        return root
    if quality == "min":
        return f"{root}m"
    if quality == "min7":
        return f"{root}m7"
    # 7, maj7, dim, sus4, aug — append directly
    return f"{root}{quality}"


def label_to_root_quality(label: str) -> tuple[str, str]:
    if label == "N":
        return NO_CHORD, ""
    # Roots can be 1 or 2 characters (with optional sharp).
    if len(label) >= 2 and label[1] == "#":
        root = label[:2]
        suffix = label[2:]
    else:
        root = label[:1]
        suffix = label[1:]
    if suffix == "":
        return root, "maj"
    if suffix == "m":
        return root, "min"
    if suffix == "m7":
        return root, "min7"
    if suffix in ("7", "maj7", "dim", "sus4", "aug"):
        return root, suffix
    raise ValueError(f"cannot parse chord label: {label!r}")
