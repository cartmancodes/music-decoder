"""Late fusion of three independent key cues.

Each source yields 24 scores (index ``pitch_class * 2 + mode``, mode 0 =
major, 1 = minor). Scores are z-normalized per source and summed:

- **cnn** — log-probabilities from madmom's CNN key classifier;
- **profiles** — Krumhansl-Kessler and Temperley correlations with the
  mean chroma (two cues);
- **chords** — duration-weighted diatonic fit of the recognized chord
  progression (tonic/subdominant/dominant weighted highest).

On GuitarSet dev (300 tracks) the fused estimate scores 0.685 MIREX-weighted
vs 0.56-0.66 for any single cue, because the cues make different mistakes
(profiles confuse relative keys, the CNN fifths, chord fits modes).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np

from music_decoder.key.ks import correlate_against_profiles
from music_decoder.key.profiles import KEYS
from music_decoder.types import ChordSegment, KeyEstimate

_Arr = np.ndarray[Any, np.dtype[np.float64]]
_PC = {k: i for i, k in enumerate(KEYS)}
# Accept flat spellings too (all current chord backends emit sharps).
_PC.update({"Db": 1, "Eb": 3, "Gb": 6, "Ab": 8, "Bb": 10})
_MODES: tuple[Literal["major", "minor"], Literal["major", "minor"]] = ("major", "minor")
# madmom.features.key.KEY_LABELS tonic order (A .. G#), sharp spelling.
_MADMOM_TONICS = ("A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#")

# (interval above tonic, triad quality, weight)
_MAJOR_DEGREES = (
    (0, "maj", 1.0),
    (5, "maj", 0.8),
    (7, "maj", 0.8),
    (9, "min", 0.6),
    (2, "min", 0.5),
    (4, "min", 0.4),
)
_MINOR_DEGREES = (
    (0, "min", 1.0),
    (5, "min", 0.8),
    (7, "min", 0.6),
    (7, "maj", 0.6),
    (3, "maj", 0.6),
    (8, "maj", 0.5),
    (10, "maj", 0.5),
)
_TRIAD = {
    "maj": "maj",
    "7": "maj",
    "maj7": "maj",
    "aug": "maj",
    "sus4": "maj",
    "min": "min",
    "min7": "min",
    "dim": "min",
}


def key_index(tonic: str, mode: str) -> int:
    return _PC[tonic] * 2 + (1 if mode == "minor" else 0)


def cnn_scores(probs: np.ndarray[Any, np.dtype[Any]]) -> _Arr:
    """madmom's 24 key probabilities → log-scores in our index order."""
    p = np.asarray(probs, dtype=float).reshape(-1)
    out = np.empty(24)
    for i, prob in enumerate(p):
        out[key_index(_MADMOM_TONICS[i % 12], _MODES[i // 12])] = np.log(max(prob, 1e-9))
    return out


def profile_scores(
    pitch_class_distribution: np.ndarray[Any, np.dtype[Any]],
    profile: Literal["krumhansl_kessler", "temperley"] = "krumhansl_kessler",
) -> _Arr:
    out = np.zeros(24)
    for k in correlate_against_profiles(pitch_class_distribution, profile=profile):
        out[key_index(k.tonic, k.mode)] = k.correlation
    return out


def chord_scores(segments: Sequence[ChordSegment]) -> _Arr | None:
    """Diatonic fit of a chord progression per key; ``None`` if it has no chords."""
    chords = [s for s in segments if s.root != "N" and s.root in _PC]
    total = sum(s.end_s - s.start_s for s in chords)
    if not chords or total <= 0:
        return None
    out = np.zeros(24)
    for tonic in range(12):
        for mode, degrees in enumerate((_MAJOR_DEGREES, _MINOR_DEGREES)):
            fit = 0.0
            for s in chords:
                interval = (_PC[s.root] - tonic) % 12
                triad = _TRIAD.get(s.quality, "maj")
                weight = max(
                    (w for d, q, w in degrees if d == interval and q == triad), default=0.0
                )
                fit += (s.end_s - s.start_s) * weight
            out[tonic * 2 + mode] = fit / total
    return out


def fuse(sources: Sequence[_Arr]) -> KeyEstimate:
    """Sum of per-source z-scores; flat (uninformative) sources are ignored.

    ``correlation`` is the winner's mean z-score and ``margin`` its lead over
    the runner-up (also per source), so both stay comparable across source
    counts. Raises ``ValueError`` if no source is informative.
    """
    used = [s for s in sources if float(np.std(s)) > 1e-9]
    if not used:
        raise ValueError("no informative key source")
    total = sum(((s - s.mean()) / s.std() for s in used), np.zeros(24))
    order = np.argsort(total)[::-1]
    best = int(order[0])
    n = len(used)
    return KeyEstimate(
        tonic=KEYS[best // 2],
        mode=_MODES[best % 2],
        profile="fusion",
        correlation=float(total[best] / n),
        margin=float((total[best] - total[int(order[1])]) / n),
    )
