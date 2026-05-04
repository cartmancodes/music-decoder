"""Canonical fingering for each of the 48 chord types.

Each voicing is a 6-tuple, low-to-high (E, A, D, G, B, e), where -1 = muted,
0 = open, and positive ints are fret numbers. Open-position voicings are used
where standard; barre voicings (with the bass on E or A string) cover the rest.

These are display-only -- the runtime tab assigner is unrelated.
"""
from __future__ import annotations

# Open-position naturals + standard barre voicings for the rest.
VOICINGS: dict[str, tuple[int, int, int, int, int, int]] = {
    # ---- Major chords
    "C":   (-1, 3, 2, 0, 1, 0),
    "C#":  (-1, 4, 3, 1, 2, 1),  # C# barre at 4
    "D":   (-1, -1, 0, 2, 3, 2),
    "D#":  (-1, 6, 5, 3, 4, 3),  # Eb barre
    "E":   (0, 2, 2, 1, 0, 0),
    "F":   (1, 3, 3, 2, 1, 1),
    "F#":  (2, 4, 4, 3, 2, 2),
    "G":   (3, 2, 0, 0, 0, 3),
    "G#":  (4, 6, 6, 5, 4, 4),
    "A":   (-1, 0, 2, 2, 2, 0),
    "A#":  (-1, 1, 3, 3, 3, 1),  # Bb
    "B":   (-1, 2, 4, 4, 4, 2),

    # ---- Minor chords
    "Cm":   (-1, 3, 5, 5, 4, 3),
    "C#m":  (-1, 4, 6, 6, 5, 4),
    "Dm":   (-1, -1, 0, 2, 3, 1),
    "D#m":  (-1, 6, 8, 8, 7, 6),
    "Em":   (0, 2, 2, 0, 0, 0),
    "Fm":   (1, 3, 3, 1, 1, 1),
    "F#m":  (2, 4, 4, 2, 2, 2),
    "Gm":   (3, 5, 5, 3, 3, 3),
    "G#m":  (4, 6, 6, 4, 4, 4),
    "Am":   (-1, 0, 2, 2, 1, 0),
    "A#m":  (-1, 1, 3, 3, 2, 1),
    "Bm":   (-1, 2, 4, 4, 3, 2),

    # ---- Dominant 7th chords
    "C7":   (-1, 3, 2, 3, 1, 0),
    "C#7":  (-1, 4, 3, 4, 2, 1),
    "D7":   (-1, -1, 0, 2, 1, 2),
    "D#7":  (-1, 6, 5, 6, 4, 3),
    "E7":   (0, 2, 0, 1, 0, 0),
    "F7":   (1, 3, 1, 2, 1, 1),
    "F#7":  (2, 4, 2, 3, 2, 2),
    "G7":   (3, 2, 0, 0, 0, 1),
    "G#7":  (4, 6, 4, 5, 4, 4),
    "A7":   (-1, 0, 2, 0, 2, 0),
    "A#7":  (-1, 1, 3, 1, 3, 1),
    "B7":   (-1, 2, 1, 2, 0, 2),

    # ---- Major 7th chords
    "Cmaj7":   (-1, 3, 2, 0, 0, 0),
    "C#maj7":  (-1, 4, 3, 1, 1, 1),
    "Dmaj7":   (-1, -1, 0, 2, 2, 2),
    "D#maj7":  (-1, 6, 5, 3, 3, 3),
    "Emaj7":   (0, 2, 1, 1, 0, 0),
    "Fmaj7":   (-1, 3, 3, 2, 1, 0),
    "F#maj7":  (2, 4, 3, 3, 2, 2),
    "Gmaj7":   (3, 2, 0, 0, 0, 2),
    "G#maj7":  (4, 6, 5, 5, 4, 4),
    "Amaj7":   (-1, 0, 2, 1, 2, 0),
    "A#maj7":  (-1, 1, 3, 2, 3, 1),
    "Bmaj7":   (-1, 2, 4, 3, 4, 2),

    # ---- No-chord
    "N":     (-1, -1, -1, -1, -1, -1),
}


def voicing_for(root: str, quality: str) -> tuple[int, int, int, int, int, int]:
    from .templates import chord_label

    label = chord_label(root, quality)
    if label not in VOICINGS:
        raise KeyError(f"no voicing for {label!r}")
    return VOICINGS[label]
