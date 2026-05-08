"""JAMS-style chord-label parsing, shared between the GuitarSet GT loader and
the madmom backend.

A JAMS chord label is one of:

- ``"N"`` — no chord
- ``"<root>:<quality>"`` — e.g. ``"C:maj"``, ``"A:min"``, ``"G:7"``, ``"D:maj7"``
- ``"<root>:<quality>(<extensions>)"`` — extensions ignored
- ``"<root>:<quality>/<bass>"`` — slash chord; bass ignored, quality applied
- ``"X"`` — unknown / unsupported (returned as None)

This v1 chord vocabulary is restricted to ``{maj, min, 7, maj7}``. Unsupported
qualities map to None unless they have a defensible downgrade
(e.g. ``min7 → min``). The downgrade table is conservative: only mappings that
preserve the dominant pitch-class profile are allowed.
"""

from __future__ import annotations

_QUALITY_MAP: dict[str, str] = {
    # Direct identity matches for our 8-quality vocabulary.
    "maj": "maj",
    "min": "min",
    "7": "7",
    "maj7": "maj7",
    "min7": "min7",  # Phase B-3: now first-class
    "dim": "dim",  # Phase B-3
    "sus4": "sus4",  # Phase B-3
    "aug": "aug",  # Phase B-3
    # Conservative downgrades — same root / third / fifth as our vocabulary.
    "minmaj7": "min7",  # close enough for chord recognition purposes
    "maj6": "maj",
    "min6": "min",
    "9": "7",
    "maj9": "maj7",
    "min9": "min7",  # promoted to its own quality
    "11": "7",
    "13": "7",
    "sus2": "sus4",  # both suspensions; sus4 covers it acceptably
    "dim7": "dim",  # close: same triad, different 7th
    "hdim7": "dim",  # half-diminished
    "+": "aug",
    # Triads only present in some annotation systems.
    "": "maj",  # bare root means major
    # Unsupported qualities (5+, +9, alt, etc.) deliberately omitted;
    # the parser returns None for those so the caller can fall back to N or
    # skip the segment entirely.
}


def parse_jams_chord_label(label: str) -> tuple[str, str] | None:
    """Parse a JAMS-style chord label into ``(root, quality)``.

    Returns ``("N", "")`` for the no-chord state, ``(root, quality)`` for any
    chord whose quality is in our supported vocabulary (after optional
    downgrade), or ``None`` for unsupported qualities or malformed labels.
    """
    if not isinstance(label, str):
        return None
    label = label.strip()
    if not label or label == "N" or label == "X":
        return ("N", "") if label == "N" else None

    # Strip any extension specifier and slash bass for our purposes.
    if "(" in label:
        label = label.split("(", 1)[0]
    if "/" in label:
        label = label.split("/", 1)[0]

    if ":" in label:
        root, quality = label.split(":", 1)
    else:
        # Bare root → assume major.
        root, quality = label, "maj"
    root = root.strip()
    quality = quality.strip()

    # Validate root in pitch-class set.
    if not _is_valid_root(root):
        return None
    mapped = _QUALITY_MAP.get(quality)
    if mapped is None:
        return None
    return (root, mapped)


def _is_valid_root(root: str) -> bool:
    valid_roots = {
        "C",
        "C#",
        "Db",
        "D",
        "D#",
        "Eb",
        "E",
        "F",
        "F#",
        "Gb",
        "G",
        "G#",
        "Ab",
        "A",
        "A#",
        "Bb",
        "B",
    }
    return root in valid_roots
