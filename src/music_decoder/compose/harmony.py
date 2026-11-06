"""Pitch-class helpers shared by the melody generator, the accompaniment and
the musicality metrics: scale and chord pitch classes, each chord's
identity-defining ("essential") tones, and the scale adapted to a chord."""

from __future__ import annotations

from music_decoder.types import VALID_TONICS, ChordSymbol, Scale

# Pitch-class sets relative to the tonic
_MAJOR_PCS = (0, 2, 4, 5, 7, 9, 11)
_MINOR_PCS = (0, 2, 3, 5, 7, 8, 10)  # natural minor

# Chord quality -> intervals above the root
CHORD_INTERVALS: dict[str, tuple[int, ...]] = {
    "maj": (0, 4, 7),
    "min": (0, 3, 7),
    "maj7": (0, 4, 7, 11),
    "min7": (0, 3, 7, 10),
    "7": (0, 4, 7, 10),
    "dim": (0, 3, 6),
    "sus4": (0, 5, 7),
    "aug": (0, 4, 8),
}

# The tones that give each quality its identity (third, seventh, altered fifth,
# suspended fourth); a sounding chord missing one of these is ambiguous.
ESSENTIAL_INTERVALS: dict[str, tuple[int, ...]] = {
    "maj": (4,),
    "min": (3,),
    "maj7": (4, 11),
    "min7": (3, 10),
    "7": (4, 10),
    "dim": (3, 6),
    "sus4": (5,),
    "aug": (4, 8),
}

# Scale-degree steps of each chord member above the root: chords are stacks of
# thirds (root, third, fifth, seventh); sus4 replaces the third with a fourth.
_DEGREE_STEPS: dict[str, tuple[int, ...]] = {
    "maj": (0, 2, 4),
    "min": (0, 2, 4),
    "dim": (0, 2, 4),
    "aug": (0, 2, 4),
    "maj7": (0, 2, 4, 6),
    "min7": (0, 2, 4, 6),
    "7": (0, 2, 4, 6),
    "sus4": (0, 3, 4),
}


def tonic_pc(scale: Scale) -> int:
    return VALID_TONICS.index(scale.tonic)


def leading_tone_pc(scale: Scale) -> int:
    """The pitch class a semitone below the tonic."""
    return (tonic_pc(scale) + 11) % 12


def scale_pcs(scale: Scale) -> tuple[int, ...]:
    base = _MAJOR_PCS if scale.mode == "major" else _MINOR_PCS
    return tuple((p + tonic_pc(scale)) % 12 for p in base)


def _root_pc(chord: ChordSymbol) -> int:
    return VALID_TONICS.index(chord.root)


def chord_pcs(chord: ChordSymbol) -> tuple[int, ...]:
    return tuple((i + _root_pc(chord)) % 12 for i in CHORD_INTERVALS[chord.quality])


def essential_pcs(chord: ChordSymbol) -> frozenset[int]:
    return frozenset((i + _root_pc(chord)) % 12 for i in ESSENTIAL_INTERVALS[chord.quality])


def _spelling(scale: Scale, chord: ChordSymbol) -> list[tuple[int, int]]:
    """Each chord tone paired with the pitch class of the scale degree it
    occupies, spelling the chord as a stack of thirds over the scale.

    An in-scale root fixes the spelling. An out-of-scale root sits between two
    scale degrees (B-flat between A and B in C major); the spelling that alters
    the chord's tones by the fewest semitones in total wins, and a tie goes to
    the upper degree (flat-side spelling, as in bVII, bVI, bII).
    """
    degrees = scale_pcs(scale)
    root = _root_pc(chord)
    if root in degrees:
        starts = [degrees.index(root)]
    else:
        upper = [i for i, d in enumerate(degrees) if (d - root) % 12 == 1]
        lower = [i for i, d in enumerate(degrees) if (root - d) % 12 == 1]
        starts = upper + lower
    steps = _DEGREE_STEPS[chord.quality]
    best: tuple[int, int, list[tuple[int, int]]] | None = None
    for rank, start in enumerate(starts):
        pairs = [
            (pc, degrees[(start + step) % len(degrees)])
            for pc, step in zip(chord_pcs(chord), steps, strict=True)
        ]
        cost = sum(min((pc - d) % 12, (d - pc) % 12) for pc, d in pairs)
        if best is None or (cost, rank) < best[:2]:
            best = (cost, rank, pairs)
    assert best is not None
    return best[2]


def chord_scale_pcs(scale: Scale, chord: ChordSymbol) -> frozenset[int]:
    """The scale with each out-of-scale chord tone replacing the scale degree
    it occupies in the chord's stack-of-thirds spelling (see ``_spelling``).

    Every other degree is kept, including the tonic unless the chord itself
    alters it. E7 in A minor gives A harmonic minor (G -> G#); D7 in C major
    replaces F with F# and keeps G; B7 in C major replaces D and F with D# and
    F# and keeps G (the third of Em, where it resolves); A major in C replaces
    C with C#. Diatonic chords leave the scale unchanged.
    """
    scale_set = set(scale_pcs(scale))
    chord_set = set(chord_pcs(chord))
    pool = set(scale_set)
    for pc, degree in _spelling(scale, chord):
        if pc in scale_set:
            continue
        if degree not in chord_set:
            pool.discard(degree)
        pool.add(pc)
    return frozenset(pool)
