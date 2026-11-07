"""ChordSymbol + Tuning → ranked playable VoicedChord candidates."""

from __future__ import annotations

from collections.abc import Sequence

from music_decoder.chords import voicings as canonical
from music_decoder.compose import harmony
from music_decoder.errors import InvalidProgressionError
from music_decoder.tabs.tuning import STANDARD_EADGBE
from music_decoder.types import (
    VALID_TONICS,
    ChordSymbol,
    TabPosition,
    Tuning,
    VoicedChord,
)

# Chord-quality → chord-tone pitch classes (root = 0)
_CHORD_PCS: dict[str, tuple[int, ...]] = {
    "maj": (0, 4, 7),
    "min": (0, 3, 7),
    "maj7": (0, 4, 7, 11),
    "min7": (0, 3, 7, 10),
    "7": (0, 4, 7, 10),
    "dim": (0, 3, 6),
    "sus4": (0, 5, 7),
    "aug": (0, 4, 8),
}

FAULT_TOO_FEW = "too few strings"
FAULT_FOREIGN = "foreign tone"
FAULT_BASS = "root not in bass"
FAULT_MISSING = "missing tone"

_MIN_STRINGS = 3
_FIFTH = 7  # a four-tone chord may omit root + 7


def shape_faults(frets: Sequence[int], tuning: Tuning, chord: ChordSymbol) -> tuple[str, ...]:
    """Spec 3.1 faithfulness defects of a shape, in a fixed order (empty = faithful).

    ``frets`` has one entry per string, ``-1`` muted. A four-tone chord may
    omit its perfect fifth.
    """
    pitches = [tuning.open_pitches[s] + f for s, f in enumerate(frets) if f >= 0]
    pcs = harmony.chord_pcs(chord)
    sounded = {p % 12 for p in pitches}
    required = set(pcs)
    if len(pcs) == 4:
        required.discard((pcs[0] + _FIFTH) % 12)
    faults: list[str] = []
    if len(pitches) < _MIN_STRINGS:
        faults.append(FAULT_TOO_FEW)
    if sounded - set(pcs):
        faults.append(FAULT_FOREIGN)
    if pitches and min(pitches) % 12 != pcs[0]:
        faults.append(FAULT_BASS)
    if required - sounded:
        faults.append(FAULT_MISSING)
    return tuple(faults)


def fingers_needed(frets: Sequence[int]) -> int:
    """Fretting fingers under spec 3.1's straight-barre model.

    For each distinct fret ``f > 0``, every maximal run of adjacent strings all
    fretted at ``f`` or higher costs one finger if it holds a note at exactly
    ``f``. An open or muted string ends a run.
    """
    total = 0
    for f in {x for x in frets if x > 0}:
        run_has_f = False
        in_run = False
        for x in (*frets, 0):  # trailing open string closes the last run
            if x >= f:
                in_run = True
                run_has_f = run_has_f or x == f
            elif in_run:
                total += run_has_f
                in_run = run_has_f = False
    return total


def _chord_tone_pcs(chord: ChordSymbol) -> set[int]:
    rel = _CHORD_PCS.get(chord.quality, ())
    root_pc = VALID_TONICS.index(chord.root)
    return {(r + root_pc) % 12 for r in rel}


def _is_playable(positions: Sequence[TabPosition]) -> bool:
    frets = [p.fret for p in positions if p.fret >= 0]
    if not frets:
        return False
    if min(frets) < 0:
        return False
    if max(frets) > 22:
        return False
    if max(frets) - min(frets) > 5:
        return False
    return True


def _retune(
    canonical_positions: Sequence[TabPosition],
    from_tuning: Tuning,
    to_tuning: Tuning,
    chord_pcs: set[int] | None = None,
) -> tuple[TabPosition, ...]:
    if from_tuning.open_pitches == to_tuning.open_pitches:
        return tuple(canonical_positions)
    out: list[TabPosition] = []
    for s, p in enumerate(canonical_positions):
        if p.fret < 0:
            # Muted in the canonical voicing. If the new open string happens to
            # be a chord tone (e.g. low E -> low D in Drop D for a D chord),
            # un-mute it as an open-string addition to the voicing.
            if chord_pcs is not None and (to_tuning.open_pitches[s] % 12) in chord_pcs:
                out.append(TabPosition(string=s, fret=0))
            else:
                out.append(p)
            continue
        delta = from_tuning.open_pitches[s] - to_tuning.open_pitches[s]
        new_fret = p.fret + delta
        out.append(TabPosition(string=s, fret=new_fret))
    return tuple(out)


def voicings_for(chord: ChordSymbol, tuning: Tuning) -> list[VoicedChord]:
    """Return 1-3 playable voicings for `chord` in `tuning`, lowest first."""
    canonicals = canonical.canonical_voicings_for(chord)  # in EADGBE
    if not canonicals:
        raise InvalidProgressionError(f"No canonical voicing for {chord.to_label()}")

    chord_pcs = _chord_tone_pcs(chord)
    candidates: list[VoicedChord] = []
    for c in canonicals:
        positions = _retune(c.positions, STANDARD_EADGBE, tuning, chord_pcs)
        if _is_playable(positions):
            candidates.append(VoicedChord(chord=chord, positions=positions))

    if not candidates:
        # Fallback: root + 5th + octave on the three lowest strings.
        root_pc = VALID_TONICS.index(chord.root)
        root_midi = tuning.open_pitches[0] + ((root_pc - tuning.open_pitches[0]) % 12)
        candidates.append(
            VoicedChord(
                chord=chord,
                positions=tuple(
                    TabPosition(string=i, fret=root_midi - tuning.open_pitches[i])
                    if 0 <= root_midi - tuning.open_pitches[i] <= 12
                    else TabPosition(string=i, fret=-1)
                    for i in range(len(tuning.open_pitches))
                ),
            )
        )

    candidates.sort(
        key=lambda v: (
            max((p.fret for p in v.positions if p.fret >= 0), default=99),
            sum((p.fret for p in v.positions if p.fret >= 0), 0),
        )
    )
    return candidates[:3]
