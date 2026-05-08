"""ChordSymbol + Tuning → ranked playable VoicedChord candidates."""

from __future__ import annotations

from collections.abc import Sequence

from music_decoder.chords import voicings as canonical
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
