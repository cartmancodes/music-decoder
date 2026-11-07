"""ChordSymbol + Tuning → the faithful, playable VoicedChord the composer plays."""

from __future__ import annotations

import functools
import itertools
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

FAULT_TOO_FEW = "too few strings"
FAULT_FOREIGN = "foreign tone"
FAULT_BASS = "root not in bass"
FAULT_MISSING = "missing tone"

_MIN_STRINGS = 3
_FIFTH = 7  # a four-tone chord may omit root + 7

MAX_FRETTED_SPAN = 3  # a four-fret hand frame; open strings do not count
MAX_FINGERS = 4
SEARCH_MAX_BASE_FRET = 9
SEARCH_MIN_STRINGS = 4
_MAX_FRET = 22


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


def _is_playable(frets: Sequence[int]) -> bool:
    """Spec 5.2 playability: a fretted span of at most ``MAX_FRETTED_SPAN``
    (open strings excluded) and at most ``MAX_FINGERS`` fretting fingers."""
    sounding = [f for f in frets if f >= 0]
    if not sounding or max(sounding) > _MAX_FRET:
        return False
    fretted = [f for f in sounding if f > 0]
    if fretted and max(fretted) - min(fretted) > MAX_FRETTED_SPAN:
        return False
    return fingers_needed(frets) <= MAX_FINGERS


def _retune(frets: Sequence[int], tuning: Tuning, chord: ChordSymbol) -> tuple[int, ...] | None:
    """Move an EADGBE shape to ``tuning`` keeping every fretted pitch.

    ``None`` when a string would need a fret below 0 (a tuning above
    standard). Muted strings are un-muted in two passes: first every one
    whose open note is the root, then every one whose open note is a chord
    tone above the lowest pitch sounding after the first pass. An added open
    string so never puts a non-root tone in the bass, and Drop D's D keeps
    its open D2-A2-D3 drone.
    """
    if tuning.open_pitches == STANDARD_EADGBE.open_pitches:
        return tuple(frets)
    std = STANDARD_EADGBE.open_pitches
    opens = tuning.open_pitches
    shifted = [f if f < 0 else f + std[s] - opens[s] for s, f in enumerate(frets)]
    if any(f < 0 for f, orig in zip(shifted, frets, strict=True) if orig >= 0):
        return None
    pcs = harmony.chord_pcs(chord)
    out = [0 if f < 0 and opens[s] % 12 == pcs[0] else f for s, f in enumerate(shifted)]
    lowest = min((opens[s] + f for s, f in enumerate(out) if f >= 0), default=None)
    if lowest is not None:
        for s, f in enumerate(out):
            if f < 0 and opens[s] % 12 in pcs and opens[s] > lowest:
                out[s] = 0
    return tuple(out)


@functools.cache  # lru_cache(maxsize=None); both arguments are frozen dataclasses
def search_voicing(chord: ChordSymbol, tuning: Tuning) -> tuple[int, ...] | None:
    """The best faithful, playable shape for ``chord`` in ``tuning`` (spec 5.2).

    For each base fret the hand covers the open strings plus frets
    ``base .. base+3``. The root is the bass on one of the three lowest
    strings, and every string from there up to the highest sounding one
    plays a chord tone (at least ``SEARCH_MIN_STRINGS``, no gaps). The
    smallest (omitted tones, lowest fret, -strings, frets) wins; ``None``
    when nothing qualifies.
    """
    pcs = harmony.chord_pcs(chord)
    tones = set(pcs)
    opens = tuning.open_pitches
    n = len(opens)
    best: tuple[tuple[int, int, int, tuple[int, ...]], tuple[int, ...]] | None = None
    for base in range(SEARCH_MAX_BASE_FRET + 1):
        window = sorted({0, *range(max(base, 1), base + MAX_FRETTED_SPAN + 1)})
        options = [[f for f in window if (opens[s] + f) % 12 in tones] for s in range(n)]
        for bass in range(min(3, n)):
            bass_frets = [f for f in options[bass] if (opens[bass] + f) % 12 == pcs[0]]
            for top in range(bass + SEARCH_MIN_STRINGS - 1, n):
                for combo in itertools.product(bass_frets, *options[bass + 1 : top + 1]):
                    frets = (-1,) * bass + combo + (-1,) * (n - top - 1)
                    if not _is_playable(frets) or shape_faults(frets, tuning, chord):
                        continue
                    fretted = [f for f in combo if f > 0]
                    key = (
                        len(tones - {(opens[s] + f) % 12 for s, f in enumerate(frets) if f >= 0}),
                        min(fretted, default=0),
                        -len(combo),
                        frets,
                    )
                    if best is None or key < best[0]:
                        best = (key, frets)
    return None if best is None else best[1]


def _root_fallback(chord: ChordSymbol, tuning: Tuning) -> VoicedChord:
    # v2 fallback: the root pitch class on every string that reaches it at frets 0-12.
    root_pc = VALID_TONICS.index(chord.root)
    root_midi = tuning.open_pitches[0] + ((root_pc - tuning.open_pitches[0]) % 12)
    return VoicedChord(
        chord=chord,
        positions=tuple(
            TabPosition(string=i, fret=root_midi - tuning.open_pitches[i])
            if 0 <= root_midi - tuning.open_pitches[i] <= 12
            else TabPosition(string=i, fret=-1)
            for i in range(len(tuning.open_pitches))
        ),
    )


def _voiced(chord: ChordSymbol, frets: Sequence[int]) -> VoicedChord:
    return VoicedChord(
        chord=chord, positions=tuple(TabPosition(string=s, fret=f) for s, f in enumerate(frets))
    )


def voicings_for(chord: ChordSymbol, tuning: Tuning) -> list[VoicedChord]:
    """The voicing the composer plays for ``chord`` in ``tuning`` (one element).

    The retuned library shape when it is playable and faithful, else the
    shape search, else the v2 root fallback (only for exotic custom tunings).
    """
    canonicals = canonical.canonical_voicings_for(chord)  # in EADGBE
    if not canonicals:
        raise InvalidProgressionError(f"No canonical voicing for {chord.to_label()}")

    # A Tuning built with a list must still hash for the cached search and
    # compare equal to EADGBE in _retune.
    tuning = Tuning(tuning.name, tuple(tuning.open_pitches))
    for c in canonicals:
        frets = _retune([p.fret for p in c.positions], tuning, chord)
        if frets is not None and _is_playable(frets) and not shape_faults(frets, tuning, chord):
            return [_voiced(chord, frets)]
    searched = search_voicing(chord, tuning)
    if searched is not None:
        return [_voiced(chord, searched)]
    return [_root_fallback(chord, tuning)]
