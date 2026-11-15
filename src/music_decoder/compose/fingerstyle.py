"""One-guitar-part arrangement for the ``fingerstyle`` style.

This module starts with the hand shape: what a held chord shape lets the
melody reach (spec 5.5). Later tasks add the per-bar arrangement on top.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from music_decoder.compose.metrics import moment_playable
from music_decoder.compose.voicings import MAX_FRETTED_SPAN
from music_decoder.types import TabPosition, Tuning, VoicedChord

MELODY_FLOOR = 55  # G3: the lowest melody pitch in the fingerstyle part
THUMB_MARGIN = 3  # the melody sits at least a minor third over the thumb notes


@dataclass(frozen=True)
class HandShape:
    """A chord shape held by the fretting hand, and the melody it can reach.

    ``root`` is the lowest sounding string and ``alternate`` the next one up
    (``root`` itself when the shape sounds one string): the thumb notes.
    ``melody_strings`` are all strings above ``alternate``, sounded or not;
    ``fill_strings`` are those the shape sounds. ``window`` is the lowest and
    highest fret a fretted melody note may use; with a ``barre_fret`` the
    window starts at the barre and open melody notes are excluded. ``allowed``
    is the set of reachable melody pitches; use :meth:`positions` for where.
    """

    voicing: VoicedChord
    tuning: Tuning
    root: TabPosition
    alternate: TabPosition
    melody_strings: tuple[int, ...]
    fill_strings: tuple[int, ...]
    window: tuple[int, int]
    barre_fret: int | None
    allowed: frozenset[int]
    # (pitch, positions) pairs, derived from the fields above; not compared.
    _reach: tuple[tuple[int, tuple[TabPosition, ...]], ...] = field(
        default=(), repr=False, compare=False
    )

    def positions(self, pitch: int) -> tuple[TabPosition, ...]:
        """Reachable positions of ``pitch``, highest string first, or ``()``."""
        for p, found in self._reach:
            if p == pitch:
                return found
        return ()


def hand_shape(voicing: VoicedChord, tuning: Tuning) -> HandShape:
    """Build the :class:`HandShape` of ``voicing``. Spec 5.5.

    The shape is taken as given: faithfulness to the chord is not checked.
    """
    sounding = sorted((p for p in voicing.positions if p.fret >= 0), key=lambda p: p.string)
    if not sounding:
        raise ValueError("a hand shape needs at least one sounding string")
    root = sounding[0]
    alternate = sounding[1] if len(sounding) > 1 else root
    sounded = {p.string for p in sounding}
    melody_strings = tuple(range(alternate.string + 1, len(tuning.open_pitches)))
    fill_strings = tuple(s for s in melody_strings if s in sounded)

    thumb_frets = [p.fret for p in (root, alternate) if p.fret > 0]
    shape_frets = [p.fret for p in sounding if p.fret > 0]
    window: tuple[int, int]
    if thumb_frets:
        window = (max(1, max(thumb_frets) - MAX_FRETTED_SPAN), min(thumb_frets) + MAX_FRETTED_SPAN)
    elif shape_frets:
        window = (min(shape_frets), min(shape_frets) + MAX_FRETTED_SPAN)
    else:
        window = (1, 1 + MAX_FRETTED_SPAN)

    barre_fret: int | None = None
    if shape_frets and all(p.fret > 0 for p in sounding):
        lowest = min(shape_frets)
        if shape_frets.count(lowest) >= 2:
            barre_fret = lowest
            window = (max(window[0], lowest), window[1])

    open_pitches = tuning.open_pitches
    floor = max(
        MELODY_FLOOR, max(open_pitches[p.string] + p.fret for p in (root, alternate)) + THUMB_MARGIN
    )
    frets = (
        range(window[0], window[1] + 1)
        if barre_fret is not None
        else (0, *range(window[0], window[1] + 1))
    )
    reach: dict[int, list[TabPosition]] = {}
    for s in reversed(melody_strings):  # highest string first
        for f in frets:
            pitch = open_pitches[s] + f
            cand = TabPosition(string=s, fret=f)
            if pitch >= floor and moment_playable([root, alternate, cand]):
                reach.setdefault(pitch, []).append(cand)
    return HandShape(
        voicing=voicing,
        tuning=tuning,
        root=root,
        alternate=alternate,
        melody_strings=melody_strings,
        fill_strings=fill_strings,
        window=window,
        barre_fret=barre_fret,
        allowed=frozenset(reach),
        _reach=tuple((p, tuple(ps)) for p, ps in sorted(reach.items())),
    )
