"""One-guitar-part arrangement for the ``fingerstyle`` style.

The hand shape says what a held chord shape lets the melody reach (spec 5.5).
A body bar puts the melody on that shape's strings and picks the accompaniment
under it: a Travis thumb plus fills that keep the chord's identity (5.3, 5.4).
The final bar holds the last melody note over one rolled, possibly re-voiced
chord (5.3, final bar); :func:`arrange_fingerstyle` joins the bars.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass, field

from music_decoder.compose.arrangement import FINAL_STAGGER_S, THIN_MARGIN_SEMITONES, _thin_below
from music_decoder.compose.harmony import chord_pcs, essential_pcs
from music_decoder.compose.metrics import moment_playable
from music_decoder.compose.voicings import MAX_FRETTED_SPAN, ranked_shapes
from music_decoder.types import Note, TabbedNote, TabPosition, Tuning, VoicedChord

MELODY_FLOOR = 55  # G3: the lowest melody pitch in the fingerstyle part
THUMB_MARGIN = 3  # the melody sits at least a minor third over the thumb notes
THUMB_EIGHTHS = (0, 2, 4, 6)  # root on beats 1 and 3, alternate bass on 2 and 4
FILL_EIGHTHS = (3, 7)  # regular fills: the "and" of beats 2 and 4
IDENTITY_EIGHTHS = (3, 7, 1, 5, 0, 2, 4, 6)  # where a missing essential tone is tried
THUMB_VELOCITY = 72
FILL_VELOCITY = 64
FINAL_VELOCITY = 80  # v3's final stroke
FINAL_MAX_FRET = 12  # highest fret for a final note outside the hand's reach
_EPS_S = 1e-4  # notes must share more than this much time to overlap


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


@dataclass(frozen=True)
class BarPart:
    """One bar of the fingerstyle part, as tab (both tuples time-sorted).

    ``identity_met`` says whether the accompaniment alone sounds every
    essential tone of the bar's chord (spec 5.4).
    """

    melody_tab: tuple[TabbedNote, ...]
    accompaniment_tab: tuple[TabbedNote, ...]
    identity_met: bool


def _tab_note(
    hand: HandShape, position: TabPosition, start_s: float, end_s: float, velocity: int
) -> TabbedNote:
    pitch = hand.tuning.open_pitches[position.string] + position.fret
    note = Note(start_s=start_s, end_s=end_s, pitch=pitch, velocity=velocity, confidence=1.0)
    return TabbedNote(note=note, position=position)


def _overlaps(n: TabbedNote, t0: float, t1: float) -> bool:
    """``n`` sounds at some time in ``[t0, t1)``."""
    return n.note.start_s < t1 - _EPS_S and n.note.end_s > t0 + _EPS_S


def _overlapping(part: Iterable[TabbedNote], t0: float, t1: float) -> list[TabbedNote]:
    return [x for x in part if _overlaps(x, t0, t1)]


def _sounding_at(part: Iterable[TabbedNote], t: float) -> list[TabbedNote]:
    """The notes of ``part`` sounding at time ``t`` (those starting then included)."""
    return [x for x in part if x.note.start_s <= t + _EPS_S and x.note.end_s > t + _EPS_S]


def _starts_at(n: TabbedNote, t: float) -> bool:
    return abs(n.note.start_s - t) < _EPS_S


def _pitch_classes(part: Iterable[TabbedNote]) -> set[int]:
    return {x.note.pitch % 12 for x in part}


def _playable_at(part: Iterable[TabbedNote], t: float) -> bool:
    return moment_playable([x.position for x in _sounding_at(part, t)])


def _held_playable(n: TabbedNote, part: Sequence[TabbedNote]) -> bool:
    """Every moment while ``n`` sounds is playable; ``part`` includes ``n``."""
    onsets = {n.note.start_s} | {
        x.note.start_s for x in part if n.note.start_s < x.note.start_s < n.note.end_s - _EPS_S
    }
    return all(_playable_at(part, t) for t in onsets)


def _fill_options(hand: HandShape, start_s: float, end_s: float) -> list[TabbedNote]:
    """The shape's own note on each fill string, lowest string first."""
    shape = {p.string: p for p in hand.voicing.positions}
    return [_tab_note(hand, shape[s], start_s, end_s, FILL_VELOCITY) for s in hand.fill_strings]


def _fill_eligible(
    fill: TabbedNote,
    melody: Sequence[TabbedNote],
    thumb: Sequence[TabbedNote],
    fills: Sequence[TabbedNote],
    missing: Collection[int],
) -> bool:
    """Spec 5.3: may ``fill`` join the bar? ``missing`` holds the essential
    pitch classes the accompaniment (``thumb`` plus ``fills``) still lacks."""
    t0, p = fill.note.start_s, fill.note.pitch
    if any(_starts_at(f, t0) for f in fills):
        return False
    supplies_missing = p % 12 in missing
    for m in _overlapping(melody, t0, fill.note.end_s):
        q = m.note.pitch
        if m.position.string == fill.position.string:
            return False
        if p > q - THIN_MARGIN_SEMITONES and not (supplies_missing and p % 12 != q % 12):
            return False
        if _starts_at(m, t0) and p >= q:
            return False
    return _playable_at([*melody, *thumb, *fills, fill], t0)


def _melody_pitch_at(melody: Iterable[TabbedNote], t: float) -> int | None:
    sounding = _sounding_at(melody, t)
    return min(m.note.pitch for m in sounding) if sounding else None


def _choose_fill(
    eligible: Sequence[TabbedNote], melody_pitch: int | None, missing: Collection[int]
) -> TabbedNote | None:
    """A missing essential tone first, then the pitch nearest below the melody
    note sounding then (the lowest pitch when none is sounding)."""
    pool = [f for f in eligible if f.note.pitch % 12 in missing] or list(eligible)
    if not pool:
        return None
    below = [f for f in pool if melody_pitch is not None and f.note.pitch < melody_pitch]
    if below:
        return max(below, key=lambda f: f.note.pitch)
    return min(pool, key=lambda f: f.note.pitch)


def _tone_fill(
    tone: int,
    options: Iterable[TabbedNote],
    melody: Sequence[TabbedNote],
    thumb: Sequence[TabbedNote],
    fills: Sequence[TabbedNote],
    missing: Collection[int],
) -> TabbedNote | None:
    """The first eligible fill of pitch class ``tone`` among ``options``."""
    for f in options:
        if f.note.pitch % 12 == tone and _fill_eligible(f, melody, thumb, fills, missing):
            return f
    return None


def _fill_by_moving(
    tone: int,
    hand: HandShape,
    options: Iterable[TabbedNote],
    melody: Sequence[TabbedNote],
    thumb: Sequence[TabbedNote],
    fills: Sequence[TabbedNote],
    missing: Collection[int],
) -> tuple[list[TabbedNote], TabbedNote] | None:
    """Spec 5.4 step 3 at one eighth (``options`` are its fills).

    The one melody note sounding on a string that carries ``tone`` moves to
    another reachable string, if it stays playable for its whole length there
    and the fill of ``tone`` then becomes eligible. Returns the new melody and
    that fill, or ``None`` with nothing moved.
    """
    for fill in options:
        if fill.note.pitch % 12 != tone:
            continue
        on_string = [
            i
            for i, m in enumerate(melody)
            if m.position.string == fill.position.string
            and _overlaps(m, fill.note.start_s, fill.note.end_s)
        ]
        if len(on_string) != 1:
            continue
        i = on_string[0]
        for alt in hand.positions(melody[i].note.pitch):
            if alt.string == melody[i].position.string:
                continue
            moved = [*melody[:i], TabbedNote(note=melody[i].note, position=alt), *melody[i + 1 :]]
            if _held_playable(moved[i], [*moved, *thumb, *fills]) and _fill_eligible(
                fill, moved, thumb, fills, missing
            ):
                return moved, fill
    return None


def body_bar(
    hand: HandShape,
    melody: Sequence[Note],
    *,
    bar_start_s: float,
    beat_s: float,
    essential: frozenset[int],
) -> BarPart:
    """Arrange one body bar over ``hand``. Spec 5.3-5.5.

    ``melody`` holds the notes that start in this bar; each must be in
    ``hand.allowed``. ``essential`` holds the chord's essential pitch classes.
    """
    for m in melody:
        if m.pitch not in hand.allowed:
            raise ValueError(
                f"melody pitch {m.pitch} is not reachable over {hand.voicing.chord.to_label()}"
            )
    e = beat_s / 2

    def onset(k: int) -> float:
        return bar_start_s + k * e

    # Spec 5.5 rule 1: the highest reachable string. Rule 2 (least fret
    # movement) only breaks ties between positions on one string, and a pitch
    # has one position per string, so it never applies. Rule 3 holds by
    # construction: every melody string lies above both thumb strings.
    mel = [
        TabbedNote(note=m, position=hand.positions(m.pitch)[0])
        for m in sorted(melody, key=lambda m: m.start_s)
    ]

    # The thumb is never thinned: the reachable window keeps the melody at
    # least THUMB_MARGIN semitones above both thumb notes.
    thumb = [
        _tab_note(
            hand,
            hand.root if k % 4 == 0 else hand.alternate,
            onset(k),
            onset(k) + 0.95 * beat_s,
            THUMB_VELOCITY,
        )
        for k in THUMB_EIGHTHS
    ]
    fills: list[TabbedNote] = []

    def missing() -> set[int]:
        return set(essential) - _pitch_classes([*thumb, *fills])

    def options(k: int) -> list[TabbedNote]:
        return _fill_options(hand, onset(k), onset(k) + 0.95 * e)

    # Regular fills, where no melody note starts.
    for k in FILL_EIGHTHS:
        if any(_starts_at(m, onset(k)) for m in mel):
            continue
        miss = missing()
        eligible = [f for f in options(k) if _fill_eligible(f, mel, thumb, fills, miss)]
        choice = _choose_fill(eligible, _melody_pitch_at(mel, onset(k)), miss)
        if choice is not None:
            fills.append(choice)

    # Identity, step 2: a fill of each missing tone on any eighth that takes it.
    for tone in sorted(missing()):
        for k in IDENTITY_EIGHTHS:
            fill = _tone_fill(tone, options(k), mel, thumb, fills, missing())
            if fill is not None:
                fills.append(fill)
                break

    # Identity, step 3: move one melody note off the tone's string to make room.
    for tone in sorted(missing()):
        for k in IDENTITY_EIGHTHS:
            moved = _fill_by_moving(tone, hand, options(k), mel, thumb, fills, missing())
            if moved is not None:
                mel, fill = moved
                fills.append(fill)
                break

    accompaniment = sorted([*thumb, *fills], key=lambda x: (x.note.start_s, x.position.string))
    return BarPart(
        melody_tab=tuple(mel),
        accompaniment_tab=tuple(accompaniment),
        identity_met=set(essential) <= _pitch_classes(accompaniment),
    )


# ---- the final bar ---------------------------------------------------------


def _thumb_floor(hand: HandShape) -> int:
    """The lowest melody pitch over ``hand``'s thumb notes."""
    opens = hand.tuning.open_pitches
    thumb_high = max(opens[p.string] + p.fret for p in (hand.root, hand.alternate))
    return max(MELODY_FLOOR, thumb_high + THUMB_MARGIN)


def final_bar_pitches(hand: HandShape, cadence: int, essential: frozenset[int]) -> frozenset[int]:
    """The pitches the final melody note may take over ``hand``. Spec 5.3.

    The cadence candidates are the ``cadence``-class pitches of
    ``hand.allowed`` and, past them, the extension: ``cadence``-class pitches
    on a melody string at frets 0-12, at least a minor third over the thumb
    (and at least G3). A pitch works when :func:`final_bar` voices it without
    the fallback. The chosen pitches are the reachable ones that work, else
    the extension ones that work, else the reachable ones, else the
    extension. Returns ``hand.allowed`` with its ``cadence``-class pitches
    replaced by the chosen ones.
    """
    opens = hand.tuning.open_pitches
    floor = _thumb_floor(hand)
    reachable = {p for p in hand.allowed if p % 12 == cadence}
    extension = {
        opens[s] + f
        for s in hand.melody_strings
        for f in range(FINAL_MAX_FRET + 1)
        if (opens[s] + f) % 12 == cadence and opens[s] + f >= floor
    } - hand.allowed

    def works(pitch: int) -> bool:
        held = Note(start_s=0.0, end_s=1.0, pitch=pitch, velocity=FINAL_VELOCITY, confidence=1.0)
        _, fallback = final_bar(hand, held, bar_start_s=0.0, bar_s=1.0, essential=essential)
        return not fallback

    chosen = (
        {p for p in reachable if works(p)}
        or {p for p in extension if works(p)}
        or reachable
        or extension
    )
    return frozenset({p for p in hand.allowed if p % 12 != cadence} | chosen)


def final_positions(hand: HandShape, pitch: int) -> tuple[TabPosition, ...]:
    """Where the final note ``pitch`` may sit, highest string first.

    ``hand.positions(pitch)`` when the hand reaches it; otherwise every
    position of ``pitch`` on a melody string at frets 0-12.
    """
    reached = hand.positions(pitch)
    if reached:
        return reached
    opens = hand.tuning.open_pitches
    return tuple(
        TabPosition(string=s, fret=pitch - opens[s])
        for s in reversed(hand.melody_strings)
        if 0 <= pitch - opens[s] <= FINAL_MAX_FRET
    )


def _chord_under(
    hand: HandShape,
    frets: Sequence[int],
    fp: TabPosition,
    final: Note,
    essential: frozenset[int],
    bar_start_s: float,
    bar_s: float,
) -> list[TabPosition]:
    """``frets`` on the strings below ``fp``, thinned as v3 thins the final
    stroke against the held ``final`` note; lowest string first."""
    opens = hand.tuning.open_pitches
    under = [TabPosition(string=s, fret=f) for s, f in enumerate(frets) if f >= 0 and s < fp.string]
    by_pitch = sorted(under, key=lambda p: (opens[p.string] + p.fret, p.string))
    kept = Counter(
        _thin_below(
            [opens[p.string] + p.fret for p in by_pitch],
            essential,
            [final],
            bar_start_s,
            bar_start_s + bar_s,
        )
    )
    chosen: list[TabPosition] = []
    for p in by_pitch:
        pitch = opens[p.string] + p.fret
        if kept[pitch] > 0:
            kept[pitch] -= 1
            chosen.append(p)
    return sorted(chosen, key=lambda p: p.string)


def _final_chord_ok(
    hand: HandShape,
    chord: Sequence[TabPosition],
    fp: TabPosition,
    final: Note,
    essential: frozenset[int],
) -> bool:
    """Spec 5.3: the root in the bass, every essential tone (but the final
    note's own) under the final note, and playable with it."""
    if not chord:
        return False
    opens = hand.tuning.open_pitches
    pitches = [opens[p.string] + p.fret for p in chord]
    if min(pitches) % 12 != chord_pcs(hand.voicing.chord)[0]:
        return False
    under = {q % 12 for q in pitches if q < final.pitch}
    if not essential - {final.pitch % 12} <= under:
        return False
    return moment_playable([*chord, fp])


def final_bar(
    hand: HandShape,
    final: Note,
    *,
    bar_start_s: float,
    bar_s: float,
    essential: frozenset[int],
) -> tuple[BarPart, bool]:
    """The final bar: ``final`` held over one rolled chord. Spec 5.3.

    For each final position, the bar's own shape and then every other
    :func:`ranked_shapes` shape is tried, taking its notes on the strings
    below the final note's and thinning them as v3 does. The first chord with
    the root in the bass, every essential tone under ``final`` and a playable
    moment wins. Otherwise (the returned flag is ``True``) the bar's own shape
    below the first final position is played, thinned.
    """
    positions = final_positions(hand, final.pitch)
    if not positions:
        raise ValueError(
            f"final pitch {final.pitch} has no position over {hand.voicing.chord.to_label()}"
        )
    tuning = Tuning(hand.tuning.name, tuple(hand.tuning.open_pitches))
    own = [-1] * len(tuning.open_pitches)
    for p in hand.voicing.positions:
        own[p.string] = p.fret
    own_frets = tuple(own)
    others = [f for f in ranked_shapes(hand.voicing.chord, tuning) if f != own_frets]

    def under(frets: Sequence[int], fp: TabPosition) -> list[TabPosition]:
        return _chord_under(hand, frets, fp, final, essential, bar_start_s, bar_s)

    found: tuple[TabPosition, list[TabPosition]] | None = None
    for fp in positions:
        for frets in (own_frets, *others):
            chord = under(frets, fp)
            if _final_chord_ok(hand, chord, fp, final, essential):
                found = (fp, chord)
                break
        if found is not None:
            break
    fallback = found is None
    fp, chord = found if found is not None else (positions[0], under(own_frets, positions[0]))

    end_s = bar_start_s + bar_s
    accompaniment = tuple(
        _tab_note(hand, p, bar_start_s + k * FINAL_STAGGER_S, end_s, FINAL_VELOCITY)
        for k, p in enumerate(chord)
    )
    part = BarPart(
        melody_tab=(TabbedNote(note=final, position=fp),),
        accompaniment_tab=accompaniment,
        identity_met=set(essential) <= _pitch_classes(accompaniment),
    )
    return part, fallback


# ---- the whole part --------------------------------------------------------


@dataclass(frozen=True)
class FingerstylePart:
    """The one-guitar part, as tab (both tuples time-sorted).

    ``identity_misses`` lists the bars whose accompaniment lacks an essential
    tone; ``final_fallback`` says the final chord could not be re-voiced.
    """

    melody_tab: tuple[TabbedNote, ...]
    accompaniment_tab: tuple[TabbedNote, ...]
    identity_misses: tuple[int, ...]
    final_fallback: bool


def arrange_fingerstyle(
    voicings: Sequence[VoicedChord],
    melody: Sequence[Note],
    hands: Sequence[HandShape],
    *,
    bars_per_chord: int,
    tempo_bpm: float,
) -> FingerstylePart:
    """Arrange the whole fingerstyle part. Spec 5.3-5.5.

    ``hands[i]`` is the hand shape of ``voicings[i]``. Every bar but the last
    is a :func:`body_bar`; the last is a :func:`final_bar` and must hold
    exactly one melody note, starting with the bar.
    """
    if len(hands) != len(voicings):
        raise ValueError("one hand shape per voicing is needed")
    beat_s = 60.0 / tempo_bpm
    bar_s = 4.0 * beat_s
    n_bars = len(voicings) * bars_per_chord
    if n_bars == 0:
        raise ValueError("the part needs at least one bar")
    bars: list[list[Note]] = [[] for _ in range(n_bars)]
    for m in melody:
        index = int((m.start_s + _EPS_S) // bar_s)
        if not 0 <= index < n_bars:
            raise ValueError(f"melody note at {m.start_s:.3f}s lies outside the part")
        bars[index].append(m)
    if len(bars[-1]) != 1:
        raise ValueError(f"the final bar must hold exactly one melody note, not {len(bars[-1])}")
    if abs(bars[-1][0].start_s - (n_bars - 1) * bar_s) > 1e-6:
        raise ValueError("the final melody note must start with the final bar")

    melody_tab: list[TabbedNote] = []
    accompaniment: list[TabbedNote] = []
    misses: list[int] = []
    fallback = False
    for bar, notes in enumerate(bars):
        chord = bar // bars_per_chord
        hand = hands[chord]
        essential = essential_pcs(voicings[chord].chord)
        start = bar * bar_s
        if bar == n_bars - 1:
            part, fallback = final_bar(
                hand, notes[0], bar_start_s=start, bar_s=bar_s, essential=essential
            )
        else:
            part = body_bar(hand, notes, bar_start_s=start, beat_s=beat_s, essential=essential)
        melody_tab.extend(part.melody_tab)
        accompaniment.extend(part.accompaniment_tab)
        if not part.identity_met:
            misses.append(bar)
    accompaniment.sort(key=lambda x: (x.note.start_s, x.position.string))
    return FingerstylePart(
        melody_tab=tuple(melody_tab),
        accompaniment_tab=tuple(accompaniment),
        identity_misses=tuple(misses),
        final_fallback=fallback,
    )
