"""Phrase model tokens: one token per sixteenth slot, plus ten conditioning streams.

Token 0 means no onset; tokens 1..R are onsets at relative pitch ``pitch_lo..pitch_hi``,
where relative pitch is measured from the anchor: the key's tonic at or below a middle
pitch (the song's median melody pitch in training, ``tessitura_center`` when writing,
the melody's own median when scoring).
"""

from __future__ import annotations

import bisect
import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from music_decoder.compose.harmony import CHORD_INTERVALS, chord_pcs, tonic_pc
from music_decoder.compose.rhythm import form_labels, split_phrases
from music_decoder.compose.tune_corpus import Song
from music_decoder.types import ChordSymbol, Note, Scale

SLOTS_PER_BAR = 16
NO_ONSET = 0
CONTEXT_SLOTS = 512
MAX_PHRASE_BARS = 15
TEMPO_EDGES = (80.0, 100.0, 120.0)
QUALITIES = ("maj", "min", "maj7", "min7", "7", "dim", "sus4", "aug", "other", "none")
CONDITION_KEYS = (
    "mode",
    "root",
    "quality",
    "next_root",
    "next_quality",
    "slot",
    "since",
    "until",
    "restate",
    "tempo",
)
CONDITION_SIZES: dict[str, int] = {
    "mode": 2,
    "root": 12,
    "quality": len(QUALITIES),
    "next_root": 12,
    "next_quality": len(QUALITIES),
    "slot": SLOTS_PER_BAR,
    "since": MAX_PHRASE_BARS + 1,
    "until": MAX_PHRASE_BARS + 1,
    "restate": 2,
    "tempo": len(TEMPO_EDGES) + 1,
}
IntArray = npt.NDArray[np.int64]
_NONE = QUALITIES.index("none")
_OTHER = QUALITIES.index("other")

_BY_SET: dict[frozenset[int], list[tuple[int, int]]] = {}
for _root in range(12):
    for _q in QUALITIES[:8]:
        _BY_SET.setdefault(frozenset((_root + i) % 12 for i in CHORD_INTERVALS[_q]), []).append(
            (_root, QUALITIES.index(_q))
        )


@dataclass(frozen=True)
class Encoded:
    tokens: IntArray  # one per slot
    cond: dict[str, IntArray]  # one per slot per key
    out_of_range_bars: frozenset[int]


def anchor_pitch(tonic_pc: int, middle: int) -> int:
    """The pitch with pitch class ``tonic_pc`` at or below ``middle``."""
    return middle - ((middle - tonic_pc) % 12)


def chord_code(pcs: frozenset[int], tonic_pc: int) -> tuple[int, int]:
    """(root relative to the tonic, quality index) by exact pitch-class-set match."""
    if not pcs:
        return 0, _NONE
    hits = _BY_SET.get(frozenset(pcs), [])
    if len(hits) != 1:
        return 0, _OTHER
    root, quality = hits[0]
    return (root - tonic_pc) % 12, quality


def tempo_bucket(tempo_bpm: float) -> int:
    return bisect.bisect_right(TEMPO_EDGES, tempo_bpm)


def _phrase_streams(
    phrases: Sequence[tuple[int, int, bool]], n_bars: int
) -> tuple[IntArray, IntArray, IntArray]:
    """since/until/restate per bar from (first bar, bars, is restatement) triples."""
    since = np.full(n_bars, MAX_PHRASE_BARS, dtype=np.int64)
    until = np.full(n_bars, MAX_PHRASE_BARS, dtype=np.int64)
    restate = np.zeros(n_bars, dtype=np.int64)
    for first, bars, again in phrases:
        for b in range(max(first, 0), min(first + bars, n_bars)):
            since[b] = min(b - first, MAX_PHRASE_BARS)
            until[b] = min(first + bars - 1 - b, MAX_PHRASE_BARS)
            restate[b] = int(again)
    return since, until, restate


def _streams(
    n_bars: int,
    mode_of: Callable[[int], int],
    code_of: Callable[[int, int, int], tuple[int, int]],
    phrases: Sequence[tuple[int, int, bool]],
    tempo_bpm: float,
) -> dict[str, IntArray]:
    since, until, restate = _phrase_streams(phrases, n_bars)
    cond = {k: np.zeros(n_bars * SLOTS_PER_BAR, dtype=np.int64) for k in CONDITION_KEYS}
    tb = tempo_bucket(tempo_bpm)
    for bar in range(n_bars):
        sl = slice(bar * SLOTS_PER_BAR, (bar + 1) * SLOTS_PER_BAR)
        cond["mode"][sl] = mode_of(bar)
        nxt = code_of(bar + 1, 0, bar)  # the next chord, in this bar's key
        for s in range(SLOTS_PER_BAR):
            t = bar * SLOTS_PER_BAR + s
            cond["root"][t], cond["quality"][t] = code_of(bar, s, bar)
            cond["next_root"][t], cond["next_quality"][t] = nxt
        cond["slot"][sl] = np.arange(SLOTS_PER_BAR)
        cond["since"][sl], cond["until"][sl], cond["restate"][sl] = (
            since[bar],
            until[bar],
            restate[bar],
        )
        cond["tempo"][sl] = tb
    return cond


def piece_conditioning(
    *, scale: Scale, progression: Sequence[ChordSymbol], bars_per_chord: int, tempo_bpm: float
) -> dict[str, IntArray]:
    """Conditioning streams for a piece the composer is writing or scoring."""
    n_bars = len(progression) * bars_per_chord
    tonic = tonic_pc(scale)
    codes = [
        chord_code(frozenset(chord_pcs(progression[b // bars_per_chord])), tonic)
        for b in range(n_bars)
    ]
    phrases, first = [], 0
    for length, label in zip(split_phrases(n_bars), form_labels(n_bars), strict=True):
        phrases.append((first, length, label == "A'"))
        first += length
    return _streams(
        n_bars,
        lambda _b: int(scale.mode == "minor"),
        lambda b, _s, _k: codes[b] if b < n_bars else (0, _NONE),
        phrases,
        tempo_bpm,
    )


def _tokens(
    slots: Sequence[int], rel: Sequence[int], n_slots: int, pitch_lo: int, pitch_hi: int
) -> tuple[IntArray, frozenset[int]]:
    tokens = np.zeros(n_slots, dtype=np.int64)
    bad: set[int] = set()
    for slot, r in zip(slots, rel, strict=True):
        if not 0 <= slot < n_slots:
            continue
        if pitch_lo <= r <= pitch_hi:
            tokens[slot] = r - pitch_lo + 1
        else:
            bad.add(slot // SLOTS_PER_BAR)
    return tokens, frozenset(bad)


def encode_piece(
    melody: Sequence[Note],
    *,
    scale: Scale,
    progression: Sequence[ChordSymbol],
    bars_per_chord: int,
    tempo_bpm: float,
    middle: int,
    pitch_lo: int,
    pitch_hi: int,
) -> Encoded:
    cond = piece_conditioning(
        scale=scale, progression=progression, bars_per_chord=bars_per_chord, tempo_bpm=tempo_bpm
    )
    anchor = anchor_pitch(tonic_pc(scale), middle)
    sixteenth = 15.0 / tempo_bpm
    slots = [round(n.start_s / sixteenth) for n in melody]
    tokens, bad = _tokens(
        slots, [n.pitch - anchor for n in melody], len(cond["slot"]), pitch_lo, pitch_hi
    )
    return Encoded(tokens=tokens, cond=cond, out_of_range_bars=bad)


def _song_bars(song: Song) -> int:
    last = max(n.onset + n.duration for n in song.notes)
    labelled = max((first + bars for _, first, bars in song.phrases), default=0)
    return max(labelled, -(-last // SLOTS_PER_BAR))


def song_relative_pitches(song: Song) -> IntArray:
    """Each note's pitch relative to its bar key's anchor (middle = the song's median pitch)."""
    middle = int(np.median([n.pitch for n in song.notes]))
    return np.array(
        [
            n.pitch - anchor_pitch(song.key_at(n.onset // SLOTS_PER_BAR)[0], middle)
            for n in song.notes
        ],
        dtype=np.int64,
    )


def phrase_starts(song: Song) -> list[int]:
    """First bars of capital-letter phrases."""
    return sorted({first for letter, first, _ in song.phrases if letter.isupper()})


def encode_song(song: Song, pitch_lo: int, pitch_hi: int) -> Encoded:
    n_bars = _song_bars(song)
    seen: set[str] = set()
    phrases = []
    for letter, first, bars in song.phrases:
        phrases.append((first, bars, letter in seen))
        seen.add(letter)

    def code_of(bar: int, slot: int, key_bar: int) -> tuple[int, int]:
        return chord_code(song.chord_at(bar * SLOTS_PER_BAR + slot), song.key_at(key_bar)[0])

    cond = _streams(
        n_bars, lambda b: int(song.key_at(b)[1] == "minor"), code_of, phrases, song.tempo
    )
    tokens, bad = _tokens(
        [n.onset for n in song.notes],
        song_relative_pitches(song).tolist(),
        n_bars * SLOTS_PER_BAR,
        pitch_lo,
        pitch_hi,
    )
    return Encoded(tokens=tokens, cond=cond, out_of_range_bars=bad)


def eighth_only(enc: Encoded, first_bar: int, bars: int) -> bool:
    """True when the span lies inside the piece and has no onset on an odd sixteenth."""
    a, e = first_bar * SLOTS_PER_BAR, (first_bar + bars) * SLOTS_PER_BAR
    if first_bar < 0 or e > len(enc.tokens):
        return False
    return not enc.tokens[a:e][1::2].any()


def figure_hashes(
    onsets: Sequence[int], pitches: Sequence[int], length: int
) -> npt.NDArray[np.uint32]:
    """One uint32 per run of ``length`` notes, over its intervals and onset gaps in sixteenths."""
    ons, ps = np.asarray(onsets, dtype=np.int64), np.asarray(pitches, dtype=np.int64)
    out = []
    for i in range(len(ps) - length + 1):
        key = (
            np.concatenate([np.diff(ps[i : i + length]), np.diff(ons[i : i + length])])
            .astype("<i8")
            .tobytes()
        )
        out.append(int.from_bytes(hashlib.blake2b(key, digest_size=8).digest()[:4], "little"))
    return np.array(out, dtype=np.uint32)
