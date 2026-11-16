"""Read POP909 songs and turn them into the composer's melody statistics (v6).

The dataset is the hierarchical-structure release of POP909: one folder per
song holding ``melody.txt`` (``pitch duration`` in sixteenths, pitch 0 a
rest), ``finalized_chord.txt`` (``name [pcs] root beats``, chords back to back
from the start of the melody), ``analyzed_key.txt`` (one or more keys such as
``Gb:maj``), ``tempo.txt`` and ``human_label1.txt`` (phrases such as
``i4A4B8``; capital letters are melodic phrases). Only the statistics leave
the dataset.

The view is the eighth grid of spec 5.1: a bar is 16 sixteenths and is *pure*
when every onset in it is on an even sixteenth; its pattern is the tuple of
``sixteenth // 2``. Bands are the 25th, 50th and 75th percentiles computed
with ``numpy.percentile``'s default (linear interpolation, the same as
``statistics.quantiles(method="inclusive")``).

A song that lists several keys (a modulation, about 120 of the 909 songs)
gives each labelled segment the listed key that covers most of its notes;
ties keep the previous segment's key, else the first listed one.
"""

from __future__ import annotations

import datetime
import re
from bisect import bisect_right
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from music_decoder.compose.tune_metrics import (
    FEATURES,
    PENTATONIC,
    GridNote,
    Mode,
    grid_features,
)
from music_decoder.compose.tune_stats import (
    METRIC_CLASSES,
    MODES,
    ROLES,
    START,
    MetricClass,
    Pattern,
    Role,
    metric_class,
    pattern_key,
)

Key = tuple[int, Mode]  # tonic pitch class, mode

SIXTEENTHS_PER_BAR = 16
WINDOW_BARS = 8
MIN_WINDOW_NOTES = 12
MAX_MOVE = 7
MIN_PILOT_SONGS = 100
PILOT_WINDOW_SLOTS = 64  # 8 bars of eighths
SOURCES = (
    "POP909 (Wang et al., ISMIR 2020), hierarchical-structure release "
    "(Dai, Zhang and Dannenberg, 2020): melody, chords, keys, tempo, human_label1; "
    "licence: MIT",
)

SCALES: dict[Mode, tuple[int, ...]] = {
    "major": (0, 2, 4, 5, 7, 9, 11),
    "minor": (0, 2, 3, 5, 7, 8, 10),
}
_LETTERS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_LABEL = re.compile(r"([A-Za-z])(\d+)")
_CHORD = re.compile(r"(\S+)\s+\[([^\]]*)\]\s+(.*)")


@dataclass(frozen=True)
class SongNote:
    onset: int  # sixteenths from the start of the melody
    duration: int  # sixteenths
    pitch: int  # MIDI


@dataclass(frozen=True)
class Chord:
    start: int  # sixteenths
    end: int
    pcs: frozenset[int]


@dataclass(frozen=True)
class Song:
    name: str
    keys: tuple[Key, ...]  # as listed in analyzed_key.txt
    tempo: int  # quarter notes per minute
    phrases: tuple[tuple[str, int, int], ...]  # (letter, first bar, bars), every label
    notes: tuple[SongNote, ...]  # pitched notes in order
    chords: tuple[Chord, ...]
    bar_keys: tuple[Key, ...]  # key of each bar

    @property
    def tonic_pc(self) -> int:
        return self.keys[0][0]

    @property
    def mode(self) -> Mode:
        return self.keys[0][1]

    def key_at(self, bar: int) -> Key:
        return self.bar_keys[min(bar, len(self.bar_keys) - 1)]

    def chord_at(self, sixteenth: int) -> frozenset[int]:
        """Pitch classes sounding at ``sixteenth``; empty beyond the last chord."""
        i = bisect_right([c.start for c in self.chords], sixteenth) - 1
        if i < 0 or sixteenth >= self.chords[i].end:
            return frozenset()
        return self.chords[i].pcs


def _parse_key(text: str) -> Key:
    name, _, kind = text.strip().partition(":")
    if not name or name[0] not in _LETTERS:
        raise ValueError(f"bad key {text!r}")
    pc = _LETTERS[name[0]] + name[1:].count("#") - name[1:].count("b")
    if name[1:].strip("#b"):
        raise ValueError(f"bad key {text!r}")
    if kind.startswith("maj"):
        return pc % 12, "major"
    if kind.startswith("min"):
        return pc % 12, "minor"
    raise ValueError(f"bad key {text!r}")


def _parse_melody(text: str) -> list[SongNote]:
    notes: list[SongNote] = []
    t = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        pitch_s, dur_s = line.split()
        pitch, dur = int(pitch_s), int(dur_s)
        if dur <= 0:
            raise ValueError(f"bad duration {line!r}")
        if pitch > 0:
            notes.append(SongNote(onset=t, duration=dur, pitch=pitch))
        t += dur
    return notes


def _parse_chords(text: str) -> list[Chord]:
    chords: list[Chord] = []
    t = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        m = _CHORD.fullmatch(line.strip())
        if m is None:
            raise ValueError(f"bad chord {line!r}")
        pcs = frozenset(int(p) % 12 for p in m.group(2).split(",") if p.strip())
        length = round(float(m.group(3).split()[-1]) * 4)  # beats -> sixteenths
        chords.append(Chord(start=t, end=t + length, pcs=pcs))
        t += length
    return chords


def _in_scale(pitch: int, key: Key) -> bool:
    return (pitch - key[0]) % 12 in SCALES[key[1]]


def _bar_keys(
    keys: Sequence[Key], phrases: Sequence[tuple[str, int, int]], notes: Sequence[SongNote]
) -> tuple[Key, ...]:
    n_bars = max(
        sum(n for _, _, n in phrases),
        (notes[-1].onset // SIXTEENTHS_PER_BAR + 1) if notes else 0,
        1,
    )
    if len(set(keys)) == 1:
        return (keys[0],) * n_bars
    segments = [(start, start + n) for _, start, n in phrases]
    labelled = segments[-1][1] if segments else 0
    if labelled < n_bars:
        segments.append((labelled, n_bars))
    out: list[Key] = []
    current = keys[0]
    for lo, hi in segments:
        pitches = [n.pitch for n in notes if lo <= n.onset // SIXTEENTHS_PER_BAR < hi]
        scores = [sum(_in_scale(p, k) for p in pitches) for k in keys]
        best = max(scores)
        if scores[keys.index(current)] < best:
            current = keys[scores.index(best)]
        out.extend([current] * (hi - lo))
    return tuple(out)


def read_song(folder: Path) -> Song | None:
    """One POP909 song folder, or ``None`` when a file is missing or unparsable."""
    try:
        keys = tuple(
            _parse_key(line)
            for line in (folder / "analyzed_key.txt").read_text().splitlines()
            if line.strip()
        )
        tempo = int((folder / "tempo.txt").read_text().split()[0])
        label = (folder / "human_label1.txt").read_text().strip()
        if not keys or tempo <= 0 or not re.fullmatch(r"(?:[A-Za-z]\d+)+", label):
            return None
        phrases: list[tuple[str, int, int]] = []
        bar = 0
        for letter, n in _LABEL.findall(label):
            phrases.append((letter, bar, int(n)))
            bar += int(n)
        notes = _parse_melody((folder / "melody.txt").read_text())
        chords = _parse_chords((folder / "finalized_chord.txt").read_text())
    except (OSError, ValueError, IndexError):
        return None
    return Song(
        name=folder.name,
        keys=keys,
        tempo=tempo,
        phrases=tuple(phrases),
        notes=tuple(notes),
        chords=tuple(chords),
        bar_keys=_bar_keys(keys, phrases, notes),
    )


# --- statistics -----------------------------------------------------------


def _capital(song: Song) -> list[tuple[str, int, int]]:
    return [p for p in song.phrases if p[0].isupper() and p[2] > 0]


def _roles(song: Song) -> dict[int, Role]:
    roles: dict[int, Role] = {}
    for _, start, n in _capital(song):
        for bar in range(start, start + n):
            roles[bar] = "body"
        roles[start] = "open"
        roles[start + n - 1] = "cadence"
    return roles


def _bar_onsets(song: Song) -> dict[int, list[int]]:
    onsets: dict[int, list[int]] = defaultdict(list)
    for note in song.notes:
        onsets[note.onset // SIXTEENTHS_PER_BAR].append(note.onset % SIXTEENTHS_PER_BAR)
    return onsets


def _pattern(onsets: Sequence[int]) -> Pattern | None:
    """The bar's pattern, or ``None`` when it is empty or not pure."""
    if not onsets or any(o % 2 for o in onsets):
        return None
    return tuple(sorted({o // 2 for o in onsets}))


def _count_rhythm(
    song: Song,
    rhythm: Mapping[Role, Counter[Pattern]],
    pairs: Mapping[Role, defaultdict[Pattern, Counter[Pattern]]],
) -> None:
    onsets = _bar_onsets(song)
    for bar, role in _roles(song).items():
        this = _pattern(onsets.get(bar, []))
        if this is None:
            continue
        rhythm[role][this] += 1
        previous = _pattern(onsets.get(bar - 1, []))
        if previous is not None:
            pairs[role][previous][this] += 1


def _scale_index(pitch: int, key: Key) -> int | None:
    rel = pitch - key[0]
    try:
        return SCALES[key[1]].index(rel % 12) + 7 * (rel // 12)
    except ValueError:
        return None


def _count_moves(
    song: Song,
    steps: Mapping[MetricClass, defaultdict[int, Counter[int]]],
    base: Mapping[MetricClass, Counter[int]],
) -> None:
    """Scale-step moves along chains (controller ruling R1).

    A chain's first note has no move; the second note's move is counted with
    previous move ``START``; later moves with the real previous move. A note
    off the scale or off the eighth grid, or after a rest of a bar or more,
    ends the chain (an off-scale or off-grid note belongs to no chain).
    """
    prev_index: int | None = None
    prev_move = START
    prev_key: Key | None = None
    prev_end: int | None = None
    for note in song.notes:
        key = song.key_at(note.onset // SIXTEENTHS_PER_BAR)
        rest = note.onset - prev_end if prev_end is not None else 0
        prev_end = note.onset + note.duration
        index = _scale_index(note.pitch, key) if note.onset % 2 == 0 else None
        if index is None:
            prev_index = None
            continue
        if prev_index is None or rest >= SIXTEENTHS_PER_BAR or key != prev_key:
            prev_index, prev_move, prev_key = index, START, key
            continue
        move = max(-MAX_MOVE, min(MAX_MOVE, index - prev_index))
        cls = metric_class(note.onset // 2)
        steps[cls][prev_move][move] += 1
        base[cls][move] += 1
        prev_index, prev_move = index, move


def _count_tones(
    song: Song,
    chord_tone: Mapping[MetricClass, list[int]],
    degrees: Mapping[Mode, list[int]],
    finals: Mapping[Mode, list[int]],
) -> None:
    for note in song.notes:
        tonic, mode = song.key_at(note.onset // SIXTEENTHS_PER_BAR)
        degrees[mode][(note.pitch - tonic) % 12] += 1
        if note.onset % 2 == 0:
            hits = chord_tone[metric_class(note.onset // 2)]
            hits[0] += note.pitch % 12 in song.chord_at(note.onset)
            hits[1] += 1
    for _, start, n in _capital(song):
        lo, hi = start * SIXTEENTHS_PER_BAR, (start + n) * SIXTEENTHS_PER_BAR
        inside = [note for note in song.notes if lo <= note.onset < hi]
        if inside:
            last = inside[-1]
            tonic, mode = song.key_at(last.onset // SIXTEENTHS_PER_BAR)
            finals[mode][(last.pitch - tonic) % 12] += 1


def _window_starts(song: Song) -> list[int]:
    """Bars that start an 8-bar phrase or two back-to-back 4-bar phrases (paired greedily)."""
    phrases = _capital(song)
    starts: list[int] = []
    i = 0
    while i < len(phrases):
        _, start, n = phrases[i]
        if n == WINDOW_BARS:
            starts.append(start)
        elif n == 4 and i + 1 < len(phrases):
            _, next_start, next_n = phrases[i + 1]
            if next_n == 4 and next_start == start + 4:
                starts.append(start)
                i += 2
                continue
        i += 1
    return starts


def _windows(song: Song) -> list[tuple[Mode, dict[str, float]]]:
    out: list[tuple[Mode, dict[str, float]]] = []
    onsets = _bar_onsets(song)
    for start in _window_starts(song):
        bars = range(start, start + WINDOW_BARS)
        if any(onsets.get(b) and _pattern(onsets[b]) is None for b in bars):
            continue
        if len({song.key_at(b) for b in bars}) != 1:
            continue
        lo = start * SIXTEENTHS_PER_BAR
        hi = lo + WINDOW_BARS * SIXTEENTHS_PER_BAR
        notes = [
            GridNote(slot=(n.onset - lo) // 2, pitch=n.pitch)
            for n in song.notes
            if lo <= n.onset < hi
        ]
        if len(notes) < MIN_WINDOW_NOTES:
            continue
        tonic, mode = song.key_at(start)
        features = grid_features(
            notes,
            n_bars=WINDOW_BARS,
            tonic_pc=tonic,
            mode=mode,
            chord_tones=[song.chord_at(lo + 2 * s) for s in range(WINDOW_BARS * 8)],
            seconds_per_bar=240.0 / song.tempo,
        )
        out.append((mode, features))
    return out


def _band(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"low": 0.0, "mid": 0.0, "high": 0.0}
    low, mid, high = np.percentile(np.asarray(values, dtype=float), [25, 50, 75])
    return {"low": float(low), "mid": float(mid), "high": float(high)}


def _shares(counts: Sequence[int]) -> list[float]:
    total = sum(counts)
    return [c / total if total else 0.0 for c in counts]


def _median(values: Sequence[float]) -> float:
    return float(np.median(values)) if values else 0.0


def _pilot_line_range(slots_pitches: Sequence[tuple[int, int]]) -> float | None:
    """Mean highest-minus-lowest pitch over the 8-bar windows holding two or more notes."""
    windows: dict[int, list[int]] = defaultdict(list)
    for slot, pitch in slots_pitches:
        windows[slot // PILOT_WINDOW_SLOTS].append(pitch)
    spans = [max(p) - min(p) for p in windows.values() if len(p) >= 2]
    return sum(spans) / len(spans) if spans else None


def pilot_stats(lines: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Pitch statistics of vocal lines extracted from the pilot's own songs.

    Each line is ``{"tonic_pc", "mode", "notes": [[eighth_slot, pitch], ...]}``.
    ``interval_classes`` (the share of repeats, steps of 1-2 semitones, thirds of
    3-4 and leaps of 5 or more among consecutive notes) and ``pentatonic`` (per
    mode) are medians over songs; ``degrees`` is the mean over songs of each
    song's share per semitone above the tonic; ``range`` is the 25th, 50th and 75th
    percentile over songs of the song's mean 8-bar-window range (as ``range`` in
    ``tune_metrics``).
    """
    classes: dict[str, list[float]] = {c: [] for c in ("repeat", "step", "third", "leap")}
    pentatonic: dict[Mode, list[float]] = {m: [] for m in MODES}
    shares: dict[Mode, list[list[float]]] = {m: [] for m in MODES}
    ranges: list[float] = []
    n_songs = 0
    for line in lines:
        n_songs += 1
        mode: Mode = "minor" if line["mode"] == "minor" else "major"
        tonic = int(line["tonic_pc"])
        notes = sorted((int(s), int(p)) for s, p in line["notes"])
        if not notes:
            continue
        degrees = [(p - tonic) % 12 for _, p in notes]
        counts = [degrees.count(d) for d in range(12)]
        shares[mode].append(_shares(counts))
        pent = PENTATONIC[mode]
        pentatonic[mode].append(sum(d in pent for d in degrees) / len(degrees))
        sizes = [abs(b[1] - a[1]) for a, b in pairwise(notes)]
        if sizes:
            classes["repeat"].append(sum(s == 0 for s in sizes) / len(sizes))
            classes["step"].append(sum(1 <= s <= 2 for s in sizes) / len(sizes))
            classes["third"].append(sum(3 <= s <= 4 for s in sizes) / len(sizes))
            classes["leap"].append(sum(s >= 5 for s in sizes) / len(sizes))
        span = _pilot_line_range(notes)
        if span is not None:
            ranges.append(span)
    quartiles = [float(q) for q in np.percentile(ranges, [25, 50, 75])] if ranges else [0.0] * 3
    return {
        "songs": n_songs,
        "interval_classes": {c: _median(v) for c, v in classes.items()},
        "pentatonic": {m: _median(pentatonic[m]) for m in MODES},
        "degrees": {
            m: [float(x) for x in np.mean(shares[m], axis=0)] if shares[m] else [0.0] * 12
            for m in MODES
        },
        "range": quartiles,
    }


def _merge_pilot(out: dict[str, Any], pilot: Mapping[str, Any]) -> None:
    """Widen the pitch bands to the pilot's values and average the degree shares."""
    values: dict[Mode, dict[str, tuple[float, float]]] = {}
    low_q, _, high_q = (float(x) for x in pilot["range"])
    for mode in MODES:
        values[mode] = {c: (float(v), float(v)) for c, v in pilot["interval_classes"].items()}
        values[mode]["pentatonic"] = (float(pilot["pentatonic"][mode]),) * 2
        values[mode]["range"] = (low_q, high_q)
        for feature, (lo, hi) in values[mode].items():
            band = out["bands"][mode].get(feature)
            if band is not None:
                band["low"] = min(band["low"], lo)
                band["high"] = max(band["high"], hi)
        theirs = [float(x) for x in pilot["degrees"][mode]]
        if sum(theirs) > 0:
            mean = [(a + b) / 2 for a, b in zip(out["degrees"][mode], theirs, strict=True)]
            out["degrees"][mode] = _shares(mean)
    out["meta"]["sources"].append(
        f"own YouTube pilot, vocal lines of {int(pilot['songs'])} songs; "
        "licence: own YouTube pilot, statistics only"
    )


def build_stats(
    songs: Iterable[Song],
    pilot: Mapping[str, Any] | None = None,
    *,
    min_pattern_count: int = 5,
) -> dict[str, Any]:
    """The statistics as a JSON-ready dict (see ``tune_stats`` for the typed view).

    Patterns seen fewer than ``min_pattern_count`` times in all of ``rhythm``
    are removed from ``rhythm`` and ``rhythm_pairs``. ``pilot`` (``pilot_stats``
    output) is stored under ``"pilot"`` as given; when it covers at least
    ``MIN_PILOT_SONGS`` songs it also widens the pitch bands and is averaged
    into ``degrees`` (``_merge_pilot``). Rhythm and move tables stay POP909's.
    """
    rhythm: dict[Role, Counter[Pattern]] = {r: Counter() for r in ROLES}
    pairs: dict[Role, defaultdict[Pattern, Counter[Pattern]]] = {
        r: defaultdict(Counter) for r in ROLES
    }
    steps: dict[MetricClass, defaultdict[int, Counter[int]]] = {
        c: defaultdict(Counter) for c in METRIC_CLASSES
    }
    base: dict[MetricClass, Counter[int]] = {c: Counter() for c in METRIC_CLASSES}
    chord_tone: dict[MetricClass, list[int]] = {c: [0, 0] for c in METRIC_CLASSES}
    degrees: dict[Mode, list[int]] = {m: [0] * 12 for m in MODES}
    finals: dict[Mode, list[int]] = {m: [0] * 12 for m in MODES}
    windows: dict[Mode, list[dict[str, float]]] = {m: [] for m in MODES}
    n_songs = 0
    for song in songs:
        n_songs += 1
        _count_rhythm(song, rhythm, pairs)
        _count_moves(song, steps, base)
        _count_tones(song, chord_tone, degrees, finals)
        for mode, features in _windows(song):
            windows[mode].append(features)

    total: Counter[Pattern] = Counter()
    for counts in rhythm.values():
        total.update(counts)
    kept = {p for p, n in total.items() if n >= min_pattern_count}

    def moves(counts: Counter[int]) -> dict[str, int]:
        return {str(m): counts[m] for m in sorted(counts)}

    all_windows = [w for ws in windows.values() for w in ws]
    out: dict[str, Any] = {
        "rhythm": {
            r: {pattern_key(p): rhythm[r][p] for p in sorted(rhythm[r]) if p in kept} for r in ROLES
        },
        "rhythm_pairs": {
            r: {
                pattern_key(prev): {pattern_key(p): row[p] for p in sorted(row) if p in kept}
                for prev, row in sorted(pairs[r].items())
                if prev in kept and kept.intersection(row)
            }
            for r in ROLES
        },
        "steps": {
            c: {str(prev): moves(row) for prev, row in sorted(steps[c].items())}
            for c in METRIC_CLASSES
        },
        "steps_base": {c: moves(base[c]) for c in METRIC_CLASSES},
        "chord_tone": {
            c: chord_tone[c][0] / chord_tone[c][1] if chord_tone[c][1] else 0.0
            for c in METRIC_CLASSES
        },
        "degrees": {m: _shares(degrees[m]) for m in MODES},
        "final_degrees": {m: _shares(finals[m]) for m in MODES},
        "notes_per_second": _band([w["notes_per_second"] for w in all_windows]),
        "bands": {
            m: {
                f: _band([w[f] for w in windows[m] if f in w])
                for f in FEATURES
                if any(f in w for w in windows[m])
            }
            for m in MODES
        },
        "meta": {
            "sources": list(SOURCES),
            "songs": n_songs,
            "windows": {m: len(windows[m]) for m in MODES},
            "built": datetime.date.today().isoformat(),
        },
    }
    if pilot is not None:
        out["pilot"] = dict(pilot)
        if int(pilot.get("songs", 0)) >= MIN_PILOT_SONGS:
            _merge_pilot(out, pilot)
    return out
