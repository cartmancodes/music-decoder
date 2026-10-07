"""Dependency-free GuitarSet (JAMS) reader for the accuracy benchmark.

JAMS files are plain JSON. GuitarSet stores six ``note_midi`` annotations
(``data_source`` "0".."5" = low E .. high e), two ``chord`` annotations
(first = instructed lead-sheet, second = performed), one ``key_mode`` and one
``beat_position``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from music_decoder.paths import project_root

DEV_PLAYERS = ("00", "01", "02", "03", "04")
TEST_PLAYERS = ("05",)
DEFAULT_ROOT = project_root() / "tests" / "fixtures" / "guitarset"

_FLAT_TO_SHARP = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}
_AUDIO_SUFFIX = {
    "audio_mono-mic": "_mic.wav",
    "audio_mono-pickup_mix": "_mix.wav",
}


@dataclass(frozen=True)
class GTNote:
    start_s: float
    end_s: float
    pitch: int
    string: int


@dataclass(frozen=True)
class GuitarSetTrack:
    track_id: str
    player: str
    audio_path: Path
    notes: tuple[GTNote, ...]
    chords: tuple[tuple[float, float, str], ...]
    key: tuple[str, str]
    beats: np.ndarray[Any, np.dtype[np.float64]]
    downbeats: np.ndarray[Any, np.dtype[np.float64]]


def _by_ns(jams: dict[str, Any], ns: str) -> list[dict[str, Any]]:
    return [a for a in jams["annotations"] if a["namespace"] == ns]


def parse_jams(jams: dict[str, Any], *, track_id: str, audio_path: Path) -> GuitarSetTrack:
    notes: list[GTNote] = []
    for ann in _by_ns(jams, "note_midi"):
        string = int(ann["annotation_metadata"]["data_source"])
        for d in ann["data"]:
            start = float(d["time"])
            end = start + float(d["duration"])
            notes.append(GTNote(start, end, round(float(d["value"])), string))
    notes.sort(key=lambda n: (n.start_s, n.pitch))

    chord_anns = _by_ns(jams, "chord")
    chords = tuple(
        (float(d["time"]), float(d["time"]) + float(d["duration"]), str(d["value"]))
        for d in (chord_anns[0]["data"] if chord_anns else [])
    )

    key_anns = _by_ns(jams, "key_mode")
    tonic, mode = "C", "major"
    if key_anns:
        tonic, mode = str(key_anns[0]["data"][0]["value"]).split(":")
    tonic = _FLAT_TO_SHARP.get(tonic, tonic)

    beat_anns = _by_ns(jams, "beat_position")
    rows = beat_anns[0]["data"] if beat_anns else []
    beats = np.array([float(d["time"]) for d in rows], dtype=float)
    downbeats = np.array(
        [float(d["time"]) for d in rows if int(d["value"]["position"]) == 1], dtype=float
    )
    return GuitarSetTrack(
        track_id=track_id,
        player=track_id.split("_", 1)[0],
        audio_path=audio_path,
        notes=tuple(notes),
        chords=chords,
        key=(tonic, mode),
        beats=beats,
        downbeats=downbeats,
    )


def iter_tracks(
    root: Path = DEFAULT_ROOT,
    *,
    players: Iterable[str],
    audio: str = "audio_mono-mic",
) -> Iterator[GuitarSetTrack]:
    """Yield tracks for the given players whose audio file exists, sorted by id."""
    wanted = set(players)
    for jams_path in sorted((root / "annotation").glob("*.jams")):
        track_id = jams_path.stem
        if track_id.split("_", 1)[0] not in wanted:
            continue
        audio_path = root / audio / f"{track_id}{_AUDIO_SUFFIX[audio]}"
        if not audio_path.exists():
            continue
        jams = json.loads(jams_path.read_text())
        yield parse_jams(jams, track_id=track_id, audio_path=audio_path)
