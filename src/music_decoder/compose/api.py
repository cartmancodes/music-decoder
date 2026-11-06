"""Public synchronous compose() entry point."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from music_decoder.compose.arrangement import _accompaniment_notes, build_midi
from music_decoder.compose.melody import generate_melody
from music_decoder.compose.voicings import voicings_for
from music_decoder.errors import InvalidProgressionError, InvalidScaleError
from music_decoder.synth import render_wav
from music_decoder.tabs.render import render_ascii_tab
from music_decoder.tabs.tuning import STANDARD_EADGBE
from music_decoder.types import (
    ChordSymbol,
    Composition,
    Note,
    Scale,
    TabbedNote,
    TabPosition,
    Tuning,
    VoicedChord,
)


@dataclass(frozen=True)
class Arrangement:
    """Everything compose() writes, computed without touching the filesystem."""

    melody: tuple[Note, ...]
    voicings: tuple[VoicedChord, ...]
    accompaniment: tuple[Note, ...]
    melody_tab: tuple[TabbedNote, ...]


def _validate(scale: Scale, progression: Sequence[ChordSymbol]) -> None:
    if scale.mode not in ("major", "minor"):
        raise InvalidScaleError(f"Mode must be major or minor, got {scale.mode!r}")
    if not progression:
        raise InvalidProgressionError("Progression must contain at least one chord.")


def _lowest_fret_position(pitch: int, tuning: Tuning) -> TabPosition | None:
    """The most playable single position: the string giving the lowest fret."""
    best: tuple[int, int] | None = None  # (fret, string)
    for s, open_pitch in enumerate(tuning.open_pitches):
        fret = pitch - open_pitch
        if 0 <= fret <= 22 and (best is None or fret < best[0]):
            best = (fret, s)
    return None if best is None else TabPosition(string=best[1], fret=best[0])


def _melody_tab(melody: Sequence[Note], tuning: Tuning) -> tuple[TabbedNote, ...]:
    out: list[TabbedNote] = []
    for n in melody:
        position = _lowest_fret_position(n.pitch, tuning)
        if position is not None:
            out.append(TabbedNote(note=n, position=position))
    return tuple(out)


def accompany(
    voicings: Sequence[VoicedChord],
    melody: Sequence[Note],
    *,
    bars_per_chord: int,
    tempo_bpm: float,
    style: Literal["arpeggio", "strum", "fingerstyle"],
    tuning: Tuning,
) -> tuple[Note, ...]:
    """The accompaniment track as ``Note``s (the melody is not consulted yet)."""
    return tuple(
        Note(
            start_s=float(n.start),
            end_s=float(n.end),
            pitch=int(n.pitch),
            velocity=int(n.velocity),
            confidence=1.0,
        )
        for n in _accompaniment_notes(voicings, bars_per_chord, tempo_bpm, style, tuning)
    )


def arrange(
    scale: Scale,
    progression: Sequence[ChordSymbol],
    *,
    bars_per_chord: int = 1,
    tempo_bpm: float = 100.0,
    style: Literal["arpeggio", "strum", "fingerstyle"] = "fingerstyle",
    tuning: Tuning = STANDARD_EADGBE,
    seed: int | None = None,
) -> Arrangement:
    """Pure core of compose(): voicings, melody, accompaniment and melody tab."""
    _validate(scale, progression)
    voicings = tuple(voicings_for(c, tuning)[0] for c in progression)
    melody = tuple(
        generate_melody(
            scale=scale,
            progression=progression,
            bars_per_chord=bars_per_chord,
            tempo_bpm=tempo_bpm,
            seed=seed,
        )
    )
    return Arrangement(
        melody=melody,
        voicings=voicings,
        accompaniment=accompany(
            voicings,
            melody,
            bars_per_chord=bars_per_chord,
            tempo_bpm=tempo_bpm,
            style=style,
            tuning=tuning,
        ),
        melody_tab=_melody_tab(melody, tuning),
    )


def _ascii_tab_for(arrangement: Arrangement, tuning: Tuning) -> str:
    """Melody tab plus each chord voicing, spread evenly over the melody span
    (a visual approximation; the MIDI is the source of truth for timing)."""
    melody = arrangement.melody
    voicings = arrangement.voicings
    notes: list[TabbedNote] = list(arrangement.melody_tab)
    if voicings and melody:
        span_start = min(n.start_s for n in melody)
        span_end = max(n.end_s for n in melody)
        seg = (span_end - span_start) / len(voicings) if span_end > span_start else 0.0
        for ci, vc in enumerate(voicings):
            t0 = span_start + ci * seg
            anchor = Note(start_s=t0, end_s=t0 + seg, pitch=0, velocity=0, confidence=1.0)
            for pos in vc.positions:
                if pos.fret < 0:  # muted string — not played
                    continue
                notes.append(TabbedNote(note=anchor, position=pos))
    return render_ascii_tab(notes, num_strings=len(tuning.open_pitches))


def compose(
    scale: Scale,
    progression: Sequence[ChordSymbol],
    *,
    bars_per_chord: int = 1,
    tempo_bpm: float = 100.0,
    style: Literal["arpeggio", "strum", "fingerstyle"] = "fingerstyle",
    tuning: Tuning = STANDARD_EADGBE,
    seed: int | None = None,
    out_dir: Path,
) -> Composition:
    arrangement = arrange(
        scale,
        progression,
        bars_per_chord=bars_per_chord,
        tempo_bpm=tempo_bpm,
        style=style,
        tuning=tuning,
        seed=seed,
    )

    payload_hash = hashlib.sha1(
        f"{scale}|{[c.to_label() for c in progression]}|{tempo_bpm}|{style}|{seed}".encode()
    ).hexdigest()[:8]
    ts = time.strftime("%Y%m%dT%H%M%S")
    work = out_dir / f"{ts}-{payload_hash}"
    work.mkdir(parents=True, exist_ok=True)

    midi_path = work / "composition.mid"
    build_midi(
        melody=arrangement.melody,
        voicings=arrangement.voicings,
        bars_per_chord=bars_per_chord,
        tempo_bpm=tempo_bpm,
        style=style,
        out_path=midi_path,
        tuning=tuning,
    )
    wav_path = work / "composition.wav"
    render_wav(midi_path, wav_path)

    ascii_tab = _ascii_tab_for(arrangement, tuning)

    return Composition(
        midi_path=midi_path,
        wav_path=wav_path,
        ascii_tab=ascii_tab,
        melody_notes=arrangement.melody,
        chord_voicings=arrangement.voicings,
        metadata={
            "scale": f"{scale.tonic}:{scale.mode}",
            "progression": [c.to_label() for c in progression],
            "bars_per_chord": bars_per_chord,
            "tempo_bpm": tempo_bpm,
            "style": style,
            "seed": seed,
            "tuning": tuning.name,
        },
    )
