"""Public synchronous compose() entry point."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from music_decoder.compose.arrangement import build_midi
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


def _validate(scale: Scale, progression: Sequence[ChordSymbol]) -> None:
    if scale.mode not in ("major", "minor"):
        raise InvalidScaleError(f"Mode must be major or minor, got {scale.mode!r}")
    if not progression:
        raise InvalidProgressionError("Progression must contain at least one chord.")


def _ascii_tab_for(
    melody: list[Note],
    voicings: list[VoicedChord],
    tuning: Tuning,
) -> str:
    """Build a TabbedNote stream for the melody + each chord voicing, then
    hand to render_ascii_tab."""
    notes: list[TabbedNote] = []
    n_str = len(tuning.open_pitches)
    # Melody: pick the most playable position — the string giving the lowest
    # non-negative fret. (The old greedy-from-high search collapsed every
    # melody note onto the high-e string.)
    for n in melody:
        best: tuple[int, int] | None = None  # (fret, string)
        for s in range(n_str):
            fret = n.pitch - tuning.open_pitches[s]
            if 0 <= fret <= 22 and (best is None or fret < best[0]):
                best = (fret, s)
        if best is not None:
            notes.append(TabbedNote(note=n, position=TabPosition(string=best[1], fret=best[0])))
    # Chord voicings: emit each non-muted string position at the chord's
    # approximate start time, so the accompaniment shows on the lower strings.
    # (A visual approximation spread evenly over the melody span; the MIDI is
    # the source of truth for exact timing.)
    if voicings and melody:
        span_start = min(n.start_s for n in melody)
        span_end = max(n.end_s for n in melody)
        seg = (span_end - span_start) / len(voicings) if span_end > span_start else 0.0
        for ci, vc in enumerate(voicings):
            t0 = span_start + ci * seg
            anchor = Note(
                start_s=t0, end_s=t0 + seg, pitch=0, velocity=0, confidence=1.0
            )
            for pos in vc.positions:
                if pos.fret < 0:  # muted string — not played
                    continue
                notes.append(TabbedNote(note=anchor, position=pos))
    return render_ascii_tab(notes, num_strings=n_str)


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
    _validate(scale, progression)

    voicings_list: list[VoicedChord] = [voicings_for(c, tuning)[0] for c in progression]
    melody = generate_melody(
        scale=scale,
        progression=progression,
        bars_per_chord=bars_per_chord,
        tempo_bpm=tempo_bpm,
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
        melody=melody,
        voicings=voicings_list,
        bars_per_chord=bars_per_chord,
        tempo_bpm=tempo_bpm,
        style=style,
        out_path=midi_path,
        tuning=tuning,
    )
    wav_path = work / "composition.wav"
    render_wav(midi_path, wav_path)

    ascii_tab = _ascii_tab_for(melody, voicings_list, tuning)

    return Composition(
        midi_path=midi_path,
        wav_path=wav_path,
        ascii_tab=ascii_tab,
        melody_notes=tuple(melody),
        chord_voicings=tuple(voicings_list),
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
