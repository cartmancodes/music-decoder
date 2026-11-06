"""Public synchronous compose() entry point."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from music_decoder.compose.arrangement import (
    THIN_MARGIN_SEMITONES,
    _accompaniment_notes,
    _voiced_pitches,
    build_midi,
)
from music_decoder.compose.harmony import essential_pcs
from music_decoder.compose.melody import generate_melody
from music_decoder.compose.tabtext import render_composition_tab
from music_decoder.compose.voicings import voicings_for
from music_decoder.errors import InvalidProgressionError, InvalidScaleError
from music_decoder.synth import render_wav
from music_decoder.tabs.assigner import assign_tab
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

# Bumped whenever the composer changes what a given seed produces.
COMPOSER_VERSION = 2

# A* fingering parameters for the melody tab. Pinned here (values as tuned on
# GuitarSet dev for analyze(), 2026-10) rather than read from
# config/hyperparameters.yaml, so compose output depends only on its inputs,
# its seed and COMPOSER_VERSION.
COMPOSE_TAB_WEIGHTS: dict[str, float] = {
    "w_move": 0.5,
    "w_string": 0.15,
    "w_span": 0.5,
    "w_high": 1.6,
    "w_open": 0.05,
    "w_chord_intra": 0.3,
}
COMPOSE_TAB_MAX_FRET = 22


@dataclass(frozen=True)
class Arrangement:
    """Everything compose() writes, computed without touching the filesystem."""

    melody: tuple[Note, ...]
    voicings: tuple[VoicedChord, ...]
    accompaniment: tuple[Note, ...]
    melody_tab: tuple[TabbedNote, ...]


def _validate(scale: Scale, progression: Sequence[ChordSymbol], bars_per_chord: int) -> None:
    if scale.mode not in ("major", "minor"):
        raise InvalidScaleError(f"Mode must be major or minor, got {scale.mode!r}")
    if not progression:
        raise InvalidProgressionError("Progression must contain at least one chord.")
    if bars_per_chord < 1:
        raise InvalidProgressionError(f"bars_per_chord must be >= 1, got {bars_per_chord}")


def _lowest_fret_position(pitch: int, tuning: Tuning) -> TabPosition | None:
    """The most playable single position: the string giving the lowest fret."""
    best: tuple[int, int] | None = None  # (fret, string)
    for s, open_pitch in enumerate(tuning.open_pitches):
        fret = pitch - open_pitch
        if 0 <= fret <= COMPOSE_TAB_MAX_FRET and (best is None or fret < best[0]):
            best = (fret, s)
    return None if best is None else TabPosition(string=best[1], fret=best[0])


def _melody_tab(melody: Sequence[Note], tuning: Tuning) -> tuple[TabbedNote, ...]:
    """A* fingering with compose's pinned weights; any note A* drops falls
    back to its lowest-fret position so the tab never loses notes."""
    result = assign_tab(
        melody, tuning=tuning, weights=COMPOSE_TAB_WEIGHTS, max_fret=COMPOSE_TAB_MAX_FRET
    )
    assigned = {(tn.note.start_s, tn.note.pitch): tn for tn in result.tabbed_notes}
    out: list[TabbedNote] = []
    for n in melody:
        tabbed = assigned.get((n.start_s, n.pitch))
        if tabbed is None:
            position = _lowest_fret_position(n.pitch, tuning)
            if position is None:
                continue
            tabbed = TabbedNote(note=n, position=position)
        out.append(tabbed)
    return tuple(out)


def _melody_floor(voicing: VoicedChord, tuning: Tuning) -> int | None:
    """Lowest melody pitch that keeps the shape's defining tones (third,
    seventh…) at least a minor third below the tune."""
    pitches = _voiced_pitches(voicing, tuning)
    lows = [
        min(p for p in pitches if p % 12 == pc)
        for pc in essential_pcs(voicing.chord)
        if any(p % 12 == pc for p in pitches)
    ]
    return max(lows) + THIN_MARGIN_SEMITONES if lows else None


def accompany(
    voicings: Sequence[VoicedChord],
    melody: Sequence[Note],
    *,
    bars_per_chord: int,
    tempo_bpm: float,
    style: Literal["arpeggio", "strum", "fingerstyle"],
    tuning: Tuning,
) -> tuple[Note, ...]:
    """The accompaniment track as ``Note``s, thinned around ``melody``."""
    return tuple(
        Note(
            start_s=float(n.start),
            end_s=float(n.end),
            pitch=int(n.pitch),
            velocity=int(n.velocity),
            confidence=1.0,
        )
        for n in _accompaniment_notes(
            voicings, bars_per_chord, tempo_bpm, style, tuning, melody=melody
        )
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
    """Pure core of compose(): voicings, melody, accompaniment and melody tab.

    Reads no files or configuration: the result depends only on the
    arguments and ``COMPOSER_VERSION``.
    """
    _validate(scale, progression, bars_per_chord)
    voicings = tuple(voicings_for(c, tuning)[0] for c in progression)
    melody = tuple(
        generate_melody(
            scale=scale,
            progression=progression,
            bars_per_chord=bars_per_chord,
            tempo_bpm=tempo_bpm,
            seed=seed,
            chord_floors=[_melody_floor(v, tuning) for v in voicings],
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

    ascii_tab = render_composition_tab(
        arrangement.melody_tab,
        arrangement.voicings,
        bars_per_chord=bars_per_chord,
        tempo_bpm=tempo_bpm,
        tuning=tuning,
    )

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
            "composer_version": COMPOSER_VERSION,
        },
    )
