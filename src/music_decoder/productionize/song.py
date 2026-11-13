"""The song a production renders: grid, form, bass roots and notes.

Two sources:

- :func:`song_from_midi` / :func:`song_from_composition` take what the
  composer wrote (one pass over the progression: a ``melody`` and an
  ``accompaniment`` track) and lay it out over the production form.
- :func:`hisaab_song` is the reference song the recipe was tuned on. Its
  notes come from the final note file when one is given, else they are
  rebuilt from the written description of that file.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pretty_midi

from music_decoder.errors import MusicDecoderError
from music_decoder.productionize.config import (
    COMPOSER_SECTIONS,
    HISAAB_BAR_ROOTS,
    HISAAB_GRID,
    HISAAB_SECTIONS,
    HISAAB_V3_REFERENCE,
    ComposerArrangeConfig,
    Grid,
    QcReference,
    SectionSpec,
)
from music_decoder.types import VALID_TONICS, ChordSymbol, Composition


class ProductionError(MusicDecoderError):
    """The production chain could not render the song."""


@dataclass(frozen=True)
class NoteEvent:
    start_s: float
    end_s: float
    pitch: int
    velocity: int


def _onset_order(note: NoteEvent) -> tuple[float, int]:
    return (note.start_s, note.pitch)


@dataclass(frozen=True)
class Song:
    title: str
    key: str  # for the MP3 tag, e.g. "F#m"
    grid: Grid
    sections: tuple[SectionSpec, ...]
    bar_roots: tuple[int, ...]  # bass MIDI pitch for each bar of a section
    piano: tuple[NoteEvent, ...]
    strings: tuple[NoteEvent, ...]
    reference: QcReference | None = None
    source: str = ""

    def __post_init__(self) -> None:
        if not self.sections:
            raise ProductionError("a song needs at least one section")
        if len(self.bar_roots) != self.grid.bars_per_section:
            raise ProductionError(
                f"{len(self.bar_roots)} bass roots for {self.grid.bars_per_section} bars a section"
            )

    @property
    def body_s(self) -> float:
        """Length of the sections, without the tail."""
        return len(self.sections) * self.grid.section_s

    def section_span(self, index: int) -> tuple[int, int]:
        """First and one-past-last sample of section ``index``."""
        g = self.grid
        return g.samples(g.time(index)), g.samples(g.time(index + 1))

    @property
    def verse_sections(self) -> tuple[int, ...]:
        """Sections that levels are measured over: the verses if there are
        any, else every section with full drums, else all of them."""
        verses = tuple(i for i, s in enumerate(self.sections) if s.role == "verse")
        if verses:
            return verses
        full = tuple(i for i, s in enumerate(self.sections) if s.drums == "full")
        return full or tuple(range(len(self.sections)))

    @property
    def strings_sections(self) -> tuple[int, ...]:
        return tuple(i for i, s in enumerate(self.sections) if s.strings)

    @property
    def tail_start_s(self) -> float:
        return self.body_s


# ---- Reading note files ----------------------------------------------------

_STRING_PROGRAMS = range(40, 56)  # General MIDI strings and ensembles


def _events(instrument: pretty_midi.Instrument) -> list[NoteEvent]:
    return [
        NoteEvent(float(n.start), float(n.end), int(n.pitch), int(n.velocity))
        for n in instrument.notes
        if n.end > n.start
    ]


def _load_midi(path: Path) -> pretty_midi.PrettyMIDI:
    try:
        return pretty_midi.PrettyMIDI(str(path))
    except Exception as e:  # pretty_midi raises a mix of OSError/ValueError/EOFError
        raise ProductionError(f"could not read MIDI file {path}: {e}") from e


def read_note_file(path: Path) -> tuple[tuple[NoteEvent, ...], tuple[NoteEvent, ...]]:
    """(piano, strings) notes of a two-part note file, sorted by onset.

    A track is the strings part if its name says so or it uses a General
    MIDI string program; everything else pitched is the piano part.
    Zero-length notes are dropped.
    """
    piano: list[NoteEvent] = []
    strings: list[NoteEvent] = []
    for inst in _load_midi(path).instruments:
        if inst.is_drum:
            continue
        is_strings = "string" in inst.name.lower() or inst.program in _STRING_PROGRAMS
        (strings if is_strings else piano).extend(_events(inst))
    if not piano:
        raise ProductionError(f"{path} has no piano notes")
    return tuple(sorted(piano, key=_onset_order)), tuple(sorted(strings, key=_onset_order))


# ---- Composer output -------------------------------------------------------


def _fold(pitch_class: int, low: int) -> int:
    """The lowest pitch with ``pitch_class`` at or above ``low``."""
    return low + (pitch_class - low) % 12


def _key_label(scale: str) -> str:
    """``"A:minor"`` -> ``"Am"``; ``"C:major"`` -> ``"C"``."""
    tonic, _, mode = scale.partition(":")
    return tonic + ("m" if mode == "minor" else "")


def _roll(
    pitches: Sequence[int], start_s: float, spread_s: float, hold_s: float, velocity: int
) -> list[NoteEvent]:
    """The pitches entering one by one, lowest first, all ending together."""
    n = len(pitches)
    return [
        NoteEvent(start_s + spread_s * k / n, start_s + hold_s, int(p), velocity)
        for k, p in enumerate(sorted(pitches))
    ]


def _clamp_velocity(velocity: int) -> int:
    return max(1, min(127, velocity))


def song_from_midi(
    midi_path: Path,
    *,
    tempo_bpm: float | None = None,
    progression: Sequence[ChordSymbol] | None = None,
    title: str = "Composition (Instrumental)",
    key: str = "",
    sections: tuple[SectionSpec, ...] = COMPOSER_SECTIONS,
    arrange: ComposerArrangeConfig | None = None,
) -> Song:
    """Lay a composer MIDI file (one pass over its progression) over the form.

    The tempo comes from the file unless given. Each bar's bass root is the
    root of ``progression`` when given, else the lowest accompaniment note
    of the bar.
    """
    cfg = arrange or ComposerArrangeConfig()
    pm = _load_midi(midi_path)
    by_name = {inst.name.lower(): inst for inst in pm.instruments if not inst.is_drum}
    pitched = [inst for inst in pm.instruments if not inst.is_drum]
    if "melody" in by_name and "accompaniment" in by_name:
        melody = _events(by_name["melody"])
        accompaniment = _events(by_name["accompaniment"])
    elif len(pitched) >= 2:
        melody = _events(pitched[0])
        accompaniment = [n for inst in pitched[1:] for n in _events(inst)]
    else:
        raise ProductionError(
            f"{midi_path} is not a composer file: expected a melody and an accompaniment track"
        )
    if not accompaniment:
        raise ProductionError(f"{midi_path} has no accompaniment notes")

    if tempo_bpm is None:
        tempi = pm.get_tempo_changes()[1]
        tempo_bpm = float(tempi[0]) if len(tempi) else 120.0
    if tempo_bpm <= 0:
        raise ProductionError(f"tempo must be positive, got {tempo_bpm}")
    beat = 60.0 / tempo_bpm
    bar = 4.0 * beat
    snap = bar / 64.0  # MIDI ticks round note times by a millisecond or two
    n_bars = max(1, round(max(n.end_s for n in accompaniment) / bar))
    grid = Grid(beat_s=beat, bars_per_section=n_bars)

    def bar_of(note: NoteEvent) -> int:
        return min(n_bars - 1, max(0, int((note.start_s + snap) // bar)))

    chords: list[list[int]] = [[] for _ in range(n_bars)]
    for n in accompaniment:
        chords[bar_of(n)].append(n.pitch)
    chords = [sorted(set(c)) for c in chords]
    for i, chord in enumerate(chords):  # an empty bar keeps the previous chord
        if not chord:
            chords[i] = chords[i - 1] if i else next(c for c in chords if c)

    if progression:
        if n_bars % len(progression):
            raise ProductionError(
                f"{len(progression)} chords do not divide the file's {n_bars} bars evenly"
            )
        per_chord = n_bars // len(progression)
        root_pcs = [VALID_TONICS.index(progression[b // per_chord].root) for b in range(n_bars)]
    else:
        root_pcs = [chord[0] % 12 for chord in chords]
    bar_roots = tuple(_fold(pc, cfg.bass_low) for pc in root_pcs)
    if any(r > cfg.bass_high for r in bar_roots):
        raise ProductionError("bass range must span an octave (bass_high >= bass_low + 11)")

    last_bar_start = (n_bars - 1) * bar - snap
    piano: list[NoteEvent] = []
    strings: list[NoteEvent] = []
    for index, section in enumerate(sections):
        t0 = grid.time(index)

        def place(notes: Sequence[NoteEvent], offset: int, t0: float = t0) -> list[NoteEvent]:
            return [
                NoteEvent(
                    n.start_s + t0, n.end_s + t0, n.pitch, _clamp_velocity(n.velocity + offset)
                )
                for n in notes
            ]

        offset = section.piano_velocity_offset
        if section.piano_texture == "held_chords":
            for b in range(n_bars - 1):
                piano.extend(
                    _roll(
                        chords[b],
                        t0 + b * bar,
                        cfg.roll_steps * grid.step_s,
                        cfg.held_chord_beats * beat,
                        _clamp_velocity(cfg.held_chord_velocity + offset),
                    )
                )
            # The last bar is the composer's own ringing chord.
            piano.extend(place([n for n in accompaniment if n.start_s >= last_bar_start], offset))
            piano.extend(place(melody, offset))
        elif section.piano_texture == "accompaniment_only":
            piano.extend(place(accompaniment, offset))
        else:
            piano.extend(place(accompaniment, offset))
            piano.extend(place(melody, offset))
        if section.strings:
            for b in range(n_bars):
                start = t0 + b * bar
                strings.extend(
                    NoteEvent(start, start + cfg.strings_beats * beat, p, cfg.strings_velocity)
                    for p in chords[b][-cfg.strings_voices :]
                )

    tail = len(sections) * grid.section_s
    piano.extend(
        _roll(
            chords[-1],
            tail,
            cfg.tail_roll_steps * grid.step_s,
            cfg.tail_beats * beat,
            cfg.tail_velocity,
        )
    )
    if any(s.strings for s in sections):
        strings_end = tail + min(cfg.tail_beats * beat, cfg.strings_tail_max_s)
        strings.extend(
            NoteEvent(tail, strings_end, p, cfg.strings_velocity)
            for p in chords[-1][-cfg.strings_voices :]
        )

    return Song(
        title=title,
        key=key,
        grid=grid,
        sections=sections,
        bar_roots=bar_roots,
        piano=tuple(sorted(piano, key=_onset_order)),
        strings=tuple(sorted(strings, key=_onset_order)),
        source=f"composer file {midi_path.name}",
    )


def song_from_composition(
    composition: Composition,
    *,
    sections: tuple[SectionSpec, ...] = COMPOSER_SECTIONS,
    arrange: ComposerArrangeConfig | None = None,
) -> Song:
    """A :class:`Song` from what ``compose()`` returned (uses its metadata)."""
    meta = composition.metadata
    labels = meta.get("progression") or []
    scale = str(meta.get("scale", ""))
    tempo = meta.get("tempo_bpm")
    return song_from_midi(
        composition.midi_path,
        tempo_bpm=float(tempo) if tempo is not None else None,
        progression=[ChordSymbol.parse(str(label)) for label in labels] or None,
        title=f"Composition in {scale.replace(':', ' ')}" if scale else "Composition",
        key=_key_label(scale) if scale else "",
        sections=sections,
        arrange=arrange,
    )


# ---- The reference song ----------------------------------------------------

# MIDI pitches: F#1=30, B1=35, C#2=37, F#2=42, B2=47, C#3=49, F#3=54, G#3=56,
# A3=57, B3=59, C#4=61, D4=62, F4=65, F#4=66, G#4=68, A4=69, B4=71, C#5=73,
# D5=74, F5=77, F#5=78.
_HIT_CHORD = (30, 42, 49, 57, 61, 66, 69, 73)  # the rolled F#m of bar 4 and the tail
_INTRO_LEFT = ((42, 49), (47, 54), (49, 56))
_INTRO_RIGHT = ((69, 73, 78), (71, 74, 78), (71, 73, 77))
_HOOK_LEFT = ((30, 42), (35, 47), (37, 49))
_HOOK_RIGHT = (
    (57, 61, 66, 69, 66, 61, 57),
    (59, 62, 66, 71, 66, 62, 59),
    (61, 65, 68, 73, 68, 65, 61),
)
_VERSE_CHORDS = ((30, 42, 57, 61, 66), (35, 47, 59, 62, 66), (37, 49, 59, 61, 65, 68))
_VERSE_MELODY = (73, 74, 77)
_STRING_CHORDS = ((42, 57, 61, 66), (47, 59, 62, 66), (49, 59, 65), (42, 57, 61, 66))


def _hisaab_notes() -> tuple[tuple[NoteEvent, ...], tuple[NoteEvent, ...]]:
    """The final piano (263 notes) and strings (34 notes) parts, rebuilt from
    the written description of the note file. Velocities are drawn from the
    documented ranges with a fixed seed; pitches and rhythm are as described.
    """
    g = HISAAB_GRID
    beat, step = g.beat_s, g.step_s
    rng = np.random.default_rng(1)
    piano: list[NoteEvent] = []
    strings: list[NoteEvent] = []

    def vel(low: int, high: int) -> int:
        return int(rng.integers(low, high + 1))

    def hit(section: int, low: int, high: int) -> None:
        start = g.time(section, 3)
        for k, pitch in enumerate(_HIT_CHORD):
            piano.append(
                NoteEvent(
                    start + step * k / len(_HIT_CHORD), start + 3.8 * beat, pitch, vel(low, high)
                )
            )

    def held_bars(section: int, low: int, high: int) -> None:
        for bar, chord in enumerate(_VERSE_CHORDS):
            start = g.time(section, bar)
            for k, pitch in enumerate(chord):
                piano.append(
                    NoteEvent(
                        start + step * k / len(chord), start + 3.9 * beat, pitch, vel(low, high)
                    )
                )
            melody_start = g.time(section, bar, 10)
            piano.append(
                NoteEvent(
                    melody_start, melody_start + 1.4 * beat, _VERSE_MELODY[bar], vel(low, high)
                )
            )

    def broken_bars(section: int, low: int, high: int) -> None:
        for bar in range(3):
            for beat_step in (0, 8):  # the octave on beat 1, again on beat 3
                start = g.time(section, bar, beat_step)
                for pitch in _HOOK_LEFT[bar]:
                    piano.append(NoteEvent(start, start + 1.9 * beat, pitch, vel(low, high)))
            for k, pitch in enumerate(_HOOK_RIGHT[bar]):
                start = g.time(section, bar, 2 * (k + 1))
                piano.append(NoteEvent(start, start + 2 * step, pitch, vel(low, high)))

    # Intro: arpeggio over a held left hand.
    for bar in range(3):
        start = g.time(0, bar)
        for pitch in _INTRO_LEFT[bar]:
            piano.append(NoteEvent(start, start + 3.8 * beat, pitch, vel(38, 54)))
        tones = _INTRO_RIGHT[bar]
        for k, index in enumerate((0, 1, 2, 1, 0, 1, 2, 1)):
            start = g.time(0, bar, 2 * k)
            piano.append(NoteEvent(start, start + 2 * step, tones[index], vel(38, 54)))
    hit(0, 76, 92)

    broken_bars(1, 58, 76)
    hit(1, 88, 105)
    for section in (2, 3, 4, 5):
        held_bars(section, 56, 67)
        hit(section, 95, 102)
    broken_bars(6, 62, 80)
    hit(6, 88, 105)
    held_bars(7, 44, 55)
    hit(7, 80, 94)

    tail = g.time(8)
    for k, pitch in enumerate(_HIT_CHORD):
        piano.append(
            NoteEvent(tail + 1.5 * step * k / len(_HIT_CHORD), tail + 8 * beat, pitch, vel(70, 84))
        )

    for section in (6, 7):
        for bar, chord in enumerate(_STRING_CHORDS):
            start = g.time(section, bar)
            strings.extend(NoteEvent(start, start + 4 * beat, p, vel(46, 66)) for p in chord)
    strings.extend(NoteEvent(tail, tail + 8 * beat, p, vel(46, 66)) for p in _STRING_CHORDS[0])

    return tuple(sorted(piano, key=_onset_order)), tuple(sorted(strings, key=_onset_order))


def hisaab_song(notes_midi: Path | None = None) -> Song:
    """ "Hisaab": 1:43 boom-bap instrumental in F# minor at 80.1 BPM.

    Pass the final note file (``hisaab_final_notes.mid``) to play it as is;
    without it the notes are the rebuild from its description.
    """
    if notes_midi is not None:
        piano, strings = read_note_file(notes_midi)
        source = f"note file {notes_midi.name}"
    else:
        piano, strings = _hisaab_notes()
        source = "notes rebuilt from the build spec"
    return Song(
        title="Hisaab (Instrumental)",
        key="F#m",
        grid=HISAAB_GRID,
        sections=HISAAB_SECTIONS,
        bar_roots=HISAAB_BAR_ROOTS,
        piano=piano,
        strings=strings,
        reference=HISAAB_V3_REFERENCE,
        source=source,
    )
