"""The pop-rap song as data: sections, drum hits, pad events and note lists.

Nothing here makes sound. :func:`poprap_song` reads a composer file once and
lays its one pass over the pop-rap form; the other functions turn that song
into what the renderers play (trap drums, 808, pad, pluck, lead).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from music_decoder.productionize.config import (
    ComposerArrangeConfig,
    Grid,
    PopRapConfig,
    PopRapFormConfig,
    QcReference,
    SectionRole,
    TrapDrumsConfig,
)
from music_decoder.productionize.song import ComposerParts, NoteEvent, read_composer_parts
from music_decoder.types import ChordSymbol, Scale

PopRapDrums = Literal["none", "full", "drop"]  # "drop": full, but silent in the last bar
PopRapLeadPart = Literal["none", "melody", "melody_double", "tail"]
DrumKind = Literal["kick", "clap", "hat", "open_hat", "crash", "riser"]

DEFAULT_RISER_BARS = PopRapFormConfig().riser_bars


@dataclass(frozen=True)
class PopRapSection:
    name: str
    role: SectionRole
    start_bar: int
    bars: int
    drums: PopRapDrums
    lead: PopRapLeadPart
    hats_from_bar: int | None = None  # hats before the drums come in (the intro)
    riser: bool = False
    crash: bool = False
    rolls: bool = False  # a triplet hat roll in every drum bar


@dataclass(frozen=True)
class PopRapSong:
    title: str
    key: str
    grid: Grid  # bars_per_section is the composer pass
    sections: tuple[PopRapSection, ...]
    parts: ComposerParts
    source: str = ""
    reference: QcReference | None = None

    @property
    def body_s(self) -> float:
        """Seconds of every section; the final chord starts here."""
        return sum(s.bars for s in self.sections) * self.grid.bar_s

    def section_span(self, index: int) -> tuple[int, int]:
        """(start, end) of a section in samples."""
        section = self.sections[index]
        bar_s = self.grid.bar_s
        return (
            self.grid.samples(section.start_bar * bar_s),
            self.grid.samples((section.start_bar + section.bars) * bar_s),
        )

    @property
    def verse_sections(self) -> tuple[int, ...]:
        return tuple(i for i, s in enumerate(self.sections) if s.role == "verse")

    def pass_bar(self, bar_in_section: int) -> int:
        """The composer bar that ``bar_in_section`` repeats."""
        return bar_in_section % self.parts.n_bars


@dataclass(frozen=True)
class DrumHit:
    time_s: float
    kind: DrumKind
    velocity: float = 1.0
    pitch: int = 0  # kick: the 808's MIDI pitch
    length_s: float = 0.0  # kick: to the next kick or bar end; riser: its length
    pan: float = 0.0


@dataclass(frozen=True)
class PadEvent:
    start_s: float
    length_s: float
    pitches: tuple[int, ...]
    gain: float = 1.0


def hook_bars(pass_bars: int, target: int) -> int:
    """Whole passes closest to ``target`` bars (at least one)."""
    return max(1, round(target / pass_bars)) * pass_bars


def poprap_form(pass_bars: int, cfg: PopRapFormConfig) -> tuple[PopRapSection, ...]:
    """Intro, hook, verse, hook, outro: the spec's table."""
    body = hook_bars(pass_bars, cfg.section_target_bars)
    layout: tuple[tuple[str, SectionRole, int, dict[str, object]], ...] = (
        (
            "Intro",
            "intro",
            cfg.intro_bars,
            {
                "drums": "none",
                "lead": "none",
                "hats_from_bar": cfg.intro_hats_from_bar,
                "riser": True,
            },
        ),
        ("Hook 1", "hook", body, {"drums": "full", "lead": "melody", "crash": True}),
        ("Verse", "verse", body, {"drums": "drop", "lead": "none", "rolls": True, "crash": True}),
        ("Hook 2", "hook", body, {"drums": "full", "lead": "melody_double", "crash": True}),
        ("Outro", "outro", cfg.outro_bars, {"drums": "none", "lead": "tail"}),
    )
    sections: list[PopRapSection] = []
    start = 0
    for name, role, bars, flags in layout:
        sections.append(PopRapSection(name, role, start, bars, **flags))  # type: ignore[arg-type]
        start += bars
    return tuple(sections)


def poprap_song(
    midi_path: Path,
    *,
    cfg: PopRapConfig,
    tempo_bpm: float | None = None,
    progression: Sequence[ChordSymbol] | None = None,
    scale: Scale | None = None,
    title: str = "Composition (Instrumental)",
    key: str = "",
) -> PopRapSong:
    """Lay a composer MIDI file (one pass over its progression) over the pop-rap form."""
    parts = read_composer_parts(
        midi_path,
        tempo_bpm=tempo_bpm,
        progression=progression,
        scale=scale,
        key=key,
        arrange=ComposerArrangeConfig(),
    )
    return PopRapSong(
        title=title,
        key=key,
        grid=Grid(beat_s=parts.beat_s, bars_per_section=parts.n_bars),
        sections=poprap_form(parts.n_bars, cfg.form),
        parts=parts,
        source=f"composer file {midi_path.name}",
    )


# ---- Drums -----------------------------------------------------------------


def _is_drum_bar(section: PopRapSection, b: int) -> bool:
    if section.drums == "full":
        return True
    return section.drums == "drop" and b != section.bars - 1


def _kick_hits(bar_start: float, root: int, steps: tuple[int, ...], grid: Grid) -> list[DrumHit]:
    ends = (*steps[1:], grid.steps_per_bar)
    return [
        DrumHit(
            bar_start + step * grid.step_s,
            "kick",
            pitch=root,
            length_s=(end - step) * grid.step_s,
        )
        for step, end in zip(steps, ends, strict=True)
    ]


def _hat_steps(b: int, cfg: TrapDrumsConfig, grid: Grid) -> list[int]:
    if b % 2 == 0:
        return list(range(0, grid.steps_per_bar, 2))
    return [
        *range(0, cfg.hat_odd_bar_16ths_from, 2),
        *range(cfg.hat_odd_bar_16ths_from, grid.steps_per_bar),
    ]


def drum_hits(
    song: PopRapSong, cfg: TrapDrumsConfig, *, riser_bars: int = DEFAULT_RISER_BARS
) -> list[DrumHit]:
    """Every drum hit of the song in absolute seconds, in section order.

    ``riser_bars`` is the length of the intro riser (``PopRapFormConfig.riser_bars``).
    """
    grid = song.grid
    bar_s, step_s = grid.bar_s, grid.step_s
    hits: list[DrumHit] = []
    for section in song.sections:
        for b in range(section.bars):
            bar_start = (section.start_bar + b) * bar_s
            drum_bar = _is_drum_bar(section, b)
            if drum_bar:
                steps = cfg.kick_steps_even if b % 2 == 0 else cfg.kick_steps_odd
                root = song.parts.bar_roots[song.pass_bar(b)]
                hits.extend(_kick_hits(bar_start, root, steps, grid))
                hits.append(DrumHit(bar_start + cfg.clap_step * step_s, "clap", cfg.clap_gain))
                if b % 4 == 3:
                    hits.append(
                        DrumHit(
                            bar_start + cfg.ghost_clap_step * step_s, "clap", cfg.ghost_clap_gain
                        )
                    )
            early_hats = section.hats_from_bar is not None and b >= section.hats_from_bar
            if drum_bar or early_hats:
                for step in _hat_steps(b, cfg, grid):
                    accent = cfg.hat_accent if step % 4 == 0 else cfg.hat_soft
                    hits.append(DrumHit(bar_start + step * step_s, "hat", accent, pan=cfg.hat_pan))
            if drum_bar and (section.rolls or b % 4 == 3):
                roll_start = bar_start + cfg.roll_start_beat * grid.beat_s
                for k in range(cfg.roll_hits):
                    hits.append(
                        DrumHit(
                            roll_start + k * grid.beat_s / 6,
                            "hat",
                            cfg.roll_velocity + cfg.roll_velocity_step * k,
                            pan=cfg.hat_pan,
                        )
                    )
            if drum_bar and b % 4 == 2:
                hits.append(
                    DrumHit(
                        bar_start + cfg.open_hat_step * step_s,
                        "open_hat",
                        cfg.open_hat_velocity,
                        pan=cfg.open_hat_pan,
                    )
                )
            if section.crash and b == 0:
                hits.append(DrumHit(bar_start, "crash", pan=cfg.crash_pan))
        if section.riser:
            hits.append(
                DrumHit(
                    (section.start_bar + section.bars - riser_bars) * bar_s,
                    "riser",
                    cfg.riser_gain,
                    length_s=riser_bars * bar_s,
                )
            )
    return hits


# ---- Pad -------------------------------------------------------------------


def _fold(pitch_class: int, low: int) -> int:
    """The lowest pitch with ``pitch_class`` at or above ``low``."""
    return low + (pitch_class - low) % 12


def pad_events(song: PopRapSong, cfg: PopRapConfig) -> list[PadEvent]:
    """One pad chord per run of equal voicing, ending on the home chord."""
    parts, bar_s = song.parts, song.grid.bar_s
    root_low = ComposerArrangeConfig().rh_low - 12

    def chord(pass_bar: int) -> tuple[int, ...]:
        root = _fold(parts.harmonies[pass_bar].root, root_low)
        return tuple(sorted({*parts.voicings[pass_bar], root}))

    events: list[PadEvent] = []
    for section in song.sections:
        starts: list[tuple[int, tuple[int, ...]]] = []
        for b in range(section.bars):
            pitches = chord(song.pass_bar(b))
            if not starts or starts[-1][1] != pitches:
                starts.append((b, pitches))
        ends = [b for b, _ in starts[1:]] + [section.bars]
        for (b, pitches), end in zip(starts, ends, strict=True):
            events.append(PadEvent((section.start_bar + b) * bar_s, (end - b) * bar_s, pitches))
    home_root = _fold(parts.home.root, root_low)
    events.append(
        PadEvent(
            song.body_s,
            cfg.form.final_chord_bars * bar_s,
            tuple(sorted({*parts.home_voicing, home_root})),
            cfg.pad.final_chord_gain,
        )
    )
    return events


# ---- Notes -----------------------------------------------------------------


def _pass_notes(
    song: PopRapSong, section: PopRapSection, notes: Sequence[NoteEvent]
) -> list[NoteEvent]:
    """``notes`` once per composer pass from the section start, cut at their pass end."""
    bar_s = song.grid.bar_s
    placed: list[NoteEvent] = []
    for first_bar in range(0, section.bars, song.parts.n_bars):
        offset = (section.start_bar + first_bar) * bar_s
        pass_end = offset + min(song.parts.n_bars, section.bars - first_bar) * bar_s
        for n in notes:
            start = offset + n.start_s
            if start < pass_end:
                placed.append(
                    NoteEvent(start, min(offset + n.end_s, pass_end), n.pitch, n.velocity)
                )
    return placed


def pluck_notes(song: PopRapSong) -> list[NoteEvent]:
    """The accompaniment once per pass in every section, cut at its pass end."""
    return [
        n for section in song.sections for n in _pass_notes(song, section, song.parts.accompaniment)
    ]


def lead_notes(song: PopRapSong) -> tuple[list[NoteEvent], list[NoteEvent]]:
    """(main, hook-2 double): the melody where a section has a lead, at melody pitch."""
    main: list[NoteEvent] = []
    double: list[NoteEvent] = []
    for section in song.sections:
        if section.lead == "none":
            continue
        notes = _pass_notes(song, section, song.parts.melody)
        main.extend(notes)
        if section.lead == "melody_double":
            double.extend(notes)
    return main, double
