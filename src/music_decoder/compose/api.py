"""Public synchronous compose() entry point."""

from __future__ import annotations

import dataclasses
import hashlib
import statistics
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

import numpy as np
import pretty_midi

from music_decoder.compose.arrangement import (
    THIN_MARGIN_SEMITONES,
    _accompaniment_notes,
    _voiced_pitches,
    build_midi,
)
from music_decoder.compose.fingerstyle import (
    THUMB_EIGHTHS,
    THUMB_VELOCITY,
    arrange_fingerstyle,
    final_bar_pitches,
    hand_shape,
)
from music_decoder.compose.harmony import essential_pcs
from music_decoder.compose.melody import cadence_pc, generate_melody
from music_decoder.compose.ml import MelodyProposer
from music_decoder.compose.performance import humanize, let_ring, shape_dynamics
from music_decoder.compose.scoring import MelodyEngine as MelodyEngine
from music_decoder.compose.scoring import candidate_seeds, score_arrangement
from music_decoder.compose.shapes import choose_bar_shapes, repair_identity
from music_decoder.compose.tabtext import render_composition_tab, render_fingerstyle_tab
from music_decoder.compose.tune import generate_tune
from music_decoder.compose.voicings import voicings_for
from music_decoder.errors import InvalidProgressionError, InvalidScaleError
from music_decoder.synth import render_wav
from music_decoder.synth.plucked import render_plucked, tab_for_midi_notes
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
COMPOSER_VERSION = 6

# MelodyEngine is re-exported from scoring, which defines it because api imports scoring.

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

# How many of the best candidates by symbolic score the opt-in audio re-rank
# renders and scores (v5 spec 5.4).
RANK_AUDIO_TOP_K = 4


@dataclass(frozen=True)
class Arrangement:
    """Everything compose() writes, computed without touching the filesystem.

    ``accompaniment_tab``, ``identity_misses`` and ``final_fallback`` are
    filled only for fingerstyle, where melody and accompaniment are one tabbed
    guitar part (``accompaniment`` is then ``accompaniment_tab``'s notes).
    ``identity_misses`` lists the bars whose accompaniment alone lacks an
    essential tone (spec 5.4 counts them); ``final_fallback`` says the final
    chord could not be re-voiced under the final note (spec 5.3).
    ``bar_voicings`` holds the shape of each bar, fingerstyle only (v5 spec
    5.2); ``voicings`` is one per chord, its first bar's shape.
    ``candidates`` is how many fingerstyle candidates were arranged,
    ``chosen_candidate`` the index of the one kept and ``score`` its
    symbolic score (v5 spec 5.4); ``score`` is ``None`` for strum and
    arpeggio, which arrange a single candidate.
    """

    melody: tuple[Note, ...]
    voicings: tuple[VoicedChord, ...]
    accompaniment: tuple[Note, ...]
    melody_tab: tuple[TabbedNote, ...]
    accompaniment_tab: tuple[TabbedNote, ...] = ()
    identity_misses: tuple[int, ...] = ()
    final_fallback: bool = False
    bar_voicings: tuple[VoicedChord, ...] = ()
    candidates: int = 1
    chosen_candidate: int = 0
    score: float | None = None


def _validate(scale: Scale, progression: Sequence[ChordSymbol], bars_per_chord: int) -> None:
    if scale.mode not in ("major", "minor"):
        raise InvalidScaleError(f"Mode must be major or minor, got {scale.mode!r}")
    if not progression:
        raise InvalidProgressionError("Progression must contain at least one chord.")
    if bars_per_chord < 1:
        raise InvalidProgressionError(f"bars_per_chord must be >= 1, got {bars_per_chord}")


def _validate_candidates(candidates: int) -> None:
    if candidates < 1:
        raise ValueError(f"candidates must be >= 1, got {candidates}")


def _validate_engine(melody_engine: str) -> None:
    if melody_engine not in get_args(MelodyEngine):
        raise ValueError(f"melody_engine must be 'tune' or 'rules', got {melody_engine!r}")


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
    """The accompaniment track as ``Note``s, thinned around ``melody``.

    Strum and arpeggio only: the fingerstyle accompaniment is arranged with
    its own melody, so ``style="fingerstyle"`` raises ``ValueError`` (use
    :func:`arrange`).
    """
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


def _thumb_context(
    bar_voicings: Sequence[VoicedChord], tuning: Tuning, *, tempo_bpm: float
) -> tuple[Note, ...]:
    """The thumb pattern of each bar's hand: the root on beats 1 and 3, the
    alternate bass on 2 and 4, as the fingerstyle part plays it before any
    melody exists. A melody model's conditioning (v5 spec 5.5)."""
    beat_s = 60.0 / tempo_bpm
    out: list[Note] = []
    for b, voicing in enumerate(bar_voicings):
        hand = hand_shape(voicing, tuning)
        for k in THUMB_EIGHTHS:
            position = hand.root if k % 4 == 0 else hand.alternate
            start = b * 4 * beat_s + k * beat_s / 2
            out.append(
                Note(
                    start_s=start,
                    end_s=start + 0.95 * beat_s,
                    pitch=tuning.open_pitches[position.string] + position.fret,
                    velocity=THUMB_VELOCITY,
                    confidence=1.0,
                )
            )
    return tuple(out)


def _propose_melody(
    proposer: MelodyProposer,
    progression: Sequence[ChordSymbol],
    bars_per_chord: int,
    tempo_bpm: float,
    tuning: Tuning,
    seed: int,
) -> tuple[Note, ...]:
    """``proposer``'s melody over the thumb pattern of the piece's per-bar
    shapes, made once per piece (v5 rulings F5a, F5b)."""
    bars = choose_bar_shapes(progression, bars_per_chord=bars_per_chord, tuning=tuning)
    return tuple(
        proposer.propose(
            _thumb_context(bars, tuning, tempo_bpm=tempo_bpm),
            end_s=len(bars) * 4 * 60.0 / tempo_bpm,
            tempo_bpm=tempo_bpm,
            seed=seed,
        )
    )


def _arrange_fingerstyle(
    scale: Scale,
    progression: Sequence[ChordSymbol],
    bars_per_chord: int,
    tempo_bpm: float,
    tuning: Tuning,
    seed: int | None,
    pitch_prior: Sequence[Note] | None = None,
    melody_engine: MelodyEngine = "rules",
) -> Arrangement:
    """The fingerstyle arrangement: a shape per bar (v5 spec 5.2), a melody
    drawn from the pitches each bar's shape can reach (spec 5.5), pulled
    toward ``pitch_prior`` if given (v5 spec 5.5), then the one-guitar part
    around it, its identity repair (v5 spec 5.2) and its performance:
    dynamics and let-ring (v5 spec 5.1). The melody comes from
    ``melody_engine``: the tune engine's statistics of real tunes (v6), or
    the rules engine's hooked, arched phrases (v5 spec 5.3)."""
    bars = choose_bar_shapes(progression, bars_per_chord=bars_per_chord, tuning=tuning)
    hands = [hand_shape(v, tuning) for v in bars]
    n_bars = len(bars)
    allowed = [hand.allowed for hand in hands]
    allowed[-1] = final_bar_pitches(
        hands[-1], cadence_pc(scale, progression[-1]), essential_pcs(progression[-1])
    )
    center = statistics.median_low([p for pitches in allowed for p in sorted(pitches)])
    if melody_engine == "tune":
        melody = tuple(
            generate_tune(
                scale=scale,
                progression=progression,
                bars_per_chord=bars_per_chord,
                tempo_bpm=tempo_bpm,
                seed=seed,
                tessitura_center=center,
                allowed_pitches=allowed,
                final_whole_note=True,
                pitch_prior=pitch_prior,
            )
        )
    else:
        melody = tuple(
            generate_melody(
                scale=scale,
                progression=progression,
                bars_per_chord=bars_per_chord,
                tempo_bpm=tempo_bpm,
                seed=seed,
                tessitura_center=center,
                allowed_pitches=allowed,
                final_whole_note=True,
                phrase_shaping=True,
                pitch_prior=pitch_prior,
            )
        )
    # One shape and hand per bar, so the part is arranged bar by bar.
    part = arrange_fingerstyle(bars, melody, hands, bars_per_chord=1, tempo_bpm=tempo_bpm)
    part, bars = repair_identity(
        part,
        bars,
        progression,
        bars_per_chord=bars_per_chord,
        tempo_bpm=tempo_bpm,
        tuning=tuning,
    )
    # v5 spec 5.1: the performance (velocities, then let-ring lengths).
    part = shape_dynamics(part, n_bars=n_bars, tempo_bpm=tempo_bpm)
    part = let_ring(part, progression, bars_per_chord=bars_per_chord, tempo_bpm=tempo_bpm)
    return Arrangement(
        melody=tuple(t.note for t in part.melody_tab),
        voicings=bars[::bars_per_chord],
        accompaniment=tuple(t.note for t in part.accompaniment_tab),
        melody_tab=part.melody_tab,
        accompaniment_tab=part.accompaniment_tab,
        identity_misses=part.identity_misses,
        final_fallback=part.final_fallback,
        bar_voicings=bars,
    )


def arrange_candidates(
    scale: Scale,
    progression: Sequence[ChordSymbol],
    *,
    bars_per_chord: int = 1,
    tempo_bpm: float = 100.0,
    style: Literal["arpeggio", "strum", "fingerstyle"] = "fingerstyle",
    tuning: Tuning = STANDARD_EADGBE,
    seed: int | None = None,
    candidates: int = 1,
    melody_proposer: MelodyProposer | None = None,
    melody_engine: MelodyEngine = "rules",
) -> tuple[Arrangement, ...]:
    """Every candidate :func:`arrange` considers, best first.

    Fingerstyle arranges ``candidates`` pieces, candidate k from the k-th of
    :func:`candidate_seeds`, each recording its own k (``chosen_candidate``)
    and :func:`score_arrangement` (``score``). They are sorted by score,
    highest first, ties going to the lower k (v5 spec 5.4). With
    ``seed=None`` a base seed is drawn once. With a ``melody_proposer``
    (fingerstyle only), its melody over the thumb pattern of the piece's
    per-bar shapes, seeded with the base seed, is made once and is every
    candidate's ``pitch_prior`` (v5 spec 5.5). ``melody_engine`` picks the
    fingerstyle melody generator and the matching score (v6 spec 5.3, 5.4);
    the default ``"rules"`` is v5's. Strum and arpeggio ignore ``candidates``,
    ``melody_proposer`` and ``melody_engine`` and give their single
    arrangement, its melody from the rules engine.
    """
    _validate(scale, progression, bars_per_chord)
    _validate_candidates(candidates)
    _validate_engine(melody_engine)
    if style != "fingerstyle":
        return (
            _arrange_single(scale, progression, bars_per_chord, tempo_bpm, style, tuning, seed),
        )
    base = seed if seed is not None else int(np.random.default_rng().integers(2**31))
    prior = (
        None
        if melody_proposer is None
        else _propose_melody(melody_proposer, progression, bars_per_chord, tempo_bpm, tuning, base)
    )
    arrangements = [
        _arrange_fingerstyle(
            scale, progression, bars_per_chord, tempo_bpm, tuning, s, prior, melody_engine
        )
        for s in candidate_seeds(base, candidates)
    ]
    scores = [
        score_arrangement(
            a,
            progression,
            bars_per_chord=bars_per_chord,
            tempo_bpm=tempo_bpm,
            tuning=tuning,
            scale=scale,
            melody_engine=melody_engine,
        )
        for a in arrangements
    ]
    order = sorted(range(candidates), key=lambda k: (-scores[k], k))  # ties: the lower k
    return tuple(
        dataclasses.replace(
            arrangements[k], candidates=candidates, chosen_candidate=k, score=scores[k]
        )
        for k in order
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
    candidates: int = 1,
    melody_proposer: MelodyProposer | None = None,
    melody_engine: MelodyEngine = "rules",
) -> Arrangement:
    """Pure core of compose(): voicings, melody, accompaniment and melody tab.

    Reads no files or configuration: the result depends only on the
    arguments and ``COMPOSER_VERSION``. ``compose()`` and
    :func:`compose_with_options` arrange with ``candidates=12``; ``arrange()``
    defaults to ``candidates=1``, a single fingerstyle candidate, so pass
    ``candidates=12`` for what ``compose()`` writes.

    Fingerstyle arranges ``candidates`` pieces, candidate k from the k-th of
    :func:`candidate_seeds`, and keeps the one with the best
    :func:`score_arrangement` (ties go to the lowest k; v5 spec 5.4): the
    first of :func:`arrange_candidates`. With ``seed=None`` a base seed is
    drawn once. A ``melody_proposer`` (fingerstyle only) proposes one melody
    per piece, every candidate's soft pitch prior (v5 spec 5.5).
    ``melody_engine`` (fingerstyle only) picks the melody generator: the
    default ``"rules"`` is v5's, note for note; :func:`compose_with_options`
    defaults to ``"tune"`` (v6 spec 5.3). Strum and arpeggio ignore
    ``candidates``, ``melody_proposer`` and ``melody_engine``.
    """
    return arrange_candidates(
        scale,
        progression,
        bars_per_chord=bars_per_chord,
        tempo_bpm=tempo_bpm,
        style=style,
        tuning=tuning,
        seed=seed,
        candidates=candidates,
        melody_proposer=melody_proposer,
        melody_engine=melody_engine,
    )[0]


def _arrange_single(
    scale: Scale,
    progression: Sequence[ChordSymbol],
    bars_per_chord: int,
    tempo_bpm: float,
    style: Literal["arpeggio", "strum", "fingerstyle"],
    tuning: Tuning,
    seed: int | None,
) -> Arrangement:
    """The strum or arpeggio arrangement: one shape per chord, a melody above
    the shapes' defining tones, the accompaniment thinned around it."""
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


def _humanized_tab(tab: Sequence[TabbedNote], seed: int) -> tuple[TabbedNote, ...]:
    """``tab`` with its notes micro-timed by :func:`humanize`, positions kept."""
    notes = humanize([t.note for t in tab], seed=seed)
    return tuple(dataclasses.replace(t, note=n) for t, n in zip(tab, notes, strict=True))


def _midi_notes(midi_path: Path) -> tuple[Note, ...]:
    """Every note of the MIDI file, ordered by onset then pitch."""
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes = sorted(
        (n for inst in pm.instruments for n in inst.notes), key=lambda n: (n.start, n.pitch)
    )
    return tuple(
        Note(
            start_s=float(n.start),
            end_s=float(n.end),
            pitch=int(n.pitch),
            velocity=int(n.velocity),
            confidence=1.0,
        )
        for n in notes
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
    return compose_with_options(
        scale,
        progression,
        bars_per_chord=bars_per_chord,
        tempo_bpm=tempo_bpm,
        style=style,
        tuning=tuning,
        seed=seed,
        out_dir=out_dir,
        sound="soundfont",
    )


def _render_fingerstyle_wav(
    arrangement: Arrangement,
    wav_path: Path,
    *,
    sound: Literal["soundfont", "plucked"],
    bars_per_chord: int,
    tempo_bpm: float,
    tuning: Tuning,
    humanize_seed: int,
) -> None:
    """The fingerstyle WAV of ``arrangement``, its strokes micro-timed by
    :func:`humanize` (v5 spec 5.1): plucked from the humanized tab, or the
    soundfont from a temporary performance MIDI next to ``wav_path``,
    deleted after rendering."""
    if sound == "plucked":
        tabbed = [
            *_humanized_tab(arrangement.melody_tab, humanize_seed),
            *_humanized_tab(arrangement.accompaniment_tab, humanize_seed),
        ]
        render_plucked(tabbed, tuning, wav_path, seed=humanize_seed)
        return
    performance_path = wav_path.with_suffix(".performance.mid")
    build_midi(
        melody=humanize(arrangement.melody, seed=humanize_seed),
        voicings=arrangement.voicings,
        bars_per_chord=bars_per_chord,
        tempo_bpm=tempo_bpm,
        style="fingerstyle",
        out_path=performance_path,
        tuning=tuning,
        accompaniment=humanize(arrangement.accompaniment, seed=humanize_seed),
    )
    try:
        render_wav(performance_path, wav_path)
    finally:
        performance_path.unlink(missing_ok=True)


def _rank_by_audio(
    scale: Scale,
    progression: Sequence[ChordSymbol],
    *,
    bars_per_chord: int,
    tempo_bpm: float,
    tuning: Tuning,
    seed: int | None,
    candidates: int,
    sound: Literal["soundfont", "plucked"],
    humanize_seed: int,
    melody_proposer: MelodyProposer | None = None,
    melody_engine: MelodyEngine = "rules",
) -> tuple[Arrangement, dict[str, float]]:
    """The fingerstyle candidate the audio re-rank keeps, and its ``CE`` and
    ``PQ``: the top ``RANK_AUDIO_TOP_K`` by symbolic score are rendered to
    temporary WAVs, removed once scored (v5 spec 5.4)."""
    from music_decoder.compose.ml import aesthetics

    scorer = aesthetics.load_scorer()  # fails before any work without the extra
    top = arrange_candidates(
        scale,
        progression,
        bars_per_chord=bars_per_chord,
        tempo_bpm=tempo_bpm,
        style="fingerstyle",
        tuning=tuning,
        seed=seed,
        candidates=candidates,
        melody_proposer=melody_proposer,
        melody_engine=melody_engine,
    )[:RANK_AUDIO_TOP_K]
    with tempfile.TemporaryDirectory(prefix="music-decoder-rank-") as tmp:
        wavs = [Path(tmp) / f"candidate-{i}.wav" for i in range(len(top))]
        for candidate, wav in zip(top, wavs, strict=True):
            _render_fingerstyle_wav(
                candidate,
                wav,
                sound=sound,
                bars_per_chord=bars_per_chord,
                tempo_bpm=tempo_bpm,
                tuning=tuning,
                humanize_seed=humanize_seed,
            )
        rows = scorer.score(wavs)
    best = aesthetics.choose_by_audio(top, rows)
    return top[best], {"CE": rows[best]["CE"], "PQ": rows[best]["PQ"]}


def compose_with_options(
    scale: Scale,
    progression: Sequence[ChordSymbol],
    *,
    bars_per_chord: int = 1,
    tempo_bpm: float = 100.0,
    style: Literal["arpeggio", "strum", "fingerstyle"] = "fingerstyle",
    tuning: Tuning = STANDARD_EADGBE,
    seed: int | None = None,
    out_dir: Path,
    sound: Literal["soundfont", "plucked"] = "soundfont",
    rank_audio: bool = False,
    melody_model: Literal["amt"] | None = None,
    candidates: int = 12,
    melody_engine: MelodyEngine = "tune",
) -> Composition:
    """compose() plus the choice of ``sound``, of the number of fingerstyle
    ``candidates`` and of the audio re-rank (v5 spec 5.4; strum and arpeggio
    ignore both).

    ``melody_engine`` picks the fingerstyle melody generator (v6 spec 5.3):
    ``"tune"`` (the default, and what compose() uses) builds it from the
    statistics of real tunes; ``"rules"`` is composer v5's, note for note.
    Strum and arpeggio always use the rules engine.
    ``metadata["melody_engine"]`` records the engine that made the melody.

    ``metadata["seed_used"]`` is the seed the piece was made with: ``seed``,
    or the one drawn when ``seed`` is ``None``. Passing it back as ``seed``
    with the same other arguments rebuilds the same piece (MIDI, WAV and
    tab). ``metadata["identity_misses"]`` is ``None`` for strum and arpeggio,
    which do not count them.

    ``composition.mid`` and the tab are the on-grid part, whatever the sound.
    With ``sound="soundfont"`` the fingerstyle WAV is rendered from a
    temporary performance MIDI whose strokes are micro-timed by
    :func:`humanize` (v5 spec 5.1), deleted after rendering. With
    ``sound="plucked"`` the WAV comes from :func:`render_plucked`: for
    fingerstyle from the humanized tab, for strum and arpeggio from the notes
    of ``composition.mid`` on their lowest-fret strings.

    With ``rank_audio`` (fingerstyle only, needs the ``ml`` extra and the
    weights, else :class:`~music_decoder.compose.ml.MLUnavailableError`), the
    top ``RANK_AUDIO_TOP_K`` candidates by symbolic score are rendered with
    ``sound`` to temporary WAVs and scored by Audiobox Aesthetics; only the
    chosen one is written.

    With ``melody_model="amt"`` (fingerstyle only, needs the ``ml`` extra and
    the weights, else :class:`~music_decoder.compose.ml.MLUnavailableError`),
    the Anticipatory Music Transformer proposes one melody per piece over the
    thumb pattern, a soft pitch prior for every candidate (v5 spec 5.5). It
    proposes for the first ``AMT_MAX_SECONDS`` (40 s) of the piece at most;
    ``metadata["melody_model_seconds"]`` says how many seconds it covered
    (``None`` without the model).
    """
    _validate(scale, progression, bars_per_chord)  # before any model loads
    _validate_candidates(candidates)
    _validate_engine(melody_engine)
    rank_audio = rank_audio and style == "fingerstyle"
    melody_model = melody_model if style == "fingerstyle" else None
    melody_engine = melody_engine if style == "fingerstyle" else "rules"
    seed_used = seed if seed is not None else int(np.random.default_rng().integers(2**31))
    proposer: MelodyProposer | None = None
    melody_model_seconds: float | None = None
    if melody_model == "amt":
        from music_decoder.compose.ml import amt

        proposer = amt.load_proposer()  # fails before any work without the extra
        piece_s = len(progression) * bars_per_chord * 4 * 60.0 / tempo_bpm
        melody_model_seconds = amt.proposal_seconds(piece_s)
    humanize_seed = seed_used
    audio_scores: dict[str, float] | None = None
    if rank_audio:
        arrangement, audio_scores = _rank_by_audio(
            scale,
            progression,
            bars_per_chord=bars_per_chord,
            tempo_bpm=tempo_bpm,
            tuning=tuning,
            seed=seed_used,
            candidates=candidates,
            sound=sound,
            humanize_seed=humanize_seed,
            melody_proposer=proposer,
            melody_engine=melody_engine,
        )
    else:
        arrangement = arrange(
            scale,
            progression,
            bars_per_chord=bars_per_chord,
            tempo_bpm=tempo_bpm,
            style=style,
            tuning=tuning,
            seed=seed_used,
            candidates=candidates,
            melody_proposer=proposer,
            melody_engine=melody_engine,
        )

    payload_hash = hashlib.sha1(
        (
            f"{scale}|{[c.to_label() for c in progression]}|{tempo_bpm}|{style}|{seed}"
            f"|{melody_engine}|{tuning.name}|{candidates}|{sound}|{bars_per_chord}"
            f"|{rank_audio}|{melody_model}"
        ).encode()
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
        accompaniment=arrangement.accompaniment if style == "fingerstyle" else None,
    )
    wav_path = work / "composition.wav"
    if style == "fingerstyle":
        _render_fingerstyle_wav(
            arrangement,
            wav_path,
            sound=sound,
            bars_per_chord=bars_per_chord,
            tempo_bpm=tempo_bpm,
            tuning=tuning,
            humanize_seed=humanize_seed,
        )
    elif sound == "plucked":
        tabbed = list(tab_for_midi_notes(_midi_notes(midi_path), tuning))
        render_plucked(tabbed, tuning, wav_path, seed=humanize_seed)
    else:
        render_wav(midi_path, wav_path)

    if style == "fingerstyle":
        # One shape per bar, so each bar is named by its own chord.
        ascii_tab = render_fingerstyle_tab(
            [*arrangement.melody_tab, *arrangement.accompaniment_tab],
            arrangement.bar_voicings,
            bars_per_chord=1,
            tempo_bpm=tempo_bpm,
            tuning=tuning,
        )
    else:
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
            "seed_used": seed_used,
            "tuning": tuning.name,
            "composer_version": COMPOSER_VERSION,
            "sound": sound,
            "candidates": arrangement.candidates,
            "chosen_candidate": arrangement.chosen_candidate,
            "score": arrangement.score,
            "final_fallback": arrangement.final_fallback,
            "identity_misses": (
                list(arrangement.identity_misses) if style == "fingerstyle" else None
            ),
            "rank_audio": rank_audio,
            "audio_scores": audio_scores,
            "melody_model": melody_model,
            "melody_model_seconds": melody_model_seconds,
            "melody_engine": melody_engine,
        },
    )
