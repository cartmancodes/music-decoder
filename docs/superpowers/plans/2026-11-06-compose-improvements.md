# Compose Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `music_decoder.compose` output noticeably more musical, and prove it. The composer gains phrase-structured rhythm with motifs and cadences, scored pitch choice with leading-tone resolution and altered-chord awareness, and an on-grid accompaniment that stays under the tune without losing chord identity. Its playable A\* melody tab is printed without collisions. A paired evaluation compares the composer against a committed baseline of the pre-change composer on identical held-out inputs.

**Architecture:**
- `compose/harmony.py` (pitch-class helpers) and `compose/rhythm.py` (phrase layout and bar rhythms) feed a rewritten `compose/melody.py`.
- `compose/arrangement.py` plays grid-aligned, melody-aware patterns.
- `compose/tabtext.py` renders the tab.
- `compose/api.py` gains a pure `arrange()` that `compose()` wraps with file I/O.
- `compose/metrics.py` plus `evaluation/compose_musicality.py` measure everything.
- `tests/fixtures/compose/musicality_baseline.json` holds the baseline, written before the generator changes.

**Tech Stack:** Python 3.11, numpy (seeded `default_rng`), pretty_midi, the existing A\* assigner (`music_decoder.tabs.assigner.assign_tab`), pytest, ruff, mypy --strict. No new dependencies.

**Execution method:** superpowers:subagent-driven-development (pre-approved by the pipeline: Opus implements, GPT-6 Astra reviews).

**Spec:** `docs/superpowers/specs/2026-11-06-compose-improvements-design.md` (revision 2). Read §3 (paired baseline and targets), §4 (definitions), §5 (design), §10 (decisions) and §11 (review response).

**Dry run:** every code block below was built in a scratch copy of this repo, and the tasks were then replayed in order on a fresh copy. After every task all four gates were green, ending at `pytest -q tests/unit tests/integration` with 517 passed and 3 skipped. Task 3's baseline JSON regenerated identically on replay. If a step's result differs from its "Expected" line, stop and investigate; do not loosen a threshold to make it pass.

## Global Constraints

- `compose()`'s signature is frozen: `tests/unit/test_public_contract.py::test_compose_signature_is_frozen` must pass unchanged. Add no `compose()` parameters and no CLI flags.
- Output is deterministic for a given seed: `numpy.random.default_rng(seed)` is the only randomness. `arrange()` must read no files or configuration.
- No new dependencies and no model downloads. Do not install packages.
- Gate commands, green after every task: `pytest -q tests/unit tests/integration`, `ruff check src tests`, `ruff format --check src tests`, `mypy --strict src`.
- Line length 100; ruff rules `E F W I B UP RUF`.
- Work only in this worktree on `feat/compose-improve`. Never push, merge or switch branches.
- **Never `git add -A`, `git add .` or `git commit -a`.** Test runs rewrite `tests/fixtures/synthetic/.*.synth_marker`; never commit those. Stage exactly the paths listed in each task.
- Every commit message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Do not change `compose/voicings.py`, `chords/`, `tabs/` (including `tabs/render.py`), `synth/` or the analyze pipeline. The only config change is Task 7's deletion of the unread `composition:` block.
- Do not add wall-clock assertions. `tests/unit/test_compose_musicality.py` took 2.3–4.7 s in the dry run, depending on machine load.

## Review Focus

1. **`bars_per_chord` ≤ 0.** It must raise `InvalidProgressionError`; it used to give an empty melody silently. Pinned by `test_bars_per_chord_below_one_raises` (Task 7).
2. **Single-chord, single-bar progressions** (`--progression "D"`). They must still produce a melody that ends on the tonic, over a ringing chord. Pinned by `test_single_bar_progression` (Task 4), `test_single_bar_accompaniment_is_the_final_chord` (Task 5) and the edge matrix (Task 8).
3. **Non-standard tunings** (Drop D, DADGAD, D standard). Every melody note must be tabbed at a position that produces its pitch, with correct string labels. Pinned by `test_arrange_in_other_tunings_tabs_every_note` (Task 7) and `test_string_labels` (Task 6).
4. **Extreme tempos** (40 and 220 BPM, the UI slider's limits). The grid and the total length must stay exact. Pinned by `test_extreme_tempos_keep_the_grid_and_the_length` (Task 4).
5. **Altered chords in a major key** (secondary dominant D7, borrowed Fm, ♭VII). The melody adapts only the altered degree and keeps the target note. Pinned by `test_secondary_dominant_in_major_keeps_its_target` (Task 4), `test_borrowed_and_secondary_chords_in_c` (Task 2) and the edge matrix (Task 8).

---

### Task 0: Restore the `ruff format` gate (pre-existing drift)

On the base commit `ruff format --check src tests` already fails on 7 files (spec D12).

**Files:**
- Modify (formatting only): `src/music_decoder/compose/api.py`, `src/music_decoder/config/hyperparameters.py`, `src/music_decoder/synth/__init__.py`, `tests/unit/test_config_hyperparameters_v3.py`, `tests/unit/test_public_contract.py`, `tests/unit/test_tab_assignment.py`, `tests/unit/test_transcription_decode.py`

**Interfaces:**
- Consumes: nothing. Produces: a green `ruff format --check src tests`.

- [ ] **Step 1: Confirm the drift**

Run: `ruff format --check src tests`
Expected: `7 files would be reformatted`, exactly the files above.

- [ ] **Step 2: Format exactly those files**

```bash
ruff format src/music_decoder/compose/api.py src/music_decoder/config/hyperparameters.py \
  src/music_decoder/synth/__init__.py tests/unit/test_config_hyperparameters_v3.py \
  tests/unit/test_public_contract.py tests/unit/test_tab_assignment.py \
  tests/unit/test_transcription_decode.py
```

- [ ] **Step 3: Verify the gates**

Run: `ruff format --check src tests && ruff check src tests && mypy --strict src && pytest -q tests/unit tests/integration`
Expected: all clean; pytest gives 0 failures (dry run: 327 passed, 3 skipped).
Run: `git status --short`. Only the 7 files plus the regenerated `*.synth_marker` files should be listed. Do not stage the markers.

- [ ] **Step 4: Commit**

```bash
git add src/music_decoder/compose/api.py src/music_decoder/config/hyperparameters.py \
  src/music_decoder/synth/__init__.py tests/unit/test_config_hyperparameters_v3.py \
  tests/unit/test_public_contract.py tests/unit/test_tab_assignment.py \
  tests/unit/test_transcription_decode.py
git commit -m "style: apply ruff format to files with pre-existing drift

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 1: Phrase layout and rhythm planner

**Files:**
- Create: `src/music_decoder/compose/rhythm.py`
- Test: `tests/unit/test_compose_rhythm.py`

**Interfaces:**
- Consumes: `numpy.random.Generator`.
- Produces, in `music_decoder.compose.rhythm`:
  - `BODY_RHYTHMS`, `CADENCE_RHYTHMS: tuple[tuple[int, ...], ...]`, in eighths, each summing to 8
  - `@dataclass(frozen=True) PhraseLayout(phrase_lengths: tuple[int, ...], cadence_bars: tuple[int, ...], echo_pairs: tuple[tuple[int, int], ...])`
  - `@dataclass(frozen=True) BarPlan(rhythm: tuple[int, ...], cadence: bool, final: bool, motif_from: int | None)`
  - `split_phrases(n_bars) -> list[int]`, which raises `ValueError` if `n_bars < 1`
  - `phrase_layout(n_bars) -> PhraseLayout`, which depends on `n_bars` only
  - `plan_bars(n_bars, rng) -> list[BarPlan]`, which follows `phrase_layout`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_compose_rhythm.py`:

```python
import numpy as np
import pytest

from music_decoder.compose.rhythm import (
    BODY_RHYTHMS,
    CADENCE_RHYTHMS,
    BarPlan,
    phrase_layout,
    plan_bars,
    split_phrases,
)


def test_templates_fill_one_bar():
    for rhythm in BODY_RHYTHMS + CADENCE_RHYTHMS:
        assert sum(rhythm) == 8, rhythm
        assert all(v >= 1 for v in rhythm)


def test_cadence_templates_end_long():
    assert all(r[-1] >= 4 for r in CADENCE_RHYTHMS)


@pytest.mark.parametrize(
    ("n_bars", "expected"),
    [
        (1, [1]),
        (2, [2]),
        (3, [3]),
        (4, [4]),
        (5, [3, 2]),
        (6, [4, 2]),
        (7, [4, 3]),
        (8, [4, 4]),
        (9, [4, 3, 2]),
        (12, [4, 4, 4]),
    ],
)
def test_split_phrases(n_bars, expected):
    assert split_phrases(n_bars) == expected
    assert sum(split_phrases(n_bars)) == n_bars


def test_split_phrases_rejects_zero():
    with pytest.raises(ValueError):
        split_phrases(0)


@pytest.mark.parametrize(
    ("n_bars", "cadences", "echoes"),
    [
        (1, (0,), ()),
        (3, (2,), ()),
        (4, (3,), ((0, 2),)),
        (5, (2, 4), ((0, 3),)),  # 3 + 2: the consequent echoes antecedent bar 1
        (7, (3, 6), ((0, 2), (0, 4), (1, 5))),  # 4 + 3
        (8, (3, 7), ((0, 2), (0, 4), (1, 5), (2, 6))),
        (9, (3, 6, 8), ((0, 2), (0, 4), (1, 5))),  # 4 + 3 + 2 (third phrase is fresh)
    ],
)
def test_phrase_layout(n_bars, cadences, echoes):
    layout = phrase_layout(n_bars)
    assert layout.cadence_bars == cadences
    assert layout.echo_pairs == echoes
    assert sum(layout.phrase_lengths) == n_bars


def test_plan_bars_structure_for_eight_bars():
    plans = plan_bars(8, np.random.default_rng(3))
    assert len(plans) == 8
    assert all(isinstance(p, BarPlan) for p in plans)
    assert [p.cadence for p in plans] == [False, False, False, True] * 2
    assert [p.final for p in plans] == [False] * 7 + [True]
    assert plans[2].rhythm == plans[0].rhythm
    assert plans[2].motif_from == 0
    assert [p.rhythm for p in plans[4:7]] == [p.rhythm for p in plans[0:3]]
    assert [p.motif_from for p in plans[4:7]] == [0, 1, 2]
    assert plans[7].rhythm[-1] >= 4


def test_plan_bars_single_bar_is_a_final_cadence():
    (only,) = plan_bars(1, np.random.default_rng(0))
    assert only.cadence and only.final
    assert only.rhythm in CADENCE_RHYTHMS


@pytest.mark.parametrize("n_bars", range(1, 14))
def test_plan_bars_follows_the_layout(n_bars):
    plans = plan_bars(n_bars, np.random.default_rng(n_bars))
    layout = phrase_layout(n_bars)
    assert len(plans) == n_bars
    assert tuple(i for i, p in enumerate(plans) if p.cadence) == layout.cadence_bars
    pairs = tuple((p.motif_from, i) for i, p in enumerate(plans) if p.motif_from is not None)
    assert tuple(sorted(pairs, key=lambda x: x[1])) == tuple(
        sorted(layout.echo_pairs, key=lambda x: x[1])
    )
    for source, target in layout.echo_pairs:
        assert plans[source].rhythm == plans[target].rhythm


def test_plan_bars_is_deterministic_per_seed():
    assert plan_bars(8, np.random.default_rng(11)) == plan_bars(8, np.random.default_rng(11))
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/unit/test_compose_rhythm.py -q`
Expected: collection ERROR, `ModuleNotFoundError: No module named 'music_decoder.compose.rhythm'`.

- [ ] **Step 3: Write the implementation**

Create `src/music_decoder/compose/rhythm.py`:

```python
"""One-bar rhythm templates and phrase planning for the melody generator.

Units are eighth notes; every template sums to 8 (one 4/4 bar). The phrase
*layout* depends only on the number of bars (see :func:`phrase_layout`), so
the musicality metrics can evaluate any melody, old or new, against the same
phrase boundaries and echo pairs the planner uses:

- antecedent phrases (even index) play ``[A, B, A, cadence]``; bar 3 echoes
  bar 1's intervals;
- the following consequent phrase reuses the antecedent's body rhythms and
  echoes its intervals bar for bar;
- every phrase ends on a cadence bar whose last note is long.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BODY_RHYTHMS: tuple[tuple[int, ...], ...] = (
    (2, 2, 2, 2),
    (2, 1, 1, 2, 2),
    (1, 1, 2, 2, 2),
    (2, 2, 1, 1, 2),
    (3, 1, 2, 2),
    (2, 2, 3, 1),
    (4, 2, 2),
    (1, 1, 1, 1, 2, 2),
    (2, 1, 1, 1, 1, 2),
    (3, 1, 3, 1),
    (4, 1, 1, 2),
)

# Each ends on a note of at least a half note (4 eighths).
CADENCE_RHYTHMS: tuple[tuple[int, ...], ...] = (
    (2, 2, 4),
    (4, 4),
    (3, 1, 4),
    (2, 1, 1, 4),
    (8,),
)

_FINAL_MIN_LAST_EIGHTHS = 4


@dataclass(frozen=True)
class PhraseLayout:
    """Bar-level phrase structure for a piece of ``n_bars`` bars."""

    phrase_lengths: tuple[int, ...]
    cadence_bars: tuple[int, ...]  # last bar of each phrase (absolute index)
    echo_pairs: tuple[tuple[int, int], ...]  # (source bar, echoing bar)


@dataclass(frozen=True)
class BarPlan:
    """Rhythm and role of one bar of melody."""

    rhythm: tuple[int, ...]  # note lengths in eighths, summing to 8
    cadence: bool  # last bar of a phrase
    final: bool  # last bar of the piece
    motif_from: int | None  # absolute index of the bar whose intervals to echo


def split_phrases(n_bars: int) -> list[int]:
    """Phrase lengths (bars) covering ``n_bars``: 4s, then a 2- or 3-bar tail."""
    if n_bars < 1:
        raise ValueError(f"n_bars must be >= 1, got {n_bars}")
    if n_bars <= 4:
        return [n_bars]
    phrases = [4] * (n_bars // 4)
    remainder = n_bars % 4
    if remainder == 1:
        phrases[-1] = 3
        phrases.append(2)
    elif remainder:
        phrases.append(remainder)
    return phrases


def phrase_layout(n_bars: int) -> PhraseLayout:
    """Phrase boundaries and motif echo pairs (independent of any randomness)."""
    lengths = split_phrases(n_bars)
    cadences: list[int] = []
    echoes: list[tuple[int, int]] = []
    start = 0
    antecedent_start, antecedent_body = 0, 0
    for index, length in enumerate(lengths):
        body = length - 1
        if index % 2 == 0:
            if length == 4:
                echoes.append((start, start + 2))
            antecedent_start, antecedent_body = start, body
        else:
            echoes.extend(
                (antecedent_start + i, start + i) for i in range(min(body, antecedent_body))
            )
        cadences.append(start + length - 1)
        start += length
    return PhraseLayout(tuple(lengths), tuple(cadences), tuple(echoes))


def _draw(rng: np.random.Generator, templates: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    return templates[int(rng.integers(len(templates)))]


def plan_bars(n_bars: int, rng: np.random.Generator) -> list[BarPlan]:
    """One ``BarPlan`` per bar, in order, following :func:`phrase_layout`."""
    layout = phrase_layout(n_bars)
    echo_of = {target: source for source, target in layout.echo_pairs}
    rhythms: list[tuple[int, ...]] = []
    plans: list[BarPlan] = []
    start = 0
    for index, length in enumerate(layout.phrase_lengths):
        final = index == len(layout.phrase_lengths) - 1
        if index % 2 == 0:
            a = _draw(rng, BODY_RHYTHMS)
            b = _draw(rng, BODY_RHYTHMS)
            fresh = [a, b, a]
        for i in range(length - 1):
            bar = start + i
            source = echo_of.get(bar)
            if source is not None:
                rhythm = rhythms[source]
            elif index % 2 == 0:
                rhythm = fresh[i]
            else:
                rhythm = _draw(rng, BODY_RHYTHMS)
            rhythms.append(rhythm)
            plans.append(BarPlan(rhythm=rhythm, cadence=False, final=False, motif_from=source))
        cadences = tuple(
            c for c in CADENCE_RHYTHMS if not final or c[-1] >= _FINAL_MIN_LAST_EIGHTHS
        )
        rhythm = _draw(rng, cadences)
        rhythms.append(rhythm)
        plans.append(BarPlan(rhythm=rhythm, cadence=True, final=final, motif_from=None))
        start += length
    return plans
```

- [ ] **Step 4: Run the tests and gates**

Run: `pytest tests/unit/test_compose_rhythm.py -q`. Expected: `36 passed`.
Run: `ruff check src tests && ruff format --check src tests && mypy --strict src`. Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add src/music_decoder/compose/rhythm.py tests/unit/test_compose_rhythm.py
git commit -m "feat(compose): phrase layout and bar rhythm planner

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Harmony helpers and musicality metrics

**Files:**
- Create: `src/music_decoder/compose/harmony.py`, `src/music_decoder/compose/metrics.py`
- Test: `tests/unit/test_compose_harmony.py`, `tests/unit/test_compose_metrics.py`

**Interfaces:**
- Consumes: `rhythm.phrase_layout` (Task 1).
- Produces, in `music_decoder.compose.harmony`:
  - `CHORD_INTERVALS`, `ESSENTIAL_INTERVALS: dict[str, tuple[int, ...]]`
  - `tonic_pc(scale) -> int`, `leading_tone_pc(scale) -> int`
  - `scale_pcs(scale) -> tuple[int, ...]`, `chord_pcs(chord) -> tuple[int, ...]` (root first)
  - `essential_pcs(chord) -> frozenset[int]`
  - `chord_scale_pcs(scale, chord) -> frozenset[int]`
- Produces, in `music_decoder.compose.metrics`. Melody and accompaniment arguments are `Sequence[Note]`, and ratios return `float | None`:
  - `chord_at`, `metric_position`, `intervals`
  - `strong_beat_chord_tone_ratio(melody, progression, bars_per_chord, tempo_bpm)`
  - `step_ratio(melody)`, `repeat_ratio(melody)`, `large_leap_ratio(melody)`, `max_interval(melody) -> int`
  - `leap_recovery_counts(melody) -> tuple[int, int]`
  - `ambitus(melody) -> int`
  - `final_resolution(melody, scale, progression) -> bool`, `final_note_beats(melody, tempo_bpm) -> float`
  - `distinct_durations(melody, tempo_bpm) -> int`, `ioi_entropy(melody, tempo_bpm) -> float`
  - `motif_pair_coverage(melody, tempo_bpm, n_bars)`, `motif_contour_match(melody, tempo_bpm, n_bars)`
  - `phrase_end_long_ratio(melody, tempo_bpm, n_bars) -> float`
  - `exposed_chromatic_clash_ratio(melody, scale, progression, bars_per_chord, tempo_bpm)`
  - `leading_tone_resolution_counts(melody, scale, progression, bars_per_chord, tempo_bpm) -> tuple[int, int]`
  - `onset_grid_ratio(onsets, tempo_bpm, per_beat)`, `stroke_heads(onsets) -> list[float]`
  - `melody_above_accompaniment_ratio(melody, accompaniment)`
  - `exposed_semitone_clash_ratio(melody, accompaniment, tempo_bpm)`
  - `final_chord_rings(accompaniment, n_bars, tempo_bpm) -> bool`
  - `essential_tone_coverage(melody, accompaniment, voicings, bars_per_chord, tempo_bpm, tuning) -> list[bool]`
  - `hand_shift_rate(frets)`
  - `nearest_pitch_proximity(voicings, tuning)`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_compose_harmony.py`:

```python
import pytest

from music_decoder.compose.harmony import (
    chord_pcs,
    chord_scale_pcs,
    essential_pcs,
    leading_tone_pc,
    scale_pcs,
    tonic_pc,
)
from music_decoder.types import ChordSymbol, Scale

C = Scale("C", "major")
A_MINOR = Scale("A", "minor")


def ch(label: str) -> ChordSymbol:
    return ChordSymbol.parse(label)


def test_scale_and_chord_pitch_classes():
    assert scale_pcs(C) == (0, 2, 4, 5, 7, 9, 11)
    assert scale_pcs(A_MINOR) == (9, 11, 0, 2, 4, 5, 7)
    assert chord_pcs(ch("G7")) == (7, 11, 2, 5)
    assert tonic_pc(A_MINOR) == 9
    assert leading_tone_pc(A_MINOR) == 8  # G#


@pytest.mark.parametrize(
    ("label", "expected"),
    [("C", {4}), ("Am", {0}), ("G7", {11, 5}), ("Bdim", {2, 5}), ("Dsus4", {7}), ("Caug", {4, 8})],
)
def test_essential_pcs(label, expected):
    assert essential_pcs(ch(label)) == frozenset(expected)


def test_chord_scale_is_the_scale_over_diatonic_chords():
    for label in ("C", "Dm7", "Em", "F", "G7", "Am", "Bdim"):
        assert chord_scale_pcs(C, ch(label)) == frozenset(scale_pcs(C))


def test_e7_in_a_minor_gives_harmonic_minor_and_keeps_the_tonic():
    assert chord_scale_pcs(A_MINOR, ch("E7")) == frozenset({9, 11, 0, 2, 4, 5, 8})


def test_secondary_dominant_keeps_its_target():
    pool = chord_scale_pcs(C, ch("D7"))  # D F# A C
    assert 6 in pool and 5 not in pool  # F -> F#
    assert 7 in pool  # G (where D7 resolves) is kept


@pytest.mark.parametrize(
    ("label", "added", "removed"),
    [
        ("C7", 10, 11),  # Bb replaces B
        ("Fm", 8, 9),  # Ab replaces A
        ("A", 1, 0),  # C# replaces C over A major
        ("A#", 10, 11),  # bVII: foreign root Bb replaces B
    ],
)
def test_borrowed_and_secondary_chords_in_c(label, added, removed):
    pool = chord_scale_pcs(C, ch(label))
    assert added in pool and removed not in pool
    assert len(pool) == 7


def test_bii_maj7_keeps_its_own_chord_tones():
    pool = chord_scale_pcs(C, ch("C#maj7"))  # C# F G# C
    assert {1, 5, 8, 0} <= pool
    assert 2 not in pool and 9 not in pool  # D and A are the lowered degrees
```

Create `tests/unit/test_compose_metrics.py`. Each metric is checked on a hand-built example with a known answer. The file includes review round 1's counterexamples: staggered strum voices, a late entry, non-overlap controls, the 5-bar layout, unequal echo bars, and the proximity blind spot.

```python
import pytest

from music_decoder.compose import metrics as m
from music_decoder.compose.voicings import voicings_for
from music_decoder.tabs.tuning import STANDARD_EADGBE
from music_decoder.types import ChordSymbol, Note, Scale, TabPosition, VoicedChord

TEMPO = 120.0  # one beat = 0.5 s, one bar = 2 s
C = Scale("C", "major")
A_MINOR = Scale("A", "minor")


def n(start_beats: float, dur_beats: float, pitch: int) -> Note:
    return Note(
        start_s=start_beats * 0.5,
        end_s=(start_beats + dur_beats) * 0.5,
        pitch=pitch,
        velocity=80,
        confidence=1.0,
    )


def prog(*labels: str) -> list[ChordSymbol]:
    return [ChordSymbol.parse(x) for x in labels]


def shape(label: str) -> VoicedChord:
    return voicings_for(ChordSymbol.parse(label), STANDARD_EADGBE)[0]


def test_chord_at_and_metric_position():
    p = prog("C", "G")
    assert m.chord_at(0.0, p, 1, TEMPO).to_label() == "C"
    assert m.chord_at(2.0, p, 1, TEMPO).to_label() == "G"  # bar 2 starts at 2.0 s
    assert m.chord_at(2.0, p, 2, TEMPO).to_label() == "C"  # 2 bars per chord
    assert m.chord_at(99.0, p, 1, TEMPO).to_label() == "G"  # clamped
    assert m.metric_position(0.0, TEMPO) == 0
    assert m.metric_position(1.0, TEMPO) == 4  # beat 3
    assert m.metric_position(2.25, TEMPO) == 1  # bar 2, the "and" of beat 1


def test_strong_beat_chord_tone_ratio():
    # beat 1: E (chord tone of C), beat 2: F (weak, ignored), beat 3: D (non-chord tone)
    mel = [n(0, 1, 64), n(1, 1, 65), n(2, 1, 62)]
    assert m.strong_beat_chord_tone_ratio(mel, prog("C"), 1, TEMPO) == pytest.approx(0.5)
    assert m.strong_beat_chord_tone_ratio([n(1, 1, 60)], prog("C"), 1, TEMPO) is None


def test_interval_ratios_and_max_interval():
    mel = [n(i, 1, p) for i, p in enumerate([60, 62, 62, 64, 72, 71])]
    # intervals: +2, 0, +2, +8, -1
    assert m.step_ratio(mel) == pytest.approx(3 / 5)
    assert m.repeat_ratio(mel) == pytest.approx(1 / 5)
    assert m.large_leap_ratio(mel) == pytest.approx(1 / 5)
    assert m.max_interval(mel) == 8
    assert m.step_ratio([n(0, 1, 60)]) is None


def test_leap_recovery_counts():
    # +7 then -2 (recovered); +5 then +2 (not recovered); final +4 is not a leap
    mel = [n(i, 1, p) for i, p in enumerate([60, 67, 65, 70, 72, 76])]
    assert m.leap_recovery_counts(mel) == (2, 1)


def test_ambitus_and_final_note():
    mel = [n(0, 1, 62), n(1, 1, 79), n(2, 2, 72)]
    assert m.ambitus(mel) == 17
    assert m.final_note_beats(mel, TEMPO) == pytest.approx(2.0)


def test_final_resolution_tonic_or_root():
    assert m.final_resolution([n(0, 4, 72)], C, prog("G", "C"))  # ends on C over C
    assert not m.final_resolution([n(0, 4, 64)], C, prog("G", "C"))  # E over C: not tonic
    assert m.final_resolution([n(0, 4, 67)], C, prog("C", "G"))  # G root (no tonic in G)
    assert not m.final_resolution([], C, prog("C"))


def test_rhythm_variety():
    uniform = [n(i, 0.95, 60) for i in range(8)]
    assert m.distinct_durations(uniform, TEMPO) == 1
    assert m.ioi_entropy(uniform, TEMPO) == pytest.approx(0.0)
    varied = [n(0, 1.5, 60), n(1.5, 0.5, 62), n(2, 1, 64), n(3, 1, 65), n(4, 2, 67)]
    # spacings in eighths: 3, 1, 2, 2 and last note 4 -> {1, 2, 3, 4}
    assert m.distinct_durations(varied, TEMPO) == 4
    assert m.ioi_entropy(varied, TEMPO) == pytest.approx(1.5)


def _bar(bar: int, pitches: list[int], last_beats: float = 0.95) -> list[Note]:
    k = len(pitches)
    step = 4 / k
    notes = [n(bar * 4 + i * step, step * 0.95, p) for i, p in enumerate(pitches[:-1])]
    return [*notes, n(bar * 4 + (k - 1) * step, last_beats, pitches[-1])]


def test_motif_metrics_on_a_four_bar_phrase():
    # layout for 4 bars echoes bar 0 in bar 2
    mel = (
        _bar(0, [60, 62, 64, 62])  # up up down
        + _bar(1, [67])
        + _bar(2, [65, 67, 65, 64])  # up down down
        + _bar(3, [60])
    )
    assert m.motif_pair_coverage(mel, TEMPO, 4) == pytest.approx(1.0)
    assert m.motif_contour_match(mel, TEMPO, 4) == pytest.approx(2 / 3)


def test_motif_metrics_follow_the_five_bar_layout():
    # 5 bars = 3 + 2: the only echo pair is (bar 0, bar 3)
    mel = (
        _bar(0, [60, 62, 64])
        + _bar(1, [65, 64])
        + _bar(2, [62], 2)
        + _bar(3, [64, 65, 67])  # same contour as bar 0
        + _bar(4, [60], 2)
    )
    assert m.motif_pair_coverage(mel, TEMPO, 5) == pytest.approx(1.0)
    assert m.motif_contour_match(mel, TEMPO, 5) == pytest.approx(1.0)


def test_unequal_echo_bars_lower_coverage_and_are_undefined_for_contour():
    mel = _bar(0, [60, 62, 64, 62]) + _bar(1, [67]) + _bar(2, [65, 67]) + _bar(3, [60])
    assert m.motif_pair_coverage(mel, TEMPO, 4) == pytest.approx(0.0)
    assert m.motif_contour_match(mel, TEMPO, 4) is None
    assert m.motif_pair_coverage(_bar(0, [60, 62]), TEMPO, 1) is None  # no pairs at all


def test_phrase_end_long_ratio_uses_the_layout():
    # 5 bars = 3 + 2: phrase ends are bars 2 and 4 (not bar 3)
    mel = [*_bar(0, [60] * 4), *_bar(1, [60] * 4), *_bar(2, [62], 2)]
    mel += [*_bar(3, [60] * 4), *_bar(4, [60], 2)]
    assert m.phrase_end_long_ratio(mel, TEMPO, 5) == pytest.approx(1.0)
    short = [*_bar(0, [60] * 4), *_bar(1, [60] * 4), *_bar(2, [62] * 4)]
    short += [*_bar(3, [60] * 4), *_bar(4, [60], 2)]
    assert m.phrase_end_long_ratio(short, TEMPO, 5) == pytest.approx(0.5)


def test_exposed_chromatic_clash_ratio():
    p = prog("Am", "E7")  # E7 = E G# B D; G# is foreign to A natural minor
    mel = [
        n(0, 1, 67),  # over Am: not counted
        n(4, 1, 67),  # beat 1 over E7, G vs G#: clash
        n(5, 0.5, 69),  # weak, short A: a passing note, not exposed
        n(6, 1, 68),  # beat 3, G# itself is a chord tone: fine
        n(7, 1, 69),  # beat 4 but a whole beat long: exposed A vs G#: clash
    ]
    assert m.exposed_chromatic_clash_ratio(mel, A_MINOR, p, 1, TEMPO) == pytest.approx(2 / 3)
    assert m.exposed_chromatic_clash_ratio(mel, C, prog("C", "G"), 1, TEMPO) is None


def test_chromatic_clash_ignores_chord_tones_next_to_a_foreign_tone():
    # C#maj7 in C major = C# F G# C: C is a chord tone even though C# is foreign
    assert m.exposed_chromatic_clash_ratio([n(0, 2, 72)], C, prog("C#maj7"), 1, TEMPO) == 0.0


def test_leading_tone_resolution_counts():
    p = prog("E7", "Am")
    resolved = [n(0, 4, 68), n(4, 4, 69)]  # G# -> A across E7 -> Am
    assert m.leading_tone_resolution_counts(resolved, A_MINOR, p, 1, TEMPO) == (1, 1)
    falls = [n(0, 4, 68), n(4, 4, 64)]  # G# -> E
    assert m.leading_tone_resolution_counts(falls, A_MINOR, p, 1, TEMPO) == (1, 0)
    same_chord = [n(0, 2, 68), n(2, 2, 69)]  # still inside E7: not eligible
    assert m.leading_tone_resolution_counts(same_chord, A_MINOR, p, 1, TEMPO) == (0, 0)


def test_onset_grid_and_stroke_heads():
    assert m.onset_grid_ratio([0.0, 0.25, 0.5], TEMPO, 2) == pytest.approx(1.0)
    assert m.onset_grid_ratio([0.0, 0.1], TEMPO, 2) == pytest.approx(0.5)
    assert m.stroke_heads([0.0, 0.008, 0.016, 0.25, 0.26]) == [0.0, 0.25]


def test_melody_above_and_exposed_clash():
    acc = [n(0, 4, 48), n(0, 4, 64)]  # C3 + E4 ringing for the bar
    mel = [n(0, 1, 65), n(1, 0.5, 67), n(2, 1, 60)]
    # 65 above 64 (yes), 67 (yes), 60 below 64 (no)
    assert m.melody_above_accompaniment_ratio(mel, acc) == pytest.approx(2 / 3)
    # exposed: 65 (beat 1, F vs E = semitone -> clash), 60 (beat 3, no clash); 67 is weak+short
    assert m.exposed_semitone_clash_ratio(mel, acc, TEMPO) == pytest.approx(0.5)
    assert m.melody_above_accompaniment_ratio(mel, []) == pytest.approx(1.0)


def test_staggered_and_late_entering_accompaniment_is_seen():
    # C4 held for 1.9 s; C3 from 0, C#4 enters 8 ms late (a strummed upper string)
    mel = [Note(start_s=0.0, end_s=1.9, pitch=60, velocity=80, confidence=1.0)]
    acc = [
        Note(start_s=0.0, end_s=2.0, pitch=48, velocity=80, confidence=1.0),
        Note(start_s=0.008, end_s=2.0, pitch=61, velocity=80, confidence=1.0),
    ]
    assert m.melody_above_accompaniment_ratio(mel, acc) == 0.0
    assert m.exposed_semitone_clash_ratio(mel, acc, TEMPO) == 1.0
    entering = [Note(start_s=1.0, end_s=2.0, pitch=61, velocity=80, confidence=1.0)]
    assert m.exposed_semitone_clash_ratio(mel, entering, TEMPO) == 1.0


def test_non_overlapping_accompaniment_is_ignored():
    mel = [Note(start_s=0.0, end_s=1.0, pitch=60, velocity=80, confidence=1.0)]
    before = [Note(start_s=-1.0, end_s=0.0, pitch=61, velocity=80, confidence=1.0)]
    after = [Note(start_s=1.0, end_s=2.0, pitch=61, velocity=80, confidence=1.0)]
    for acc in (before, after):
        assert m.melody_above_accompaniment_ratio(mel, acc) == 1.0
        assert m.exposed_semitone_clash_ratio(mel, acc, TEMPO) == 0.0


def test_final_chord_rings():
    ring = [n(4, 4, 48), n(4.02, 3.98, 52)]
    assert m.final_chord_rings(ring, 2, TEMPO)
    restruck = [n(4, 2, 48), n(6, 2, 48)]
    assert not m.final_chord_rings(restruck, 2, TEMPO)
    assert not m.final_chord_rings([], 2, TEMPO)


def test_essential_tone_coverage():
    d = shape("D")  # xx0232: D3 A3 D4 F#4
    full = [n(0, 4, 50), n(0, 4, 57), n(0, 4, 62), n(0, 4, 66)]
    assert m.essential_tone_coverage([], full, [d], 1, TEMPO, STANDARD_EADGBE) == [True]
    no_third = [n(0, 4, 50), n(0, 4, 57), n(0, 4, 62)]
    assert m.essential_tone_coverage([], no_third, [d], 1, TEMPO, STANDARD_EADGBE) == [False]
    melody_has_it = [n(0, 4, 78)]  # F#5 in the melody supplies the third
    assert m.essential_tone_coverage(melody_has_it, no_third, [d], 1, TEMPO, STANDARD_EADGBE) == [
        True
    ]


def test_hand_shift_rate():
    assert m.hand_shift_rate([5, 7, 8, 9]) == pytest.approx(0.0)
    assert m.hand_shift_rate([5, 7, 12, 13]) == pytest.approx(1 / 3)
    assert m.hand_shift_rate([0, 0, 5]) is None  # open strings do not count


def test_nearest_pitch_proximity():
    v = [shape(c) for c in ("C", "C", "Am")]
    d = m.nearest_pitch_proximity(v, STANDARD_EADGBE)
    assert d is not None and 0.0 < d < 2.0
    assert m.nearest_pitch_proximity(v[:2], STANDARD_EADGBE) is None  # same chord only


def test_nearest_pitch_proximity_is_not_a_voice_leading_score():
    # Documented blind spot: a 3-note shape collapsing onto one doubled pitch
    # scores 0 even though two voices vanished.
    pos = [TabPosition(string=s, fret=-1) for s in range(6)]
    wide = VoicedChord(
        ChordSymbol.parse("C"),
        (
            TabPosition(0, -1),
            TabPosition(1, 3),
            TabPosition(2, 2),
            TabPosition(3, 0),
            *pos[4:],
        ),
    )  # C3 E3 G3
    collapsed = VoicedChord(
        ChordSymbol.parse("Em"),
        (TabPosition(0, -1), TabPosition(1, -1), TabPosition(2, 2), *pos[3:]),
    )  # E3 only
    assert m.nearest_pitch_proximity([wide, collapsed], STANDARD_EADGBE) == 0.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/unit/test_compose_harmony.py tests/unit/test_compose_metrics.py -q`
Expected: collection ERRORs, `ModuleNotFoundError: No module named 'music_decoder.compose.harmony'`.

- [ ] **Step 3: Write the implementation**

Create `src/music_decoder/compose/harmony.py`:

```python
"""Pitch-class helpers shared by the melody generator, the accompaniment and
the musicality metrics: scale and chord pitch classes, each chord's
identity-defining ("essential") tones, and the scale adapted to a chord."""

from __future__ import annotations

from music_decoder.types import VALID_TONICS, ChordSymbol, Scale

# Pitch-class sets relative to the tonic
_MAJOR_PCS = (0, 2, 4, 5, 7, 9, 11)
_MINOR_PCS = (0, 2, 3, 5, 7, 8, 10)  # natural minor

# Chord quality -> intervals above the root
CHORD_INTERVALS: dict[str, tuple[int, ...]] = {
    "maj": (0, 4, 7),
    "min": (0, 3, 7),
    "maj7": (0, 4, 7, 11),
    "min7": (0, 3, 7, 10),
    "7": (0, 4, 7, 10),
    "dim": (0, 3, 6),
    "sus4": (0, 5, 7),
    "aug": (0, 4, 8),
}

# The tones that give each quality its identity (third, seventh, altered fifth,
# suspended fourth); a sounding chord missing one of these is ambiguous.
ESSENTIAL_INTERVALS: dict[str, tuple[int, ...]] = {
    "maj": (4,),
    "min": (3,),
    "maj7": (4, 11),
    "min7": (3, 10),
    "7": (4, 10),
    "dim": (3, 6),
    "sus4": (5,),
    "aug": (4, 8),
}

# For a chord tone outside the scale, which neighbour is the scale degree it
# chromatically alters, by the tone's interval above the chord root: a major
# third or major seventh is a raised degree (G# in E7 replaces G), a minor
# third / minor seventh / diminished fifth / fourth is a lowered one (Ab in Fm
# replaces A in C major). Roots and perfect fifths are ambiguous: see below.
_ALTERED_NEIGHBOUR: dict[int, int] = {4: -1, 11: -1, 8: -1, 3: 1, 10: 1, 6: 1, 5: 1}


def tonic_pc(scale: Scale) -> int:
    return VALID_TONICS.index(scale.tonic)


def leading_tone_pc(scale: Scale) -> int:
    """The pitch class a semitone below the tonic."""
    return (tonic_pc(scale) + 11) % 12


def scale_pcs(scale: Scale) -> tuple[int, ...]:
    base = _MAJOR_PCS if scale.mode == "major" else _MINOR_PCS
    return tuple((p + tonic_pc(scale)) % 12 for p in base)


def _root_pc(chord: ChordSymbol) -> int:
    return VALID_TONICS.index(chord.root)


def chord_pcs(chord: ChordSymbol) -> tuple[int, ...]:
    return tuple((i + _root_pc(chord)) % 12 for i in CHORD_INTERVALS[chord.quality])


def essential_pcs(chord: ChordSymbol) -> frozenset[int]:
    return frozenset((i + _root_pc(chord)) % 12 for i in ESSENTIAL_INTERVALS[chord.quality])


def chord_scale_pcs(scale: Scale, chord: ChordSymbol) -> frozenset[int]:
    """The scale with each out-of-scale chord tone replacing the one scale
    degree it alters; every other degree (including the tonic) is kept.

    E7 in A minor gives A harmonic minor (G -> G#, A kept); D7 in C major
    keeps G and replaces F with F#. For a foreign root or perfect fifth the
    altered degree is the neighbour in the scale; if both neighbours are,
    the upper one (flat-side borrowing such as bII or bVI is the common case).
    """
    tones = chord_pcs(chord)
    chord_set = set(tones)
    scale_set = set(scale_pcs(scale))
    pool = set(scale_set)
    root = _root_pc(chord)
    for pc in tones:
        if pc in scale_set:
            continue
        direction = _ALTERED_NEIGHBOUR.get((pc - root) % 12)
        if direction is None:
            upper = (pc + 1) % 12
            direction = 1 if upper in scale_set and upper not in chord_set else -1
        neighbour = (pc + direction) % 12
        if neighbour not in chord_set:
            pool.discard(neighbour)
        pool.add(pc)
    return frozenset(pool)
```

Create `src/music_decoder/compose/metrics.py`:

```python
"""Objective musicality metrics for composer output.

Pure functions over melody / accompaniment notes. They are applied to the
pre-2026-11 composer (baseline, ``tests/fixtures/compose/musicality_baseline.json``)
and to the current one on the same inputs by
``music_decoder.evaluation.compose_musicality``; definitions follow
``docs/superpowers/specs/2026-11-06-compose-improvements-design.md`` section 4.

Times are in seconds and ``tempo_bpm`` fixes the 4/4 beat grid. Ratio metrics
return ``None`` when nothing qualifies to be counted, which is distinct from
a score of 0.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from itertools import pairwise

from music_decoder.compose.harmony import (
    chord_pcs,
    essential_pcs,
    leading_tone_pc,
    scale_pcs,
    tonic_pc,
)
from music_decoder.compose.rhythm import phrase_layout
from music_decoder.types import ChordSymbol, Note, Scale, Tuning, VoicedChord

_GRID_TOL_S = 1e-3  # onset counts as "on the grid" within 1 ms
_STROKE_WINDOW_S = 0.06  # onsets this close to a stroke's first onset belong to it
_OVERLAP_EPS_S = 1e-4  # notes must share more than this much time to overlap


def _beat_s(tempo_bpm: float) -> float:
    return 60.0 / tempo_bpm


def chord_at(
    t: float, progression: Sequence[ChordSymbol], bars_per_chord: int, tempo_bpm: float
) -> ChordSymbol:
    """Chord sounding at time ``t`` (bar index // bars_per_chord, clamped)."""
    bar = math.floor(t / (4.0 * _beat_s(tempo_bpm)) + 1e-6)
    return progression[min(bar // bars_per_chord, len(progression) - 1)]


def metric_position(start_s: float, tempo_bpm: float) -> int:
    """Eighth-note slot within the bar (0-7); 0 and 4 are beats 1 and 3."""
    return round(start_s / (_beat_s(tempo_bpm) / 2.0)) % 8


def _is_exposed(n: Note, tempo_bpm: float) -> bool:
    """On beat 1 or 3, or at least 0.9 beats long."""
    return metric_position(n.start_s, tempo_bpm) in (0, 4) or (
        n.end_s - n.start_s >= 0.9 * _beat_s(tempo_bpm)
    )


def intervals(melody: Sequence[Note]) -> list[int]:
    return [b.pitch - a.pitch for a, b in pairwise(melody)]


def _ratio(hits: int, total: int) -> float | None:
    return hits / total if total else None


# ---- melody ---------------------------------------------------------------


def strong_beat_chord_tone_ratio(
    melody: Sequence[Note],
    progression: Sequence[ChordSymbol],
    bars_per_chord: int,
    tempo_bpm: float,
) -> float | None:
    """M1: share of beat-1/beat-3 onsets whose pitch class is a chord tone."""
    total = hits = 0
    for n in melody:
        if metric_position(n.start_s, tempo_bpm) not in (0, 4):
            continue
        total += 1
        chord = chord_at(n.start_s, progression, bars_per_chord, tempo_bpm)
        hits += n.pitch % 12 in chord_pcs(chord)
    return _ratio(hits, total)


def step_ratio(melody: Sequence[Note]) -> float | None:
    """M2: share of intervals of 1-2 semitones."""
    ivs = intervals(melody)
    return _ratio(sum(1 <= abs(i) <= 2 for i in ivs), len(ivs))


def repeat_ratio(melody: Sequence[Note]) -> float | None:
    """M3: share of repeated-pitch intervals."""
    ivs = intervals(melody)
    return _ratio(sum(i == 0 for i in ivs), len(ivs))


def large_leap_ratio(melody: Sequence[Note]) -> float | None:
    """M4: share of intervals larger than a perfect fifth."""
    ivs = intervals(melody)
    return _ratio(sum(abs(i) > 7 for i in ivs), len(ivs))


def max_interval(melody: Sequence[Note]) -> int:
    return max((abs(i) for i in intervals(melody)), default=0)


def leap_recovery_counts(melody: Sequence[Note]) -> tuple[int, int]:
    """M5: (leaps >= 5 semitones followed by another note, of those recovered).

    Recovered = the next interval moves the opposite way by 1-2 semitones.
    Counts, so callers can pool them across compositions.
    """
    ivs = intervals(melody)
    leaps = recovered = 0
    for i, j in pairwise(ivs):
        if abs(i) >= 5:
            leaps += 1
            recovered += i * j < 0 and 1 <= abs(j) <= 2
    return leaps, recovered


def ambitus(melody: Sequence[Note]) -> int:
    """M6: melody range in semitones."""
    pitches = [n.pitch for n in melody]
    return max(pitches) - min(pitches) if pitches else 0


def final_resolution(
    melody: Sequence[Note], scale: Scale, progression: Sequence[ChordSymbol]
) -> bool:
    """M7: last note is the tonic if the final chord holds it, else the final root."""
    if not melody:
        return False
    last_chord = progression[-1]
    if tonic_pc(scale) in chord_pcs(last_chord):
        target = tonic_pc(scale)
    else:
        target = chord_pcs(last_chord)[0]
    return melody[-1].pitch % 12 == target


def final_note_beats(melody: Sequence[Note], tempo_bpm: float) -> float:
    """M8: sounding length of the last melody note in beats."""
    if not melody:
        return 0.0
    return (melody[-1].end_s - melody[-1].start_s) / _beat_s(tempo_bpm)


def _durations_in_eighths(melody: Sequence[Note], tempo_bpm: float) -> list[int]:
    eighth = _beat_s(tempo_bpm) / 2.0
    out = [round((b.start_s - a.start_s) / eighth) for a, b in pairwise(melody)]
    if melody:
        out.append(round((melody[-1].end_s - melody[-1].start_s) / eighth))
    return out


def distinct_durations(melody: Sequence[Note], tempo_bpm: float) -> int:
    """M9: number of distinct note lengths (onset spacing, in eighths)."""
    return len(set(_durations_in_eighths(melody, tempo_bpm)))


def ioi_entropy(melody: Sequence[Note], tempo_bpm: float) -> float:
    """M10: Shannon entropy (bits) of inter-onset intervals quantised to eighths."""
    iois = _durations_in_eighths(melody, tempo_bpm)[:-1]
    if not iois:
        return 0.0
    counts = Counter(iois)
    total = len(iois)
    return -sum(c / total * math.log2(c / total) for c in counts.values())


def _bar_notes(melody: Sequence[Note], bar: int, tempo_bpm: float) -> list[Note]:
    bar_s = 4.0 * _beat_s(tempo_bpm)
    lo, hi = bar * bar_s - 1e-6, (bar + 1) * bar_s - 1e-6
    return [n for n in melody if lo <= n.start_s < hi]


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _echo_pairs(
    melody: Sequence[Note], tempo_bpm: float, n_bars: int
) -> list[tuple[list[Note], list[Note]]]:
    return [
        (_bar_notes(melody, source, tempo_bpm), _bar_notes(melody, target, tempo_bpm))
        for source, target in phrase_layout(n_bars).echo_pairs
    ]


def motif_pair_coverage(melody: Sequence[Note], tempo_bpm: float, n_bars: int) -> float | None:
    """M11a: share of the layout's echo pairs whose two bars have the same
    number of notes (>= 2), i.e. that can carry a repeated motif."""
    pairs = _echo_pairs(melody, tempo_bpm, n_bars)
    covered = sum(len(a) == len(b) >= 2 for a, b in pairs)
    return _ratio(covered, len(pairs))


def motif_contour_match(melody: Sequence[Note], tempo_bpm: float, n_bars: int) -> float | None:
    """M11b: interval-direction agreement, averaged over the covered echo pairs."""
    scores: list[float] = []
    for a, b in _echo_pairs(melody, tempo_bpm, n_bars):
        if len(a) != len(b) or len(a) < 2:
            continue
        sa = [_sign(i) for i in intervals(a)]
        sb = [_sign(i) for i in intervals(b)]
        scores.append(sum(x == y for x, y in zip(sa, sb, strict=True)) / len(sa))
    return sum(scores) / len(scores) if scores else None


def phrase_end_long_ratio(melody: Sequence[Note], tempo_bpm: float, n_bars: int) -> float:
    """M12: share of the layout's phrase-final bars whose last note lasts >= 1.5 beats."""
    beat = _beat_s(tempo_bpm)
    cadences = phrase_layout(n_bars).cadence_bars
    hits = 0
    for bar in cadences:
        notes = _bar_notes(melody, bar, tempo_bpm)
        hits += bool(notes) and notes[-1].end_s - notes[-1].start_s >= 1.5 * beat
    return hits / len(cadences)


def exposed_chromatic_clash_ratio(
    melody: Sequence[Note],
    scale: Scale,
    progression: Sequence[ChordSymbol],
    bars_per_chord: int,
    tempo_bpm: float,
) -> float | None:
    """M13: over chords with an out-of-scale tone, the share of exposed melody
    notes (beat 1/3 or >= 0.9 beat) that are non-chord tones a semitone from
    one of those foreign tones (e.g. a held G against E7's G# in A minor).
    Chord tones and short unaccented passing/neighbour notes never count.
    ``None`` when no exposed note sounds over such a chord."""
    scale_set = set(scale_pcs(scale))
    total = hits = 0
    for n in melody:
        chord = chord_at(n.start_s, progression, bars_per_chord, tempo_bpm)
        tones = set(chord_pcs(chord))
        foreign = tones - scale_set
        if not foreign or not _is_exposed(n, tempo_bpm):
            continue
        total += 1
        pc = n.pitch % 12
        hits += pc not in tones and any((pc - f) % 12 in (1, 11) for f in foreign)
    return _ratio(hits, total)


def leading_tone_resolution_counts(
    melody: Sequence[Note],
    scale: Scale,
    progression: Sequence[ChordSymbol],
    bars_per_chord: int,
    tempo_bpm: float,
) -> tuple[int, int]:
    """M15: (leading tones sounding as a chord tone that are followed by a note
    in a new, tonic-containing chord; of those, resolved up a semitone)."""
    lead, tonic = leading_tone_pc(scale), tonic_pc(scale)
    eligible = resolved = 0
    for a, b in pairwise(melody):
        chord_a = chord_at(a.start_s, progression, bars_per_chord, tempo_bpm)
        chord_b = chord_at(b.start_s, progression, bars_per_chord, tempo_bpm)
        if a.pitch % 12 != lead or lead not in chord_pcs(chord_a):
            continue
        if chord_b == chord_a or tonic not in chord_pcs(chord_b):
            continue
        eligible += 1
        resolved += b.pitch == a.pitch + 1
    return eligible, resolved


def onset_grid_ratio(onsets: Sequence[float], tempo_bpm: float, per_beat: int) -> float | None:
    """M14 / A1: share of onsets within 1 ms of a 1/``per_beat``-beat grid."""
    step = _beat_s(tempo_bpm) / per_beat
    hits = sum(abs(t / step - round(t / step)) * step <= _GRID_TOL_S for t in onsets)
    return _ratio(hits, len(onsets))


# ---- accompaniment ----------------------------------------------------------


def stroke_heads(onsets: Sequence[float]) -> list[float]:
    """First onset of each stroke (onsets within 60 ms of a stroke's first onset)."""
    heads: list[float] = []
    for t in sorted(onsets):
        if not heads or t - heads[-1] > _STROKE_WINDOW_S:
            heads.append(t)
    return heads


def _overlapping(accompaniment: Sequence[Note], n: Note) -> list[int]:
    """Accompaniment pitches sounding at any time during melody note ``n``."""
    return [
        a.pitch
        for a in accompaniment
        if a.start_s < n.end_s - _OVERLAP_EPS_S and a.end_s > n.start_s + _OVERLAP_EPS_S
    ]


def melody_above_accompaniment_ratio(
    melody: Sequence[Note], accompaniment: Sequence[Note]
) -> float | None:
    """A2: share of melody notes higher than every accompaniment pitch that
    overlaps them at any point (late-entering strum voices included)."""
    hits = 0
    for n in melody:
        overlapping = _overlapping(accompaniment, n)
        hits += not overlapping or n.pitch > max(overlapping)
    return _ratio(hits, len(melody))


def exposed_semitone_clash_ratio(
    melody: Sequence[Note], accompaniment: Sequence[Note], tempo_bpm: float
) -> float | None:
    """A3: share of exposed melody notes (beat 1/3 or >= 0.9 beat) that overlap
    an accompaniment pitch a minor second / major seventh away (pitch-class
    distance 1 or 11) at any point while they sound."""
    total = hits = 0
    for n in melody:
        if not _is_exposed(n, tempo_bpm):
            continue
        total += 1
        hits += any((n.pitch - p) % 12 in (1, 11) for p in _overlapping(accompaniment, n))
    return _ratio(hits, total)


def final_chord_rings(accompaniment: Sequence[Note], n_bars: int, tempo_bpm: float) -> bool:
    """A4: final-bar accompaniment is one attack (within 60 ms) ringing to the bar end."""
    bar_s = 4.0 * _beat_s(tempo_bpm)
    start, end = (n_bars - 1) * bar_s, n_bars * bar_s
    last = [a for a in accompaniment if a.start_s >= start - 1e-6]
    if not last:
        return False
    return all(a.start_s - start <= _STROKE_WINDOW_S for a in last) and all(
        a.end_s >= end - _GRID_TOL_S for a in last
    )


def _voicing_pitches(voicing: VoicedChord, tuning: Tuning) -> list[int]:
    return [tuning.open_pitches[p.string] + p.fret for p in voicing.positions if p.fret >= 0]


def essential_tone_coverage(
    melody: Sequence[Note],
    accompaniment: Sequence[Note],
    voicings: Sequence[VoicedChord],
    bars_per_chord: int,
    tempo_bpm: float,
    tuning: Tuning,
) -> list[bool]:
    """H1: per bar, whether every essential tone (third, seventh, altered
    fifth, suspended fourth) that the chord's shape contains is heard during
    the bar in the accompaniment or the melody."""
    bar_s = 4.0 * _beat_s(tempo_bpm)
    out: list[bool] = []
    for bar in range(len(voicings) * bars_per_chord):
        voicing = voicings[bar // bars_per_chord]
        shape_pcs = {p % 12 for p in _voicing_pitches(voicing, tuning)}
        required = essential_pcs(voicing.chord) & shape_pcs
        t0, t1 = bar * bar_s, (bar + 1) * bar_s
        heard = {
            n.pitch % 12
            for n in (*accompaniment, *melody)
            if n.start_s < t1 - _OVERLAP_EPS_S and n.end_s > t0 + _OVERLAP_EPS_S
        }
        out.append(required <= heard)
    return out


# ---- tab and voicings -----------------------------------------------------


def hand_shift_rate(frets: Sequence[int]) -> float | None:
    """T2: hand-position shifts per fretted transition (window spans <= 4 frets)."""
    fretted = [f for f in frets if f > 0]
    if len(fretted) < 2:
        return None
    lo = hi = fretted[0]
    shifts = 0
    for f in fretted[1:]:
        new_lo, new_hi = min(lo, f), max(hi, f)
        if new_hi - new_lo > 4:
            shifts += 1
            lo = hi = f
        else:
            lo, hi = new_lo, new_hi
    return shifts / (len(fretted) - 1)


def nearest_pitch_proximity(voicings: Sequence[VoicedChord], tuning: Tuning) -> float | None:
    """V1: mean distance (semitones) from each pitch of a chord shape to the
    nearest pitch of the previous, different shape.

    A proximity statistic only, not a voice-leading score: several voices may
    share one nearest source, dropped voices cost nothing and crossings are
    ignored. Used solely as a guard that the (unchanged) shapes stay close.
    """
    moves: list[float] = []
    for a, b in pairwise(voicings):
        pa, pb = _voicing_pitches(a, tuning), _voicing_pitches(b, tuning)
        if a.chord == b.chord or not pa or not pb:
            continue
        moves.append(sum(min(abs(x - y) for y in pa) for x in pb) / len(pb))
    return sum(moves) / len(moves) if moves else None
```

- [ ] **Step 4: Run the tests and gates**

Run: `pytest tests/unit/test_compose_harmony.py tests/unit/test_compose_metrics.py -q`. Expected: `38 passed` (15 + 23).
Run: `ruff check src tests && ruff format --check src tests && mypy --strict src`. Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add src/music_decoder/compose/harmony.py src/music_decoder/compose/metrics.py \
  tests/unit/test_compose_harmony.py tests/unit/test_compose_metrics.py
git commit -m "feat(compose): harmony helpers and musicality metrics

Role-based chord-scale rule (an altered chord tone replaces only the degree it
alters), essential tones, and pure metric functions: overlap-based
accompaniment metrics, phrase-layout-aware motif/cadence metrics, chromatic
clash, leading-tone resolution, essential-tone coverage, tab hand shifts and
a nearest-pitch proximity guard.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Pure `arrange()` refactor, evaluation matrices and the committed baseline

This is a **behaviour-preserving** refactor of the **current** composer, followed by
recording its metrics. Do it before any generator change: the baseline must come
from the old melody, accompaniment and tab.

**Files:**
- Modify (full rewrite, same behaviour): `src/music_decoder/compose/api.py`
- Create: `src/music_decoder/evaluation/compose_musicality.py`, `scripts/compose_musicality_report.py`, `tests/fixtures/compose/musicality_baseline.json` (generated)
- Modify (full rewrite, keeping the 3 existing tests at the top unchanged): `tests/unit/test_compose_api.py`
- Test: `tests/unit/test_compose_evaluation.py`

**Interfaces:**
- Consumes: `metrics` (Task 2); the current `generate_melody`, `_accompaniment_notes`, `voicings_for`.
- Produces, in `music_decoder.compose.api`:
  - `@dataclass(frozen=True) Arrangement(melody: tuple[Note, ...], voicings: tuple[VoicedChord, ...], accompaniment: tuple[Note, ...], melody_tab: tuple[TabbedNote, ...])`
  - `arrange(scale, progression, *, bars_per_chord=1, tempo_bpm=100.0, style="fingerstyle", tuning=STANDARD_EADGBE, seed=None) -> Arrangement`
  - `accompany(voicings, melody, *, bars_per_chord, tempo_bpm, style, tuning) -> tuple[Note, ...]`
  - `_lowest_fret_position(pitch, tuning) -> TabPosition | None`
  - `_melody_tab(melody, tuning)`
- Produces, in `music_decoder.evaluation.compose_musicality`:
  - `STYLES`, `TEMPO_BPM`, `VALIDATION_SEEDS`, `EDGE_SEEDS`
  - `realize(scale, numeral) -> ChordSymbol`
  - `validation_inputs()` (400 inputs) and `edge_inputs()` (80)
  - `Case`, `build_cases(inputs) -> list[Case]`, `case_metrics(case) -> dict[str, Value]`
  - `summarize(cases) -> dict[group, dict[metric, dict[stat, float]]]`; the groups are overall, major, minor, bars1 and bars2
  - `evaluate() -> {"validation": ..., "edge": ...}`

- [ ] **Step 1: Record the current `compose()` output (golden, not committed)**

```bash
python - > "$TMPDIR/compose_golden_before.txt" <<'EOF'
import tempfile
from pathlib import Path
from unittest import mock
from music_decoder import ChordSymbol, Scale, compose
with mock.patch("music_decoder.compose.api.render_wav"):
    r = compose(Scale("A", "minor"), [ChordSymbol.parse(c) for c in "Am Dm E7 Am".split()],
                bars_per_chord=2, tempo_bpm=90.0, style="strum", seed=7,
                out_dir=Path(tempfile.mkdtemp()))
print(r.ascii_tab)
print([(n.pitch, round(n.start_s, 6)) for n in r.melody_notes])
print([v.positions for v in r.chord_voicings])
EOF
```

- [ ] **Step 2: Write the failing tests**

Create `tests/unit/test_compose_evaluation.py`:

```python
import json
from pathlib import Path

import pytest

from music_decoder.evaluation import compose_musicality as ev
from music_decoder.types import ChordSymbol, Scale

BASELINE = Path(__file__).parents[1] / "fixtures" / "compose" / "musicality_baseline.json"


@pytest.mark.parametrize(
    ("key", "mode", "numeral", "label"),
    [
        ("C", "major", "I", "C"),
        ("C", "major", "vi7", "Am7"),
        ("C", "major", "V7", "G7"),
        ("G", "major", "Imaj7", "Gmaj7"),
        ("A", "minor", "V7", "E7"),
        ("A", "minor", "VI", "F"),
        ("A", "minor", "iidim", "Bdim"),
        ("F#", "minor", "v", "C#m"),
    ],
)
def test_realize(key, mode, numeral, label):
    assert ev.realize(Scale(key, mode), numeral) == ChordSymbol.parse(label)


def test_matrices_have_the_documented_size_and_held_out_seeds():
    validation = ev.validation_inputs()
    assert len(validation) == 400
    assert len(ev.edge_inputs()) == 80
    assert min(seed for *_, seed in validation) >= 1000


def test_case_metrics_cover_every_style():
    (case,) = ev.build_cases(ev.edge_inputs()[:1])
    row = ev.case_metrics(case)
    for style in ev.STYLES:
        for prefix in ("A1", "A2", "A3", "A4", "H1", "H1final"):
            assert f"{prefix}.{style}" in row
    assert {"M1", "M7", "M11a", "M12", "M13", "T1", "T2", "V1"} <= set(row)


def test_summarize_groups_and_pools_counts():
    summary = ev.summarize(ev.build_cases(ev.validation_inputs()[:10]))
    assert {"overall", "major", "bars1"} <= set(summary)
    assert set(summary["overall"]["M5"]) == {"pooled", "n"}
    assert set(summary["overall"]["M2"]) == {"mean", "min", "max"}


def test_committed_baseline_has_both_matrices():
    baseline = json.loads(BASELINE.read_text())
    assert {"source", "validation", "edge"} <= set(baseline)
    assert baseline["validation"]["overall"]["M9"]["mean"] == 1.0  # uniform quarters
```

Replace the whole of `tests/unit/test_compose_api.py` with this content (the first three tests are the existing ones):

```python
from pathlib import Path
from unittest import mock

import pytest

from music_decoder.compose.api import arrange, compose
from music_decoder.errors import InvalidProgressionError, InvalidScaleError
from music_decoder.types import ChordSymbol, Scale


def test_compose_returns_paths_and_data(tmp_path):
    with mock.patch("music_decoder.compose.api.render_wav") as render:
        render.side_effect = lambda midi_path, out_path: out_path.write_bytes(b"RIFFWAV")
        result = compose(
            scale=Scale("C", "major"),
            progression=[ChordSymbol.parse(c) for c in ("Cmaj7", "Am7", "Dm7", "G7")],
            tempo_bpm=120.0,
            bars_per_chord=1,
            style="strum",
            seed=7,
            out_dir=tmp_path,
        )
    assert result.midi_path.exists()
    assert result.wav_path.exists()
    assert result.ascii_tab.strip() != ""
    assert len(result.melody_notes) > 0
    assert len(result.chord_voicings) == 4
    assert result.metadata["seed"] == 7


def test_compose_invalid_progression_raises():
    with pytest.raises(InvalidProgressionError):
        compose(
            scale=Scale("C", "major"),
            progression=[],  # empty
            out_dir=Path("/tmp"),
        )


def test_compose_invalid_scale_raises():
    # non-Literal mode is only constructible by going around the parser:
    bad = Scale(tonic="C", mode="dorian")  # type: ignore[arg-type]
    with pytest.raises(InvalidScaleError):
        compose(scale=bad, progression=[ChordSymbol.parse("C")], out_dir=Path("/tmp"))


# ---- arrange(): the filesystem-free core of compose() ----------------------

_PROG = [ChordSymbol.parse(c) for c in ("Cmaj7", "Am7", "Dm7", "G7")]


def test_arrange_is_deterministic_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    a = arrange(Scale("C", "major"), _PROG, tempo_bpm=120.0, style="strum", seed=3)
    b = arrange(Scale("C", "major"), _PROG, tempo_bpm=120.0, style="strum", seed=3)
    assert a == b
    assert list(tmp_path.iterdir()) == []
    assert len(a.voicings) == 4 and a.accompaniment
    assert [tn.note for tn in a.melody_tab] == list(a.melody)


def test_compose_returns_the_arrangement(tmp_path):
    with mock.patch("music_decoder.compose.api.render_wav"):
        result = compose(scale=Scale("C", "major"), progression=_PROG, seed=1, out_dir=tmp_path)
    arr = arrange(Scale("C", "major"), _PROG, seed=1)
    assert result.melody_notes == arr.melody
    assert result.chord_voicings == arr.voicings
```

Run: `pytest tests/unit/test_compose_api.py tests/unit/test_compose_evaluation.py -q`
Expected: collection ERRORs, `ImportError: cannot import name 'arrange' from 'music_decoder.compose.api'` and `ModuleNotFoundError: ... evaluation.compose_musicality`.

- [ ] **Step 3: Refactor `compose/api.py` (same behaviour)**

Replace the whole of `src/music_decoder/compose/api.py` with:

```python
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
```

Verify byte-identical behaviour by re-running the Step 1 snippet into `"$TMPDIR/compose_golden_after.txt"`, then:
Run: `diff "$TMPDIR/compose_golden_before.txt" "$TMPDIR/compose_golden_after.txt" && echo IDENTICAL`
Expected: `IDENTICAL`.

- [ ] **Step 4: Add the evaluation module and the report script**

Create `src/music_decoder/evaluation/compose_musicality.py`:

```python
"""Musicality evaluation of the composer over fixed, named input matrices.

Shared by ``scripts/compose_musicality_report.py`` (which wrote the committed
pre-change baseline, ``tests/fixtures/compose/musicality_baseline.json``) and
``tests/unit/test_compose_musicality.py`` (which compares the current
composer against it on exactly the same inputs).

Seeds 1000+ are disjoint from the seeds 0-19 used while tuning the
generator's weights, so the matrices are held-out validation data.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Literal, cast

from music_decoder.compose import metrics
from music_decoder.compose.api import Arrangement, accompany, arrange
from music_decoder.tabs.tuning import STANDARD_EADGBE
from music_decoder.types import VALID_TONICS, ChordSymbol, Note, Scale

Style = Literal["arpeggio", "strum", "fingerstyle"]
STYLES: tuple[Style, ...] = ("arpeggio", "strum", "fingerstyle")
TEMPO_BPM = 120.0
VALIDATION_SEEDS = tuple(range(1000, 1005))
EDGE_SEEDS = tuple(range(1000, 1010))
MAJOR_KEYS = ("C", "G", "A#", "E")
MINOR_KEYS = ("A", "E", "C", "F#")
MAJOR_PROGRESSIONS = (
    ("I", "V", "vi", "IV"),
    ("Imaj7", "vi7", "ii7", "V7"),
    ("I", "IV", "V", "I"),
    ("ii7", "V7", "Imaj7", "Imaj7"),
    ("I", "vi", "IV", "V"),
)
MINOR_PROGRESSIONS = (
    ("i", "iv", "V7", "i"),
    ("i", "VI", "III", "VII"),
    ("i", "iv", "v", "i"),
    ("i", "VII", "VI", "V"),
    ("iidim", "V7", "i", "i"),
)
# Odd lengths (3/5/7/9 bars), a single chord, secondary dominants and a
# borrowed chord; one bar per chord.
EDGE_PROGRESSIONS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("C", "major", ("C", "F", "G")),
    ("G", "major", ("G", "C", "D", "Em", "G")),
    ("A", "minor", ("Am", "Dm", "E7", "Am", "F", "Dm", "E7")),
    ("D", "major", ("D", "Bm", "G", "A", "D", "G", "Em", "A", "D")),
    ("D", "major", ("D",)),
    ("C", "major", ("C", "D7", "G", "C")),
    ("C", "major", ("C", "Fm", "C", "G7")),
    ("E", "minor", ("Em", "B7", "Em", "C", "B7", "Em")),
)

_MAJOR_DEGREES = {"I": 0, "ii": 2, "iii": 4, "IV": 5, "V": 7, "vi": 9, "vii": 11}
_MINOR_DEGREES = {"i": 0, "ii": 2, "III": 3, "iv": 5, "v": 7, "V": 7, "VI": 8, "VII": 10}
_LABEL_SUFFIX = {"maj": "", "min": "m", "min7": "m7", "7": "7", "maj7": "maj7", "dim": "dim"}


def realize(scale: Scale, numeral: str) -> ChordSymbol:
    """Roman numeral (``I``, ``vi7``, ``V7``, ``iidim``…) to a chord in ``scale``."""
    match = re.fullmatch(r"([ivIV]+)(.*)", numeral)
    if match is None:
        raise ValueError(f"bad numeral {numeral!r}")
    degree, suffix = match.groups()
    table = _MAJOR_DEGREES if scale.mode == "major" else _MINOR_DEGREES
    root = VALID_TONICS[(VALID_TONICS.index(scale.tonic) + table[degree]) % 12]
    upper = degree.isupper()
    if suffix in ("dim", "maj7"):
        quality = suffix
    elif suffix == "7":
        quality = "7" if upper else "min7"
    else:
        quality = "maj" if upper else "min"
    return ChordSymbol.parse(root + _LABEL_SUFFIX[quality])


@dataclass(frozen=True)
class Case:
    """One generated composition plus its accompaniment in every style."""

    scale: Scale
    progression: tuple[ChordSymbol, ...]
    bars_per_chord: int
    seed: int
    arrangement: Arrangement
    accompaniment: dict[str, tuple[Note, ...]]

    @property
    def n_bars(self) -> int:
        return len(self.progression) * self.bars_per_chord


Inputs = tuple[Scale, tuple[ChordSymbol, ...], int, int]


def validation_inputs() -> list[Inputs]:
    out: list[Inputs] = []
    for mode, keys, progressions in (
        ("major", MAJOR_KEYS, MAJOR_PROGRESSIONS),
        ("minor", MINOR_KEYS, MINOR_PROGRESSIONS),
    ):
        for key in keys:
            scale = Scale(key, cast(Literal["major", "minor"], mode))
            for numerals in progressions:
                progression = tuple(realize(scale, n) for n in numerals)
                for bars in (1, 2):
                    out.extend((scale, progression, bars, seed) for seed in VALIDATION_SEEDS)
    return out


def edge_inputs() -> list[Inputs]:
    out: list[Inputs] = []
    for key, mode, labels in EDGE_PROGRESSIONS:
        scale = Scale(key, cast(Literal["major", "minor"], mode))
        progression = tuple(ChordSymbol.parse(c) for c in labels)
        out.extend((scale, progression, 1, seed) for seed in EDGE_SEEDS)
    return out


def build_cases(inputs: Iterable[Inputs]) -> list[Case]:
    cases: list[Case] = []
    for scale, progression, bars, seed in inputs:
        arr = arrange(
            scale,
            progression,
            bars_per_chord=bars,
            tempo_bpm=TEMPO_BPM,
            style="fingerstyle",
            seed=seed,
        )
        acc: dict[str, tuple[Note, ...]] = {
            style: arr.accompaniment
            if style == "fingerstyle"
            else accompany(
                arr.voicings,
                arr.melody,
                bars_per_chord=bars,
                tempo_bpm=TEMPO_BPM,
                style=style,
                tuning=STANDARD_EADGBE,
            )
            for style in STYLES
        }
        cases.append(Case(scale, progression, bars, seed, arr, acc))
    return cases


Value = float | int | bool | None


def case_metrics(case: Case) -> dict[str, Value]:
    """Every metric for one case; accompaniment metrics are keyed ``A2.strum`` etc."""
    mel, prog = case.arrangement.melody, case.progression
    bars, tempo = case.bars_per_chord, TEMPO_BPM
    frets = [tn.position.fret for tn in case.arrangement.melody_tab]
    leaps, recovered = metrics.leap_recovery_counts(mel)
    lt_eligible, lt_resolved = metrics.leading_tone_resolution_counts(
        mel, case.scale, prog, bars, tempo
    )
    out: dict[str, Value] = {
        "M1": metrics.strong_beat_chord_tone_ratio(mel, prog, bars, tempo),
        "M2": metrics.step_ratio(mel),
        "M3": metrics.repeat_ratio(mel),
        "M4": metrics.large_leap_ratio(mel),
        "M4max": metrics.max_interval(mel),
        "M5.leaps": leaps,
        "M5.recovered": recovered,
        "M6": metrics.ambitus(mel),
        "M7": metrics.final_resolution(mel, case.scale, prog),
        "M8": metrics.final_note_beats(mel, tempo),
        "M9": metrics.distinct_durations(mel, tempo),
        "M10": metrics.ioi_entropy(mel, tempo),
        "M11a": metrics.motif_pair_coverage(mel, tempo, case.n_bars),
        "M11b": metrics.motif_contour_match(mel, tempo, case.n_bars),
        "M12": metrics.phrase_end_long_ratio(mel, tempo, case.n_bars),
        "M13": metrics.exposed_chromatic_clash_ratio(mel, case.scale, prog, bars, tempo),
        "M14": metrics.onset_grid_ratio([n.start_s for n in mel], tempo, 2),
        "M15.eligible": lt_eligible,
        "M15.resolved": lt_resolved,
        "T1": max(frets) if frets else None,
        "T2": metrics.hand_shift_rate(frets),
        "Tcount": len(frets) == len(mel),
        "V1": metrics.nearest_pitch_proximity(case.arrangement.voicings, STANDARD_EADGBE),
    }
    for style, acc in case.accompaniment.items():
        coverage = metrics.essential_tone_coverage(
            mel, acc, case.arrangement.voicings, bars, tempo, STANDARD_EADGBE
        )
        heads = metrics.stroke_heads([n.start_s for n in acc])
        out[f"A1.{style}"] = metrics.onset_grid_ratio(heads, tempo, 4)
        out[f"A2.{style}"] = metrics.melody_above_accompaniment_ratio(mel, acc)
        out[f"A3.{style}"] = metrics.exposed_semitone_clash_ratio(mel, acc, tempo)
        out[f"A4.{style}"] = metrics.final_chord_rings(acc, case.n_bars, tempo)
        out[f"H1.{style}"] = sum(coverage) / len(coverage)
        out[f"H1final.{style}"] = coverage[-1]
    return out


def _aggregate(rows: Sequence[dict[str, Value]]) -> dict[str, dict[str, float]]:
    """Mean/min/max of every metric over ``rows`` (``None`` values skipped),
    plus pooled ratios for the count-based M5 and M15."""
    out: dict[str, dict[str, float]] = {}
    for key in rows[0]:
        if key.startswith(("M5.", "M15.")):
            continue
        values = [float(r[key]) for r in rows if r[key] is not None]  # type: ignore[arg-type]
        if values:
            out[key] = {
                "mean": round(statistics.fmean(values), 4),
                "min": round(min(values), 4),
                "max": round(max(values), 4),
            }
    for name, total_key, hit_key in (
        ("M5", "M5.leaps", "M5.recovered"),
        ("M15", "M15.eligible", "M15.resolved"),
    ):
        total = sum(int(r[total_key] or 0) for r in rows)
        hits = sum(int(r[hit_key] or 0) for r in rows)
        out[name] = {"pooled": round(hits / total, 4) if total else -1.0, "n": float(total)}
    return out


def summarize(cases: Sequence[Case]) -> dict[str, dict[str, dict[str, float]]]:
    """Aggregates overall and per subgroup (mode, bars per chord)."""
    rows = [case_metrics(c) for c in cases]
    groups: dict[str, Callable[[Case], bool]] = {
        "overall": lambda c: True,
        "major": lambda c: c.scale.mode == "major",
        "minor": lambda c: c.scale.mode == "minor",
        "bars1": lambda c: c.bars_per_chord == 1,
        "bars2": lambda c: c.bars_per_chord == 2,
    }
    out: dict[str, dict[str, dict[str, float]]] = {}
    for name, keep in groups.items():
        selected = [r for c, r in zip(cases, rows, strict=True) if keep(c)]
        if selected:
            out[name] = _aggregate(selected)
    return out


def evaluate() -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    """Summaries for the validation and edge matrices."""
    return {
        "validation": summarize(build_cases(validation_inputs())),
        "edge": summarize(build_cases(edge_inputs())),
    }
```

Create `scripts/compose_musicality_report.py`:

```python
"""Print (and optionally save) composer musicality metrics.

    python scripts/compose_musicality_report.py
    python scripts/compose_musicality_report.py \
        --write tests/fixtures/compose/musicality_baseline.json

The committed baseline was written with ``--write`` on the pre-change
composer (commit recorded in the JSON's ``"source"``); re-running it now
measures the current composer on the same inputs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from music_decoder.evaluation.compose_musicality import evaluate


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", type=Path, help="write the summary JSON here")
    args = parser.parse_args()
    report = evaluate()
    for matrix, groups in report.items():
        print(f"== {matrix}")
        for key, stats in groups["overall"].items():
            print(f"  {key:16s} " + "  ".join(f"{k}={v:g}" for k, v in stats.items()))
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        payload = {"source": _git_head(), **report}
        args.write.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
        print(f"wrote {args.write}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Write the baseline (old composer) and check it**

Run: `python scripts/compose_musicality_report.py --write tests/fixtures/compose/musicality_baseline.json`
Expected: a table and `wrote tests/fixtures/compose/musicality_baseline.json`. The run is deterministic, so these validation-overall values must match **exactly**. If any differs, the refactor changed behaviour; stop.

| metric | value |
|---|---|
| M1 mean | 0.8634 |
| M2 mean | 0.3523 |
| M3 mean | 0.2785 |
| M7 mean | 0.2375 |
| M9 mean | 1.0 |
| M11b mean | 0.336 |
| M13 mean | 0.0938 |
| A1.arpeggio mean | 0.1037 |
| A2.strum mean | 0.8214 |
| A3.strum mean | 0.1448 |
| T1 max | 20.0 |
| T2 mean | 0.1709 |
| V1 mean | 1.621 |
| M5 pooled | 0.0 (n 5) |
| M15 pooled | 0.3846 (n 39) |

- [ ] **Step 6: Run the tests and gates**

Run: `pytest tests/unit/test_compose_api.py tests/unit/test_compose_evaluation.py -q`. Expected: `17 passed` (5 + 12).
Run: `pytest -q tests/unit tests/integration && ruff check src tests && ruff format --check src tests && mypy --strict src`. Expected: 0 failures, clean.

- [ ] **Step 7: Commit**

```bash
git add src/music_decoder/compose/api.py src/music_decoder/evaluation/compose_musicality.py \
  scripts/compose_musicality_report.py tests/fixtures/compose/musicality_baseline.json \
  tests/unit/test_compose_api.py tests/unit/test_compose_evaluation.py
git commit -m "refactor(compose): pure arrange(); record the pre-change musicality baseline

arrange() is the filesystem-free core of compose() (output unchanged, checked
against a golden run). evaluation/compose_musicality.py defines held-out
validation (400) and edge (80) matrices; the committed baseline JSON is the
current composer's summary on them.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Phrase-structured melody with scored pitch selection

Rewrites `generate_melody` per spec §5.3 (D4). It removes `strong_beat_chord_tone_prob`, `weak_beat_chord_tone_prob` and `markov_max_interval_semitones`; first run `grep -rn "chord_tone_prob\|markov_max" src tests scripts`, which must print matches only in `compose/melody.py` and `config/hyperparameters.yaml`. The YAML block goes in Task 7. It adds `tessitura_center` and `chord_floors`, makes `notes_per_bar` optional, and sets the window defaults to 62–79.

**Files:**
- Modify (full rewrite): `src/music_decoder/compose/melody.py`
- Modify (full rewrite, keeping the 4 existing tests at the top unchanged): `tests/unit/test_compose_melody.py`

**Interfaces:**
- Consumes: `harmony` (Task 2), `rhythm.plan_bars` (Task 1), `metrics` (tests).
- Produces, in `music_decoder.compose.melody`:
  - `generate_melody(*, scale, progression, bars_per_chord=1, tempo_bpm=100.0, notes_per_bar: int | None = None, pitch_low=62, pitch_high=79, seed=None, velocity_strong=80, velocity_weak=70, tessitura_center=70, chord_floors: Sequence[int | None] | None = None) -> list[Note]`
  - `MAX_INTERVAL = 12`
  - `final_target(scale, chord, candidates, reference) -> int`
  - `@dataclass(frozen=True) NoteContext(..., resolve_up: bool = False)`
  - `score_pitch(candidate, ctx, tessitura_center) -> float`
- The old private helpers `_chord_tones`, `_scale_pcs` and `_CHORD_PCS` are gone; `harmony` replaces them. Nothing else imports them after Task 2.

- [ ] **Step 1: Write the failing test**

Replace the whole of `tests/unit/test_compose_melody.py` with this content (the first four tests are the existing ones, unchanged). It includes the review's seeds 283 and 5774, and a 400-seed sweep:

```python
import math

import pytest

from music_decoder.compose import metrics
from music_decoder.compose.melody import (
    MAX_INTERVAL,
    NoteContext,
    final_target,
    generate_melody,
    score_pitch,
)
from music_decoder.types import ChordSymbol, Scale

C_MAJOR_PCS = {0, 2, 4, 5, 7, 9, 11}


def test_all_notes_in_scale():
    progression = [ChordSymbol.parse(c) for c in ("C", "Am", "F", "G")]
    notes = generate_melody(
        scale=Scale("C", "major"),
        progression=progression,
        bars_per_chord=1,
        tempo_bpm=120.0,
        seed=42,
    )
    assert notes
    for n in notes:
        assert n.pitch % 12 in C_MAJOR_PCS


def test_deterministic_with_seed():
    progression = [ChordSymbol.parse(c) for c in ("C", "Am", "F", "G")]
    a = generate_melody(
        scale=Scale("C", "major"),
        progression=progression,
        bars_per_chord=1,
        tempo_bpm=120.0,
        seed=42,
    )
    b = generate_melody(
        scale=Scale("C", "major"),
        progression=progression,
        bars_per_chord=1,
        tempo_bpm=120.0,
        seed=42,
    )
    assert [(n.pitch, n.start_s) for n in a] == [(n.pitch, n.start_s) for n in b]


def test_different_seeds_produce_different_melodies():
    progression = [ChordSymbol.parse(c) for c in ("C", "Am", "F", "G")]
    a = generate_melody(
        scale=Scale("C", "major"),
        progression=progression,
        bars_per_chord=1,
        tempo_bpm=120.0,
        seed=1,
    )
    b = generate_melody(
        scale=Scale("C", "major"),
        progression=progression,
        bars_per_chord=1,
        tempo_bpm=120.0,
        seed=2,
    )
    assert [n.pitch for n in a] != [n.pitch for n in b]


def test_strong_beats_favor_chord_tones_statistically():
    # Cmaj7 = {0,4,7,11}; 4 bars all Cmaj7, 4 notes per bar; strong beats are
    # beat 1 and 3 (i.e. notes 0 and 2 of every bar).
    progression = [ChordSymbol.parse("Cmaj7")] * 8
    notes = generate_melody(
        scale=Scale("C", "major"),
        progression=progression,
        bars_per_chord=1,
        tempo_bpm=120.0,
        seed=0,
        notes_per_bar=4,
    )
    chord_tones = {0, 4, 7, 11}
    strong_idxs = [i for i in range(len(notes)) if i % 4 in (0, 2)]
    strong_pcs = [notes[i].pitch % 12 for i in strong_idxs]
    fraction_chord = sum(p in chord_tones for p in strong_pcs) / len(strong_pcs)
    assert fraction_chord >= 0.55, f"strong-beat chord-tone rate {fraction_chord}"


# ---- phrase-planned generator ----------------------------------------------


def _ctx(**overrides):
    base = dict(
        strong=False,
        eighths=1.0,
        chord_pcs=frozenset({0, 4, 7}),
        prev_pitch=72,
        prev_interval=0,
        motif_interval=None,
        target=None,
        cadence_end=False,
    )
    base.update(overrides)
    return NoteContext(**base)


def _gen(scale, labels, bars=1, seed=0, tempo=120.0, **kw):
    return generate_melody(
        scale=scale,
        progression=[ChordSymbol.parse(c) for c in labels],
        bars_per_chord=bars,
        tempo_bpm=tempo,
        seed=seed,
        **kw,
    )


def test_final_target_prefers_tonic_then_root_nearest_reference():
    cands = list(range(62, 80))
    c_major = Scale("C", "major")
    assert final_target(c_major, ChordSymbol.parse("C"), cands, 70) == 72
    assert final_target(c_major, ChordSymbol.parse("Am"), cands, 66) == 72  # Am holds C
    assert final_target(c_major, ChordSymbol.parse("G"), cands, 70) == 67  # no tonic: root G
    with pytest.raises(ValueError):
        final_target(c_major, ChordSymbol.parse("C"), [62, 64], 63)


def test_score_pitch_prefers_chord_tones_on_strong_beats():
    # From C5 (72) over a C chord: E5 (a third, chord tone) beats D5 (a step, non-chord tone)
    assert score_pitch(76, _ctx(strong=True), 72) > score_pitch(74, _ctx(strong=True), 72)
    # ...but on a short weak note the step wins
    assert score_pitch(74, _ctx(), 72) > score_pitch(76, _ctx(), 72)


def test_score_pitch_excludes_leaps_over_an_octave():
    assert score_pitch(85, _ctx(), 72) == -math.inf


def test_score_pitch_keeps_the_final_target_within_an_octave():
    # previous note 72, target 60: 74 is a fine step from 72 but 14 from the target
    ctx = _ctx(target=60)
    assert score_pitch(74, ctx, 70) == -math.inf
    assert score_pitch(72, ctx, 70) > -math.inf


def test_score_pitch_rewards_leap_recovery():
    after_leap_up = _ctx(prev_interval=7)
    assert score_pitch(71, after_leap_up, 72) > score_pitch(74, after_leap_up, 72)


def test_score_pitch_resolves_the_leading_tone_up():
    ctx = _ctx(prev_pitch=71, chord_pcs=frozenset({0, 4, 7}), resolve_up=True)  # B -> C chord
    assert score_pitch(72, ctx, 72) > score_pitch(72, _ctx(prev_pitch=71), 72)
    assert score_pitch(72, ctx, 72) > score_pitch(67, ctx, 72)


def test_score_pitch_follows_the_motif():
    ctx = _ctx(motif_interval=-2)
    assert score_pitch(70, ctx, 72) > score_pitch(74, ctx, 72)


@pytest.mark.parametrize(
    ("key", "labels", "seed"),
    [("E", ("E", "A", "B7", "E"), 283), ("D", ("D", "G", "A", "D"), 5774)],
)
def test_review_reproductions_end_within_an_octave(key, labels, seed):
    notes = _gen(Scale(key, "major"), labels, seed=seed)
    assert metrics.max_interval(notes) <= MAX_INTERVAL


@pytest.mark.parametrize(
    ("key", "mode", "labels"),
    [
        ("E", "major", ("E", "A", "B7", "E")),
        ("D", "major", ("D", "G", "A", "D")),
        ("A", "minor", ("Am", "Dm", "E7", "Am")),
    ],
)
def test_no_interval_exceeds_an_octave_over_400_seeds(key, mode, labels):
    for seed in range(400):
        notes = _gen(Scale(key, mode), labels, seed=seed)
        assert metrics.max_interval(notes) <= MAX_INTERVAL, seed


def test_phrase_melody_has_varied_rhythm_on_the_eighth_grid():
    notes = _gen(Scale("C", "major"), ("C", "G", "Am", "F"), bars=2, seed=5)
    assert metrics.distinct_durations(notes, 120.0) >= 2
    assert metrics.onset_grid_ratio([n.start_s for n in notes], 120.0, 2) == 1.0
    assert notes[-1].end_s <= 8 * 2.0 + 1e-9  # stays within 8 bars at 120 BPM


@pytest.mark.parametrize("seed", range(10))
def test_phrase_melody_resolves_and_stays_in_window(seed):
    notes = _gen(Scale("A", "minor"), ("Am", "Dm", "E7", "Am"), seed=seed)
    assert notes[-1].pitch % 12 == 9  # ends on A
    assert (notes[-1].end_s - notes[-1].start_s) / 0.5 >= 1.5
    assert all(62 <= n.pitch <= 79 for n in notes)


def test_e7_in_a_minor_uses_g_sharp_and_may_keep_a():
    pitch_classes = set()
    for seed in range(40):
        notes = _gen(Scale("A", "minor"), ("Am", "E7"), seed=seed)
        bar2 = [n.pitch % 12 for n in notes if 2.0 <= n.start_s < 4.0]
        assert 7 not in bar2, seed  # no G natural against G#
        pitch_classes.update(bar2)
    assert 8 in pitch_classes  # G# is used
    assert 9 in pitch_classes  # the tonic A is still available over E7


def test_secondary_dominant_in_major_keeps_its_target():
    pitch_classes = set()
    for seed in range(40):
        notes = _gen(Scale("C", "major"), ("C", "D7", "G", "C"), seed=seed)
        bar2 = [n.pitch % 12 for n in notes if 2.0 <= n.start_s < 4.0]  # D7 = D F# A C
        assert 5 not in bar2, seed  # no F natural against F#
        pitch_classes.update(bar2)
    assert 7 in pitch_classes  # G (where D7 is heading) is still available


def test_leading_tone_mostly_resolves_up_into_the_tonic_chord():
    eligible = resolved = 0
    for seed in range(100):
        notes = _gen(Scale("A", "minor"), ("Am", "Dm", "E7", "Am"), seed=seed)
        prog = [ChordSymbol.parse(c) for c in ("Am", "Dm", "E7", "Am")]
        e, r = metrics.leading_tone_resolution_counts(notes, Scale("A", "minor"), prog, 1, 120.0)
        eligible, resolved = eligible + e, resolved + r
    assert eligible >= 10
    assert resolved / eligible >= 0.8


def test_chord_floors_lift_the_melody_over_that_chord():
    floors = [None, 69, None, None]
    for seed in range(20):
        notes = _gen(Scale("D", "major"), ("G", "D", "A", "G"), seed=seed, chord_floors=floors)
        assert all(n.pitch >= 69 for n in notes if 2.0 <= n.start_s < 4.0), seed


def test_a_floor_leaving_too_few_pitches_is_ignored():
    notes = _gen(Scale("C", "major"), ("C",), seed=1, chord_floors=[78])
    assert any(n.pitch < 78 for n in notes)
    with pytest.raises(ValueError):
        _gen(Scale("C", "major"), ("C", "G"), seed=1, chord_floors=[70])


def test_legacy_grid_keeps_uniform_timing():
    notes = _gen(Scale("C", "major"), ("C", "G"), notes_per_bar=4)
    assert len(notes) == 8
    assert [round(n.start_s, 6) for n in notes] == [i * 0.5 for i in range(8)]
    assert notes[-1].pitch % 12 == 7  # ends on the G root (no tonic in G)


def test_single_bar_progression():
    notes = _gen(Scale("D", "major"), ("D",), seed=1)
    assert notes and notes[-1].pitch % 12 == 2


@pytest.mark.parametrize("tempo", [40.0, 220.0])
def test_extreme_tempos_keep_the_grid_and_the_length(tempo):
    notes = _gen(Scale("E", "major"), ("E", "A", "B7"), seed=4, tempo=tempo)
    bar_s = 4 * 60.0 / tempo
    assert metrics.onset_grid_ratio([n.start_s for n in notes], tempo, 2) == 1.0
    assert notes[-1].end_s <= 3 * bar_s + 1e-9
    assert notes[-1].pitch % 12 == 11  # B7 holds no E: ends on the B root
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/unit/test_compose_melody.py -q`
Expected: collection ERROR, `ImportError: cannot import name 'MAX_INTERVAL' from 'music_decoder.compose.melody'`.

- [ ] **Step 3: Write the implementation**

Replace the whole of `src/music_decoder/compose/melody.py` with:

```python
"""Phrase-structured melody generator over a chord progression.

Rhythm comes from :func:`music_decoder.compose.rhythm.plan_bars` (one-bar
templates arranged in antecedent/consequent phrases that end on cadences).
Each pitch is sampled from a softmax over musical scores: interval size,
chord fit weighted by metric position, leap recovery, leading-tone
resolution, motif echo, register gravity and, in the final bar, a pull
toward the cadence target. The only randomness is
``numpy.random.default_rng(seed)``, so output is deterministic for a seed.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence
from dataclasses import dataclass

import numpy as np

from music_decoder.compose.harmony import (
    chord_pcs,
    chord_scale_pcs,
    leading_tone_pc,
    scale_pcs,
    tonic_pc,
)
from music_decoder.compose.rhythm import plan_bars
from music_decoder.types import ChordSymbol, Note, Scale

# Log-weight by interval size in semitones: steps preferred, repeats and wide
# leaps discouraged, tritone strongly avoided. Intervals > 12 are never chosen.
INTERVAL_WEIGHT: dict[int, float] = {
    0: -0.8,
    1: 0.8,
    2: 0.8,
    3: -0.2,
    4: -0.6,
    5: -1.2,
    6: -4.0,
    7: -1.8,
    8: -3.5,
    9: -3.5,
    10: -4.0,
    11: -4.5,
    12: -3.0,
}
MAX_INTERVAL = 12  # hard limit, including the move into the final note
STRONG_CHORD_TONE = 2.0  # beats 1 and 3
STRONG_NON_CHORD_TONE = -3.0
LONG_EIGHTHS = 3  # notes at least a dotted quarter long
LONG_CHORD_TONE = 1.0
LONG_NON_CHORD_TONE = -1.5
QUARTER_EIGHTHS = 2
QUARTER_SEMITONE_CLASH = -1.0  # quarter+ non-chord tone a semitone off a chord tone
WEAK_CHORD_TONE = 0.3
LEAP_SEMITONES = 5  # a leap of a fourth or more wants recovery
LEAP_RECOVERY_BONUS = 2.5  # opposite-direction step after a leap
LEAP_CONTINUE_PENALTY = -2.5  # same direction (or repeat) after a leap
LEADING_TONE_RESOLUTION = 3.0  # leading tone -> tonic a semitone up, into a tonic chord
MOTIF_SIGN_BONUS = 1.5  # same direction as the echoed interval
MOTIF_SIZE_BONUS = 1.0  # within a semitone of the echoed interval
GRAVITY_WEIGHT = 0.5  # register pull toward the tessitura centre
GRAVITY_WIDTH = 6.0
CADENCE_PULL = 0.25  # per semitone from the final target, final bar only
CADENCE_END_NON_CHORD_TONE = -3.0  # last note of a cadence bar
MIN_FLOOR_CANDIDATES = 5  # a chord floor applies only if it leaves this many pitches


def _scale_pitches_in_range(pcs: Collection[int], low: int, high: int) -> list[int]:
    return [p for p in range(low, high + 1) if (p % 12) in pcs]


def final_target(
    scale: Scale, chord: ChordSymbol, candidates: Sequence[int], reference: int
) -> int:
    """Final pitch: the tonic if ``chord`` contains it, else the chord root;
    the candidate of that pitch class nearest ``reference`` (ties -> lower)."""
    tones = chord_pcs(chord)
    target_pc = tonic_pc(scale) if tonic_pc(scale) in tones else tones[0]
    matches = [p for p in candidates if p % 12 == target_pc]
    if not matches:
        raise ValueError("No cadence target pitch in the requested range.")
    return min(matches, key=lambda p: (abs(p - reference), p))


@dataclass(frozen=True)
class NoteContext:
    """Everything the scorer needs to know about the note being chosen."""

    strong: bool
    eighths: float  # nominal length in eighth notes
    chord_pcs: frozenset[int]
    prev_pitch: int | None
    prev_interval: int
    motif_interval: int | None
    target: int | None  # final-bar cadence target
    cadence_end: bool  # last note of a cadence bar
    resolve_up: bool = False  # previous note is a leading tone entering a tonic chord


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def score_pitch(candidate: int, ctx: NoteContext, tessitura_center: int) -> float:
    """Log-weight of ``candidate``; ``-inf`` excludes it."""
    if ctx.target is not None and abs(candidate - ctx.target) > MAX_INTERVAL:
        return -math.inf  # the final note must stay reachable
    score = 0.0
    is_chord_tone = candidate % 12 in ctx.chord_pcs
    if ctx.prev_pitch is not None:
        interval = candidate - ctx.prev_pitch
        size = abs(interval)
        if size > MAX_INTERVAL:
            return -math.inf
        score += INTERVAL_WEIGHT[size]
        if abs(ctx.prev_interval) >= LEAP_SEMITONES:
            if interval * ctx.prev_interval < 0 and 1 <= size <= 2:
                score += LEAP_RECOVERY_BONUS
            elif interval * ctx.prev_interval >= 0:
                score += LEAP_CONTINUE_PENALTY
        if ctx.resolve_up and interval == 1:
            score += LEADING_TONE_RESOLUTION
        if ctx.motif_interval is not None:
            if _sign(interval) == _sign(ctx.motif_interval):
                score += MOTIF_SIGN_BONUS
            if abs(interval - ctx.motif_interval) <= 1:
                score += MOTIF_SIZE_BONUS
    if ctx.strong:
        score += STRONG_CHORD_TONE if is_chord_tone else STRONG_NON_CHORD_TONE
    elif ctx.eighths >= LONG_EIGHTHS:
        score += LONG_CHORD_TONE if is_chord_tone else LONG_NON_CHORD_TONE
    elif (
        ctx.eighths >= QUARTER_EIGHTHS
        and not is_chord_tone
        and any((candidate - c) % 12 in (1, 11) for c in ctx.chord_pcs)
    ):
        score += QUARTER_SEMITONE_CLASH
    elif is_chord_tone:
        score += WEAK_CHORD_TONE
    score -= GRAVITY_WEIGHT * ((candidate - tessitura_center) / GRAVITY_WIDTH) ** 2
    if ctx.target is not None:
        score -= CADENCE_PULL * abs(candidate - ctx.target)
    if ctx.cadence_end and not is_chord_tone:
        score += CADENCE_END_NON_CHORD_TONE
    return score


def _sample(
    candidates: Sequence[int], scores: Sequence[float], rng: np.random.Generator, prev: int | None
) -> int:
    arr = np.array(scores, dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():  # nothing reachable: take the nearest pitch
        ref = prev if prev is not None else candidates[len(candidates) // 2]
        return min(candidates, key=lambda p: (abs(p - ref), p))
    weights = np.where(finite, np.exp(arr - arr[finite].max()), 0.0)
    weights /= weights.sum()
    return int(candidates[int(rng.choice(len(candidates), p=weights))])


@dataclass(frozen=True)
class _Slot:
    offset_s: float  # onset within the bar
    length_s: float  # nominal (pre-articulation) length
    eighths: float
    strong: bool
    velocity: int


@dataclass(frozen=True)
class _Bar:
    slots: tuple[_Slot, ...]
    cadence: bool
    final: bool
    motif_from: int | None


def _phrase_bars(
    n_bars: int, beat_s: float, rng: np.random.Generator, v_strong: int, v_mid: int, v_weak: int
) -> list[_Bar]:
    eighth = beat_s / 2.0
    bars: list[_Bar] = []
    for plan in plan_bars(n_bars, rng):
        slots: list[_Slot] = []
        pos = 0
        for value in plan.rhythm:
            velocity = v_strong if pos == 0 else v_mid if pos == 4 else v_weak
            slots.append(_Slot(pos * eighth, value * eighth, value, pos in (0, 4), velocity))
            pos += value
        bars.append(_Bar(tuple(slots), plan.cadence, plan.final, plan.motif_from))
    return bars


def _grid_bars(
    n_bars: int, beat_s: float, notes_per_bar: int, v_strong: int, v_mid: int, v_weak: int
) -> list[_Bar]:
    """Legacy uniform grid: ``notes_per_bar`` equal notes per bar."""
    step = 4.0 * beat_s / notes_per_bar
    half = max(notes_per_bar // 2, 1)
    slots = tuple(
        _Slot(
            k * step,
            step,
            8.0 / notes_per_bar,
            k % half == 0,
            v_strong if k == 0 else v_mid if k % half == 0 else v_weak,
        )
        for k in range(notes_per_bar)
    )
    return [_Bar(slots, False, b == n_bars - 1, None) for b in range(n_bars)]


def _candidates(
    scale: Scale, chord: ChordSymbol, low: int, high: int, floor: int | None
) -> list[int]:
    pitches = _scale_pitches_in_range(chord_scale_pcs(scale, chord), low, high)
    if floor is not None:
        above = [p for p in pitches if p >= floor]
        if len(above) >= MIN_FLOOR_CANDIDATES:
            return above
    return pitches


def generate_melody(
    *,
    scale: Scale,
    progression: Sequence[ChordSymbol],
    bars_per_chord: int = 1,
    tempo_bpm: float = 100.0,
    notes_per_bar: int | None = None,
    pitch_low: int = 62,  # D4
    pitch_high: int = 79,  # G5
    seed: int | None = None,
    velocity_strong: int = 80,
    velocity_weak: int = 70,
    tessitura_center: int = 70,  # A#4
    chord_floors: Sequence[int | None] | None = None,
) -> list[Note]:
    """Generate a list of Notes spanning the given progression.

    ``notes_per_bar=None`` (default) uses phrase-planned rhythms; an integer
    uses the legacy uniform grid of that many notes per bar.
    ``chord_floors`` (one entry per chord) is the lowest melody pitch allowed
    over that chord, so the melody can sit above the accompaniment's
    defining tones; a floor leaving fewer than ``MIN_FLOOR_CANDIDATES``
    pitches is ignored.
    """
    rng = np.random.default_rng(seed)
    if not _scale_pitches_in_range(scale_pcs(scale), pitch_low, pitch_high):
        raise ValueError("Scale has no pitches in the requested range.")
    if chord_floors is not None and len(chord_floors) != len(progression):
        raise ValueError("chord_floors needs one entry per chord.")

    beat_s = 60.0 / tempo_bpm
    bar_s = 4.0 * beat_s
    n_bars = len(progression) * bars_per_chord
    v_mid = (velocity_strong + velocity_weak) // 2
    if notes_per_bar is None:
        bars = _phrase_bars(n_bars, beat_s, rng, velocity_strong, v_mid, velocity_weak)
    else:
        bars = _grid_bars(n_bars, beat_s, notes_per_bar, velocity_strong, v_mid, velocity_weak)
    lead, tonic = leading_tone_pc(scale), tonic_pc(scale)

    out: list[Note] = []
    prev: int | None = None
    prev_interval = 0
    prev_chord: ChordSymbol | None = None
    bar_intervals: list[list[int]] = []
    for bar_idx, bar in enumerate(bars):
        chord_idx = min(bar_idx // bars_per_chord, len(progression) - 1)
        chord = progression[chord_idx]
        tones = frozenset(chord_pcs(chord))
        floor = chord_floors[chord_idx] if chord_floors is not None else None
        candidates = _candidates(scale, chord, pitch_low, pitch_high, floor)
        if not candidates:
            raise ValueError("Scale has no pitches in the requested range.")
        target = (
            final_target(scale, chord, candidates, prev if prev is not None else tessitura_center)
            if bar.final
            else None
        )
        motif = bar_intervals[bar.motif_from] if bar.motif_from is not None else None
        here: list[int] = []
        for k, slot in enumerate(bar.slots):
            last = k == len(bar.slots) - 1
            resolve_up = (
                prev is not None
                and prev_chord is not None
                and prev % 12 == lead
                and lead in chord_pcs(prev_chord)
                and chord != prev_chord
                and tonic in tones
            )
            if target is not None and last:
                pitch = target
            else:
                ctx = NoteContext(
                    strong=slot.strong,
                    eighths=slot.eighths,
                    chord_pcs=tones,
                    prev_pitch=prev,
                    prev_interval=prev_interval,
                    motif_interval=motif[k] if motif is not None and 0 < k < len(motif) else None,
                    target=target,
                    cadence_end=bar.cadence and last,
                    resolve_up=resolve_up,
                )
                scores = [score_pitch(c, ctx, tessitura_center) for c in candidates]
                pitch = _sample(candidates, scores, rng, prev)
            interval = pitch - prev if prev is not None else 0
            here.append(interval)
            start = bar_idx * bar_s + slot.offset_s
            out.append(
                Note(
                    start_s=start,
                    end_s=start + slot.length_s * 0.95,
                    pitch=pitch,
                    velocity=velocity_strong if (target is not None and last) else slot.velocity,
                    confidence=1.0,
                )
            )
            prev_interval = interval
            prev = pitch
            prev_chord = chord
        bar_intervals.append(here)
    return out
```

- [ ] **Step 4: Run the tests and gates**

Run: `pytest tests/unit/test_compose_melody.py -q`. Expected: `36 passed`.
Run: `pytest -q tests/unit tests/integration && ruff check src tests && ruff format --check src tests && mypy --strict src`. Expected: 0 failures, clean. `test_compose_evaluation.py` still passes, because it reads the committed baseline.

- [ ] **Step 5: Commit**

```bash
git add src/music_decoder/compose/melody.py tests/unit/test_compose_melody.py
git commit -m "feat(compose): phrase-structured melody with scored pitch selection

Rhythm from the phrase planner; pitches sampled from a softmax over interval
size, metric-weighted chord fit, leap recovery, leading-tone resolution, motif
echo, register gravity and a cadence pull. The final bar stays within an
octave of its target, the chord scale replaces only altered degrees, and
optional per-chord floors keep the tune above the accompaniment's identity
tones.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: On-grid, melody-aware, identity-preserving accompaniment

**Files:**
- Modify (full rewrite): `src/music_decoder/compose/arrangement.py`
- Modify: `src/music_decoder/compose/api.py` (`accompany()` passes the melody through)
- Modify (full rewrite, keeping the 2 existing tests at the top unchanged): `tests/unit/test_compose_arrangement.py`

**Interfaces:**
- Consumes: `harmony.essential_pcs` (Task 2), `generate_melody` (Task 4), `metrics` (tests).
- Produces, in `music_decoder.compose.arrangement`:
  - `THIN_MARGIN_SEMITONES = 3`
  - `_thin_below(pitches, essential, melody, t0, t1) -> list[int]`
  - `_accompaniment_notes(voicings, bars_per_chord, tempo_bpm, style, tuning, melody=()) -> list[pretty_midi.Note]`
  - `build_midi(...)` (signature unchanged)
  - It keeps `_voiced_pitches`, `GUITAR_PROGRAM` and `Style`.

- [ ] **Step 1: Write the failing test**

Replace the whole of `tests/unit/test_compose_arrangement.py` with:

```python
import pretty_midi
import pytest

from music_decoder.compose import metrics
from music_decoder.compose.arrangement import _accompaniment_notes, _thin_below, build_midi
from music_decoder.compose.melody import generate_melody
from music_decoder.compose.voicings import voicings_for
from music_decoder.tabs.tuning import STANDARD_EADGBE
from music_decoder.types import ChordSymbol, Note, Scale


def test_build_midi_produces_two_tracks(tmp_path):
    progression = [ChordSymbol.parse(c) for c in ("Cmaj7", "Am7", "Dm7", "G7")]
    voicings = [voicings_for(c, STANDARD_EADGBE)[0] for c in progression]
    melody = generate_melody(
        scale=Scale("C", "major"),
        progression=progression,
        bars_per_chord=1,
        tempo_bpm=120.0,
        seed=42,
    )
    out = tmp_path / "x.mid"
    build_midi(
        melody=melody,
        voicings=voicings,
        bars_per_chord=1,
        tempo_bpm=120.0,
        style="strum",
        out_path=out,
    )
    assert out.exists()
    pm = pretty_midi.PrettyMIDI(str(out))
    assert len(pm.instruments) == 2
    assert sum(len(i.notes) for i in pm.instruments) > 0


def test_build_midi_styles_run(tmp_path):
    progression = [ChordSymbol.parse(c) for c in ("C", "G")]
    voicings = [voicings_for(c, STANDARD_EADGBE)[0] for c in progression]
    melody = generate_melody(
        scale=Scale("C", "major"),
        progression=progression,
        bars_per_chord=1,
        tempo_bpm=120.0,
        seed=42,
    )
    for style in ("strum", "arpeggio", "fingerstyle"):
        out = tmp_path / f"{style}.mid"
        build_midi(
            melody=melody,
            voicings=voicings,
            bars_per_chord=1,
            tempo_bpm=120.0,
            style=style,
            out_path=out,
        )
        assert out.exists()


# ---- melody-aware, on-grid accompaniment (compose improvements) ------------


def _note(start, end, pitch):
    return Note(start_s=start, end_s=end, pitch=pitch, velocity=80, confidence=1.0)


def _acc(labels, style, melody=(), bars=1):
    voicings = [voicings_for(ChordSymbol.parse(c), STANDARD_EADGBE)[0] for c in labels]
    notes = _accompaniment_notes(voicings, bars, 120.0, style, STANDARD_EADGBE, melody=melody)
    return [_note(n.start, n.end, n.pitch) for n in notes]


E = frozenset({4})  # the third of C major


def test_thin_below_keeps_accompaniment_a_minor_third_under_the_melody():
    pitches = [48, 52, 55, 60, 64]  # C3 E3 G3 C4 E4 (open C)
    melody = [_note(0.0, 0.5, 65)]  # F4
    assert _thin_below(pitches, E, melody, 0.0, 0.5) == [48, 52, 55, 60]
    assert _thin_below(pitches, E, [], 0.0, 0.5) == pitches
    assert _thin_below(pitches, E, [_note(1.0, 1.5, 50)], 0.0, 0.5) == pitches  # not sounding


def test_thin_below_always_keeps_the_bass():
    assert _thin_below([48, 52, 55], E, [_note(0.0, 0.5, 49)], 0.0, 0.5) == [48, 52]


def test_thin_below_restores_a_missing_third():
    # D chord (D3 A3 D4 F#4) under a D4 melody: F#4 would be dropped with D4
    d_shape = [50, 57, 62, 66]
    kept = _thin_below(d_shape, frozenset({6}), [_note(0.0, 2.0, 62)], 0.0, 2.0)
    assert kept == [50, 57, 66]  # third restored, the unison D4 removed


def test_thin_below_does_not_restore_what_the_melody_sounds():
    d_shape = [50, 57, 62, 66]
    kept = _thin_below(d_shape, frozenset({6}), [_note(0.0, 2.0, 66)], 0.0, 2.0)
    assert kept == [50, 57, 62]  # the melody's F#4 is the third


def test_thin_below_keeps_minor_thirds_and_sevenths():
    am = [45, 52, 57, 60, 64]  # A2 E3 A3 C4 E4 (open Am)
    assert 60 in _thin_below(am, frozenset({0}), [_note(0.0, 1.0, 62)], 0.0, 1.0)
    g7 = [43, 47, 50, 55, 59, 65]  # G2 B2 D3 G3 B3 F4 (open G7)
    kept = _thin_below(g7, frozenset({11, 5}), [_note(0.0, 1.0, 62)], 0.0, 1.0)
    assert {p % 12 for p in kept} >= {11, 5}  # B and F both still heard


@pytest.mark.parametrize("style", ["arpeggio", "strum", "fingerstyle"])
def test_stroke_heads_sit_on_the_sixteenth_grid(style):
    acc = _acc(["C", "Am", "F", "G7"], style)
    heads = metrics.stroke_heads([a.start_s for a in acc])
    assert metrics.onset_grid_ratio(heads, 120.0, 4) == 1.0


def test_arpeggio_plays_eight_eighths_per_bar():
    acc = _acc(["C", "G"], "arpeggio")  # bar 1 arpeggio, bar 2 final chord
    bar1 = sorted(a.start_s for a in acc if a.start_s < 2.0)
    assert bar1 == [k * 0.25 for k in range(8)]
    assert acc[0].pitch == 48  # C3 bass on the downbeat


def test_strum_pattern_and_up_strokes():
    acc = _acc(["C", "G"], "strum")
    bar1 = [a for a in acc if a.start_s < 2.0]
    heads = metrics.stroke_heads([a.start_s for a in bar1])
    assert [round(h / 0.25) for h in heads] == [0, 2, 3, 5, 6, 7]
    up = [a for a in bar1 if 0.75 - 1e-9 <= a.start_s < 0.75 + 0.06]
    assert len(up) == 3
    assert up[0].pitch > up[-1].pitch  # up stroke goes high -> low


def test_fingerstyle_alternates_the_bass():
    acc = _acc(["C", "G"], "fingerstyle")  # open C: C3 E3 G3 C4 E4
    beat1 = [a.pitch for a in acc if abs(a.start_s - 0.0) < 1e-9]
    beat3 = [a.pitch for a in acc if abs(a.start_s - 1.0) < 1e-9]
    assert beat1 == [48] and beat3 == [52]


@pytest.mark.parametrize("style", ["arpeggio", "strum", "fingerstyle"])
def test_final_bar_is_one_ringing_chord(style):
    acc = _acc(["C", "G", "C"], style)
    assert metrics.final_chord_rings(acc, 3, 120.0)


def test_melody_stays_on_top_of_the_strum():
    melody = [_note(0.0, 1.9, 67)]  # G4 held over the C chord
    acc = _acc(["C", "G"], "strum", melody=melody)
    sounding = [a.pitch for a in acc if a.start_s < 1.9 and a.end_s > 0.0]
    assert max(sounding) <= 64  # E4 is a minor third under G4


@pytest.mark.parametrize("style", ["arpeggio", "strum", "fingerstyle"])
def test_single_bar_accompaniment_is_the_final_chord(style):
    acc = _acc(["C"], style)
    assert acc and metrics.final_chord_rings(acc, 1, 120.0)


@pytest.mark.parametrize("style", ["arpeggio", "strum", "fingerstyle"])
def test_d_major_cadence_keeps_its_third(style):
    # review reproduction: D G A D, seed 9 used to end on bare D3 A3 D4
    progression = [ChordSymbol.parse(c) for c in ("D", "G", "A", "D")]
    voicings = [voicings_for(c, STANDARD_EADGBE)[0] for c in progression]
    melody = generate_melody(
        scale=Scale("D", "major"), progression=progression, tempo_bpm=120.0, seed=9
    )
    notes = _accompaniment_notes(voicings, 1, 120.0, style, STANDARD_EADGBE, melody=melody)
    acc = [_note(n.start, n.end, n.pitch) for n in notes]
    coverage = metrics.essential_tone_coverage(melody, acc, voicings, 1, 120.0, STANDARD_EADGBE)
    assert all(coverage)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/unit/test_compose_arrangement.py -q`
Expected: collection ERROR, `ImportError: cannot import name '_thin_below' from 'music_decoder.compose.arrangement'`.

- [ ] **Step 3: Write the implementation**

Replace the whole of `src/music_decoder/compose/arrangement.py` with:

```python
"""Combine melody + chord voicings into a 2-track PrettyMIDI file.

Accompaniment patterns sit on a straight 4/4 grid and are melody-aware: at
each attack, voicing pitches that would reach within a minor third of the
melody are dropped so the tune stays on top, except that the bass is always
kept and a chord's defining tones (third, seventh, altered fifth, fourth)
are never lost unless the melody itself is sounding them. The last bar is
always a single strummed chord ringing to the end.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Literal

import pretty_midi

from music_decoder.compose.harmony import essential_pcs
from music_decoder.tabs.tuning import STANDARD_EADGBE
from music_decoder.types import Note, Tuning, VoicedChord

GUITAR_PROGRAM = 25  # Steel Acoustic Guitar
Style = Literal["strum", "arpeggio", "fingerstyle"]

THIN_MARGIN_SEMITONES = 3  # accompaniment stays at least a minor third under the melody
STRUM_STAGGER_S = 0.008  # per string, low -> high on a down stroke
FINAL_STAGGER_S = 0.010  # <= 50 ms across six strings
# (eighth-note slot, direction): the "D . D U . U D U" pop strum
STRUM_STROKES: tuple[tuple[int, Literal["down", "up"]], ...] = (
    (0, "down"),
    (2, "down"),
    (3, "up"),
    (5, "up"),
    (6, "down"),
    (7, "up"),
)
UP_STROKE_STRINGS = 3


def _voiced_pitches(voicing: VoicedChord, tuning: Tuning) -> list[int]:
    return sorted(tuning.open_pitches[p.string] + p.fret for p in voicing.positions if p.fret >= 0)


def _thin_below(
    pitches: Sequence[int],
    essential: Collection[int],
    melody: Sequence[Note],
    t0: float,
    t1: float,
) -> list[int]:
    """The (sorted) ``pitches`` to play in [t0, t1).

    Pitches within a minor third of the lowest melody note sounding in the
    window are dropped. The lowest pitch is always kept, and each
    ``essential`` pitch class neither kept nor sounding in the melody is put
    back at its lowest voicing pitch, so the harmony keeps its identity even
    when that note has to sit near or above the tune.
    """
    sounding = [n.pitch for n in melody if n.start_s < t1 and n.end_s > t0]
    if not sounding:
        return list(pitches)
    ceiling = min(sounding) - THIN_MARGIN_SEMITONES
    kept = [p for p in pitches if p <= ceiling] or list(pitches[:1])
    heard = {p % 12 for p in kept} | {p % 12 for p in sounding}
    for pc in sorted(essential):
        if pc not in heard:
            same = [p for p in pitches if p % 12 == pc]
            if same:
                kept.append(min(same))
                heard.add(pc)
    return sorted(kept)


def _stroke(
    pitches: Sequence[int], start: float, end: float, velocity: int, stagger_s: float
) -> list[pretty_midi.Note]:
    return [
        pretty_midi.Note(velocity=velocity, pitch=int(p), start=start + k * stagger_s, end=end)
        for k, p in enumerate(pitches)
    ]


def _arpeggio_bar(
    pitches: list[int], essential: Collection[int], melody: Sequence[Note], t: float, eighth: float
) -> list[pretty_midi.Note]:
    """Bass on the downbeat, then seven eighths cycling up and down the treble."""
    treble = pitches[1:]
    cycle = treble + treble[-2:0:-1] if len(treble) > 1 else treble
    sequence = [pitches[0]] + [cycle[i % len(cycle)] for i in range(7)] if cycle else pitches[:1]
    out: list[pretty_midi.Note] = []
    for k, p in enumerate(sequence):
        start = t + k * eighth
        if k > 0 and p not in _thin_below(pitches, essential, melody, start, start + eighth):
            continue
        out.append(
            pretty_midi.Note(
                velocity=70 if k == 0 else 62,
                pitch=int(p),
                start=start,
                end=start + 0.95 * eighth,
            )
        )
    return out


def _strum_bar(
    pitches: list[int], essential: Collection[int], melody: Sequence[Note], t: float, eighth: float
) -> list[pretty_midi.Note]:
    """Down strokes play the whole (thinned) voicing; up strokes the top strings."""
    out: list[pretty_midi.Note] = []
    for i, (slot, direction) in enumerate(STRUM_STROKES):
        start = t + slot * eighth
        next_slot = STRUM_STROKES[i + 1][0] if i + 1 < len(STRUM_STROKES) else 8
        end = t + next_slot * eighth
        chord = _thin_below(pitches, essential, melody, start, end)
        if direction == "up":
            out.extend(_stroke(chord[-UP_STROKE_STRINGS:][::-1], start, end, 60, STRUM_STAGGER_S))
        else:
            out.extend(_stroke(chord, start, end, 80 if slot == 0 else 72, STRUM_STAGGER_S))
    return out


def _fingerstyle_bar(
    pitches: list[int], essential: Collection[int], melody: Sequence[Note], t: float, beat: float
) -> list[pretty_midi.Note]:
    """Alternating bass on beats 1 and 3, treble pinch on beats 2 and 4."""
    alternate = len(pitches) >= 4
    alt_bass = pitches[1] if alternate else pitches[0]
    out: list[pretty_midi.Note] = []
    for k in range(4):
        start = t + k * beat
        end = start + 0.95 * beat
        if k % 2 == 0:
            bass = pitches[0] if k == 0 else alt_bass
            out.append(pretty_midi.Note(velocity=72, pitch=int(bass), start=start, end=end))
            continue
        for p in _thin_below(pitches[1:] or pitches, essential, melody, start, start + beat):
            if alternate and p == alt_bass:
                continue
            out.append(pretty_midi.Note(velocity=64, pitch=int(p), start=start, end=end))
    return out


def _accompaniment_notes(
    voicings: Sequence[VoicedChord],
    bars_per_chord: int,
    tempo_bpm: float,
    style: Style,
    tuning: Tuning,
    melody: Sequence[Note] = (),
) -> list[pretty_midi.Note]:
    seconds_per_beat = 60.0 / tempo_bpm
    seconds_per_bar = seconds_per_beat * 4.0
    eighth = seconds_per_beat / 2.0
    n_bars = len(voicings) * bars_per_chord
    out: list[pretty_midi.Note] = []
    for bar in range(n_bars):
        voicing = voicings[bar // bars_per_chord]
        pitches = _voiced_pitches(voicing, tuning)
        essential = essential_pcs(voicing.chord)
        t = bar * seconds_per_bar
        if not pitches:
            continue
        if bar == n_bars - 1:
            chord = _thin_below(pitches, essential, melody, t, t + seconds_per_bar)
            out.extend(_stroke(chord, t, t + seconds_per_bar, 80, FINAL_STAGGER_S))
        elif style == "arpeggio":
            out.extend(_arpeggio_bar(pitches, essential, melody, t, eighth))
        elif style == "strum":
            out.extend(_strum_bar(pitches, essential, melody, t, eighth))
        elif style == "fingerstyle":
            out.extend(_fingerstyle_bar(pitches, essential, melody, t, seconds_per_beat))
    return out


def build_midi(
    *,
    melody: Sequence[Note],
    voicings: Sequence[VoicedChord],
    bars_per_chord: int,
    tempo_bpm: float,
    style: Style,
    out_path: Path,
    tuning: Tuning = STANDARD_EADGBE,
) -> Path:
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo_bpm)
    melody_inst = pretty_midi.Instrument(program=GUITAR_PROGRAM, name="melody")
    for n in melody:
        melody_inst.notes.append(
            pretty_midi.Note(
                velocity=int(n.velocity),
                pitch=int(n.pitch),
                start=float(n.start_s),
                end=float(n.end_s),
            )
        )
    accomp_inst = pretty_midi.Instrument(program=GUITAR_PROGRAM, name="accompaniment")
    accomp_inst.notes.extend(
        _accompaniment_notes(
            voicings,
            bars_per_chord,
            tempo_bpm,
            style,
            tuning,
            melody=melody,
        )
    )
    pm.instruments.extend([melody_inst, accomp_inst])
    pm.write(str(out_path))
    return out_path
```

In `src/music_decoder/compose/api.py`, inside `accompany()`, replace:

```python
    """The accompaniment track as ``Note``s (the melody is not consulted yet)."""
```

with:

```python
    """The accompaniment track as ``Note``s, thinned around ``melody``."""
```

and replace:

```python
        for n in _accompaniment_notes(voicings, bars_per_chord, tempo_bpm, style, tuning)
```

with:

```python
        for n in _accompaniment_notes(
            voicings, bars_per_chord, tempo_bpm, style, tuning, melody=melody
        )
```

- [ ] **Step 4: Run the tests and gates**

Run: `pytest tests/unit/test_compose_arrangement.py -q`. Expected: `23 passed`.
Run: `pytest -q tests/unit tests/integration && ruff check src tests && ruff format --check src tests && mypy --strict src`. Expected: 0 failures, clean.

- [ ] **Step 5: Commit**

```bash
git add src/music_decoder/compose/arrangement.py src/music_decoder/compose/api.py \
  tests/unit/test_compose_arrangement.py
git commit -m "feat(compose): on-grid accompaniment that keeps the tune on top and the chord intact

Arpeggio on straight eighths (was 11-tuplets), D-DU-UDU strum with string
stagger, alternating-bass fingerstyle, a final ringing chord. Thinning drops
voicing notes within a minor third of the melody but always keeps the bass and
restores any third/seventh the melody is not sounding.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Collision-free tab text

**Files:**
- Create: `src/music_decoder/compose/tabtext.py`
- Test: `tests/unit/test_compose_tabtext.py`

**Interfaces:**
- Consumes: `types` and `voicings_for` (tests).
- Produces, in `music_decoder.compose.tabtext`:
  - `BARS_PER_LINE = 4`, `CHORDS_PER_LINE = 8`
  - `string_labels(tuning) -> list[str]`
  - `render_melody_tab(melody_tab, *, n_bars, tempo_bpm, tuning) -> str`
  - `render_chord_shapes(voicings, tuning) -> str`
  - `render_composition_tab(melody_tab, voicings, *, bars_per_chord, tempo_bpm, tuning) -> str`, which returns `"Melody\n…\n\nChords\n…"`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_compose_tabtext.py`:

```python
import re

from music_decoder.compose.tabtext import (
    render_chord_shapes,
    render_composition_tab,
    render_melody_tab,
    string_labels,
)
from music_decoder.compose.voicings import voicings_for
from music_decoder.tabs.tuning import DADGAD, EB_HALF_STEP_DOWN, STANDARD_EADGBE
from music_decoder.types import ChordSymbol, Note, TabbedNote, TabPosition


def tabbed(start_s: float, string: int, fret: int) -> TabbedNote:
    pitch = STANDARD_EADGBE.open_pitches[string] + fret
    note = Note(start_s=start_s, end_s=start_s + 0.4, pitch=pitch, velocity=80, confidence=1.0)
    return TabbedNote(note=note, position=TabPosition(string=string, fret=fret))


def decode(block: str) -> list[tuple[int, int]]:
    """(string, fret) per melody note, reading each system's columns left to right."""
    out: list[tuple[int, int]] = []
    for system in block.split("\n\n"):
        rows = system.splitlines()[::-1]  # low string first
        hits: list[tuple[int, int, int]] = []
        for s, row in enumerate(rows):
            body = row.split("|", 1)[1]
            hits.extend((m.start(), s, int(m.group())) for m in re.finditer(r"\d+", body))
        out.extend((s, f) for _, s, f in sorted(hits))
    return out


def test_string_labels():
    assert string_labels(STANDARD_EADGBE) == ["E", "A", "D", "G", "B", "e"]
    assert string_labels(DADGAD) == ["D", "A", "D", "G", "A", "d"]
    assert string_labels(EB_HALF_STEP_DOWN) == ["D#", "G#", "C#", "F#", "A#", "d#"]


def test_melody_tab_keeps_every_note_including_two_digit_frets():
    notes = [tabbed(0.0, 3, 9), tabbed(0.5, 5, 0), tabbed(1.0, 4, 12), tabbed(2.0, 5, 15)]
    block = render_melody_tab(notes, n_bars=2, tempo_bpm=120.0, tuning=STANDARD_EADGBE)
    assert decode(block) == [(3, 9), (5, 0), (4, 12), (5, 15)]
    lines = block.splitlines()
    assert len(lines) == 6 and len({len(line) for line in lines}) == 1
    assert lines[0].count("|") == 3  # label bar line + 2 bar ends


def test_melody_tab_wraps_every_four_bars_and_marks_empty_bars():
    notes = [tabbed(2.0 * b, 5, b) for b in range(6) if b != 4]
    block = render_melody_tab(notes, n_bars=6, tempo_bpm=120.0, tuning=STANDARD_EADGBE)
    systems = block.split("\n\n")
    assert len(systems) == 2
    assert decode(block) == [(5, b) for b in range(6) if b != 4]
    assert "---|" in systems[1].splitlines()[0]  # bar 5 has no notes


def test_chord_shapes_block():
    shapes = [voicings_for(ChordSymbol.parse(c), STANDARD_EADGBE)[0] for c in ("C", "Am7")]
    block = render_chord_shapes(shapes, STANDARD_EADGBE)
    header, *rows = block.splitlines()
    assert header.split() == ["C", "Am7"]
    assert rows[0].startswith("e|-0") and rows[-1].startswith("E|-x")  # C: x32010
    assert rows[1].startswith("B|-1")


def test_composition_tab_does_not_let_chords_overwrite_the_melody():
    # review reproduction: an E4 on the G string, fret 9, at the same time as a C chord
    melody = [tabbed(0.0, 3, 9)]
    shapes = [voicings_for(ChordSymbol.parse("C"), STANDARD_EADGBE)[0]]
    text = render_composition_tab(
        melody, shapes, bars_per_chord=1, tempo_bpm=120.0, tuning=STANDARD_EADGBE
    )
    melody_block = text.split("\n\nChords\n")[0].removeprefix("Melody\n")
    assert decode(melody_block) == [(3, 9)]
    assert text.startswith("Melody\n") and "\n\nChords\n" in text
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/unit/test_compose_tabtext.py -q`
Expected: collection ERROR, `ModuleNotFoundError: No module named 'music_decoder.compose.tabtext'`.

- [ ] **Step 3: Write the implementation**

Create `src/music_decoder/compose/tabtext.py`:

```python
"""Plain-text tablature for a composition: the melody tab and the chord shapes.

Each melody note gets its own column in playing order (bars separated by
``|``, a few bars per line), so no note can overwrite another; the chord
shapes are listed separately under their names. Rhythm is not drawn: the
MIDI file is the source of truth for timing.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from music_decoder.types import VALID_TONICS, TabbedNote, Tuning, VoicedChord

BARS_PER_LINE = 4
CHORDS_PER_LINE = 8


def string_labels(tuning: Tuning) -> list[str]:
    """One label per string (low to high), equal width; a top string that
    repeats the bottom string's name is lower-cased, as in ``E A D G B e``."""
    names = [VALID_TONICS[p % 12] for p in tuning.open_pitches]
    if len(names) > 1 and names[-1] == names[0]:
        names[-1] = names[-1].lower()
    width = max(len(n) for n in names)
    return [n.ljust(width) for n in names]


def render_melody_tab(
    melody_tab: Sequence[TabbedNote], *, n_bars: int, tempo_bpm: float, tuning: Tuning
) -> str:
    labels = string_labels(tuning)
    bar_s = 4.0 * 60.0 / tempo_bpm
    by_bar: list[list[TabbedNote]] = [[] for _ in range(max(n_bars, 1))]
    for tn in sorted(melody_tab, key=lambda t: t.note.start_s):
        bar = min(math.floor(tn.note.start_s / bar_s + 1e-6), len(by_bar) - 1)
        by_bar[bar].append(tn)
    systems: list[str] = []
    for first in range(0, len(by_bar), BARS_PER_LINE):
        rows = [f"{label}|" for label in labels]
        for bar_notes in by_bar[first : first + BARS_PER_LINE]:
            for tn in bar_notes:
                text = str(tn.position.fret)
                for s in range(len(rows)):
                    rows[s] += "-" + text if s == tn.position.string else "-" * (len(text) + 1)
            rows = [row + ("-|" if bar_notes else "---|") for row in rows]
        systems.append("\n".join(reversed(rows)))
    return "\n\n".join(systems)


def render_chord_shapes(voicings: Sequence[VoicedChord], tuning: Tuning) -> str:
    labels = string_labels(tuning)
    pad = " " * (len(labels[0]) + 1)
    systems: list[str] = []
    for first in range(0, len(voicings), CHORDS_PER_LINE):
        chunk = voicings[first : first + CHORDS_PER_LINE]
        header = pad
        rows = [f"{label}|" for label in labels]
        for vc in chunk:
            name = vc.chord.to_label()
            frets = {p.string: ("x" if p.fret < 0 else str(p.fret)) for p in vc.positions}
            width = max(len(name), *(len(f) for f in frets.values())) + 2
            header += " " + name.ljust(width - 1)
            for s in range(len(rows)):
                text = frets.get(s, "x")
                rows[s] += "-" + text + "-" * (width - len(text) - 1)
        rows = [row + "|" for row in rows]
        systems.append("\n".join([header.rstrip(), *reversed(rows)]))
    return "\n\n".join(systems)


def render_composition_tab(
    melody_tab: Sequence[TabbedNote],
    voicings: Sequence[VoicedChord],
    *,
    bars_per_chord: int,
    tempo_bpm: float,
    tuning: Tuning,
) -> str:
    n_bars = len(voicings) * bars_per_chord
    melody = render_melody_tab(melody_tab, n_bars=n_bars, tempo_bpm=tempo_bpm, tuning=tuning)
    chords = render_chord_shapes(voicings, tuning)
    return f"Melody\n{melody}\n\nChords\n{chords}"
```

- [ ] **Step 4: Run the tests and gates**

Run: `pytest tests/unit/test_compose_tabtext.py -q`. Expected: `5 passed`.
Run: `ruff check src tests && ruff format --check src tests && mypy --strict src`. Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add src/music_decoder/compose/tabtext.py tests/unit/test_compose_tabtext.py
git commit -m "feat(compose): collision-free melody tab and chord-shape text

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Wire `arrange()`: pinned A\* tab, melody floors, new tab text, validation, version

**Files:**
- Modify (full rewrite): `src/music_decoder/compose/api.py`
- Modify (full rewrite, keeping the 3 existing tests at the top unchanged): `tests/unit/test_compose_api.py`
- Modify: `config/hyperparameters.yaml` (delete the unread `composition:` block)

**Interfaces:**
- Consumes:
  - `tabs.assigner.assign_tab(notes, *, tuning, weights, max_fret) -> TabAssignmentResult` (existing)
  - `render_composition_tab` (Task 6)
  - `essential_pcs` (Task 2)
  - `THIN_MARGIN_SEMITONES`, `_voiced_pitches` and `_accompaniment_notes` (Task 5)
  - `generate_melody(chord_floors=...)` (Task 4)
- Produces, in `music_decoder.compose.api`:
  - `COMPOSER_VERSION = 2`
  - `COMPOSE_TAB_WEIGHTS: dict[str, float]`, `COMPOSE_TAB_MAX_FRET = 22`
  - `_melody_floor(voicing, tuning) -> int | None`
  - `_validate(scale, progression, bars_per_chord)`
  - `arrange`, `accompany`, `Arrangement`: same names and fields as Task 3
  - `compose`: signature unchanged; `metadata` gains `"composer_version": 2`
  - `_ascii_tab_for` is removed (replaced by `render_composition_tab`)

- [ ] **Step 1: Write the failing test**

Replace the whole of `tests/unit/test_compose_api.py` with:

```python
import re
from pathlib import Path
from unittest import mock

import pytest

from music_decoder.compose import metrics
from music_decoder.compose.api import (
    COMPOSE_TAB_WEIGHTS,
    COMPOSER_VERSION,
    _melody_floor,
    _melody_tab,
    arrange,
    compose,
)
from music_decoder.compose.tabtext import render_melody_tab
from music_decoder.compose.voicings import voicings_for
from music_decoder.errors import InvalidProgressionError, InvalidScaleError
from music_decoder.tabs import _DEFAULT_WEIGHTS
from music_decoder.tabs.tuning import PRESETS, STANDARD_EADGBE
from music_decoder.types import ChordSymbol, Note, Scale


def test_compose_returns_paths_and_data(tmp_path):
    with mock.patch("music_decoder.compose.api.render_wav") as render:
        render.side_effect = lambda midi_path, out_path: out_path.write_bytes(b"RIFFWAV")
        result = compose(
            scale=Scale("C", "major"),
            progression=[ChordSymbol.parse(c) for c in ("Cmaj7", "Am7", "Dm7", "G7")],
            tempo_bpm=120.0,
            bars_per_chord=1,
            style="strum",
            seed=7,
            out_dir=tmp_path,
        )
    assert result.midi_path.exists()
    assert result.wav_path.exists()
    assert result.ascii_tab.strip() != ""
    assert len(result.melody_notes) > 0
    assert len(result.chord_voicings) == 4
    assert result.metadata["seed"] == 7


def test_compose_invalid_progression_raises():
    with pytest.raises(InvalidProgressionError):
        compose(
            scale=Scale("C", "major"),
            progression=[],  # empty
            out_dir=Path("/tmp"),
        )


def test_compose_invalid_scale_raises():
    # non-Literal mode is only constructible by going around the parser:
    bad = Scale(tonic="C", mode="dorian")  # type: ignore[arg-type]
    with pytest.raises(InvalidScaleError):
        compose(scale=bad, progression=[ChordSymbol.parse("C")], out_dir=Path("/tmp"))


# ---- arrange(): the filesystem-free core of compose() ----------------------

_PROG = [ChordSymbol.parse(c) for c in ("Cmaj7", "Am7", "Dm7", "G7")]


def _chords(*labels):
    return [ChordSymbol.parse(c) for c in labels]


def _decode(block):
    """(string, fret) per melody note from a rendered melody tab block."""
    out = []
    for system in block.split("\n\n"):
        rows = system.splitlines()[::-1]
        hits = []
        for s, row in enumerate(rows):
            body = row.split("|", 1)[1]
            hits.extend((m.start(), s, int(m.group())) for m in re.finditer(r"\d+", body))
        out.extend((s, f) for _, s, f in sorted(hits))
    return out


def test_arrange_is_deterministic_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    a = arrange(Scale("C", "major"), _PROG, tempo_bpm=120.0, style="strum", seed=3)
    b = arrange(Scale("C", "major"), _PROG, tempo_bpm=120.0, style="strum", seed=3)
    assert a == b
    assert list(tmp_path.iterdir()) == []
    assert len(a.voicings) == 4 and a.accompaniment


def test_arrange_reads_no_configuration():
    with (
        mock.patch("music_decoder.config.hyperparameters.load_hyperparameters") as load,
        mock.patch("music_decoder.tabs.assign_tabs") as configurable,
    ):
        arrange(Scale("C", "major"), _PROG, seed=3)
    load.assert_not_called()
    configurable.assert_not_called()


def test_arrange_ignores_the_analysis_hyperparameters_file(tmp_path, monkeypatch):
    before = arrange(Scale("G", "major"), _PROG, seed=5)
    other = tmp_path / "hp.yaml"
    other.write_text("id: test\ntab_assignment:\n  weights: {w_move: 9.0}\n  max_fret: 5\n")
    monkeypatch.setenv("MUSIC_DECODER_HYPERPARAMETERS", str(other))
    assert arrange(Scale("G", "major"), _PROG, seed=5) == before


def test_pinned_tab_weights_match_the_analysis_defaults():
    assert COMPOSE_TAB_WEIGHTS == _DEFAULT_WEIGHTS


def test_compose_returns_the_arrangement_and_version(tmp_path):
    with mock.patch("music_decoder.compose.api.render_wav"):
        result = compose(scale=Scale("C", "major"), progression=_PROG, seed=1, out_dir=tmp_path)
    arr = arrange(Scale("C", "major"), _PROG, seed=1)
    assert result.melody_notes == arr.melody
    assert result.chord_voicings == arr.voicings
    assert result.metadata["composer_version"] == COMPOSER_VERSION == 2


def test_arrange_tabs_every_melody_note_in_a_playable_register():
    arr = arrange(Scale("G", "major"), _PROG, bars_per_chord=2, seed=9)
    assert [tn.note for tn in arr.melody_tab] == list(arr.melody)
    for tn in arr.melody_tab:
        assert STANDARD_EADGBE.open_pitches[tn.position.string] + tn.position.fret == tn.note.pitch
    assert max(tn.position.fret for tn in arr.melody_tab) <= 15


def test_melody_tab_falls_back_when_astar_drops_a_note():
    melody = [
        Note(start_s=0.0, end_s=0.4, pitch=64, velocity=80, confidence=1.0),
        Note(start_s=0.5, end_s=0.9, pitch=67, velocity=80, confidence=1.0),
    ]
    empty = mock.Mock(tabbed_notes=[])
    with mock.patch("music_decoder.compose.api.assign_tab", return_value=empty):
        tab = _melody_tab(melody, STANDARD_EADGBE)
    assert [(t.position.string, t.position.fret) for t in tab] == [(5, 0), (5, 3)]


@pytest.mark.parametrize("bars", [0, -1])
def test_bars_per_chord_below_one_raises(bars):
    with pytest.raises(InvalidProgressionError):
        arrange(Scale("C", "major"), _PROG, bars_per_chord=bars)


@pytest.mark.parametrize("tuning_name", ["Drop_D", "DADGAD", "D_standard"])
def test_arrange_in_other_tunings_tabs_every_note(tuning_name):
    tuning = PRESETS[tuning_name]
    arr = arrange(Scale("D", "major"), _chords("D", "G", "A", "D"), tuning=tuning, seed=2)
    assert len(arr.melody_tab) == len(arr.melody)
    for tn in arr.melody_tab:
        assert tuning.open_pitches[tn.position.string] + tn.position.fret == tn.note.pitch


def test_melody_floor_sits_a_minor_third_above_the_defining_tones():
    d = voicings_for(ChordSymbol.parse("D"), STANDARD_EADGBE)[0]  # D3 A3 D4 F#4
    assert _melody_floor(d, STANDARD_EADGBE) == 66 + 3
    g7 = voicings_for(ChordSymbol.parse("G7"), STANDARD_EADGBE)[0]  # G2 B2 D3 G3 B3 F4
    assert _melody_floor(g7, STANDARD_EADGBE) == 65 + 3  # the seventh F4 is the highest


def test_d_major_cadence_has_its_third_under_the_melody():
    # review reproduction: D G A D, seed 9
    arr = arrange(Scale("D", "major"), _chords("D", "G", "A", "D"), tempo_bpm=120.0, seed=9)
    final_melody = [n for n in arr.melody if n.start_s >= 6.0]
    final_acc = [n for n in arr.accompaniment if n.start_s >= 6.0 - 1e-9]
    assert 6 in {n.pitch % 12 for n in final_acc}  # F# is heard
    assert max(n.pitch for n in final_acc) < min(n.pitch for n in final_melody)
    assert metrics.final_resolution(arr.melody, Scale("D", "major"), _chords("D", "G", "A", "D"))


@pytest.mark.parametrize(
    ("key", "labels", "bars", "seed"),
    [
        ("C", ("C", "G", "Am", "F"), 1, 0),  # review reproduction
        ("A#", ("A#", "Gm", "D#", "F", "Cm", "F", "A#", "A#"), 2, 1004),  # 16 bars
    ],
)
def test_rendered_tab_shows_every_melody_position(tmp_path, key, labels, bars, seed):
    with mock.patch("music_decoder.compose.api.render_wav"):
        result = compose(
            scale=Scale(key, "major"),
            progression=_chords(*labels),
            bars_per_chord=bars,
            tempo_bpm=120.0,
            seed=seed,
            out_dir=tmp_path,
        )
    arr = arrange(
        Scale(key, "major"), _chords(*labels), bars_per_chord=bars, tempo_bpm=120.0, seed=seed
    )
    melody_block = result.ascii_tab.split("\n\nChords\n")[0].removeprefix("Melody\n")
    expected = [(tn.position.string, tn.position.fret) for tn in arr.melody_tab]
    assert _decode(melody_block) == expected
    assert melody_block == render_melody_tab(
        arr.melody_tab, n_bars=len(labels) * bars, tempo_bpm=120.0, tuning=STANDARD_EADGBE
    )
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/unit/test_compose_api.py -q`
Expected: collection ERROR, `ImportError: cannot import name 'COMPOSE_TAB_WEIGHTS' from 'music_decoder.compose.api'`.

- [ ] **Step 3: Write the implementation**

Replace the whole of `src/music_decoder/compose/api.py` with:

```python
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
        if 0 <= fret <= 22 and (best is None or fret < best[0]):
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
```

In `config/hyperparameters.yaml`, delete exactly these lines; nothing reads them (spec D15):

```yaml
composition:
  notes_per_bar: 4
  strong_beat_chord_tone_prob: 0.7
  weak_beat_chord_tone_prob: 0.3
  markov_max_interval_semitones: 4
  velocity_strong: 80
  velocity_weak: 70
```

Then run `grep -rn "chord_tone_prob\|markov_max\|^composition:" src tests scripts config`. Expected: no output.

- [ ] **Step 4: Run the tests and gates**

Run: `pytest tests/unit/test_compose_api.py -q`. Expected: `19 passed`.
Run: `pytest -q tests/unit tests/integration && ruff check src tests && ruff format --check src tests && mypy --strict src`. Expected: 0 failures (including `test_public_contract.py`, unchanged), clean.
Run: `music-decoder compose --scale C:major --progression "Cmaj7 Am7 Dm7 G7" --style fingerstyle --tempo 100 --seed 42 --out "$TMPDIR/compose-smoke"`
Expected: exit 0. The output shows the `MIDI:` and `WAV:` paths, then `ASCII tab:`, a `Melody` block (`e|-12-10----12-|-8-10---------8-|…` on the first line) and a `Chords` block headed `Cmaj7  Am7  Dm7  G7`.

- [ ] **Step 5: Commit**

```bash
git add src/music_decoder/compose/api.py tests/unit/test_compose_api.py config/hyperparameters.yaml
git commit -m "feat(compose): config-free A* melody tab, melody floors, new tab text, composer_version

arrange() fingers the melody with assign_tab and compose-pinned weights (no
YAML read; MUSIC_DECODER_HYPERPARAMETERS cannot change compose output), keeps
the melody a minor third above each shape's identity tones, renders the
collision-free tab, rejects bars_per_chord < 1 and records composer_version=2.
Removes the unread 'composition' YAML block.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Musicality regression suite and documentation

**Files:**
- Create: `tests/unit/test_compose_musicality.py`
- Modify: `docs/architecture-technical.md` (§5 data flow, §5.2–§5.4.1, §6.2 YAML notes)
- Modify: `docs/architecture-overview.md` ("How the tool works" bullet; Compose steps 2, 3, 5)
- Modify: `docs/usage.md` (`--seed`/`--bars` bullets, example output, JSON example, style choices, seed reproducibility)

**Interfaces:**
- Consumes: `evaluate`, `STYLES` (Task 3); the committed baseline JSON (Task 3); `arrange` (Task 7).
- Produces: the paired regression suite, and docs matching the new behaviour.

- [ ] **Step 1: Write the regression test**

Create `tests/unit/test_compose_musicality.py`:

```python
"""Musicality regression test: the composer against its pre-change baseline.

``music_decoder.evaluation.compose_musicality.evaluate()`` generates the
held-out validation matrix (8 keys x 5 progressions per mode x 1-2 bars per
chord x seeds 1000-1004 = 400 melodies, each accompanied in 3 styles) and an
edge matrix (odd lengths, a single chord, secondary dominants, a borrowed
chord). ``tests/fixtures/compose/musicality_baseline.json`` holds the same
summaries for the pre-change composer, written by
``scripts/compose_musicality_report.py``. Targets and their reasons:
docs/superpowers/specs/2026-11-06-compose-improvements-design.md sections 3-4.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from music_decoder.compose.api import arrange
from music_decoder.evaluation.compose_musicality import STYLES, evaluate
from music_decoder.types import ChordSymbol, Scale

BASELINE_PATH = Path(__file__).parents[1] / "fixtures" / "compose" / "musicality_baseline.json"
GROUPS = ("overall", "major", "minor", "bars1", "bars2")

# Higher is better / lower is better; every subgroup must beat the baseline.
IMPROVE_UP = (
    "M1",
    "M2",
    "M7",
    "M8",
    "M9",
    "M10",
    "M11b",
    "M12",
    "A1.arpeggio",
    *(f"A2.{s}" for s in STYLES),
    *(f"A4.{s}" for s in STYLES),
)
IMPROVE_DOWN = ("M3", "M13", "T2", *(f"A3.{s}" for s in STYLES))


@pytest.fixture(scope="module")
def report():
    return evaluate()


@pytest.fixture(scope="module")
def baseline():
    return json.loads(BASELINE_PATH.read_text())


def _mean(report, metric, matrix="validation", group="overall"):
    return report[matrix][group][metric]["mean"]


# ---- absolute targets (validation matrix) -----------------------------------


@pytest.mark.parametrize(
    ("metric", "low", "high"),
    [
        ("M1", 0.95, None),  # strong-beat chord tones; baseline 0.863
        ("M2", 0.50, None),  # stepwise motion; baseline 0.352
        ("M3", None, 0.15),  # repeated notes; baseline 0.279
        ("M4", None, 0.02),  # leaps > a fifth; baseline 0.0005 (allowed rise, spec §3)
        ("M9", 3.0, None),  # distinct durations; baseline 1.0
        ("M10", 1.0, None),  # IOI entropy (bits); baseline 0.0
        ("M11a", 0.95, None),  # echo pairs that can carry the motif; baseline 1.0
        ("M11b", 0.60, None),  # motif contour agreement; baseline 0.336
        ("M12", 0.95, None),  # long phrase-final notes; baseline 0.0
        ("M13", None, 0.04),  # exposed chromatic clashes; baseline 0.094
        ("T2", None, 0.14),  # hand shifts per note; baseline 0.171
        *((f"A2.{s}", 0.97, None) for s in STYLES),  # melody on top; baseline 0.82-0.91
        *((f"A3.{s}", None, 0.10) for s in STYLES),  # exposed semitone clashes; 0.11-0.145
    ],
)
def test_validation_targets(report, metric, low, high):
    value = _mean(report, metric)
    if low is not None:
        assert value >= low, f"{metric} = {value}"
    if high is not None:
        assert value <= high, f"{metric} = {value}"


def test_validation_floors(report):
    o = report["validation"]["overall"]
    assert o["M1"]["min"] >= 0.60  # no piece ignores the harmony; baseline min 0.5
    assert o["M9"]["min"] >= 2  # no piece is rhythmically uniform; baseline 1


def test_pooled_leap_recovery_and_leading_tone_resolution(report):
    overall = report["validation"]["overall"]
    assert overall["M5"]["n"] >= 100 and overall["M5"]["pooled"] >= 0.70  # baseline 0.0 (n=5)
    assert overall["M15"]["n"] >= 20 and overall["M15"]["pooled"] >= 0.75  # baseline 0.385


@pytest.mark.parametrize("matrix", ["validation", "edge"])
def test_hard_invariants(report, matrix):
    o = report[matrix]["overall"]
    assert o["M4max"]["max"] <= 12  # no leap over an octave; baseline max 23
    assert o["M6"]["max"] <= 17  # ambitus; baseline max 24
    assert o["M7"]["min"] == 1.0  # every piece resolves; baseline 0.24
    assert o["M8"]["min"] >= 1.5  # final note >= 1.5 beats; baseline 0.95
    assert o["M12"]["min"] == 1.0  # every phrase ends long
    assert o["M14"]["min"] == 1.0  # melody on the eighth grid
    assert o["T1"]["max"] <= 15  # melody tab frets; baseline max 20
    assert o["Tcount"]["min"] == 1.0  # the tab has every melody note
    for style in STYLES:
        assert o[f"A1.{style}"]["min"] == 1.0  # baseline arpeggio 0.10
        assert o[f"A4.{style}"]["min"] == 1.0  # final chord rings
        assert o[f"H1.{style}"]["min"] == 1.0  # chord identity heard in every bar
        assert o[f"H1final.{style}"]["min"] == 1.0


# ---- paired comparison with the baseline ----------------------------------


@pytest.mark.parametrize("group", GROUPS)
def test_every_subgroup_improves_on_the_baseline(report, baseline, group):
    cand, base = report["validation"][group], baseline["validation"][group]
    for metric in (*IMPROVE_UP, *IMPROVE_DOWN):
        if metric not in cand or metric not in base:
            continue  # e.g. M13 is undefined over all-diatonic major progressions
        c, b = cand[metric]["mean"], base[metric]["mean"]
        assert (c > b) if metric in IMPROVE_UP else (c < b), f"{group} {metric}: {b} -> {c}"


def test_edge_matrix_improves_on_the_baseline(report, baseline):
    cand, base = report["edge"]["overall"], baseline["edge"]["overall"]
    for metric in (*IMPROVE_UP, *IMPROVE_DOWN):
        c, b = cand[metric]["mean"], base[metric]["mean"]
        assert (c > b) if metric in IMPROVE_UP else (c < b), f"edge {metric}: {b} -> {c}"


def test_relative_clash_reduction(report, baseline):
    for metric in ("M13", *(f"A3.{s}" for s in STYLES)):
        c, b = _mean(report, metric), _mean(baseline, metric)
        assert c <= 0.7 * b, f"{metric}: {b} -> {c}"


def test_guards_do_not_regress(report, baseline):
    for matrix in ("validation", "edge"):
        cand, base = report[matrix]["overall"], baseline[matrix]["overall"]
        assert cand["V1"]["mean"] <= base["V1"]["mean"] + 1e-9  # shapes are unchanged
        assert cand["M11a"]["mean"] >= base["M11a"]["mean"] - 0.05


def test_same_seed_same_music():
    progression = [ChordSymbol.parse(c) for c in ("Am", "F", "C", "G")]
    a = arrange(Scale("A", "minor"), progression, bars_per_chord=2, seed=1003)
    b = arrange(Scale("A", "minor"), progression, bars_per_chord=2, seed=1003)
    assert a == b
```

- [ ] **Step 2: Run it**

Run: `pytest tests/unit/test_compose_musicality.py -q --durations=3`
Expected: `30 passed`. Fixture setup took about 2–4.5 s in the dry run, depending on load; it is a single `evaluate()` call. This suite compares the finished composer with the baseline recorded in Task 3; the behavioural tests in Tasks 1–7 drove the implementation.

- [ ] **Step 3: Prove the suite bites (do not commit any of this)**

Apply each break below **on its own**, run the command, confirm the named test FAILS, then restore the file with `git checkout -- <file>`.

| # | Break | Command | Must fail |
|---|---|---|---|
| a | In `melody.py`, set `INTERVAL_WEIGHT` entry `0: -0.8` to `0: 3.0` | `pytest tests/unit/test_compose_musicality.py -q` | `test_validation_targets[M3-None-0.15]` (and M2) |
| b | In `arrangement.py` `_thin_below`, make the first line `return list(pitches)` | same | `test_validation_targets[A2.strum-0.97-None]` |
| c | In `api.py` `_melody_tab`, replace the `assigned = {...}` line with `assigned = {}` | same | `test_validation_targets[T2-None-0.14]` |
| d | In `harmony.py` `chord_scale_pcs`, return `frozenset(scale_set)` | same | `test_validation_targets[M13-None-0.04]` |
| e | In `melody.py`, set `LEADING_TONE_RESOLUTION = 0.0` | same | `test_pooled_leap_recovery_and_leading_tone_resolution` |
| f | In `rhythm.py`, add `(2, 2, 2, 2),` to `CADENCE_RHYTHMS` | same | `test_hard_invariants[validation]` |
| g | In `melody.py` `score_pitch`, delete the two-line final-target reachability check | `pytest tests/unit/test_compose_melody.py -q` | `test_no_interval_exceeds_an_octave_over_400_seeds` |
| h | In `api.py` `arrange`, delete the `chord_floors=...` argument | `pytest tests/unit/test_compose_api.py -q` | `test_d_major_cadence_has_its_third_under_the_melody` |

After restoring everything, run `git status --short`. Expected: no modified source files.

- [ ] **Step 4: Update `docs/architecture-technical.md`**

(a) Replace everything from the line ``## 5. Data flow: `compose()` `` up to, but not including, the line `### 5.1 Voicings` with:

````markdown
## 5. Data flow: `compose()`

Implemented in [compose/api.py](../src/music_decoder/compose/api.py). All the
musical work happens in the pure `arrange()`, which reads no files or
configuration. `compose()` adds only file I/O:

```text
(scale, progression, style, tempo, bars_per_chord, seed, tuning)
  │
  ▼
arrange(...)                                   # pure: no files, no YAML
  │ _validate(scale, progression, bars_per_chord)
  │   mode ∈ {major, minor}; progression non-empty; bars_per_chord ≥ 1.
  │   Otherwise raises InvalidScaleError / InvalidProgressionError.
  │ compose.voicings.voicings_for(chord, tuning)[0] for each chord
  │ _melody_floor(voicing) for each chord     # keep the tune above the third/seventh
  │ compose.melody.generate_melody(..., chord_floors=...)
  │ accompany(voicings, melody, ...)           # melody-aware accompaniment
  │ _melody_tab(melody, tuning)                # A* with pinned weights + fallback
  ▼
Arrangement(melody, voicings, accompaniment, melody_tab)
  │
  ▼
work = out_dir / f"{ts}-{payload_hash}"        # per-call directory
  │   payload_hash = sha1(scale|progression|tempo|style|seed)[:8]
  ▼
compose.arrangement.build_midi(...)            → composition.mid
  │
  ▼
synth.render_wav(...)                          → composition.wav
  │
  ▼
compose.tabtext.render_composition_tab(...)    → ASCII string
  │
  ▼
Composition(midi_path, wav_path, ascii_tab, melody_notes, chord_voicings, metadata)
```

`compose()` is fully synchronous. Determinism: the only source of randomness
is `numpy.random.default_rng(seed)` inside the melody generator. It drives
both the rhythm plan and the pitch choices, so the same seed gives the same
music. `metadata["composer_version"]` (currently `2`) is bumped whenever the
composer changes what a given seed produces. Seeds from before version 2
(the uniform quarter-note generator) give different melodies now. The melody
tab's A\* weights are pinned in `compose/api.py` (`COMPOSE_TAB_WEIGHTS`), so
editing `config/hyperparameters.yaml` or setting
`MUSIC_DECODER_HYPERPARAMETERS` changes `analyze()` but never compose
output. The output directory layout is content-addressed (`{timestamp}-{first
8 hex of payload SHA-1}`), so callers can keep several takes side by side.
````

(b) Replace everything from the line `### 5.2 Melody generator` up to, but not including, the line `### 5.5 Synthesis` with:

````markdown
### 5.2 Melody generator

[compose/melody.py](../src/music_decoder/compose/melody.py)
`generate_melody` builds the list of `Note` for the whole progression in two
layers. The pitch-class helpers it uses (scale, chord tones, essential tones,
chord scale) live in [compose/harmony.py](../src/music_decoder/compose/harmony.py).

**Rhythm and phrases**, from [compose/rhythm.py](../src/music_decoder/compose/rhythm.py):

1. `phrase_layout(n_bars)` fixes the phrase structure from the bar count
   alone. The piece is cut into 4-bar phrases. A remainder of 2 or 3 bars
   becomes a shorter last phrase, and a remainder of 1 turns the last 4 into
   3 + 2. The layout also gives each phrase's cadence bar and the motif
   *echo pairs*: bar 3 echoes bar 1 in a 4-bar antecedent, and a consequent
   phrase echoes the antecedent bar for bar. The metrics read the same
   layout.
2. `plan_bars(n_bars, rng)` gives every bar a rhythm (eighth-note values that
   sum to 8) from `BODY_RHYTHMS` (11 one-bar templates: quarters, eighth
   pairs, dotted figures, half notes). An echoing bar reuses its source bar's
   rhythm. Every phrase ends on a `CADENCE_RHYTHMS` bar (last note ≥ a half
   note).

**Pitches**, chosen one note at a time:

1. **Chord-scale pool.** `chord_scale_pcs(scale, chord)` is the scale (major,
   or natural minor) with each out-of-scale chord tone replacing the one
   degree it chromatically alters. The tone's role in the chord identifies
   that degree: a raised third replaces the degree below it, a lowered
   third or seventh the degree above. So E7 in A minor gives A harmonic
   minor (G→G#, the tonic A kept), and D7 in C major replaces F with F# and
   keeps G. Over diatonic chords the pool is just the scale.
2. **Register.** Candidates are the pool's pitches in
   `[pitch_low=62, pitch_high=79]` (D4–G5). Over each chord they are also
   kept at or above `chord_floors[i]`, a minor third above the voicing's
   highest identity tone (for an open D chord, F#4 + 3 = A4), unless that
   would leave fewer than 5 candidates.
3. **Final target.** In the last bar the final note is fixed in advance: the
   tonic if the final chord contains it, otherwise the chord root, nearest
   the previous pitch. Every note of that bar must stay within an octave of
   the target, so the move into the final note can never exceed 12
   semitones.
4. **Score** each candidate with `score_pitch` (weights are module constants):
   - interval size (steps preferred; repeats, wide leaps and the tritone
     penalised; more than an octave excluded)
   - chord fit by metric weight (chord tones strongly preferred on beats 1
     and 3 and on long notes; quarter-or-longer notes a semitone off a chord
     tone discouraged)
   - leap recovery (an opposite-direction step after a leap of a fourth or
     more)
   - **leading-tone resolution** (the leading tone steps up to the tonic
     when the harmony moves to a tonic-containing chord)
   - motif echo (same direction and size as the echoed bar's interval)
   - register gravity toward `tessitura_center=70`
   - a pull toward the final target in the last bar
   - a penalty on non-chord tones that end a phrase
5. **Sample** from the softmax of the scores with the seeded generator.

Velocity: 80 on beat 1, 75 on beat 3, 70 elsewhere, and 80 on the final note.
Each note sounds for 95% of its slot.

`notes_per_bar=N` (an integer) keeps the legacy uniform grid of N equal
notes per bar, still with the scorer and the final-note target. `compose()`
always uses the phrase planner.

### 5.3 Arrangement

[compose/arrangement.py](../src/music_decoder/compose/arrangement.py)
`build_midi` writes a two-track `pretty_midi.PrettyMIDI`:

- **Melody track:** program `25` (Acoustic Steel Guitar), one
  `pretty_midi.Note` per generated `Note`.
- **Accompaniment track:** same program. It is built by
  `_accompaniment_notes(..., melody=melody)` on a straight 4/4 grid:
  - `arpeggio`: 8 eighths per bar. The bass goes on the downbeat, then seven
    notes cycle up and back down the treble strings.
  - `strum`: strokes on eighths 0, 2, 3, 5, 6, 7 ("D · D U · U D U"). Down
    strokes play the voicing low→high with 8 ms between strings. Up strokes
    play the top 3 strings high→low. Each stroke rings until the next one.
  - `fingerstyle`: alternating bass. The root goes on beat 1 and the
    second-lowest voicing note on beat 3 (when the voicing has ≥ 4 notes).
    The treble pinch plays on beats 2 and 4.
  - **Final bar (every style):** one down stroke (10 ms per string) that
    rings to the end of the bar.
- **Melody-aware thinning** (`_thin_below`): at each attack, voicing pitches
  that come within a minor third of the lowest melody note sounding over
  that attack are dropped. Two rules override this:
  - The bass is always kept.
  - Each **essential** tone (third, seventh, altered fifth or suspended
    fourth, from `harmony.essential_pcs`) that neither the kept notes nor
    the melody are sounding is put back at its lowest voicing pitch.

  So the chord never loses its identity. Combined with the melody floors in
  §5.2, the tune stays on top of every attack in the evaluation matrix.

Accompaniment pitches come from `_voiced_pitches(voicing, tuning) =
sorted(open_pitches[s] + fret for fretted strings)`.

### 5.4 ASCII tab for the composition

`arrange()` fingers the melody with the A\* search that `analyze()` also uses
(`music_decoder.tabs.assigner.assign_tab`). The weights and fret cap are
pinned in `compose/api.py` rather than read from YAML. A note A\* cannot
place falls back to its lowest-fret position, so no note is lost.

[compose/tabtext.py](../src/music_decoder/compose/tabtext.py)
`render_composition_tab` prints two labelled blocks:

- **`Melody`:** one column per note, in playing order, with bars separated
  by `|` and four bars per line. No note can overwrite another.
- **`Chords`:** each chord's shape under its name (`x` = muted string).

String labels follow the tuning (`E A D G B e`, `D A D G A d`, …). Rhythm is
not drawn; the MIDI file is the source of truth for timing. (The shared
`tabs.render.render_ascii_tab` used by `analyze` is unchanged.)

### 5.4.1 Musicality metrics and evaluation

[compose/metrics.py](../src/music_decoder/compose/metrics.py) holds pure,
typed metric functions:

- melody: strong-beat chord tones, step / repeat / large-leap ratios, pooled
  leap recovery, ambitus, final resolution and length, distinct durations
  and IOI entropy, motif echo coverage and contour agreement, phrase-end
  length, exposed chromatic clashes, leading-tone resolution
- accompaniment: grid alignment, melody above the accompaniment and exposed
  semitone clashes (both judged over each melody note's whole sounding
  interval), final-chord ring, essential-tone coverage per bar
- tab: hand-position shifts
- voicings: nearest-pitch proximity (a proximity guard, not a voice-leading
  score)

[evaluation/compose_musicality.py](../src/music_decoder/evaluation/compose_musicality.py)
defines two held-out input matrices (seeds 1000+ were never used for
tuning):

- a 400-melody validation matrix: 8 keys, both modes, 10 progressions, 1–2
  bars per chord, × 3 styles
- an 80-melody edge matrix: 3/5/7/9 bars, a single chord, secondary
  dominants, a borrowed chord

It summarises every metric overall and per mode and bar length.
`scripts/compose_musicality_report.py` prints that summary.
`tests/fixtures/compose/musicality_baseline.json` is the same summary for
the pre-change composer, measured on identical inputs.
[tests/unit/test_compose_musicality.py](../tests/unit/test_compose_musicality.py)
asserts two things:

- absolute targets
- that every subgroup improves on that baseline

The definitions, numbers and reasons are in
[the design spec](superpowers/specs/2026-11-06-compose-improvements-design.md).
````

(c) In §6.2, delete these two lines:

```markdown
- `composition` — notes per bar, strong/weak chord-tone probabilities,
  Markov interval cap, strong/weak velocities.
```

Then replace the sentence

```markdown
The adapters read this YAML at call time, so editing it changes the next
`analyze()` without a code change.
```

with:

```markdown
The adapters read this YAML at call time, so editing it changes the next
`analyze()` without a code change. `compose()` reads none of it: the composer's
parameters, including its A* tab weights, are pinned in code and versioned by
`metadata["composer_version"]`.
```

- [ ] **Step 5: Update `docs/architecture-overview.md`**

(a) Replace this bullet:

```markdown
- **Composing the melody** from a chord progression by stepping
  through the bars and rolling weighted dice — chord tones on the
  strong beats, scale tones on the weak ones, with a soft preference
  for staying close to the previous note.
```

with:

```markdown
- **Composing the melody** from a chord progression in four-bar
  phrases: rhythms from a small pattern library with a repeating
  motif, each pitch picked by weighted dice that favour chord tones on
  strong beats and small steps, and a final cadence on the home note.
```

(b) Under `## What happens when you click "Compose"?`, replace steps **2** and **3** (from the line starting `2. **Write the melody, bar by bar**.` through the line ending `...fingerstyle bass-and-treble alternation).`) with:

```markdown
2. **Write the melody, phrase by phrase**. The tool splits the piece into
   four-bar phrases, the way a song verse is built:
   - Each bar gets a rhythm from a small library of patterns: quarter notes,
     pairs of eighths, dotted rhythms, held notes. The third bar of a phrase
     repeats the first bar's rhythm and its rising-and-falling shape, and the
     next phrase answers the first one with the same rhythms. That
     repetition is what makes a tune feel like a tune.
   - Every phrase ends on a long note. The very last note lands on the home
     note of the key (or, if the last chord doesn't contain it, on that
     chord's root), so the piece sounds finished.
   - For each note it scores every candidate pitch and rolls weighted dice.
     Notes that belong to the current chord get a strong preference on the
     strong beats (beats 1 and 3). Small steps are preferred over jumps, and
     after a big jump the line steps back the other way. The "leading tone"
     (the note just below home) rises to the home note when the harmony
     comes home. The line stays in a comfortable register above the chord.
   - If a chord borrows a note from outside the scale (like G# in an E7
     chord in A minor), the melody borrows it too, in place of the note it
     replaces (G), instead of clashing.
   The "seed" you pass is the only source of randomness: same seed,
   same melody, every time.

3. **Add the accompaniment**. Underneath the melody, the tool plays the
   chord on every bar in your chosen pattern. The patterns are: a pop strum
   (down, down-up, up-down-up), an eighth-note arpeggio, or fingerstyle with
   an alternating bass. It leaves out chord notes that would crowd the
   melody, but it never drops the notes that make the chord major, minor or
   a seventh. It ends with one chord left ringing.
```

(c) Replace step **5** (from `5. **Write the tab**.` through `what you might see in any guitar magazine.`) with:

```markdown
5. **Write the tab**. The melody notes get placed onto the guitar
   neck by the same fingering search used for analysis, so the line
   stays in one hand position where it can. The tab shows the melody
   note by note, then the chord shapes under their names, in the kind
   of ASCII tab you might see in any guitar magazine.
```

- [ ] **Step 6: Update `docs/usage.md`**

(a) In the `compose` options, replace:

```markdown
- `--seed` — integer; same seed → same melody. Default: a fresh random
  seed each run.
```

with:

```markdown
- `--seed` — integer; same seed → same music (for a given
  `metadata.composer_version`). Default: a fresh random seed each run.
- `--bars` must be ≥ 1; `0` or a negative number is rejected with an error.
```

(b) In the I-vi-ii-V example's "Expected pretty output" block, replace everything from `MIDI:    out/20260508T142510-7c4a3b21/composition.mid` through `E|-------------------|` with:

```text
MIDI:    out/20260508T142510-7c4a3b21/composition.mid
WAV:     out/20260508T142510-7c4a3b21/composition.wav
ASCII tab:
Melody
e|-12-10----12-|-8-10---------8-|----8-------|-10----10-15-|
B|-------12----|------12-10-8---|-10---10-12-|----12-------|
G|-------------|----------------|------------|-------------|
D|-------------|----------------|------------|-------------|
A|-------------|----------------|------------|-------------|
E|-------------|----------------|------------|-------------|

Chords
   Cmaj7  Am7  Dm7  G7
e|-0------0----1----1--|
B|-0------1----1----0--|
G|-0------0----2----0--|
D|-2------2----0----0--|
A|-3------0----x----2--|
E|-x------x----x----3--|
```

and add this paragraph right after that code block:

```markdown
The tab has two parts: **Melody** lists every melody note in playing
order (bars separated by `|`, four bars per line), fingered by the same
search `analyze` uses; **Chords** shows each chord shape under its name
(`x` = muted string). Rhythm is not drawn; the MIDI file has the exact timing.
```

(c) In the JSON example under `music-decoder compose --format json`, replace `  "ascii_tab": "e|---5...",` with `  "ascii_tab": "Melody\ne|-12-10...",`. Then replace the line `    "tuning": "EADGBE"` (inside `"metadata"`) with these two lines:

```text
    "tuning": "EADGBE",
    "composer_version": 2
```

(d) Replace the bullets under `### Composition style choices`, from ``- **`fingerstyle`**`` through `delicate, classical-leaning pieces.`, with:

```markdown
- **`fingerstyle`** — alternating bass (root on beat 1, another chord
  tone on beat 3) with a treble pinch on beats 2 and 4. Best for
  ballads, folk and slow tempos. It is the default because it sounds
  reasonable on most progressions.
- **`strum`** — a pop strum on the eighth-note grid (down, down-up,
  up-down-up), with strings staggered a few milliseconds like a real
  pick stroke. Best for pop and rock progressions.
- **`arpeggio`** — bass on the downbeat, then seven eighth notes rolling
  up and down the chord. Best for delicate, classical-leaning pieces.

In every style the accompaniment leaves out chord notes that would sit
on or just under the melody (never the bass, and never the notes that
make the chord major, minor or a seventh), and the piece ends on one
ringing chord. The melody is written in four-bar phrases with a
repeated motif, resolves leading tones, follows altered chords (G# over
E7 in A minor), and ends on a long tonic (or on the last chord's root
if that chord does not contain the tonic).
```

(e) Under `### Seed reproducibility`, after the paragraph that starts `` `--seed N` makes `compose()` fully deterministic. `` and its bullet list, add:

```markdown
Seeds are reproducible within one composer version, which is recorded
in `metadata.composer_version` (currently `2`). Version 2 (November
2026) replaced the uniform quarter-note generator with phrase-planned
melodies, so a seed saved from an older version now gives a different
(more musical) take. Compose output does not depend on
`config/hyperparameters.yaml` or `MUSIC_DECODER_HYPERPARAMETERS`.

To measure the composer, run `python scripts/compose_musicality_report.py`.
It prints every musicality metric over the fixed evaluation matrices;
`tests/fixtures/compose/musicality_baseline.json` holds the
pre-version-2 numbers for comparison.
```

- [ ] **Step 7: Run every gate and the stale-text check**

Run: `pytest -q tests/unit tests/integration && ruff check src tests && ruff format --check src tests && mypy --strict src`
Expected: 0 failures (dry run: 517 passed, 3 skipped), ruff clean, `Success: no issues found`.
Run: `grep -n "bar by bar\|greedy-assigns\|block chord on beats 1 and 3\|Markov interval cap\|11-tuplet" docs/*.md`
Expected: no output.
Run: `git status --short`. Expected: only the four files of this task, plus regenerated `*.synth_marker` files, which you must not stage.

- [ ] **Step 8: Commit**

```bash
git add tests/unit/test_compose_musicality.py docs/architecture-technical.md \
  docs/architecture-overview.md docs/usage.md
git commit -m "test(compose): paired musicality regression suite; docs for composer v2

Compares the composer with the committed pre-change baseline on identical
held-out inputs: absolute targets, hard invariants on validation and edge
matrices, per-subgroup improvement and relative clash reduction. Documents
the phrase planner, chord-scale rule, identity-preserving accompaniment, tab
text, pinned tab weights and composer_version.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## Review responses (round 1)

Review: `docs/superpowers/reviews/2026-11-06-plan-review-r1.md` (GPT-6 Astra, changes requested). Every finding was reproduced before it was acted on: the plan's own code was run in a scratch copy, the review's seeds and examples were re-run, and the YAML read path was traced. All ten reproduce. Revision 2 of the spec (§11) mirrors this table.

| # | Finding (severity) | Verified? | Outcome | What changed |
|---|---|---|---|---|
| F1 | Forced final target bypasses the octave limit (Important) | Yes. Seed 283 ends `68, 64, 63, 76` (13 semitones); seed 5774 jumps 16. | **Fixed** | Spec §5.3 / Task 4: in the final bar, candidates more than 12 from the target are `-inf`, so the target is always reachable (it satisfies both constraints itself), with a deterministic nearest-pitch fallback. Tests: the review seeds, a 400-seed × 3-progression sweep, and an adversarial `score_pitch` case. Hard M4max ≤ 12 on both evaluation matrices. |
| F2 | Chord-scale rule deletes the tonic; M13 too blunt (Important) | Yes. The pool over E7 in A minor was `{C, D, E, F, G#, B}`, with no A. | **Fixed** | New `harmony.chord_scale_pcs`: the altered tone replaces only the degree it alters, identified by its role in the chord, so E7 gives A harmonic minor and D7 keeps G. Tests cover E7, D7, C7, Fm, A, B♭ and C#maj7 in `test_compose_harmony.py`; melody tests show A over E7 and G over D7 are available. M13 is redefined as *exposed non-chord tones* a semitone from a foreign tone (chord tones and short passing notes excluded), with a target of ≤ 0.04 and ≤ 0.7 × baseline, not a hard zero. New M15 (leading-tone resolution), a +3.0 scorer term, and a pooled ≥ 0.75 target. D7's "harmonic minor" claim is now accurate. |
| F3 | A2/A3 probe only the onset (Important) | Yes. The review's C4 / C#4 +8 ms example gave A2 = 1.0, A3 = 0.0. | **Fixed** | A2/A3 now use any overlap during the melody note (ε = 0.1 ms). New tests: the review's staggered case (A2 = 0, A3 = 1), a late entry, and before/after non-overlap controls. The baseline was re-measured with the same definitions (strum A2 0.821, A3 0.145), and the targets were reset on those numbers. |
| F4 | Thinning can erase chord identity (Important) | Yes. D major `D G A D`, seed 9, ended on D3 A3 D4 under a D4 melody, with no F# and a unison doubling. | **Fixed** | `_thin_below` always keeps the bass and restores any essential tone (third, seventh, altered fifth, fourth) that neither the kept notes nor the melody supply; it no longer keeps the "lowest three". `arrange()` passes per-chord **melody floors** (a minor third above the shape's identity tones), so the restored tones sit under the tune; A2 = 1.0 in every style. New metric H1 (essential-tone coverage per bar) is asserted 1.0 everywhere, including the final bar. Tests: the D-major case in every style, minor and seventh examples, and "the melody supplies the third". No voicing change was needed. |
| F5 | Rendered ASCII tab can erase melody frets (Important) | Yes. The first E4 (G string, fret 9) was rendered as the C chord's `0`. | **Fixed** | New `compose/tabtext.py` renders a `Melody` block (one column per note; nothing can overwrite anything) and a `Chords` block. The shared `tabs/render.py` is untouched. Tests: a decode-the-rendered-tab round trip on the review's `C G Am F` seed 0 and a 16-bar piece with two-digit frets; an overwrite reproduction; label, wrapping and empty-bar tests. |
| F6 | `arrange()` reads YAML / env configuration (Important) | Yes. `assign_tabs` → `_yaml_tab_params` → `load_hyperparameters` on every call, honouring `MUSIC_DECODER_HYPERPARAMETERS`. | **Fixed** | `_melody_tab` calls the lower-level `tabs.assigner.assign_tab` with `COMPOSE_TAB_WEIGHTS` / `COMPOSE_TAB_MAX_FRET`, pinned in `compose/api.py` (a test pins them equal to the analysis defaults). Tests: `load_hyperparameters` and `assign_tabs` are never called, and pointing `MUSIC_DECODER_HYPERPARAMETERS` at a different YAML leaves `arrange()` unchanged. The unread `composition:` YAML block is deleted (D15). |
| F7 | Thresholds calibrated to the prototype, not shown against the baseline (Important) | Yes. Targets came from an uncommitted script on a different matrix and overlapping seeds. | **Fixed** | Task 3 records the **old composer's** metrics on committed matrices before any behaviour change: a behaviour-preserving `arrange()` checked against a golden run, the `evaluation/compose_musicality.py` matrices, a report script, and the baseline JSON. The validation matrix uses held-out seeds 1000–1004; the edge matrix covers 3/5/7/9 bars, a single chord, a secondary dominant, a borrowed chord and the review cases. The regression suite asserts musically justified absolute targets (spec §3.2 "Why"), improvement over the baseline in every subgroup (major, minor, 1 and 2 bars, edge), relative clash reductions and hard invariants. The one allowed regression (M4) is named and bounded. Ten deliberate breaks were each confirmed to turn the suite red; eight of them are Task 8 Step 3. Behavioural failing tests live in their implementation tasks. |
| F8 | Phrase metrics ignore the planner's boundaries (Minor) | Yes. A 5-bar C-major piece scored M12 = 0 although its phrase ends were long. | **Fixed** | `rhythm.phrase_layout(n_bars)` is shared by `plan_bars` and the metrics. M12 uses the layout's cadence bars. M11 is split into coverage (M11a) and agreement (M11b), with undefined results kept as `None`. Tests: layouts for 1/3/4/5/7/8/9 bars, a 5-bar motif case, a 5-bar phrase-end case, and unequal echo bars (coverage 0, contour `None`). |
| F9 | V1 is not a voice-leading measure (Minor) | Yes, by construction. | **Partly fixed, partly rebutted** | Renamed to `nearest_pitch_proximity`; the docstring and spec now say what it cannot show; a test pins the blind spot (a three-note shape collapsing onto one pitch scores 0); the "voice leading is already fine" claim is withdrawn. **Rebutted:** building a one-to-one voice-leading metric with birth/drop costs. Chord shapes are out of scope (§9), so the metric would only guard code this branch does not change (YAGNI). The thinned, sounding harmony is now guarded by H1 instead. |
| F10 | Five-second budget not met (Minor) | Partly. The YAML parse was more than half of the A\* time (12.9 ms vs 5.7 ms per call). | **Fixed (measured); budget restated** | The YAML reads are gone (F6). A profile shows A\* in `tabs/` (out of scope) is about 85% of the remaining time. In the dry run, `evaluate()` for 480 cases took 2.3–3.0 s standalone, and the test module 2.3–4.7 s under pytest, depending on load. As the review advised, there is no wall-clock assertion; the expectation is recorded in the Global Constraints and spec §7. **Not done:** shrinking the matrix to hit a number, because coverage was the point of F7. |
