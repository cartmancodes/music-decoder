# Compose Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `music_decoder.compose` output noticeably more musical: phrase-structured rhythm with motifs and cadences, scored pitch choice, on-grid melody-aware accompaniment and a playable A\* melody tab. Every improvement is proven by objective metrics asserted over 400 generated melodies × 3 styles.

**Architecture:** A new `compose/rhythm.py` plans bar rhythms in antecedent/consequent phrases. `compose/melody.py` samples each pitch from a softmax over musical scores, using a scale adapted to each chord. `compose/arrangement.py` plays grid-aligned patterns that drop chord notes crowding the melody. `compose/api.py` gains a pure `arrange()` that `compose()` wraps with file I/O. `compose/metrics.py` holds the metric functions that the regression test asserts.

**Tech Stack:** Python 3.11, numpy (seeded `default_rng`), pretty_midi, the existing A\* tab assigner (`music_decoder.tabs.assign_tabs`), pytest, ruff, mypy --strict. No new dependencies.

**Execution method:** superpowers:subagent-driven-development. The pipeline pre-approved this: Opus implements and GPT-6 Astra (Codex) reviews.

**Spec:** `docs/superpowers/specs/2026-11-06-compose-improvements-design.md`. Read §3 (baseline and targets), §4 (metric definitions), §5 (design) and §10 (decisions).

**Dry run:** every code block below was built and run in a scratch copy of this repo before the plan was written. The plan was also replayed task by task on a fresh copy: after every task all four gates were green, ending at `pytest -q tests/unit tests/integration` 430 passed, 3 skipped. `ruff check`, `ruff format --check` (touched files) and `mypy --strict src` were clean, and the musicality test ran in about 2.6 s. If a step's result differs from its "Expected" line, stop and investigate; do not loosen a threshold to make it pass.

## Global Constraints

- The `compose()` signature is frozen: `tests/unit/test_public_contract.py::test_compose_signature_is_frozen` must pass unchanged. Add no parameters to `compose()` and no CLI flags.
- Output is deterministic for a given seed: `numpy.random.default_rng(seed)` is the only source of randomness.
- No new dependencies and no model downloads. Use numpy and pretty_midi only. Do not install packages.
- Gate commands, all green after every task: `pytest -q tests/unit tests/integration`, `ruff check src tests`, `ruff format --check src tests`, `mypy --strict src`.
- `tests/unit/test_compose_musicality.py` must run in < 5 s.
- Line length 100; ruff rules `E F W I B UP RUF` (see `pyproject.toml`).
- Work only in this worktree on branch `feat/compose-improve`. Never push, merge or switch branches.
- **Never `git add -A` or `git add .`.** `tests/fixtures/synthetic/.c_major_scale.synth_marker` has an unrelated pre-existing modification that must not be committed. Stage the exact paths listed in each task.
- Every commit message ends with the line `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Do not change `compose/voicings.py`, `chords/`, `tabs/`, `synth/` or the analyze pipeline (spec §9).

## Review Focus

1. **`bars_per_chord` of 0 or below.** It used to give an empty melody silently. It must now raise `InvalidProgressionError`, which the CLI shows as a clean error. Pinned by `test_bars_per_chord_below_one_raises` (Task 5).
2. **Single-chord, single-bar progressions** (e.g. `--progression "D"`). A melody must still be produced, end on the tonic and finish with a ringing chord. Pinned by `test_single_bar_progression` (Task 3) and `test_single_bar_accompaniment_is_the_final_chord` (Task 4).
3. **Non-standard tunings** (Drop D, DADGAD, D standard). Every melody note must appear in the tab at a position that really produces its pitch. Pinned by `test_arrange_in_other_tunings_tabs_every_note` (Task 5).
4. **Extreme tempos** (40 and 220 BPM, the UI slider's limits). The rhythm grid and the total length must stay exact. Pinned by `test_extreme_tempos_keep_the_grid_and_the_length` (Task 3).
5. **Non-diatonic chords in a major key** (a secondary dominant D7 in C major). The melody must adapt (no F or G against the F#) and not clash. Pinned by `test_secondary_dominant_in_major_adapts_the_scale` (Task 3).

---

### Task 0: Restore the `ruff format` gate (pre-existing drift)

On the base commit `ruff format --check src tests` already fails on 7 files
(spec D12). Fix that in a format-only commit so every later task can hold
the gate.

**Files:**
- Modify (formatting only): `src/music_decoder/compose/api.py`, `src/music_decoder/config/hyperparameters.py`, `src/music_decoder/synth/__init__.py`, `tests/unit/test_config_hyperparameters_v3.py`, `tests/unit/test_public_contract.py`, `tests/unit/test_tab_assignment.py`, `tests/unit/test_transcription_decode.py`

**Interfaces:**
- Consumes: nothing.
- Produces: a green `ruff format --check src tests`.

- [ ] **Step 1: Confirm the drift**

Run: `ruff format --check src tests`
Expected: `7 files would be reformatted`, listing exactly the 7 files above.

- [ ] **Step 2: Format exactly those files**

```bash
ruff format src/music_decoder/compose/api.py src/music_decoder/config/hyperparameters.py \
  src/music_decoder/synth/__init__.py tests/unit/test_config_hyperparameters_v3.py \
  tests/unit/test_public_contract.py tests/unit/test_tab_assignment.py \
  tests/unit/test_transcription_decode.py
```

- [ ] **Step 3: Verify the gates**

Run: `ruff format --check src tests && ruff check src tests && mypy --strict src && pytest -q tests/unit tests/integration`
Expected: all clean; pytest gives the same pass and skip counts as before the change (no failures).
Run: `git diff --stat`. Only the 7 files are listed, plus the pre-existing `.c_major_scale.synth_marker`, which must not be staged.

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

### Task 1: Musicality metrics module

**Files:**
- Create: `src/music_decoder/compose/metrics.py`
- Test: `tests/unit/test_compose_metrics.py`

**Interfaces:**
- Consumes: `music_decoder.compose.melody._chord_tones(chord) -> tuple[int, ...]` and `_scale_pcs(scale) -> tuple[int, ...]`. Both exist today, and Task 3 keeps them with the same names and behaviour.
- Produces, all in `music_decoder.compose.metrics` (later tasks use these exact names):
  - `chord_at(t, progression, bars_per_chord, tempo_bpm) -> ChordSymbol`
  - `metric_position(start_s, tempo_bpm) -> int`
  - `intervals(melody) -> list[int]`
  - `strong_beat_chord_tone_ratio(melody, progression, bars_per_chord, tempo_bpm) -> float | None`
  - `step_ratio(melody)`, `repeat_ratio(melody)`, `large_leap_ratio(melody)`, each `-> float | None`
  - `max_interval(melody) -> int`
  - `leap_recovery_counts(melody) -> tuple[int, int]`
  - `ambitus(melody) -> int`
  - `final_resolution(melody, scale, progression) -> bool`
  - `final_note_beats(melody, tempo_bpm) -> float`
  - `distinct_durations(melody, tempo_bpm) -> int`
  - `ioi_entropy(melody, tempo_bpm) -> float`
  - `motif_contour_match(melody, tempo_bpm, n_bars) -> float | None`
  - `phrase_end_long_ratio(melody, tempo_bpm, n_bars) -> float | None`
  - `nondiatonic_clash_ratio(melody, scale, progression, bars_per_chord, tempo_bpm) -> float | None`
  - `onset_grid_ratio(onsets, tempo_bpm, per_beat) -> float | None`
  - `stroke_heads(onsets) -> list[float]`
  - `melody_above_accompaniment_ratio(melody, accompaniment) -> float | None`
  - `exposed_semitone_clash_ratio(melody, accompaniment, tempo_bpm) -> float | None`
  - `final_chord_rings(accompaniment, n_bars, tempo_bpm) -> bool`
  - `hand_shift_rate(frets) -> float | None`
  - `voice_leading_distance(voicings, tuning) -> float | None`

  `melody` and `accompaniment` are `Sequence[Note]` (`Note` is `music_decoder.types.Note`). `onsets` is `Sequence[float]` seconds and `frets` is `Sequence[int]`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_compose_metrics.py`. Each metric is checked on a hand-built example with a known answer (at 120 BPM one beat is 0.5 s):

```python
import math

import pytest

from music_decoder.compose import metrics as m
from music_decoder.compose.voicings import voicings_for
from music_decoder.tabs.tuning import STANDARD_EADGBE
from music_decoder.types import ChordSymbol, Note, Scale

TEMPO = 120.0  # one beat = 0.5 s
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


def test_motif_contour_match():
    bar1 = [n(0, 1, 60), n(1, 1, 62), n(2, 1, 64), n(3, 1, 62)]  # up up down
    bar2 = [n(4, 4, 67)]
    bar3 = [n(8, 1, 65), n(9, 1, 67), n(10, 1, 65), n(11, 1, 64)]  # up down down
    bar4 = [n(12, 4, 60)]
    mel = bar1 + bar2 + bar3 + bar4
    assert m.motif_contour_match(mel, TEMPO, 4) == pytest.approx(2 / 3)
    assert m.motif_contour_match(bar1, TEMPO, 1) is None


def test_phrase_end_long_ratio():
    short_end = [n(i, 0.95, 60) for i in range(16)]
    assert m.phrase_end_long_ratio(short_end, TEMPO, 4) == pytest.approx(0.0)
    long_end = [n(i, 0.95, 60) for i in range(12)] + [n(12, 2, 60), n(14, 1.9, 60)]
    assert m.phrase_end_long_ratio(long_end, TEMPO, 4) == pytest.approx(1.0)
    assert m.phrase_end_long_ratio(long_end, TEMPO, 3) is None


def test_nondiatonic_clash_ratio():
    p = prog("Am", "E7")
    # bar 2 is E7 (G# foreign to A natural minor): G (67) clashes, A (69) clashes, B (71) not
    mel = [n(0, 1, 67), n(4, 1, 67), n(5, 1, 69), n(6, 1, 71)]
    assert m.nondiatonic_clash_ratio(mel, A_MINOR, p, 1, TEMPO) == pytest.approx(2 / 4)
    assert m.nondiatonic_clash_ratio(mel, C, prog("C", "G"), 1, TEMPO) is None


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


def test_final_chord_rings():
    ring = [n(4, 4, 48), n(4.02, 3.98, 52)]
    assert m.final_chord_rings(ring, 2, TEMPO)
    restruck = [n(4, 2, 48), n(6, 2, 48)]
    assert not m.final_chord_rings(restruck, 2, TEMPO)
    assert not m.final_chord_rings([], 2, TEMPO)


def test_hand_shift_rate():
    assert m.hand_shift_rate([5, 7, 8, 9]) == pytest.approx(0.0)
    assert m.hand_shift_rate([5, 7, 12, 13]) == pytest.approx(1 / 3)
    assert m.hand_shift_rate([0, 0, 5]) is None  # open strings do not count


def test_voice_leading_distance():
    v = [voicings_for(c, STANDARD_EADGBE)[0] for c in prog("C", "C", "Am")]
    d = m.voice_leading_distance(v, STANDARD_EADGBE)
    assert d is not None and 0.0 < d < 2.0
    assert m.voice_leading_distance(v[:2], STANDARD_EADGBE) is None  # same chord only
    assert not math.isnan(d)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/unit/test_compose_metrics.py -q`
Expected: collection ERROR, `ImportError: cannot import name 'metrics' from 'music_decoder.compose'`.

- [ ] **Step 3: Write the implementation**

Create `src/music_decoder/compose/metrics.py`:

```python
"""Objective musicality metrics for composer output.

Pure functions over melody / accompaniment notes, asserted across a matrix of
keys, progressions, seeds and styles by ``tests/unit/test_compose_musicality.py``.
Definitions follow ``docs/superpowers/specs/2026-11-06-compose-improvements-design.md``
section 4. Times are in seconds; ``tempo_bpm`` fixes the 4/4 beat grid.
Ratio metrics return ``None`` when nothing qualifies to be counted.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from itertools import pairwise

from music_decoder.compose.melody import _chord_tones, _scale_pcs
from music_decoder.types import VALID_TONICS, ChordSymbol, Note, Scale, Tuning, VoicedChord

_GRID_TOL_S = 1e-3  # onset counts as "on the grid" within 1 ms
_STROKE_WINDOW_S = 0.06  # onsets this close to a stroke's first onset belong to it
_SOUNDING_EPS_S = 1e-4


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


def intervals(melody: Sequence[Note]) -> list[int]:
    return [b.pitch - a.pitch for a, b in pairwise(melody)]


def _ratio(hits: int, total: int) -> float | None:
    return hits / total if total else None


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
        hits += n.pitch % 12 in _chord_tones(chord)
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
    """M5: (leaps >= 5 semitones with a following interval, of those recovered).

    Recovered = next interval moves the opposite way by 1-2 semitones.
    Returned as counts so callers can pool across many compositions.
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
    tonic_pc = VALID_TONICS.index(scale.tonic)
    last_chord = progression[-1]
    if tonic_pc in _chord_tones(last_chord):
        target = tonic_pc
    else:
        target = VALID_TONICS.index(last_chord.root)
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


def motif_contour_match(melody: Sequence[Note], tempo_bpm: float, n_bars: int) -> float | None:
    """M11: interval-direction agreement between bars 4g and 4g+2.

    Only pairs with the same note count (>= 2) are compared.
    """
    scores: list[float] = []
    for g in range(0, n_bars - 2, 4):
        a = _bar_notes(melody, g, tempo_bpm)
        c = _bar_notes(melody, g + 2, tempo_bpm)
        if len(a) != len(c) or len(a) < 2:
            continue
        sa = [_sign(i) for i in intervals(a)]
        sc = [_sign(i) for i in intervals(c)]
        scores.append(sum(x == y for x, y in zip(sa, sc, strict=True)) / len(sa))
    return sum(scores) / len(scores) if scores else None


def phrase_end_long_ratio(melody: Sequence[Note], tempo_bpm: float, n_bars: int) -> float | None:
    """M12: share of complete 4-bar groups whose bar-4 last note lasts >= 1.5 beats."""
    beat = _beat_s(tempo_bpm)
    total = hits = 0
    for bar in range(3, n_bars, 4):
        notes = _bar_notes(melody, bar, tempo_bpm)
        total += 1
        hits += bool(notes) and notes[-1].end_s - notes[-1].start_s >= 1.5 * beat
    return _ratio(hits, total)


def nondiatonic_clash_ratio(
    melody: Sequence[Note],
    scale: Scale,
    progression: Sequence[ChordSymbol],
    bars_per_chord: int,
    tempo_bpm: float,
) -> float | None:
    """M13: share of notes a semitone from an out-of-scale tone of the current chord.

    ``None`` when every chord in the progression is diatonic.
    """
    scale_set = set(_scale_pcs(scale))
    if all(set(_chord_tones(c)) <= scale_set for c in progression):
        return None
    clashes = 0
    for n in melody:
        chord = chord_at(n.start_s, progression, bars_per_chord, tempo_bpm)
        foreign = set(_chord_tones(chord)) - scale_set
        clashes += any((n.pitch - f) % 12 in (1, 11) for f in foreign)
    return _ratio(clashes, len(melody))


def onset_grid_ratio(onsets: Sequence[float], tempo_bpm: float, per_beat: int) -> float | None:
    """M14 / A1: share of onsets within 1 ms of a 1/``per_beat``-beat grid."""
    step = _beat_s(tempo_bpm) / per_beat
    hits = sum(abs(t / step - round(t / step)) * step <= _GRID_TOL_S for t in onsets)
    return _ratio(hits, len(onsets))


def stroke_heads(onsets: Sequence[float]) -> list[float]:
    """First onset of each stroke (onsets within 60 ms of a stroke's first onset)."""
    heads: list[float] = []
    for t in sorted(onsets):
        if not heads or t - heads[-1] > _STROKE_WINDOW_S:
            heads.append(t)
    return heads


def _sounding(accompaniment: Sequence[Note], t: float) -> list[int]:
    probe = t + _SOUNDING_EPS_S
    return [a.pitch for a in accompaniment if a.start_s <= probe < a.end_s]


def melody_above_accompaniment_ratio(
    melody: Sequence[Note], accompaniment: Sequence[Note]
) -> float | None:
    """A2: share of melody notes above every accompaniment pitch sounding at onset."""
    hits = 0
    for n in melody:
        sounding = _sounding(accompaniment, n.start_s)
        hits += not sounding or n.pitch > max(sounding)
    return _ratio(hits, len(melody))


def exposed_semitone_clash_ratio(
    melody: Sequence[Note], accompaniment: Sequence[Note], tempo_bpm: float
) -> float | None:
    """A3: among strong-position or >= 0.9-beat melody notes, share forming a
    minor second / major seventh (pitch-class distance 1 or 11) with a sounding
    accompaniment pitch."""
    beat = _beat_s(tempo_bpm)
    total = hits = 0
    for n in melody:
        exposed = metric_position(n.start_s, tempo_bpm) in (0, 4) or (
            n.end_s - n.start_s >= 0.9 * beat
        )
        if not exposed:
            continue
        total += 1
        hits += any((n.pitch - p) % 12 in (1, 11) for p in _sounding(accompaniment, n.start_s))
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


def voice_leading_distance(voicings: Sequence[VoicedChord], tuning: Tuning) -> float | None:
    """V1: mean nearest-voice motion (semitones) between successive distinct chords."""

    def pitches(v: VoicedChord) -> list[int]:
        return [tuning.open_pitches[p.string] + p.fret for p in v.positions if p.fret >= 0]

    moves: list[float] = []
    for a, b in pairwise(voicings):
        pa, pb = pitches(a), pitches(b)
        if a.chord == b.chord or not pa or not pb:
            continue
        moves.append(sum(min(abs(x - y) for y in pa) for x in pb) / len(pb))
    return sum(moves) / len(moves) if moves else None
```

- [ ] **Step 4: Run the tests and gates**

Run: `pytest tests/unit/test_compose_metrics.py -q`
Expected: `15 passed`.
Run: `ruff check src tests && ruff format --check src tests && mypy --strict src`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add src/music_decoder/compose/metrics.py tests/unit/test_compose_metrics.py
git commit -m "feat(compose): musicality metrics module

Pure metric functions (chord-tone fit, interval profile, leap recovery,
cadence, rhythmic variety, motif repetition, accompaniment grid/register/
clash, tab hand shifts, voice leading) per the 2026-11-06 compose spec.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Rhythm and phrase planner

**Files:**
- Create: `src/music_decoder/compose/rhythm.py`
- Test: `tests/unit/test_compose_rhythm.py`

**Interfaces:**
- Consumes: `numpy.random.Generator`.
- Produces, in `music_decoder.compose.rhythm`:
  - `BODY_RHYTHMS: tuple[tuple[int, ...], ...]` and `CADENCE_RHYTHMS: tuple[tuple[int, ...], ...]`, in eighth-note units, each summing to 8
  - `@dataclass(frozen=True) class BarPlan: rhythm: tuple[int, ...]; cadence: bool; final: bool; motif_from: int | None`
  - `split_phrases(n_bars: int) -> list[int]`, which raises `ValueError` for `n_bars < 1`
  - `plan_bars(n_bars: int, rng: np.random.Generator) -> list[BarPlan]`, one entry per bar

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_compose_rhythm.py`:

```python
import numpy as np
import pytest

from music_decoder.compose.rhythm import (
    BODY_RHYTHMS,
    CADENCE_RHYTHMS,
    BarPlan,
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


def test_plan_bars_structure_for_eight_bars():
    plans = plan_bars(8, np.random.default_rng(3))
    assert len(plans) == 8
    assert all(isinstance(p, BarPlan) for p in plans)
    assert [p.cadence for p in plans] == [False, False, False, True] * 2
    assert [p.final for p in plans] == [False] * 7 + [True]
    # antecedent: bar 3 repeats bar 1's rhythm and echoes it
    assert plans[2].rhythm == plans[0].rhythm
    assert plans[2].motif_from == 0
    # consequent reuses the antecedent body bar for bar
    assert [p.rhythm for p in plans[4:7]] == [p.rhythm for p in plans[0:3]]
    assert [p.motif_from for p in plans[4:7]] == [0, 1, 2]
    assert plans[7].rhythm[-1] >= 4


def test_plan_bars_single_bar_is_a_final_cadence():
    (only,) = plan_bars(1, np.random.default_rng(0))
    assert only.cadence and only.final
    assert only.rhythm in CADENCE_RHYTHMS


def test_plan_bars_echo_indices_point_backwards():
    for n_bars in range(1, 14):
        plans = plan_bars(n_bars, np.random.default_rng(n_bars))
        assert len(plans) == n_bars
        for i, p in enumerate(plans):
            assert p.motif_from is None or 0 <= p.motif_from < i
            if p.motif_from is not None:
                assert plans[p.motif_from].rhythm == p.rhythm


def test_plan_bars_is_deterministic_per_seed():
    a = plan_bars(8, np.random.default_rng(11))
    b = plan_bars(8, np.random.default_rng(11))
    assert a == b
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/unit/test_compose_rhythm.py -q`
Expected: collection ERROR, `ModuleNotFoundError: No module named 'music_decoder.compose.rhythm'`.

- [ ] **Step 3: Write the implementation**

Create `src/music_decoder/compose/rhythm.py`:

```python
"""One-bar rhythm templates and phrase planning for the melody generator.

Units are eighth notes; every template sums to 8 (one 4/4 bar). Phrases are
four bars long where possible: antecedent phrases play ``[A, B, A, cadence]``
(bar 3 echoes bar 1's intervals) and the following consequent phrase reuses
the antecedent's body rhythms, echoing its intervals bar for bar. Every phrase
ends on a cadence bar whose last note is long.
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


def _draw(rng: np.random.Generator, templates: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    return templates[int(rng.integers(len(templates)))]


def plan_bars(n_bars: int, rng: np.random.Generator) -> list[BarPlan]:
    """One ``BarPlan`` per bar, in order."""
    phrases = split_phrases(n_bars)
    plans: list[BarPlan] = []
    antecedent_start = 0
    antecedent_body: list[tuple[int, ...]] = []
    for index, length in enumerate(phrases):
        start = len(plans)
        n_body = length - 1
        if index % 2 == 0:
            a = _draw(rng, BODY_RHYTHMS)
            b = _draw(rng, BODY_RHYTHMS)
            body = [a, b, a][:n_body]
            echoes: list[int | None] = [None, None, start][:n_body]
            antecedent_start, antecedent_body = start, body
        else:
            body = list(antecedent_body[:n_body])
            echoes = [antecedent_start + i for i in range(len(body))]
            while len(body) < n_body:
                body.append(_draw(rng, BODY_RHYTHMS))
                echoes.append(None)
        for rhythm, echo in zip(body, echoes, strict=True):
            plans.append(BarPlan(rhythm=rhythm, cadence=False, final=False, motif_from=echo))
        final = index == len(phrases) - 1
        cadences = tuple(
            c for c in CADENCE_RHYTHMS if not final or c[-1] >= _FINAL_MIN_LAST_EIGHTHS
        )
        plans.append(
            BarPlan(rhythm=_draw(rng, cadences), cadence=True, final=final, motif_from=None)
        )
    return plans
```

- [ ] **Step 4: Run the tests and gates**

Run: `pytest tests/unit/test_compose_rhythm.py -q`
Expected: `17 passed`.
Run: `ruff check src tests && ruff format --check src tests && mypy --strict src`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add src/music_decoder/compose/rhythm.py tests/unit/test_compose_rhythm.py
git commit -m "feat(compose): bar rhythm templates and antecedent/consequent phrase planner

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Phrase-structured melody with scored pitch selection

Rewrites `generate_melody` per spec §5.2. Note the deliberate interface
changes (spec D4):
- `notes_per_bar` becomes `int | None = None`. `None` uses the phrase planner; an integer uses the legacy uniform grid.
- `pitch_low` / `pitch_high` default to 62 / 79.
- `strong_beat_chord_tone_prob`, `weak_beat_chord_tone_prob` and `markov_max_interval_semitones` are removed. Nothing in `src/` or `tests/` passes them; verify with `grep -rn "chord_tone_prob\|markov_max" src tests scripts`, which must print nothing except `melody.py` itself.
- New keyword `tessitura_center: int = 70`.

**Files:**
- Modify (full rewrite): `src/music_decoder/compose/melody.py`
- Modify (full rewrite, keeping the 4 existing tests unchanged at the top): `tests/unit/test_compose_melody.py`

**Interfaces:**
- Consumes: `music_decoder.compose.rhythm.plan_bars` (Task 2) and the `music_decoder.compose.metrics` functions (Task 1, tests only).
- Produces, in `music_decoder.compose.melody`:
  - `generate_melody(*, scale, progression, bars_per_chord=1, tempo_bpm=100.0, notes_per_bar: int | None = None, pitch_low=62, pitch_high=79, seed=None, velocity_strong=80, velocity_weak=70, tessitura_center=70) -> list[Note]`
  - `chord_scale_pcs(scale_pcs: Iterable[int], chord_pcs: Iterable[int]) -> frozenset[int]`
  - `final_target(scale, chord, candidates: Sequence[int], reference: int) -> int`, which raises `ValueError` when no candidate qualifies
  - `@dataclass(frozen=True) class NoteContext(strong: bool, eighths: float, chord_pcs: frozenset[int], prev_pitch: int | None, prev_interval: int, motif_interval: int | None, target: int | None, cadence_end: bool)`
  - `score_pitch(candidate: int, ctx: NoteContext, tessitura_center: int) -> float`, where `-inf` excludes the candidate
  - It keeps `_chord_tones`, `_scale_pcs`, `_scale_pitches_in_range` and `_CHORD_PCS` (which metrics.py imports).

- [ ] **Step 1: Write the failing test**

Replace the whole of `tests/unit/test_compose_melody.py` with this content (the first four tests are the existing ones, unchanged):

```python
import math

import pytest

from music_decoder.compose import metrics
from music_decoder.compose.melody import (
    NoteContext,
    chord_scale_pcs,
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

A_MINOR_PCS = {9, 11, 0, 2, 4, 5, 7}


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


def test_chord_scale_pcs_is_the_scale_over_diatonic_chords():
    assert chord_scale_pcs(C_MAJOR_PCS, {7, 11, 2, 5}) == frozenset(C_MAJOR_PCS)  # G7


def test_chord_scale_pcs_swaps_the_clashing_degree_over_e7_in_a_minor():
    pool = chord_scale_pcs(A_MINOR_PCS, {4, 8, 11, 2})  # E7 = E G# B D
    assert 8 in pool  # G# added
    assert 7 not in pool  # G removed (semitone below G#)
    assert 9 not in pool  # A removed (semitone above G#, not a chord tone)
    assert {11, 0, 2, 4, 5} <= pool


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
    ctx = _ctx(strong=True)
    assert score_pitch(76, ctx, 72) > score_pitch(74, ctx, 72)
    # ...but on a short weak note the step wins
    assert score_pitch(74, _ctx(), 72) > score_pitch(76, _ctx(), 72)


def test_score_pitch_excludes_leaps_over_an_octave():
    assert score_pitch(85, _ctx(), 72) == -math.inf


def test_score_pitch_rewards_leap_recovery():
    after_leap_up = _ctx(prev_interval=7)
    down_step = score_pitch(71, after_leap_up, 72)
    up_step = score_pitch(74, after_leap_up, 72)
    assert down_step > up_step


def test_score_pitch_follows_the_motif():
    ctx = _ctx(motif_interval=-2)
    assert score_pitch(70, ctx, 72) > score_pitch(74, ctx, 72)


def _gen(scale, labels, bars=1, seed=0, **kw):
    return generate_melody(
        scale=scale,
        progression=[ChordSymbol.parse(c) for c in labels],
        bars_per_chord=bars,
        tempo_bpm=120.0,
        seed=seed,
        **kw,
    )


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
    assert metrics.max_interval(notes) <= 12


def test_no_natural_seventh_against_e7_in_a_minor():
    for seed in range(20):
        notes = _gen(Scale("A", "minor"), ("Am", "E7"), seed=seed)
        bar2 = [n for n in notes if 2.0 <= n.start_s < 4.0]
        assert all(n.pitch % 12 != 7 for n in bar2), seed  # no G over E7


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
    notes = generate_melody(
        scale=Scale("E", "major"),
        progression=[ChordSymbol.parse(c) for c in ("E", "A", "B7")],
        bars_per_chord=1,
        tempo_bpm=tempo,
        seed=4,
    )
    bar_s = 4 * 60.0 / tempo
    assert metrics.onset_grid_ratio([n.start_s for n in notes], tempo, 2) == 1.0
    assert notes[-1].end_s <= 3 * bar_s + 1e-9
    assert notes[-1].pitch % 12 == 11  # B7 holds no E: ends on the B root


def test_secondary_dominant_in_major_adapts_the_scale():
    for seed in range(20):
        notes = _gen(Scale("C", "major"), ("C", "D7", "G", "C"), seed=seed)
        bar2 = [n for n in notes if 2.0 <= n.start_s < 4.0]  # D7 = D F# A C
        assert all(n.pitch % 12 not in (5, 7) for n in bar2), seed  # no F or G against F#
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/unit/test_compose_melody.py -q`
Expected: collection ERROR, `ImportError: cannot import name 'NoteContext' from 'music_decoder.compose.melody'`.

- [ ] **Step 3: Write the implementation**

Replace the whole of `src/music_decoder/compose/melody.py` with:

```python
"""Phrase-structured melody generator over a chord progression.

Rhythm comes from :func:`music_decoder.compose.rhythm.plan_bars` (one-bar
templates arranged in antecedent/consequent phrases that end on cadences).
Each pitch is sampled from a softmax over musical scores: interval size,
chord fit weighted by metric position, leap recovery, motif echo, register
gravity and, in the final bar, a pull toward the cadence target. The only
randomness is ``numpy.random.default_rng(seed)``, so output is deterministic
for a given seed.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from music_decoder.compose.rhythm import plan_bars
from music_decoder.types import VALID_TONICS, ChordSymbol, Note, Scale

# Pitch-class sets (relative to tonic = 0)
_MAJOR_PCS = (0, 2, 4, 5, 7, 9, 11)
_MINOR_PCS = (0, 2, 3, 5, 7, 8, 10)

# Chord-quality → relative pitch classes (root = 0)
_CHORD_PCS: dict[str, tuple[int, ...]] = {
    "maj": (0, 4, 7),
    "min": (0, 3, 7),
    "maj7": (0, 4, 7, 11),
    "min7": (0, 3, 7, 10),
    "7": (0, 4, 7, 10),
    "dim": (0, 3, 6),
    "sus4": (0, 5, 7),
    "aug": (0, 4, 8),
}

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
MOTIF_SIGN_BONUS = 1.5  # same direction as the echoed interval
MOTIF_SIZE_BONUS = 1.0  # within a semitone of the echoed interval
GRAVITY_WEIGHT = 0.5  # register pull toward the tessitura centre
GRAVITY_WIDTH = 6.0
CADENCE_PULL = 0.25  # per semitone from the final target, final bar only
CADENCE_END_NON_CHORD_TONE = -3.0  # last note of a cadence bar


def _scale_pcs(scale: Scale) -> tuple[int, ...]:
    base = _MAJOR_PCS if scale.mode == "major" else _MINOR_PCS
    tonic_pc = VALID_TONICS.index(scale.tonic)
    return tuple((p + tonic_pc) % 12 for p in base)


def _chord_tones(chord: ChordSymbol) -> tuple[int, ...]:
    rel = _CHORD_PCS[chord.quality]
    root_pc = VALID_TONICS.index(chord.root)
    return tuple((r + root_pc) % 12 for r in rel)


def _scale_pitches_in_range(scale_pcs: Collection[int], low: int, high: int) -> list[int]:
    return [p for p in range(low, high + 1) if (p % 12) in scale_pcs]


def chord_scale_pcs(scale_pcs: Iterable[int], chord_pcs: Iterable[int]) -> frozenset[int]:
    """Scale adapted to the chord: each chord tone outside the scale replaces
    the non-chord scale tones a semitone away from it (e.g. G -> G# over E7 in
    A minor). Over a diatonic chord this is just the scale."""
    chord = set(chord_pcs)
    scale = set(scale_pcs)
    pool = set(scale)
    for foreign in chord - scale:
        for offset in (1, 11):
            neighbour = (foreign + offset) % 12
            if neighbour not in chord:
                pool.discard(neighbour)
        pool.add(foreign)
    return frozenset(pool)


def final_target(
    scale: Scale, chord: ChordSymbol, candidates: Sequence[int], reference: int
) -> int:
    """Final pitch: the tonic if ``chord`` contains it, else the chord root;
    the candidate of that pitch class nearest ``reference`` (ties -> lower)."""
    tonic_pc = VALID_TONICS.index(scale.tonic)
    target_pc = tonic_pc if tonic_pc in _chord_tones(chord) else VALID_TONICS.index(chord.root)
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


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def score_pitch(candidate: int, ctx: NoteContext, tessitura_center: int) -> float:
    """Log-weight of ``candidate``; ``-inf`` excludes it."""
    score = 0.0
    is_chord_tone = candidate % 12 in ctx.chord_pcs
    if ctx.prev_pitch is not None:
        interval = candidate - ctx.prev_pitch
        size = abs(interval)
        if size > 12:
            return -math.inf
        score += INTERVAL_WEIGHT[size]
        if abs(ctx.prev_interval) >= LEAP_SEMITONES:
            if interval * ctx.prev_interval < 0 and 1 <= size <= 2:
                score += LEAP_RECOVERY_BONUS
            elif interval * ctx.prev_interval >= 0:
                score += LEAP_CONTINUE_PENALTY
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
    if not finite.any():  # nothing within an octave: take the nearest pitch
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
) -> list[Note]:
    """Generate a list of Notes spanning the given progression.

    ``notes_per_bar=None`` (default) uses phrase-planned rhythms; an integer
    uses the legacy uniform grid of that many notes per bar.
    """
    rng = np.random.default_rng(seed)
    scale_pcs = _scale_pcs(scale)
    if not _scale_pitches_in_range(scale_pcs, pitch_low, pitch_high):
        raise ValueError("Scale has no pitches in the requested range.")

    beat_s = 60.0 / tempo_bpm
    bar_s = 4.0 * beat_s
    n_bars = len(progression) * bars_per_chord
    v_mid = (velocity_strong + velocity_weak) // 2
    if notes_per_bar is None:
        bars = _phrase_bars(n_bars, beat_s, rng, velocity_strong, v_mid, velocity_weak)
    else:
        bars = _grid_bars(n_bars, beat_s, notes_per_bar, velocity_strong, v_mid, velocity_weak)

    out: list[Note] = []
    prev: int | None = None
    prev_interval = 0
    bar_intervals: list[list[int]] = []
    for bar_idx, bar in enumerate(bars):
        chord = progression[min(bar_idx // bars_per_chord, len(progression) - 1)]
        chord_pcs = frozenset(_chord_tones(chord))
        candidates = _scale_pitches_in_range(
            chord_scale_pcs(scale_pcs, chord_pcs), pitch_low, pitch_high
        )
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
            if target is not None and last:
                pitch = target
            else:
                ctx = NoteContext(
                    strong=slot.strong,
                    eighths=slot.eighths,
                    chord_pcs=chord_pcs,
                    prev_pitch=prev,
                    prev_interval=prev_interval,
                    motif_interval=motif[k] if motif is not None and 0 < k < len(motif) else None,
                    target=target,
                    cadence_end=bar.cadence and last,
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
        bar_intervals.append(here)
    return out
```

- [ ] **Step 4: Run the tests and gates**

Run: `pytest tests/unit/test_compose_melody.py -q`
Expected: `28 passed`.
Run: `pytest -q tests/unit/test_compose_arrangement.py tests/unit/test_compose_api.py tests/unit/test_compose_metrics.py tests/integration/test_compose_e2e.py tests/unit/test_public_contract.py`
Expected: all pass. The existing arrangement, api and e2e tests still work with the new melody.
Run: `ruff check src tests && ruff format --check src tests && mypy --strict src`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add src/music_decoder/compose/melody.py tests/unit/test_compose_melody.py
git commit -m "feat(compose): phrase-structured melody with scored pitch selection

Rhythm from the phrase planner; pitches sampled from a softmax over interval
size, metric-weighted chord fit, leap recovery, motif echo, register gravity
and a cadence pull. Chord-scale adaptation for non-diatonic chord tones; the
final note resolves to the tonic (or the final chord's root). Melody window
62-79. notes_per_bar=N keeps the legacy uniform grid.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: On-grid, melody-aware accompaniment

**Files:**
- Modify (full rewrite): `src/music_decoder/compose/arrangement.py`
- Modify (full rewrite, keeping the 2 existing tests unchanged at the top): `tests/unit/test_compose_arrangement.py`

**Interfaces:**
- Consumes: `music_decoder.compose.metrics.stroke_heads`, `onset_grid_ratio` and `final_chord_rings` (Task 1, tests only), plus `generate_melody` (Task 3).
- Produces, in `music_decoder.compose.arrangement`:
  - `_accompaniment_notes(voicings, bars_per_chord, tempo_bpm, style, tuning, melody: Sequence[Note] = ()) -> list[pretty_midi.Note]`. The new keyword `melody` defaults to `()`, which means no thinning.
  - `_thin_below(pitches: Sequence[int], melody: Sequence[Note], t0: float, t1: float) -> list[int]`
  - `build_midi(...)`, signature unchanged; it now passes `melody=melody` through.
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


def test_thin_below_keeps_accompaniment_a_minor_third_under_the_melody():
    pitches = [48, 52, 55, 60, 64]  # C3 E3 G3 C4 E4 (open C)
    melody = [_note(0.0, 0.5, 65)]  # F4
    assert _thin_below(pitches, melody, 0.0, 0.5) == [48, 52, 55, 60]
    assert _thin_below(pitches, [_note(0.0, 0.5, 50)], 0.0, 0.5) == [48, 52, 55]  # keep 3
    assert _thin_below(pitches, [], 0.0, 0.5) == pitches
    assert _thin_below(pitches, [_note(1.0, 1.5, 50)], 0.0, 0.5) == pitches  # not sounding


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
melody are dropped (keeping at least three), so the tune stays on top. The
last bar is always a single strummed chord ringing to the end.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import pretty_midi

from music_decoder.tabs.tuning import STANDARD_EADGBE
from music_decoder.types import Note, Tuning, VoicedChord

GUITAR_PROGRAM = 25  # Steel Acoustic Guitar
Style = Literal["strum", "arpeggio", "fingerstyle"]

THIN_MARGIN_SEMITONES = 3  # accompaniment stays at least a minor third under the melody
MIN_KEPT_PITCHES = 3
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


def _thin_below(pitches: Sequence[int], melody: Sequence[Note], t0: float, t1: float) -> list[int]:
    """Drop (sorted) ``pitches`` within a minor third of the lowest melody note
    sounding in [t0, t1); keep the lowest ``MIN_KEPT_PITCHES`` if too few survive."""
    sounding = [n.pitch for n in melody if n.start_s < t1 and n.end_s > t0]
    if not sounding:
        return list(pitches)
    ceiling = min(sounding) - THIN_MARGIN_SEMITONES
    kept = [p for p in pitches if p <= ceiling]
    if len(kept) < MIN_KEPT_PITCHES:
        return list(pitches[:MIN_KEPT_PITCHES])
    return kept


def _stroke(
    pitches: Sequence[int], start: float, end: float, velocity: int, stagger_s: float
) -> list[pretty_midi.Note]:
    return [
        pretty_midi.Note(velocity=velocity, pitch=int(p), start=start + k * stagger_s, end=end)
        for k, p in enumerate(pitches)
    ]


def _arpeggio_bar(
    pitches: list[int], melody: Sequence[Note], t: float, eighth: float
) -> list[pretty_midi.Note]:
    """Bass on the downbeat, then seven eighths cycling up and down the treble."""
    treble = pitches[1:]
    cycle = treble + treble[-2:0:-1] if len(treble) > 1 else treble
    sequence = [pitches[0]] + [cycle[i % len(cycle)] for i in range(7)] if cycle else pitches[:1]
    out: list[pretty_midi.Note] = []
    for k, p in enumerate(sequence):
        start = t + k * eighth
        if k > 0 and p not in _thin_below(pitches, melody, start, start + eighth):
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
    pitches: list[int], melody: Sequence[Note], t: float, eighth: float
) -> list[pretty_midi.Note]:
    """Down strokes play the whole (thinned) voicing; up strokes the top strings."""
    out: list[pretty_midi.Note] = []
    for i, (slot, direction) in enumerate(STRUM_STROKES):
        start = t + slot * eighth
        next_slot = STRUM_STROKES[i + 1][0] if i + 1 < len(STRUM_STROKES) else 8
        end = t + next_slot * eighth
        chord = _thin_below(pitches, melody, start, end)
        if direction == "up":
            out.extend(_stroke(chord[-UP_STROKE_STRINGS:][::-1], start, end, 60, STRUM_STAGGER_S))
        else:
            out.extend(_stroke(chord, start, end, 80 if slot == 0 else 72, STRUM_STAGGER_S))
    return out


def _fingerstyle_bar(
    pitches: list[int], melody: Sequence[Note], t: float, beat: float
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
        for p in _thin_below(pitches[1:] or pitches, melody, start, start + beat):
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
        pitches = _voiced_pitches(voicings[bar // bars_per_chord], tuning)
        t = bar * seconds_per_bar
        if not pitches:
            continue
        if bar == n_bars - 1:
            chord = _thin_below(pitches, melody, t, t + seconds_per_bar)
            out.extend(_stroke(chord, t, t + seconds_per_bar, 80, FINAL_STAGGER_S))
        elif style == "arpeggio":
            out.extend(_arpeggio_bar(pitches, melody, t, eighth))
        elif style == "strum":
            out.extend(_strum_bar(pitches, melody, t, eighth))
        elif style == "fingerstyle":
            out.extend(_fingerstyle_bar(pitches, melody, t, seconds_per_beat))
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

- [ ] **Step 4: Run the tests and gates**

Run: `pytest tests/unit/test_compose_arrangement.py -q`
Expected: `16 passed`.
Run: `pytest -q tests/unit/test_compose_api.py tests/integration/test_compose_e2e.py`
Expected: all pass.
Run: `ruff check src tests && ruff format --check src tests && mypy --strict src`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add src/music_decoder/compose/arrangement.py tests/unit/test_compose_arrangement.py
git commit -m "feat(compose): on-grid, melody-aware accompaniment patterns

Arpeggio on straight eighths (was 11-tuplets), D-DU-UDU strum with string
stagger, alternating-bass fingerstyle, a final ringing chord, and thinning
of voicing notes within a minor third of the melody.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Pure `arrange()`, A\* melody tab, chord anchor times, `composer_version`

**Files:**
- Modify (full rewrite): `src/music_decoder/compose/api.py`
- Modify (full rewrite, keeping the 3 existing tests unchanged at the top): `tests/unit/test_compose_api.py`

**Interfaces:**
- Consumes: `generate_melody` (Task 3), `_accompaniment_notes` and `build_midi` (Task 4), `music_decoder.tabs.assign_tabs(notes, *, tuning) -> Sequence[TabbedNote]` (existing), `voicings_for` (existing).
- Produces, in `music_decoder.compose.api`:
  - `COMPOSER_VERSION = 2`
  - `@dataclass(frozen=True) class Arrangement: melody: tuple[Note, ...]; voicings: tuple[VoicedChord, ...]; accompaniment: tuple[Note, ...]; melody_tab: tuple[TabbedNote, ...]`
  - `arrange(scale, progression, *, bars_per_chord=1, tempo_bpm=100.0, style="fingerstyle", tuning=STANDARD_EADGBE, seed=None) -> Arrangement`, which raises `InvalidScaleError` / `InvalidProgressionError` (including for `bars_per_chord < 1`)
  - `_melody_tab(melody, tuning) -> tuple[TabbedNote, ...]`
  - `_ascii_tab_for(arrangement, bars_per_chord, tempo_bpm, tuning) -> str`
  - `compose(...)`, signature unchanged; `metadata` gains `"composer_version": 2`.

- [ ] **Step 1: Write the failing test**

Replace the whole of `tests/unit/test_compose_api.py` with:

```python
from pathlib import Path
from unittest import mock

import pytest

from music_decoder.compose.api import (
    COMPOSER_VERSION,
    _ascii_tab_for,
    _melody_tab,
    arrange,
    compose,
)
from music_decoder.errors import InvalidProgressionError, InvalidScaleError
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


# ---- arrange() and tab (compose improvements) ------------------------------

_PROG = [ChordSymbol.parse(c) for c in ("Cmaj7", "Am7", "Dm7", "G7")]


def test_arrange_is_pure_and_deterministic(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    a = arrange(Scale("C", "major"), _PROG, tempo_bpm=120.0, style="strum", seed=3)
    b = arrange(Scale("C", "major"), _PROG, tempo_bpm=120.0, style="strum", seed=3)
    assert a == b
    assert list(tmp_path.iterdir()) == []  # no files written
    assert len(a.voicings) == 4
    assert a.accompaniment and all(isinstance(n, Note) for n in a.accompaniment)


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
    with mock.patch("music_decoder.compose.api.assign_tabs", return_value=()):
        tab = _melody_tab(melody, STANDARD_EADGBE)
    assert [(t.position.string, t.position.fret) for t in tab] == [(5, 0), (5, 3)]


@pytest.mark.parametrize("bars", [0, -1])
def test_bars_per_chord_below_one_raises(bars):
    with pytest.raises(InvalidProgressionError):
        arrange(Scale("C", "major"), _PROG, bars_per_chord=bars)


def test_ascii_tab_places_chords_at_their_start_times():
    arr = arrange(Scale("C", "major"), _PROG, bars_per_chord=2, tempo_bpm=120.0, seed=1)
    with mock.patch("music_decoder.compose.api.render_ascii_tab", return_value="") as render:
        _ascii_tab_for(arr, 2, 120.0, STANDARD_EADGBE)
    (notes,), _ = render.call_args
    anchors = sorted({tn.note.start_s for tn in notes if tn.note.pitch == 0})
    assert anchors == [0.0, 4.0, 8.0, 12.0]  # 2 bars of 2 s per chord


def test_compose_metadata_records_composer_version(tmp_path):
    with mock.patch("music_decoder.compose.api.render_wav") as render:
        render.side_effect = lambda midi_path, out_path: out_path.write_bytes(b"RIFFWAV")
        result = compose(scale=Scale("C", "major"), progression=_PROG, seed=1, out_dir=tmp_path)
    assert result.metadata["composer_version"] == COMPOSER_VERSION == 2
    assert result.melody_notes == arrange(Scale("C", "major"), _PROG, seed=1).melody


@pytest.mark.parametrize("tuning_name", ["Drop_D", "DADGAD", "D_standard"])
def test_arrange_in_other_tunings_tabs_every_note(tuning_name):
    tuning = PRESETS[tuning_name]
    arr = arrange(
        Scale("D", "major"),
        [ChordSymbol.parse(c) for c in ("D", "G", "A", "D")],
        tuning=tuning,
        style="arpeggio",
        seed=2,
    )
    assert len(arr.melody_tab) == len(arr.melody)
    for tn in arr.melody_tab:
        assert tuning.open_pitches[tn.position.string] + tn.position.fret == tn.note.pitch
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/unit/test_compose_api.py -q`
Expected: collection ERROR, `ImportError: cannot import name 'COMPOSER_VERSION' from 'music_decoder.compose.api'`.

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

from music_decoder.compose.arrangement import _accompaniment_notes, build_midi
from music_decoder.compose.melody import generate_melody
from music_decoder.compose.voicings import voicings_for
from music_decoder.errors import InvalidProgressionError, InvalidScaleError
from music_decoder.synth import render_wav
from music_decoder.tabs import assign_tabs
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

# Bumped whenever the generator changes what a given seed produces.
COMPOSER_VERSION = 2


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
    best: tuple[int, int] | None = None  # (fret, string)
    for s, open_pitch in enumerate(tuning.open_pitches):
        fret = pitch - open_pitch
        if 0 <= fret <= 22 and (best is None or fret < best[0]):
            best = (fret, s)
    return None if best is None else TabPosition(string=best[1], fret=best[0])


def _melody_tab(melody: Sequence[Note], tuning: Tuning) -> tuple[TabbedNote, ...]:
    """A* fingering (shared with analyze()); any note A* drops falls back to
    its lowest-fret position so the tab never loses notes."""
    assigned = {(tn.note.start_s, tn.note.pitch): tn for tn in assign_tabs(melody, tuning=tuning)}
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
    _validate(scale, progression, bars_per_chord)
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
    accompaniment = tuple(
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
    return Arrangement(
        melody=melody,
        voicings=voicings,
        accompaniment=accompaniment,
        melody_tab=_melody_tab(melody, tuning),
    )


def _ascii_tab_for(
    arrangement: Arrangement, bars_per_chord: int, tempo_bpm: float, tuning: Tuning
) -> str:
    """Melody tab plus each chord voicing at the time the chord starts.
    (A visual reference; the MIDI is the source of truth for exact timing.)"""
    notes: list[TabbedNote] = list(arrangement.melody_tab)
    chord_s = bars_per_chord * 4.0 * 60.0 / tempo_bpm
    for ci, vc in enumerate(arrangement.voicings):
        t0 = ci * chord_s
        anchor = Note(start_s=t0, end_s=t0 + chord_s, pitch=0, velocity=0, confidence=1.0)
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

    ascii_tab = _ascii_tab_for(arrangement, bars_per_chord, tempo_bpm, tuning)

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

- [ ] **Step 4: Run the tests and gates**

Run: `pytest tests/unit/test_compose_api.py -q`
Expected: `13 passed`.
Run: `pytest -q tests/unit tests/integration`
Expected: no failures (`test_public_contract.py` included, unchanged).
Run: `ruff check src tests && ruff format --check src tests && mypy --strict src`
Expected: clean.
Run: `music-decoder compose --scale A:minor --progression "Am Dm E7 Am" --bars 2 --seed 42 --out "$TMPDIR/compose-smoke"`
Expected: it prints the `MIDI:` / `WAV:` paths and a six-line ASCII tab, and exits 0.

- [ ] **Step 5: Commit**

```bash
git add src/music_decoder/compose/api.py tests/unit/test_compose_api.py
git commit -m "feat(compose): pure arrange(), A* melody tab, real chord anchor times

compose() now wraps a filesystem-free arrange(). The melody tab uses the
shared A* assigner (lowest-fret fallback), chord shapes sit at their real
start times, bars_per_chord < 1 raises InvalidProgressionError, and
metadata records composer_version=2.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Musicality regression test and documentation

**Files:**
- Create: `tests/unit/test_compose_musicality.py`
- Modify: `docs/architecture-technical.md` (§5 data flow, §5.2–§5.4, new §5.4.1)
- Modify: `docs/architecture-overview.md` ("How the tool works" bullet; "What happens when you click Compose?" steps 2, 3 and 5)
- Modify: `docs/usage.md` (`--seed` bullet, JSON example, "Composition style choices", "Seed reproducibility")

**Interfaces:**
- Consumes: `arrange`, `Arrangement` (Task 5); `_accompaniment_notes` (Task 4); every `metrics` function (Task 1).
- Produces: the regression suite guarding every target in spec §3, and docs that match the new behaviour.

- [ ] **Step 1: Write the regression test**

Create `tests/unit/test_compose_musicality.py`:

```python
"""Musicality regression test for the composer.

Generates a matrix of compositions (8 keys x 5 progressions per mode x
bars_per_chord {1, 2} x 5 seeds = 400 melodies, each accompanied in 3 styles)
and asserts the targets from
docs/superpowers/specs/2026-11-06-compose-improvements-design.md section 3.
Baseline values of the pre-2026-11 composer are quoted next to each target.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass

import pytest

from music_decoder.compose import metrics
from music_decoder.compose.api import Arrangement, arrange
from music_decoder.compose.arrangement import _accompaniment_notes
from music_decoder.tabs.tuning import STANDARD_EADGBE
from music_decoder.types import VALID_TONICS, ChordSymbol, Note, Scale

TEMPO = 120.0
STYLES = ("arpeggio", "strum", "fingerstyle")
SEEDS = range(5)
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
_MAJOR_DEGREES = {"I": 0, "ii": 2, "iii": 4, "IV": 5, "V": 7, "vi": 9, "vii": 11}
_MINOR_DEGREES = {"i": 0, "ii": 2, "III": 3, "iv": 5, "v": 7, "V": 7, "VI": 8, "VII": 10}


def _realize(scale: Scale, numeral: str) -> ChordSymbol:
    match = re.fullmatch(r"([ivIV]+)(.*)", numeral)
    assert match is not None
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
    return ChordSymbol.parse(root + {"maj": "", "min": "m", "min7": "m7"}.get(quality, quality))


@dataclass(frozen=True)
class Case:
    scale: Scale
    progression: tuple[ChordSymbol, ...]
    bars_per_chord: int
    seed: int
    arrangement: Arrangement
    accompaniment: dict[str, tuple[Note, ...]]

    @property
    def n_bars(self) -> int:
        return len(self.progression) * self.bars_per_chord


def _cases() -> list[Case]:
    out: list[Case] = []
    for mode, keys, progressions in (
        ("major", MAJOR_KEYS, MAJOR_PROGRESSIONS),
        ("minor", MINOR_KEYS, MINOR_PROGRESSIONS),
    ):
        for key in keys:
            scale = Scale(key, mode)  # type: ignore[arg-type]
            for numerals in progressions:
                progression = tuple(_realize(scale, n) for n in numerals)
                for bars in (1, 2):
                    for seed in SEEDS:
                        arr = arrange(
                            scale,
                            progression,
                            bars_per_chord=bars,
                            tempo_bpm=TEMPO,
                            style="fingerstyle",
                            seed=seed,
                        )
                        acc = {"fingerstyle": arr.accompaniment}
                        for style in ("arpeggio", "strum"):
                            acc[style] = tuple(
                                Note(n.start, n.end, n.pitch, n.velocity, 1.0)
                                for n in _accompaniment_notes(
                                    arr.voicings,
                                    bars,
                                    TEMPO,
                                    style,  # type: ignore[arg-type]
                                    STANDARD_EADGBE,
                                    melody=arr.melody,
                                )
                            )
                        out.append(Case(scale, progression, bars, seed, arr, acc))
    return out


@pytest.fixture(scope="module")
def cases() -> list[Case]:
    return _cases()


def _mean(values: list[float | None]) -> float:
    present = [v for v in values if v is not None]
    assert present, "metric undefined for every case"
    return statistics.fmean(present)


def test_matrix_size(cases):
    assert len(cases) == 400


# ---- melody ---------------------------------------------------------------


def test_m1_strong_beat_chord_tones(cases):
    values = [
        metrics.strong_beat_chord_tone_ratio(
            c.arrangement.melody, c.progression, c.bars_per_chord, TEMPO
        )
        for c in cases
    ]
    assert _mean(values) >= 0.95  # baseline 0.841
    assert min(v for v in values if v is not None) >= 0.60


def test_m2_m3_m4_interval_profile(cases):
    melodies = [c.arrangement.melody for c in cases]
    assert _mean([metrics.step_ratio(m) for m in melodies]) >= 0.50  # baseline 0.372
    assert _mean([metrics.repeat_ratio(m) for m in melodies]) <= 0.15  # baseline 0.229
    assert _mean([metrics.large_leap_ratio(m) for m in melodies]) <= 0.04  # baseline 0.000
    assert max(metrics.max_interval(m) for m in melodies) <= 12


def test_m5_leap_recovery_pooled(cases):
    leaps = recovered = 0
    for c in cases:
        n_leaps, n_recovered = metrics.leap_recovery_counts(c.arrangement.melody)
        leaps += n_leaps
        recovered += n_recovered
    assert leaps > 0
    assert recovered / leaps >= 0.70  # baseline 0.348


def test_m6_ambitus(cases):
    assert max(metrics.ambitus(c.arrangement.melody) for c in cases) <= 17  # baseline max 24


def test_m7_m8_final_cadence(cases):
    for c in cases:
        assert metrics.final_resolution(c.arrangement.melody, c.scale, c.progression), c
        assert metrics.final_note_beats(c.arrangement.melody, TEMPO) >= 1.5, c  # baseline 0.95


def test_m9_m10_rhythmic_variety(cases):
    distinct = [metrics.distinct_durations(c.arrangement.melody, TEMPO) for c in cases]
    assert statistics.fmean(distinct) >= 3.0  # baseline 1.0
    assert min(distinct) >= 2
    entropy = [metrics.ioi_entropy(c.arrangement.melody, TEMPO) for c in cases]
    assert statistics.fmean(entropy) >= 1.0  # baseline 0.0


def test_m11_motif_repetition(cases):
    values = [metrics.motif_contour_match(c.arrangement.melody, TEMPO, c.n_bars) for c in cases]
    assert _mean(values) >= 0.60  # baseline 0.300


def test_m12_phrase_ends_are_long(cases):
    values = [metrics.phrase_end_long_ratio(c.arrangement.melody, TEMPO, c.n_bars) for c in cases]
    assert _mean(values) >= 0.95  # baseline 0.0


def test_m13_no_clash_with_non_diatonic_chord_tones(cases):
    for c in cases:
        ratio = metrics.nondiatonic_clash_ratio(
            c.arrangement.melody, c.scale, c.progression, c.bars_per_chord, TEMPO
        )
        assert ratio in (None, 0.0), c  # baseline mean 0.032, max 0.219


def test_m14_melody_on_the_eighth_grid(cases):
    for c in cases:
        onsets = [n.start_s for n in c.arrangement.melody]
        assert metrics.onset_grid_ratio(onsets, TEMPO, 2) == 1.0


# ---- accompaniment ----------------------------------------------------------


@pytest.mark.parametrize("style", STYLES)
def test_a1_accompaniment_on_the_sixteenth_grid(cases, style):
    for c in cases:
        heads = metrics.stroke_heads([n.start_s for n in c.accompaniment[style]])
        assert metrics.onset_grid_ratio(heads, TEMPO, 4) == 1.0  # baseline arpeggio 0.106


@pytest.mark.parametrize("style", STYLES)
def test_a2_a3_melody_on_top_without_clashes(cases, style):
    above = [
        metrics.melody_above_accompaniment_ratio(c.arrangement.melody, c.accompaniment[style])
        for c in cases
    ]
    clash = [
        metrics.exposed_semitone_clash_ratio(c.arrangement.melody, c.accompaniment[style], TEMPO)
        for c in cases
    ]
    assert _mean(above) >= 0.97  # baseline arpeggio 0.955 / strum 0.842 / fingerstyle 0.923
    assert _mean(clash) <= 0.08  # baseline arpeggio 0.034 / strum 0.169 / fingerstyle 0.121


@pytest.mark.parametrize("style", STYLES)
def test_a4_final_chord_rings(cases, style):
    for c in cases:
        assert metrics.final_chord_rings(c.accompaniment[style], c.n_bars, TEMPO), c


# ---- tab and voicings -----------------------------------------------------


def test_t1_t2_melody_tab_playability(cases):
    shifts: list[float | None] = []
    for c in cases:
        frets = [tn.position.fret for tn in c.arrangement.melody_tab]
        assert len(frets) == len(c.arrangement.melody)
        assert max(frets) <= 15, c  # baseline mean max 15.3, worst 20
        shifts.append(metrics.hand_shift_rate(frets))
    assert _mean(shifts) <= 0.13  # baseline 0.182


def test_v1_voice_leading_not_worse(cases):
    seen: dict[tuple[str, ...], float | None] = {}
    for c in cases:
        key = tuple(ch.to_label() for ch in c.progression)
        seen[key] = metrics.voice_leading_distance(c.arrangement.voicings, STANDARD_EADGBE)
    assert _mean(list(seen.values())) <= 1.63  # baseline on this matrix 1.621 (voicings unchanged)


def test_same_seed_same_music(cases):
    c = cases[17]
    again = arrange(
        c.scale,
        c.progression,
        bars_per_chord=c.bars_per_chord,
        tempo_bpm=TEMPO,
        style="fingerstyle",
        seed=c.seed,
    )
    assert again == c.arrangement
```

- [ ] **Step 2: Run it**

Run: `pytest tests/unit/test_compose_musicality.py -q --durations=3`
Expected: `23 passed`. The module fixture's setup takes about 2–3 s and the whole file stays under 5 s. (This is a regression test of targets that Tasks 1–5 already meet, so it passes on its first run. Step 3 proves that it bites.)

- [ ] **Step 3: Prove the test catches regressions (do not commit this)**

Temporarily change `INTERVAL_WEIGHT` entry `0: -0.8` to `0: 3.0` in `src/music_decoder/compose/melody.py`, then run:
`pytest tests/unit/test_compose_musicality.py -q -k "m2_m3_m4"`
Expected: FAIL on the `repeat_ratio` assertion (`<= 0.15`).
Restore the file: `git checkout -- src/music_decoder/compose/melody.py`. Re-run the step 2 command and expect `23 passed`.

- [ ] **Step 4: Update `docs/architecture-technical.md`**

(a) Replace everything from the line ``## 5. Data flow: `compose()` `` up to, but not including, the line `### 5.1 Voicings` with:

````markdown
## 5. Data flow: `compose()`

Implemented in [compose/api.py](../src/music_decoder/compose/api.py). All the
musical work happens in the pure `arrange()`. `compose()` adds only file I/O:

```text
(scale, progression, style, tempo, bars_per_chord, seed, tuning)
  │
  ▼
arrange(...)                                   # pure, no filesystem access
  │ _validate(scale, progression, bars_per_chord)
  │   mode ∈ {major, minor}; progression non-empty; bars_per_chord ≥ 1.
  │   Otherwise raises InvalidScaleError / InvalidProgressionError.
  │ compose.voicings.voicings_for(chord, tuning)[0] for each chord
  │ compose.melody.generate_melody(...)        # phrase-planned melody
  │ compose.arrangement._accompaniment_notes(..., melody=melody)
  │ _melody_tab(melody, tuning)                # A* fingering + fallback
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
_ascii_tab_for(arrangement, ...)               → ASCII string
  │
  ▼
Composition(midi_path, wav_path, ascii_tab, melody_notes, chord_voicings, metadata)
```

`compose()` is fully synchronous. Determinism: the only source of randomness
is `numpy.random.default_rng(seed)` inside the melody generator. It drives
both the rhythm plan and the pitch choices, so the same seed gives the same
music. `metadata["composer_version"]` (currently `2`) is bumped whenever the
generator changes what a given seed produces. Seeds from before version 2
(the uniform quarter-note generator) give different melodies now. The output
directory layout is content-addressed (`{timestamp}-{first 8 hex of payload
SHA-1}`), so callers can keep several takes side by side.
````

(b) Replace everything from the line `### 5.2 Melody generator` up to, but not including, the line `### 5.5 Synthesis` with:

````markdown
### 5.2 Melody generator

[compose/melody.py](../src/music_decoder/compose/melody.py)
`generate_melody` builds the list of `Note` for the whole progression in two
layers.

**Rhythm and phrases**, from [compose/rhythm.py](../src/music_decoder/compose/rhythm.py):

1. `split_phrases(n_bars)` cuts the piece into 4-bar phrases. A remainder of 2
   or 3 bars becomes a shorter last phrase, and a remainder of 1 turns the
   last 4 into 3 + 2.
2. `plan_bars(n_bars, rng)` gives every bar a rhythm (eighth-note values that
   sum to 8). Rhythms come from `BODY_RHYTHMS` (11 one-bar templates: quarters,
   eighth pairs, dotted figures, half notes). An **antecedent** phrase plays
   `[A, B, A, cadence]`, and its bar 3 echoes bar 1's intervals. The next
   **consequent** phrase reuses the antecedent's body rhythms and echoes its
   intervals bar for bar. Every phrase ends on a `CADENCE_RHYTHMS` bar
   (last note ≥ a half note).

**Pitches**, chosen one note at a time:

1. **Chord-scale pool.** `chord_scale_pcs(scale, chord)` is the scale (major,
   or natural minor) with any chord tone outside the scale swapped in for the
   non-chord scale tones a semitone away. Example: G# replaces G and A over
   E7 in A minor. Over diatonic chords the pool is just the scale. The
   candidates are the pool's pitches in `[pitch_low=62, pitch_high=79]`
   (D4–G5).
2. **Final target.** In the last bar the final note is fixed in advance: the
   tonic if the final chord contains it, otherwise the chord root, nearest
   the previous pitch.
3. **Score** each candidate with `score_pitch` (weights are module constants):
   - interval size (steps preferred; repeats, wide leaps and the tritone
     penalised; more than an octave excluded)
   - chord fit by metric weight (chord tones strongly preferred on beats 1
     and 3 and on long notes; quarter-or-longer notes a semitone off a chord
     tone discouraged)
   - leap recovery (an opposite-direction step after a leap of a fourth or
     more)
   - motif echo (same direction and size as the echoed bar's interval)
   - register gravity toward `tessitura_center=70`
   - a pull toward the final target in the last bar
   - a penalty on non-chord tones that end a phrase
4. **Sample** from the softmax of the scores with the seeded generator.

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
  that attack are dropped. At least the three lowest pitches are always kept,
  so the tune stays on top of the chords.

Accompaniment pitches come from `_voiced_pitches(voicing, tuning) =
sorted(open_pitches[s] + fret for fretted strings)`.

### 5.4 ASCII tab for the composition

`arrange()` fingers the melody with the same A\* assigner that `analyze()`
uses (`music_decoder.tabs.assign_tabs`, with the GuitarSet-tuned weights).
A note A\* cannot place falls back to its lowest-fret position.
`_ascii_tab_for` adds every chord voicing at the time its chord starts
(`chord_index × bars_per_chord × bar`) and calls
[render_ascii_tab](../src/music_decoder/tabs/render.py). The ASCII tab is an
approximate visual reference; the MIDI is the source of truth.

### 5.4.1 Musicality metrics

[compose/metrics.py](../src/music_decoder/compose/metrics.py) holds pure,
typed metric functions:

- melody: strong-beat chord-tone ratio, step / repeat / large-leap ratios,
  pooled leap recovery, ambitus, final resolution and length, rhythmic
  variety (distinct durations, IOI entropy), motif contour match,
  phrase-end length, non-diatonic clash
- accompaniment: grid alignment, melody above the accompaniment, exposed
  semitone clashes, final-chord ring
- tab: hand-position shifts
- voicings: voice-leading distance

[tests/unit/test_compose_musicality.py](../tests/unit/test_compose_musicality.py)
asserts targets for all of them over 400 melodies (8 keys, both modes,
10 progressions, 1–2 bars per chord, 5 seeds) × 3 styles. The definitions,
the baseline numbers and the targets are in
[the design spec](superpowers/specs/2026-11-06-compose-improvements-design.md).
````

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
  strong beats, small steps, and a final cadence on the home note.
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
     after a big jump the line steps back the other way. It stays in a
     comfortable register.
   - If a chord borrows a note from outside the scale (like G# in an E7
     chord in A minor), the melody borrows it too instead of clashing.
   The "seed" you pass is the only source of randomness: same seed,
   same melody, every time.

3. **Add the accompaniment**. Underneath the melody, the tool plays the
   chord on every bar in your chosen pattern. The patterns are: a pop strum
   (down, down-up, up-down-up), an eighth-note arpeggio, or fingerstyle with
   an alternating bass. It leaves out any chord notes that would sit on top
   of the melody, and it ends with one chord left ringing.
```

(c) Replace step **5** (from `5. **Write the tab**.` through `what you might see in any guitar magazine.`) with:

```markdown
5. **Write the tab**. The melody notes get placed onto the guitar
   neck by the same fingering search used for analysis, so the line
   stays in one hand position where it can. The chord shapes are added
   where each chord starts. The result is an ASCII tab string, the kind
   you might see in any guitar magazine.
```

- [ ] **Step 6: Update `docs/usage.md`**

(a) In the `compose` CLI options, replace:

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

(b) In the JSON example under `music-decoder compose --format json`, replace the line `    "tuning": "EADGBE"` (inside `"metadata"`) with these two lines:

```text
    "tuning": "EADGBE",
    "composer_version": 2
```

(c) Replace the bullets under `### Composition style choices`, from ``- **`fingerstyle`**`` through `delicate, classical-leaning pieces.`, with:

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
on or just under the melody, and the piece ends on one ringing chord.
The melody is written in four-bar phrases with a repeated motif, and it
ends on a long note on the tonic (or on the last chord's root if that
chord does not contain the tonic).
```

(d) Under `### Seed reproducibility`, after the paragraph that starts `` `--seed N` makes `compose()` fully deterministic. `` and its bullet list, add:

```markdown
Seeds are reproducible within one composer version, which is recorded
in `metadata.composer_version` (currently `2`). Version 2 (November
2026) replaced the uniform quarter-note generator with phrase-planned
melodies, so a seed saved from an older version now gives a different
(more musical) take.
```

- [ ] **Step 7: Run every gate**

Run: `pytest -q tests/unit tests/integration && ruff check src tests && ruff format --check src tests && mypy --strict src`
Expected: no failures, ruff clean, `Success: no issues found`. In the dry run the pytest summary read `430 passed, 3 skipped`; the exact count depends on the environment, but there must be 0 failures.
Run: `grep -n "bar by bar\|greedy-assigns\|block chord on beats 1 and 3\|11-tuplet" docs/*.md`
Expected: no output, so no stale description of the old composer remains.

- [ ] **Step 8: Commit**

```bash
git add tests/unit/test_compose_musicality.py docs/architecture-technical.md \
  docs/architecture-overview.md docs/usage.md
git commit -m "test(compose): musicality regression suite; docs for the new composer

Asserts the spec's metric targets over 400 melodies x 3 styles (strong-beat
chord tones, steps, leap recovery, ambitus, cadence, rhythmic variety, motif
repetition, accompaniment grid/register/clash, tab playability, voice-leading
guard). Documents the phrase planner, scored pitches, new accompaniment
patterns, A* melody tab and composer_version.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
