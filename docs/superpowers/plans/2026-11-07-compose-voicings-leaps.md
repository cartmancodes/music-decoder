# Faithful Voicings and Singable Leaps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every chord shape the composer plays faithful in every preset
tuning, and make melodies singable: no tritones or sevenths; leaps larger
than a fifth only when justified and always recovered; smooth chord changes
and final approaches. Every v2 musicality metric must stay within its stated
tolerance.

**Architecture** (revision 2, after plan review round 1, whose responses
are at the end of this plan):
- Repair seven entries in the shared library `chords/voicings.py`: five
  tone faults, plus two five-finger grips.
- Rebuild the tuning path in `compose/voicings.py`:
  - a pitch-preserving retune with a narrower un-mute rule
  - a fretted-only span gate and a four-finger feasibility gate
  - a faithfulness gate
  - a deterministic shape search
  - the v2 root fallback as a last resort
- Add hard admissibility rules (`admissible`) to the melody sampler, plus:
  - per-bar feasibility sets for the final bar and the approach bar
  - a deterministic floor/target repair order
  - a step bonus at chord changes
  - a graded scoring ladder (strict → R1+R2 → v2) for the rare case where
    nothing survives
- Measure everything with:
  - new metrics, including a pooled large-leap frequency `L0`
  - development matrices (validation, edge, tunings)
  - a frozen **holdout** matrix
  - a committed v2 snapshot

**Tech Stack:** Python 3, numpy, pretty_midi, click, pytest, ruff, mypy
--strict. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-11-07-compose-voicings-leaps-design.md`.
Read it first. Its section numbers (§) are cited throughout.

## Global Constraints

- `compose()` and the CLI keep their signatures and flags. The test
  `tests/unit/test_public_contract.py::test_compose_signature_is_frozen`
  must keep passing.
- The output is deterministic by seed. `COMPOSER_VERSION` becomes `3` (Task 6).
- No new dependencies, models or downloads.
- Gate, which must be green at every commit:
  - `pytest -q tests/unit tests/integration`
  - `ruff check src tests`
  - `ruff format --check src tests`
  - `mypy --strict src`

  Branch baseline: 582 passed, 3 skipped.
- Running the tests rewrites `tests/fixtures/synthetic/.*.synth_marker`.
  **Never commit those.** Stage explicit paths only; never use `git add -A`
  or `git commit -a`.
- Every commit message ends with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Never push,
  merge, switch branches or rewrite history.
- **Clean tree for generated fixtures.** Running the gate rewrites tracked
  `.synth_marker` files. Before writing any fixture whose `source` comes
  from `git describe --dirty`:
  - Restore those markers with `git checkout -- tests/fixtures/synthetic/`.
  - Check that `git status --porcelain --untracked-files=no` is empty.
- **Development versus holdout.** The validation, edge and tunings matrices
  are development data for v3 (spec §6). Never tune rules, weights or
  thresholds to the holdout matrix. A failed holdout check is reported with
  its numbers (spec §6.4).
- `music_decoder.compose` as an *attribute* is the re-exported function, not
  the package. To patch module objects, use dotted strings in `mock.patch`
  or `importlib.import_module("music_decoder.compose.api")`.
- Constants from the spec, verbatim:
  - `SMOOTH_LIMIT = 7`
  - `FORBIDDEN_INTERVALS = frozenset({6, 10, 11})`
  - `CHORD_CHANGE_STEP_BONUS = 1.0`
  - `MAX_FRETTED_SPAN = 3`
  - `MAX_FINGERS = 4`
  - search base frets 0–9; search minimum of 4 sounding strings
  - `MAX_INTERVAL = 12` is unchanged

## Review Focus

These are the five inputs most likely to break for a user that the spec
implies but the main tests do not reach. Each has a pinned test in the task
named.

1. **A custom `Tuning` higher than standard**, such as F standard
   `(41, 46, 51, 56, 60, 65)`. A shifted fret goes below 0, so the voicing
   must come from the search, never from a silently muted string. Task 5
   pins `test_tuning_above_standard_uses_the_search`.
2. **An exotic tuning with no faithful shape**, such as six strings all
   tuned to C2, with a D chord. The composer must still return a voicing
   (the v2 root fallback), not raise. Task 5 pins
   `test_unvoiceable_chord_falls_back_without_raising`.
3. **Direct `generate_melody` callers** using the legacy `notes_per_bar`
   grid and `chord_floors=None`. The new rules must hold and nothing may
   raise. Task 6 pins `test_legacy_grid_and_no_floors_obey_the_leap_rules`.
4. **One-chord and two-bar pieces** (a one-note final bar right after the
   first bar), and **chords foreign to the key** (review r1 F1: C → D#m).
   The feasibility repair must not crash, leap or relax. Task 6 pins:
   - `test_short_pieces_obey_the_leap_rules`
   - `test_review_r1_major_seventh_is_repaired`
   - `test_every_root_and_quality_composes_smoothly`
5. **The searched chords, played by the composer itself** (Drop C G, Drop D
   G7, DADGAD D#dim and the rest). They must compose with H1 and H2 intact
   in every style, not just pass the shape check. Task 5 pins
   `test_searched_voicings_compose_with_their_identity_heard`.

---

## File map

| File | Responsibility | Tasks |
|---|---|---|
| `src/music_decoder/compose/metrics.py` | New leap and shape metrics | 1 |
| `src/music_decoder/compose/voicings.py` | `shape_faults` and `fingers_needed` (Task 1); retune, gates, search, fallback (Task 5) | 1, 5 |
| `src/music_decoder/evaluation/compose_musicality.py` | `Case.tuning`, tunings and holdout matrices, new keys, L0 pooling, per-tuning groups | 2 |
| `scripts/compose_musicality_report.py` | Docstring only | 2 |
| `tests/fixtures/compose/musicality_v2.json` | New v2 snapshot (generated) | 2 |
| `tests/unit/test_compose_samples.py` | New CLI regression tests for the three samples | 3, 4, 5, 6 |
| `src/music_decoder/chords/voicings.py` | Seven repaired shapes and the invariant docstring | 4 |
| `src/music_decoder/compose/melody.py` | `admissible`, context fields, `feasible_sets`, the repair order and `ranked_scores` | 6 |
| `src/music_decoder/compose/api.py` | `COMPOSER_VERSION = 3` | 6 |
| `tests/unit/test_compose_musicality.py` | Guards (Task 4); new targets and v2 comparison (Task 7) | 4, 7 |
| `docs/usage.md`, `docs/architecture-technical.md`, the spec's Results | Documentation | 8 |

---

### Task 1: Leap and shape measurement helpers

Pure additions. The composer's behaviour does not change.

**Files:**
- Modify: `src/music_decoder/compose/metrics.py`: add functions after
  `leap_recovery_counts`, plus `bass_is_root` in the "tab and voicings"
  section. Add the new spec path to the module docstring.
- Modify: `src/music_decoder/compose/voicings.py`: add `shape_faults` and
  its labels.
- Test: `tests/unit/test_compose_metrics.py`, `tests/unit/test_compose_voicings.py`

**Interfaces:**
- Produces, in `compose.metrics`:
  - `forbidden_interval_count(melody: Sequence[Note]) -> int`: intervals of
    size 6, 10 or 11, or larger than 12.
  - `large_leap_recovery_counts(melody: Sequence[Note]) -> tuple[int, int]`:
    returns (leaps of more than 7 semitones followed by another interval,
    of those the ones whose next interval is opposite and 1–2 semitones).
  - `unjustified_large_leap_count(melody, progression: Sequence[ChordSymbol], bars_per_chord: int, tempo_bpm: float) -> int`:
    counts leaps of more than 7 where the two notes sound over different
    chords (`chord_at`), either note's pitch class is not in its chord's
    `chord_pcs`, or the leap lands on the last note.
  - `chord_change_moves(melody, progression, bars_per_chord, tempo_bpm) -> list[int]`:
    `abs` intervals between consecutive notes whose `chord_at` chords
    differ.
  - `bass_is_root(voicing: VoicedChord, tuning: Tuning) -> bool`: the lowest
    sounding pitch class equals `chord_pcs(chord)[0]`; `False` for an empty
    shape.
- Produces, in `compose.voicings`:
  - `shape_faults(frets: Sequence[int], tuning: Tuning, chord: ChordSymbol) -> tuple[str, ...]`
    returns labels in this fixed order, each at most once:
    - `"too few strings"`: fewer than 3 frets ≥ 0
    - `"foreign tone"`
    - `"root not in bass"`: only when some string sounds
    - `"missing tone"`: some tone of `harmony.chord_pcs(chord)` is absent,
      except that the perfect fifth (root + 7) may be absent when the chord
      has 4 tones
  - Export the four labels as module constants (`FAULT_TOO_FEW`,
    `FAULT_FOREIGN`, `FAULT_BASS`, `FAULT_MISSING`).
  - `fingers_needed(frets: Sequence[int]) -> int` implements spec §3.1's
    straight-barre model. Sum over each distinct fret `f > 0` the number of
    maximal runs of adjacent strings that are all fretted at `f` or higher
    and contain a note at exactly `f`. An open or muted string ends a run.

- [ ] **Step 1: Write the failing tests**

In `tests/unit/test_compose_metrics.py` (reuse its `n`, `prog`, `TEMPO`
helpers; one bar = 2 s = 4 beats):

```python
def test_forbidden_interval_count():
    mel = [n(0, 1, 60), n(1, 1, 66), n(2, 1, 76), n(3, 1, 65), n(4, 1, 78), n(5, 1, 60)]
    # +6 tritone, +10 m7, -11 M7, +13 (> octave), -18 (> octave)
    assert m.forbidden_interval_count(mel) == 5
    assert m.forbidden_interval_count([n(0, 1, 60), n(1, 1, 67), n(2, 1, 72), n(3, 1, 64)]) == 0


def test_large_leap_recovery_counts():
    up_back = [n(0, 1, 60), n(1, 1, 68), n(2, 1, 67)]  # +8 then -1: recovered
    up_on = [n(0, 1, 60), n(1, 1, 69), n(2, 1, 71)]  # +9 then +2: not
    fifth = [n(0, 1, 60), n(1, 1, 67), n(2, 1, 69)]  # +7 is not "large"
    assert m.large_leap_recovery_counts(up_back) == (1, 1)
    assert m.large_leap_recovery_counts(up_on) == (1, 0)
    assert m.large_leap_recovery_counts(fifth) == (0, 0)
    assert m.large_leap_recovery_counts([n(0, 1, 60), n(1, 3, 72)]) == (0, 0)  # last


def test_unjustified_large_leap_count():
    p = prog("C", "G")
    arpeggio = [n(0, 1, 64), n(1, 1, 72), n(2, 1, 71), n(3, 1, 67), n(4, 4, 67)]  # E4->C5 in C
    assert m.unjustified_large_leap_count(arpeggio, p, 1, TEMPO) == 0
    non_chord = [n(0, 1, 62), n(1, 1, 71), n(2, 1, 72), n(4, 4, 67)]  # D4 is not in C
    assert m.unjustified_large_leap_count(non_chord, p, 1, TEMPO) == 1
    across = [n(0, 4, 60), n(4, 1, 71), n(5, 3, 67)]  # C4 (C) -> B4 (G), +11 across the change
    assert m.unjustified_large_leap_count(across, p, 1, TEMPO) == 1
    into_final = [n(0, 1, 60), n(1, 1, 64), n(4, 1, 67), n(5, 3, 79)]  # +12 onto the last note
    assert m.unjustified_large_leap_count(into_final, p, 1, TEMPO) == 1


def test_chord_change_moves_and_bass_is_root():
    p = prog("C", "G")
    mel = [n(0, 2, 64), n(2, 2, 67), n(4, 2, 74), n(6, 2, 71)]
    assert m.chord_change_moves(mel, p, 1, TEMPO) == [7]
    assert m.chord_change_moves(mel, prog("C"), 1, TEMPO) == []
    assert m.bass_is_root(shape("C"), STANDARD_EADGBE)
    old_fmaj7 = VoicedChord(
        ChordSymbol.parse("Fmaj7"),
        tuple(TabPosition(s, f) for s, f in enumerate((-1, 3, 3, 2, 1, 0))),
    )  # C3 F3 A3 C4 E4
    assert not m.bass_is_root(old_fmaj7, STANDARD_EADGBE)
```

In `tests/unit/test_compose_voicings.py`:

```python
from music_decoder.compose.voicings import fingers_needed, shape_faults
from music_decoder.tabs.tuning import EB_HALF_STEP_DOWN


def test_shape_faults_names_each_defect_in_order():
    c = ChordSymbol.parse
    assert shape_faults((-1, 3, 2, 0, 1, 0), STANDARD_EADGBE, c("C")) == ()
    assert shape_faults((-1, 3, 2, 3, 1, 0), STANDARD_EADGBE, c("C7")) == ()  # fifth omitted: allowed
    assert shape_faults((-1, -1, 2, 3, 2, -1), STANDARD_EADGBE, c("C#dim")) == (
        "foreign tone", "root not in bass", "missing tone",
    )  # v2 C#dim: E3 A#3 C#4
    assert shape_faults((-1, 3, 3, 2, 1, 0), STANDARD_EADGBE, c("Fmaj7")) == ("root not in bass",)
    assert shape_faults((9, 4, -1, -1, -1, -1), EB_HALF_STEP_DOWN, c("Cm")) == (
        "too few strings", "missing tone",
    )  # v2 Eb fallback: C3 C3
    assert shape_faults((-1, 3, 2, 0, 1, -1), STANDARD_EADGBE, c("C")) == ()  # C E G C
    assert shape_faults((-1, 3, 5, 5, -1, -1), STANDARD_EADGBE, c("C")) == ("missing tone",)  # no E
    assert shape_faults((-1,) * 6, STANDARD_EADGBE, c("C")) == ("too few strings", "missing tone")


@pytest.mark.parametrize(
    ("frets", "fingers"),
    [
        ((0, 7, 6, 7, 5, 4), 5),  # the fret-7 notes are split by a fret-6 string
        ((-1, 6, 5, 6, 4, 3), 5),  # v2 D#7 x65643
        ((-1, 4, 3, 4, 2, 1), 5),  # v2 C#7 x43421
        ((1, 3, 3, 2, 1, 1), 3),  # barred F
        ((3, 2, 0, 0, 0, 3), 3),  # open G: the open strings split the fret-3 notes
        ((3, 0, 0, 0, 0, 3), 2),  # one barre cannot cross open strings
        ((-1, 4, 6, 4, 6, 4), 3),  # repaired C#7
        ((0, 7, 9, 7, 9, 7), 3),  # Eb-tuned D#7 after the repair
        ((-1,) * 6, 0),
    ],
)
def test_fingers_needed(frets, fingers):
    assert fingers_needed(frets) == fingers
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest -q tests/unit/test_compose_metrics.py tests/unit/test_compose_voicings.py`

Expected: FAIL with `AttributeError` / `ImportError` for the new names.

- [ ] **Step 3: Implement the five metrics functions, `shape_faults` and `fingers_needed`**

Use the signatures above. The metrics module defines its own private
constants (`_FORBIDDEN_SIZES = frozenset({6, 10, 11})`,
`_FIFTH = 7`, `_OCTAVE = 12`) rather than importing the generator.
`shape_faults` uses `harmony.chord_pcs`.

- [ ] **Step 4: Run the tests to verify they pass, then the gate**

Run: `pytest -q tests/unit/test_compose_metrics.py tests/unit/test_compose_voicings.py && ruff check src tests && ruff format --check src tests && mypy --strict src`

Expected: all pass and clean.

- [ ] **Step 5: Commit**

```bash
git add src/music_decoder/compose/metrics.py src/music_decoder/compose/voicings.py \
  tests/unit/test_compose_metrics.py tests/unit/test_compose_voicings.py
git commit -m "feat(compose): leap and shape-fault metrics for the voicing/leap fixes"
```

---

### Task 2: Evaluation: development and holdout matrices, new metric keys and the v2 snapshot

The composer is still unchanged in this task. The snapshot must be written
from it, and it is the only time the holdout is evaluated before Task 7.
Review r1 F4: this task makes **two commits**, each with a green full gate.

**Files:**
- Modify: `src/music_decoder/evaluation/compose_musicality.py`
- Modify: `scripts/compose_musicality_report.py`. Docstring only: it also
  writes `musicality_v2.json`, and covers the four matrices.
- Create (generated): `tests/fixtures/compose/musicality_v2.json`
- Test: `tests/unit/test_compose_evaluation.py`

**Interfaces:**
- Consumes: Task 1 metrics.
- Produces:
  - `TUNING_PROGRESSIONS: tuple[tuple[str, str, tuple[str, ...]], ...]`,
    the samples' progressions in sharp spelling:
    - `("B", "minor", ("Bm","G","D","A","Em7","C#dim","F#7","Bm"))`
    - `("G#", "major", ("G#","C7","Fm","C#","D#7","G#"))`
    - `("D#", "major", ("D#","Cm","Fm7","A#sus4","A#7","D#"))`
  - `tunings_inputs() -> list[Inputs]`: 15 entries (`VALIDATION_SEEDS`),
    one bar per chord.
  - `HOLDOUT_SEEDS = tuple(range(3000, 3005))`.
  - `HOLDOUT_PROGRESSIONS`: the ten progressions of spec §5.5, verbatim and
    in that order. For example `("F", "major", ("F","Dm7","Gm7","C7","F"))`
    and `("C#", "minor", ("C#m","F#m","G#7","C#m","A","G#7","C#m"))`.
  - `holdout_inputs() -> list[Inputs]`: 50 entries, one bar per chord,
    EADGBE.
  - `build_cases(inputs, tuning: Tuning = STANDARD_EADGBE) -> list[Case]`.
    `Case` gains a `tuning: Tuning` field. `case_metrics` uses
    `case.tuning` everywhere it now uses `STANDARD_EADGBE` (accompany, S1,
    V1, H1, H2).
  - `case_metrics` adds these keys:
    - `"L0.large"` and `"L0.intervals"` (ints): the count of intervals with
      `abs > 7`, and `len(intervals)`
    - `"L0.any"` (bool)
    - `"L1"`, `"L2.leaps"`, `"L2.recovered"` and `"L3"` (ints)
    - `"L4"` (the share of `chord_change_moves` ≤ 4, or `None`)
    - `"L4max"` (the max of those moves, or `None`)
    - `"L5"` (`abs` of the last interval, or `None`)
    - `"S2"` (the share of voicings where `not bass_is_root`)
  - `_aggregate` changes:
    - Pool `L2` like `M5`: skip the `"L2."` keys and emit
      `"L2": {"pooled", "n"}`.
    - Pool `L0` as `{"pooled": sum(large) / sum(intervals), "n": sum(intervals)}`.
    - Skip `"L0.large"` and `"L0.intervals"`.
    - Average `"L0.any"` normally.
  - `summarize` adds a group `f"tuning.{name}"` for each distinct
    `case.tuning.name` when the cases span more than one tuning.
  - `evaluate()` returns `{"validation", "edge", "tunings", "holdout"}`:
    - tunings: `[c for t in PRESETS.values() for c in build_cases(tunings_inputs(), t)]`,
      90 cases
    - holdout: `build_cases(holdout_inputs())`, 50 cases
  - Module docstring: the validation and edge matrices are v2-held-out but
    v3 development data; the holdout is v3's acceptance data (spec §6).

- [ ] **Step 1: Write the failing tests that do not need the snapshot** (append to `tests/unit/test_compose_evaluation.py`)

```python
def test_tunings_and_holdout_matrices():
    inputs = ev.tunings_inputs()
    assert len(inputs) == 15 and {seed for *_, seed in inputs} == set(ev.VALIDATION_SEEDS)
    assert {bars for _, _, bars, _ in inputs} == {1}
    labels = {tuple(c.to_label() for c in prog) for _, prog, _, _ in inputs}
    assert ("G#", "C7", "Fm", "C#", "D#7", "G#") in labels
    holdout = ev.holdout_inputs()
    assert len(holdout) == 50 and {seed for *_, seed in holdout} == set(range(3000, 3005))
    used = {seed for *_, seed in ev.validation_inputs() + ev.edge_inputs()}
    assert not used & {seed for *_, seed in holdout}


def test_case_metrics_use_the_case_tuning_and_report_leaps():
    from music_decoder.tabs.tuning import EB_HALF_STEP_DOWN

    (case,) = ev.build_cases(ev.tunings_inputs()[5:6], EB_HALF_STEP_DOWN)
    assert case.tuning == EB_HALF_STEP_DOWN
    row = ev.case_metrics(case)
    assert {"L0.large", "L0.intervals", "L0.any", "L1", "L2.leaps", "L2.recovered",
            "L3", "L4", "L4max", "L5", "S2"} <= set(row)


def test_summarize_pools_l2_and_groups_by_tuning():
    from music_decoder.tabs.tuning import DROP_D

    cases = ev.build_cases(ev.tunings_inputs()[:2]) + ev.build_cases(ev.tunings_inputs()[:2], DROP_D)
    summary = ev.summarize(cases)
    assert set(summary["overall"]["L2"]) == {"pooled", "n"}
    assert {"tuning.EADGBE", "tuning.Drop_D"} <= set(summary)
    assert "tuning.EADGBE" not in ev.summarize(cases[:2])


def test_l0_pools_intervals_while_m4_averages_pieces():  # review r1 F6
    rows = [
        {"M4": 0.002, "L0.large": 2, "L0.intervals": 1000, "L0.any": True},
        {"M4": 0.0, "L0.large": 0, "L0.intervals": 10, "L0.any": False},
    ]
    out = ev._aggregate(rows)
    assert out["M4"]["mean"] == 0.001  # mean of per-piece rates
    assert out["L0"] == {"pooled": 0.002, "n": 1010.0}  # 2 / 1010, rounded to 4 places
    assert out["L0.any"]["mean"] == 0.5
```

`_aggregate` must skip any pooled pair (`M5`, `M15`, `L2` or `L0`) whose
keys are absent from the rows. Today it would raise `KeyError` on rows
without `M5.*`. This keeps the hand-built rows minimal.

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest -q tests/unit/test_compose_evaluation.py`

Expected: FAIL with `AttributeError: tunings_inputs` / `holdout_inputs`.

- [ ] **Step 3: Implement the interfaces above** in `compose_musicality.py`
  and update the script docstring.

- [ ] **Step 4: Run the full gate**

Run: `pytest -q tests/unit tests/integration && ruff check src tests && ruff format --check src tests && mypy --strict src`

Expected: green. The existing musicality tests read only validation and
edge, which are unchanged.

- [ ] **Step 5: Commit 1 (code and runnable tests)**

```bash
git add src/music_decoder/evaluation/compose_musicality.py scripts/compose_musicality_report.py tests/unit/test_compose_evaluation.py
git commit -m "feat(eval): tunings and holdout matrices, leap/shape metric keys, pooled L0"
```

- [ ] **Step 6: Restore a clean tracked tree, then write the v2 snapshot**

```bash
git checkout -- tests/fixtures/synthetic/
test -z "$(git status --porcelain --untracked-files=no)" || { echo "tracked tree not clean"; exit 1; }
python scripts/compose_musicality_report.py --write tests/fixtures/compose/musicality_v2.json
```

Expected:
- `"source"` is Commit 1's short hash, with no `-dirty`.
- These v2 values were measured in the spec. Holdout values are new and
  unknown.
  - validation: `L1.max == 1`, `L4max.max == 12`, `L0.pooled ≈ 0.0044`,
    `M4.mean ≈ 0.004`, `L4.mean ≈ 0.852`
  - edge: `L4.mean ≈ 0.806`
  - tunings: `S1.missing.mean ≈ 0.088`, `S2.mean ≈ 0.107`

If `source` ends in `-dirty`, stop and find the modified tracked file.

- [ ] **Step 7: Add the snapshot test** (append to `tests/unit/test_compose_evaluation.py`)

```python
V2 = Path(__file__).parents[1] / "fixtures" / "compose" / "musicality_v2.json"


def test_committed_v2_snapshot_has_every_matrix_and_the_new_metrics():
    snap = json.loads(V2.read_text())
    assert {"source", "validation", "edge", "tunings", "holdout"} <= set(snap)
    assert not snap["source"].endswith("-dirty")
    v = snap["validation"]["overall"]
    assert v["L1"]["max"] == 1 and v["L4max"]["max"] == 12  # v2's measured defects
    assert set(v["L0"]) == {"pooled", "n"}
    assert snap["tunings"]["overall"]["S1.missing"]["mean"] > 0
```

- [ ] **Step 8: Run the full gate, then commit 2 (snapshot and its test)**

Run the full gate (green). Then restore the markers
(`git checkout -- tests/fixtures/synthetic/`) and:

```bash
git add tests/fixtures/compose/musicality_v2.json tests/unit/test_compose_evaluation.py
git commit -m "test(eval): commit composer v2 musicality snapshot (reference for v3, incl. holdout)"
```

---

### Task 3: The three sample commands as CLI regression tests (strict xfail)

**Files:**
- Create: `tests/unit/test_compose_samples.py`

**Interfaces:**
- Consumes:
  - `cli.main.main` and `cli.main._TUNINGS`
  - `compose.metrics` functions from Task 1
  - `compose.voicings.shape_faults`
- Produces: `SAMPLES: dict[str, list[str]]`, each value the exact CLI
  argument list (below). Tasks 4–6 remove the `xfail` markers.

- [ ] **Step 1: Write the tests**

```python
SAMPLES = {
    "minor-10-B": ["--scale", "B:minor", "--progression", "Bm G D A Em7 C#dim F#7 Bm",
                   "--style", "strum", "--tempo", "124", "--seed", "210", "--tuning", "EADGBE"],
    "major-08-Ab": ["--scale", "Ab:major", "--progression", "Ab C7 Fm Db Eb7 Ab",
                    "--style", "strum", "--tempo", "110", "--seed", "108", "--tuning", "Eb"],
    "major-09-Eb": ["--scale", "Eb:major", "--progression", "Eb Cm Fm7 Bbsus4 Bb7 Eb",
                    "--style", "fingerstyle", "--tempo", "72", "--seed", "109",
                    "--tuning", "D standard"],
}
```

- **Imports:** `fingers_needed` and `shape_faults` from
  `compose.voicings`; `metrics` from `compose`.
- **Helper `_run(args, tmp_path) -> dict`:**
  - Wrap the call in `mock.patch("music_decoder.compose.api.render_wav")`.
  - Invoke `CliRunner().invoke(main, ["compose", *args, "--out", str(tmp_path), "--format", "json"])`.
  - Assert `exit_code == 0`, then return `json.loads(result.output)`.
- **Helpers that rebuild objects from the JSON:**
  - `Note(**d)` for `melody_notes`.
  - `ChordSymbol(**d["chord"])` and `frets = [p["fret"] for p in d["positions"]]`
    for `chord_voicings`.
  - The tuning is `_TUNINGS[args[args.index("--tuning") + 1]]`.
  - The tempo is from the args.
  - The progression is the `ChordSymbol`s of the voicings.

```python
VOICING_XFAIL = {
    "minor-10-B": "C#dim library shape (Task 4)",
    "major-08-Ab": "Eb7 collapses in Eb tuning (Task 5)",
    "major-09-Eb": "Bb7 bass is the open D in D standard (Task 5)",
}


@pytest.mark.parametrize("name", [
    pytest.param(k, marks=pytest.mark.xfail(strict=True, reason=VOICING_XFAIL[k])) for k in SAMPLES
])
def test_sample_voicings_are_faithful(name, tmp_path):
    out = _run(SAMPLES[name], tmp_path)
    tuning = ...  # as above
    sounding = {}
    for v in out["chord_voicings"]:
        chord, frets = ChordSymbol(**v["chord"]), [p["fret"] for p in v["positions"]]
        assert shape_faults(frets, tuning, chord) == (), chord
        assert fingers_needed(frets) <= 4, chord
        sounding[chord.to_label()] = sorted(tuning.open_pitches[s] + f for s, f in enumerate(frets) if f >= 0)
    if name == "minor-10-B":
        assert {p % 12 for p in sounding["C#dim"]} == {1, 4, 7}
    if name == "major-08-Ab":
        assert len(sounding["D#7"]) >= 4 and {7, 1} <= {p % 12 for p in sounding["D#7"]}
    if name == "major-09-Eb":
        assert sounding["A#7"][0] % 12 == 10


@pytest.mark.parametrize("name", [
    pytest.param(k, marks=pytest.mark.xfail(strict=True, reason="melodic leaps (Task 6)")) for k in SAMPLES
])
def test_sample_melodies_are_singable(name, tmp_path):
    # Musical assertions only, in a fixed order (spec §5.6). No version check
    # here: it would mask these assertions while the composer is still v2.
    out = _run(SAMPLES[name], tmp_path)
    mel, prog, tempo = ...  # as above; bars_per_chord = 1
    assert metrics.forbidden_interval_count(mel) == 0, "L1"
    assert metrics.unjustified_large_leap_count(mel, prog, 1, tempo) == 0, "L3"
    leaps, recovered = metrics.large_leap_recovery_counts(mel)
    assert recovered == leaps, "L2"
    assert max(metrics.chord_change_moves(mel, prog, 1, tempo)) <= 7, "L4max"
    final = abs(mel[-1].pitch - mel[-2].pitch)
    assert final <= 7 and final != 6, "L5"


@pytest.mark.parametrize("name", list(SAMPLES))
def test_sample_commands_are_deterministic(name, tmp_path):
    a = _run(SAMPLES[name], tmp_path / "a")
    b = _run(SAMPLES[name], tmp_path / "b")
    assert a["melody_notes"] == b["melody_notes"] and a["chord_voicings"] == b["chord_voicings"]
```

- [ ] **Step 2: Run them**

Run: `pytest -q -rxX tests/unit/test_compose_samples.py`

Expected: `3 passed, 6 xfailed`, with no XPASS.

Then prove that each xfail fails *on the reported defect* (review r1 F3).
Run `pytest -q --runxfail tests/unit/test_compose_samples.py -k "singable or faithful"`
and read each failure:
- **Melody tests:** the failing assertion's message must be
  - `L3` for minor-10-B (G5→B4)
  - `L1` for major-08-Ab (G4→F5, +10)
  - `L3` for major-09-Eb (G#4→F5 at Cm→Fm7)
- **Voicing tests:**
  - minor-10-B fails on `shape_faults` for C#dim
  - major-08-Ab on `shape_faults` for D#7 (a single note)
  - major-09-Eb on `shape_faults` for A#7 (`root not in bass`)

Any other failure, such as an error in the test code or a different
assertion, must be fixed before committing. Record the six messages in the
commit body.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_compose_samples.py
git commit -m "test(compose): the three flagged sample commands as strict-xfail CLI regressions"
```

---

### Task 4: Repair the seven library shapes

**Files:**
- Modify: `src/music_decoder/chords/voicings.py`. Change the entries for
  C#dim, Ddim, D#dim, Fmaj7, G#aug, C#7 and D#7, and add an invariant
  paragraph to the module docstring stating the §3.1 rules: faithful and
  playable.
- Modify: `tests/unit/test_chord_detection_voicings.py`
- Modify: `tests/unit/test_compose_metrics.py`
  (`test_shape_tone_mismatch_reports_inherited_shape_defects`)
- Modify: `tests/unit/test_compose_musicality.py` (`test_guards_do_not_regress`)
- Modify: `tests/unit/test_compose_samples.py` (remove the minor-10-B
  voicing xfail)

**Interfaces:**
- Consumes: `compose.voicings.shape_faults` (Task 1);
  `tests/fixtures/compose/musicality_v2.json` (Task 2).
- Produces: the repaired `VOICINGS` entries:
  - `"C#dim": (-1, 4, 5, 6, 5, -1)`
  - `"Ddim": (-1, -1, 0, 1, 3, 1)`
  - `"D#dim": (-1, -1, 1, 2, 4, 2)`
  - `"Fmaj7": (-1, -1, 3, 2, 1, 0)`
  - `"G#aug": (-1, -1, 6, 5, 5, 4)`
  - `"C#7": (-1, 4, 6, 4, 6, 4)`
  - `"D#7": (-1, 6, 8, 6, 8, 6)`

- [ ] **Step 1: Write the failing tests** (in `test_chord_detection_voicings.py`)

```python
from music_decoder.compose.voicings import fingers_needed, shape_faults
from music_decoder.tabs.tuning import STANDARD_EADGBE
from music_decoder.types import ChordSymbol


def test_every_library_shape_is_faithful_and_playable():
    bad = {}
    for label, v in VOICINGS.items():
        if label == "N":
            continue
        fretted = [f for f in v if f > 0]
        span = max(fretted) - min(fretted) if fretted else 0
        faults = shape_faults(v, STANDARD_EADGBE, ChordSymbol.parse(label))
        if faults or span > 3 or fingers_needed(v) > 4:
            bad[label] = (faults, span, fingers_needed(v))
    assert bad == {}


@pytest.mark.parametrize(
    ("label", "frets"),
    [
        ("C#dim", (-1, 4, 5, 6, 5, -1)),  # C#3 G3 C#4 E4
        ("Ddim", (-1, -1, 0, 1, 3, 1)),  # D3 G#3 D4 F4
        ("D#dim", (-1, -1, 1, 2, 4, 2)),  # D#3 A3 D#4 F#4
        ("Fmaj7", (-1, -1, 3, 2, 1, 0)),  # F3 A3 C4 E4
        ("G#aug", (-1, -1, 6, 5, 5, 4)),  # G#3 C4 E4 G#4
        ("C#7", (-1, 4, 6, 4, 6, 4)),  # C#3 G#3 B3 F4 G#4 (v2 x43421 needs 5 fingers)
        ("D#7", (-1, 6, 8, 6, 8, 6)),  # D#3 A#3 C#4 G4 A#4 (v2 x65643 needs 5 fingers)
    ],
)
def test_repaired_shapes(label, frets):
    assert VOICINGS[label] == frets
```

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest -q tests/unit/test_chord_detection_voicings.py`

Expected: FAIL, listing exactly C#dim, Ddim, D#dim, Fmaj7, G#aug, C#7 and
D#7.

- [ ] **Step 3: Change the seven entries and the docstring.**

- [ ] **Step 4: Update the three dependent tests**

1. **`test_shape_tone_mismatch_reports_inherited_shape_defects`:**
   - Rename it to `test_shape_tone_mismatch_reports_shape_defects`.
   - Keep the `shape("C")` assertion.
   - Build the old C#dim as a hand-made `VoicedChord` from frets
     `(-1, -1, 2, 3, 2, -1)`, keep the `missing == {7}`, `foreign == {10}`
     and H1 assertions on it, and add
     `assert m.shape_tone_mismatch(shape("C#dim"), STANDARD_EADGBE) == (frozenset(), frozenset())`.
2. **`test_guards_do_not_regress`:**
   - Replace the S1 equality loop with
     `assert cand["S1.missing"]["max"] == 0 and cand["S1.foreign"]["max"] == 0`.
   - Replace the V1 line with
     `cand["V1"]["mean"] <= v2[matrix]["overall"]["V1"]["mean"] + 0.1`.
     The unit is semitones per voice (spec §6.2). The measured change is
     1.621 → 1.648.
   - Add a module-scoped `v2` fixture that loads `musicality_v2.json`.
   - Keep the M11a and H2 lines.
   - Update the comment: shapes are repaired, so V1 moves (spec §6.2).
3. **`test_compose_samples.py`:** turn the minor-10-B voicing param into a
   plain param.

- [ ] **Step 5: Run**

Run: `pytest -q -rxX tests/unit/test_chord_detection_voicings.py tests/unit/test_compose_metrics.py tests/unit/test_compose_musicality.py tests/unit/test_compose_samples.py tests/unit/test_chord_progression_renderers.py`

Expected:
- PASS
- samples: `4 passed, 5 xfailed`
- no XPASS. In particular the minor-10-B *melody* test stays xfailed (spec
  §5.7). If it XPASSes, stop and report.
- Re-run the Task 3 `--runxfail` check for the three melody tests. The
  failing messages must still be `L3`, `L1` and `L3` (review r1 F3).

- [ ] **Step 6: Run the full gate, then commit**

Run: `pytest -q tests/unit tests/integration && ruff check src tests && ruff format --check src tests && mypy --strict src`

```bash
git add src/music_decoder/chords/voicings.py tests/unit/test_chord_detection_voicings.py \
  tests/unit/test_compose_metrics.py tests/unit/test_compose_musicality.py tests/unit/test_compose_samples.py
git commit -m "fix(chords): repair C#dim, Ddim, D#dim, Fmaj7, G#aug, C#7 and D#7 library shapes"
```

---

### Task 5: Faithful voicings in every tuning

**Files:**
- Modify: `src/music_decoder/compose/voicings.py`
- Modify: `tests/unit/test_compose_voicings.py`
- Modify: `tests/unit/test_compose_harmony.py`. Drop
  `compose_voicings._CHORD_PCS` and its id from the parametrize of
  `test_every_chord_quality_has_a_row_in_each_compose_table`, and drop the
  now-unused import.
- Modify: `tests/unit/test_compose_samples.py` (remove the remaining two
  voicing xfails)

**Interfaces:**
- Consumes: `shape_faults` (Task 1), `harmony.chord_pcs` and
  `harmony.essential_pcs`.
- Produces:
  - `MAX_FRETTED_SPAN = 3`
  - `MAX_FINGERS = 4`
  - `SEARCH_MAX_BASE_FRET = 9`
  - `SEARCH_MIN_STRINGS = 4`
  - `_retune(frets: Sequence[int], tuning: Tuning, chord: ChordSymbol) -> tuple[int, ...] | None`
  - `_is_playable(frets: Sequence[int]) -> bool`
  - `search_voicing(chord: ChordSymbol, tuning: Tuning) -> tuple[int, ...] | None`,
    decorated with `@functools.lru_cache(maxsize=None)`
  - `voicings_for(chord, tuning) -> list[VoicedChord]`, unchanged signature,
    returning one element
  - `_CHORD_PCS` and `_chord_tone_pcs` are deleted.

Behaviour is exactly spec §5.2:
- **`_retune`:**
  - Return the input unchanged when `tuning.open_pitches == STANDARD_EADGBE.open_pitches`.
  - Otherwise shift each fretted string to keep its pitch, and return
    `None` if any shifted fret is below 0.
  - The un-mute reference is the lowest pitch of the shifted shape *before*
    any un-muting.
- **`_is_playable`:**
  - at least one fret ≥ 0
  - every fret ≤ 22
  - the span of frets > 0 is at most `MAX_FRETTED_SPAN`
  - `fingers_needed(frets) <= MAX_FINGERS`
- **`search_voicing`:**
  - window, bass rule, contiguity and selection key as in §5.2 item 5
  - its finger rule is `fingers_needed(combo) <= MAX_FINGERS`
  - acceptance is `shape_faults(...) == ()` and at least
    `SEARCH_MIN_STRINGS` sounding strings
- **`voicings_for`:**
  - Use `canonical.canonical_voicings_for(chord)` (`InvalidProgressionError`
    when empty, as before).
  - Retune. If the result is not `None`, is playable and has
    `shape_faults == ()`, use it.
  - Otherwise use `search_voicing`.
  - Otherwise use the v2 root fallback, unchanged code.

- [ ] **Step 1: Write the failing tests** (in `test_compose_voicings.py`)

```python
from music_decoder.chords.templates import QUALITIES, ROOTS, chord_label
from music_decoder.chords.voicings import VOICINGS
from music_decoder.tabs.tuning import PRESETS, DADGAD, DROP_C, D_STANDARD
from music_decoder.types import Tuning


def _frets(chord, tuning):
    return tuple(p.fret for p in voicings_for(ChordSymbol.parse(chord), tuning)[0].positions)


def test_every_chord_in_every_preset_tuning_is_faithful_and_playable():
    bad = []
    for name, tuning in PRESETS.items():
        for root in ROOTS:
            for quality in QUALITIES:
                label = chord_label(root, quality)
                frets = _frets(label, tuning)
                fretted = [f for f in frets if f > 0]
                span = max(fretted) - min(fretted) if fretted else 0
                faults = shape_faults(frets, tuning, ChordSymbol.parse(label))
                if faults or span > 3 or fingers_needed(frets) > 4:
                    bad.append((name, label, frets, faults, fingers_needed(frets)))
    assert bad == []  # v2: 98 of 576 faulty; revision 1: 22 needing 5 fingers (spec §3.3)


def test_standard_tuning_plays_the_library_shapes_unchanged():
    for label, frets in VOICINGS.items():
        if label != "N":
            assert _frets(label, STANDARD_EADGBE) == frets


@pytest.mark.parametrize(
    ("tuning", "label", "frets"),
    [
        (DROP_D, "D", (0, -1, 0, 2, 3, 2)),  # low D added (root); open A2 stays muted
        (DROP_C, "C", (0, 5, 4, 2, 3, 2)),
        (EB_HALF_STEP_DOWN, "D#7", (0, 7, 9, 7, 9, 7)),  # v2: a single D#2; not 0-7-6-7-5-4 (5 fingers)
        (EB_HALF_STEP_DOWN, "C7", (-1, 4, 3, 1, 0, 1)),  # searched: retuned x32310 would need 5 fingers
        (EB_HALF_STEP_DOWN, "Cm", (-1, 4, 6, 6, 5, 4)),  # v2: C3 C3
        (D_STANDARD, "A#7", (-1, 3, 5, 3, 5, 3)),  # v2: open D2 under the barre
        (DROP_C, "G", (-1, 0, 2, 2, 2, 0)),  # searched
        (DROP_D, "G7", (5, 2, 3, 0, 3, 3)),  # searched: the retuned span of 4 is rejected
        (DADGAD, "D#dim", (1, 0, 1, 2, 0, 4)),  # searched
    ],
)
def test_retune_and_search_choices(tuning, label, frets):
    assert _frets(label, tuning) == frets


# imports for this file: `_is_playable` and `search_voicing` from compose.voicings
def test_five_finger_retunes_are_rejected():
    assert not _is_playable((0, 7, 6, 7, 5, 4))
    assert not _is_playable((-1, 6, 5, 6, 4, 3))
    assert _is_playable((0, 7, 9, 7, 9, 7))


def test_tuning_above_standard_uses_the_search():  # Review Focus 1
    f_std = Tuning("F_standard", (41, 46, 51, 56, 60, 65))
    assert _frets("C", f_std) == (-1, 2, 1, 4, 0, 2)
    assert shape_faults(_frets("C", f_std), f_std, ChordSymbol.parse("C")) == ()


def test_unvoiceable_chord_falls_back_without_raising():  # Review Focus 2
    all_c = Tuning("C2x6", (36,) * 6)
    assert search_voicing(ChordSymbol.parse("D"), all_c) is None
    v = voicings_for(ChordSymbol.parse("D"), all_c)[0]
    assert {all_c.open_pitches[p.string] + p.fret for p in v.positions if p.fret >= 0} == {38}
```

This is the unchanged v2 root fallback: D2 at fret 2 on every string. The
point of the test is that the call returns, with the root only.

In `test_compose_api.py`, add (Review Focus 5):

```python
@pytest.mark.parametrize("style", ["arpeggio", "strum", "fingerstyle"])
@pytest.mark.parametrize(("tuning_name", "labels"), [
    ("Drop_C", ("C", "G", "Am", "G7", "Gmaj7", "C")),
    ("Drop_D", ("C", "G7", "C")),
    ("DADGAD", ("D", "A#", "D#dim", "A#sus4", "Bmaj7", "D")),
    ("Eb", ("C", "C7", "F", "B7", "C")),
])
def test_searched_voicings_compose_with_their_identity_heard(tuning_name, labels, style):
    tuning = get_preset(tuning_name)
    arr = arrange(Scale(labels[0], "major"), _chords(*labels), style=style, tuning=tuning, seed=3)
    assert all(shape_faults([p.fret for p in v.positions], tuning, v.chord) == () for v in arr.voicings)
    assert all(metrics.essential_tone_coverage(arr.melody, arr.accompaniment, arr.voicings, 1, 100.0, tuning))
    assert metrics.cadence_identity(arr.melody, arr.accompaniment, arr.voicings, tuning)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest -q tests/unit/test_compose_voicings.py tests/unit/test_compose_api.py -k "faithful or retune or above_standard or unvoiceable or searched or library_shapes"`

Expected: FAIL (98 bad voicings; `search_voicing` missing).

- [ ] **Step 3: Implement §5.2** with the interfaces above. Then fix
  `test_compose_harmony.py` and turn the two remaining voicing params in
  `test_compose_samples.py` into plain params.

- [ ] **Step 4: Run**

Run: `pytest -q -rxX tests/unit/test_compose_voicings.py tests/unit/test_compose_api.py tests/unit/test_compose_harmony.py tests/unit/test_compose_samples.py tests/unit/test_compose_musicality.py`

Expected:
- PASS
- samples: `6 passed, 3 xfailed`, with no XPASS
- the existing Drop D test still passes
- the Task 3 `--runxfail` check of the three melody tests still shows `L3`,
  `L1` and `L3` (review r1 F3)

- [ ] **Step 5: Run the full gate, then commit**

Run: `pytest -q tests/unit tests/integration && ruff check src tests && ruff format --check src tests && mypy --strict src`

```bash
git add src/music_decoder/compose/voicings.py tests/unit/test_compose_voicings.py \
  tests/unit/test_compose_api.py tests/unit/test_compose_harmony.py tests/unit/test_compose_samples.py
git commit -m "fix(compose): faithful, four-finger chord voicings in every tuning (retune, gates, shape search)"
```

---

### Task 6: Singable melodic leaps

**Files:**
- Modify: `src/music_decoder/compose/melody.py`
- Modify: `src/music_decoder/compose/api.py` (`COMPOSER_VERSION = 3`)
- Modify: `tests/unit/test_compose_melody.py`
- Modify: `tests/unit/test_compose_api.py`: the version assertion becomes
  `== 3`, plus the sweep tests below.
- Modify: `tests/unit/test_compose_samples.py`: remove the melody xfails and
  add the version test.

**Interfaces:**
- Consumes: Task 1 metrics (in tests).
- Produces, in `compose.melody`:
  - the constants `SMOOTH_LIMIT = 7`,
    `FORBIDDEN_INTERVALS = frozenset({6, 10, 11})` and
    `CHORD_CHANGE_STEP_BONUS = 1.0`
  - `NoteContext` fields `chord_change: bool = False`,
    `prev_chord_tone: bool = False`,
    `recovery_pool: frozenset[int] | None = None` and
    `allowed: frozenset[int] | None = None`
  - `admissible(candidate: int, ctx: NoteContext) -> bool`, rules R1–R4 of
    spec §5.3
  - `smooth(a: int, b: int) -> bool`: `abs(b - a) <= SMOOTH_LIMIT and abs(b - a) != 6`
  - `feasible_sets(candidates: Sequence[int], n_free: int, goal: Collection[int]) -> list[frozenset[int]]`,
    the backward construction of spec §5.3. It returns `[]` when
    `n_free == 0`.
  - `ranked_scores(candidates: Sequence[int], ctx: NoteContext, tessitura_center: int) -> tuple[list[float], int]`:
    the ladder of levels 0, 1 and 2 from spec §5.3
  - `_candidates` is **unchanged** from v2. Revision 1's `cadence_reach`
    is dropped.

**Wiring in `generate_melody`** (spec §5.3, "Repair order"):

- **Before the loop:**
  - `final_approach`: the final chord's `cadence_pc` pitches inside
    `[pitch_low, pitch_high]`, kept only at or above
    `chord_floors[-1]` when any remain (all of them if `chord_floors` is
    `None`)
  - `one_note_final = len(bars[-1].slots) == 1`
- **At each bar start:**
  - `approach_bar = one_note_final and bar_idx == len(bars) - 2`
  - options: `floored = _candidates(...)` exactly as v2, then
    `unfloored = _candidates(..., floor=None, ...)` if it differs
  - pick `(candidates, target, feas)` by the spec's order:
    - *Final bar:* cadence pitches sorted by `(abs(p - ref), p)`, with
      `feas = feasible_sets(option, n - 1, {target})`. The bar can be
      entered when `smooth(prev, target)` (for `n == 1`), or when some
      `c` in `feas[0]` has `smooth(prev, c)`. `prev is None` always
      enters.
    - *Approach bar:* `feas = feasible_sets(option, n, final_approach)`.
    - *Other bars:* `feas = None`; the option needs some pitch `smooth`
      from `prev`.
    - *If no option works:* `floored`, v2's `final_target(...)`, and
      `feas = None`.
- **Per sampled slot `k`:**
  - `chord_change = prev_chord is not None and chord != prev_chord`
  - `prev_chord_tone = prev is not None and prev % 12 in tones`
  - `allowed = feas[k] if feas else None`
  - `recovery_pool = frozenset(candidates)` if `k + 1 < n` and the bar is
    neither the final bar nor the approach bar, else `None`
  - Then `scores, _ = ranked_scores(candidates, ctx, tessitura_center)`.
- **`score_pitch`:** inside the `prev_pitch is not None` block, add
  `CHORD_CHANGE_STEP_BONUS` when `ctx.chord_change and 1 <= size <= 2`.
- **R3 recovery check:** the pool contains some `r` with
  `1 <= (candidate - r) * sign(interval) <= 2`.

- [ ] **Step 1: Write the failing unit tests** (in `test_compose_melody.py`,
  using the existing `_ctx`)

```python
from music_decoder.compose.melody import (
    CHORD_CHANGE_STEP_BONUS, admissible, feasible_sets, ranked_scores,
)


def test_admissible_forbids_tritones_sevenths_and_wide_leaps():
    for c in (78, 82, 83, 85, 66, 62, 61):  # +6 +10 +11 +13, -6 -10 -11 from 72
        assert not admissible(c, _ctx())
    assert admissible(79, _ctx()) and admissible(65, _ctx())  # fifths up and down


def test_admissible_large_leap_needs_arpeggiation_and_a_recovery():
    ok = dict(prev_chord_tone=True, recovery_pool=frozenset({74, 76, 77}))
    assert not admissible(80, _ctx(prev_pitch=72, **ok))  # G# is not a C chord tone
    assert admissible(79, _ctx(prev_pitch=67, prev_chord_tone=True, recovery_pool=frozenset({77})))
    assert not admissible(79, _ctx(prev_pitch=67, prev_chord_tone=True, recovery_pool=None))
    assert not admissible(79, _ctx(prev_pitch=67, prev_chord_tone=True, recovery_pool=frozenset({76})))
    assert not admissible(79, _ctx(prev_pitch=67, prev_chord_tone=False, recovery_pool=frozenset({77})))
    assert not admissible(79, _ctx(prev_pitch=67, prev_chord_tone=True, chord_change=True,
                                   recovery_pool=frozenset({77})))
    assert not admissible(79, _ctx(prev_pitch=67, prev_chord_tone=True, prev_interval=5,
                                   recovery_pool=frozenset({77})))  # no chained leaps
    assert not admissible(79, _ctx(prev_pitch=67, prev_chord_tone=True, target=79,
                                   recovery_pool=frozenset({77})))  # never in the final bar


def test_admissible_forces_recovery_after_a_large_leap():
    after = _ctx(prev_pitch=76, prev_interval=9)
    assert admissible(74, after) and admissible(75, after)
    assert not admissible(77, after) and not admissible(72, after) and not admissible(76, after)


def test_admissible_respects_the_feasibility_set():
    assert admissible(74, _ctx(allowed=frozenset({74}))) and not admissible(71, _ctx(allowed=frozenset({74})))


def test_feasible_sets_walk_back_from_the_goal():
    # review r1 F1: floored final bar of C -> D#m; target D#5 (75), prev E4 (64)
    floored = [69, 70, 72, 74, 75, 77, 78]
    (first,) = feasible_sets(floored, 1, {75})
    assert first == {70, 72, 74, 75, 77, 78}  # 69 is a tritone from 75
    assert not any(abs(c - 64) <= 7 and abs(c - 64) != 6 for c in first)  # dead end: repair
    assert feasible_sets(floored, 0, {75}) == []


def test_ranked_scores_ladder_keeps_r1_and_r2_before_giving_up():
    ctx = _ctx(prev_pitch=64, allowed=frozenset())  # nothing is admissible
    scores, level = ranked_scores([66, 75], ctx, 70)
    assert level == 1 and math.isfinite(scores[0]) and scores[1] == -math.inf  # +11 stays forbidden
    scores, level = ranked_scores([75], _ctx(prev_pitch=64, allowed=frozenset()), 70)
    assert level == 2 and math.isfinite(scores[0])  # only a forbidden move exists
    assert ranked_scores([66], _ctx(prev_pitch=64), 70)[1] == 0


def test_chord_change_step_bonus():
    step = score_pitch(74, _ctx(chord_change=True), 72) - score_pitch(74, _ctx(), 72)
    assert step == pytest.approx(CHORD_CHANGE_STEP_BONUS)
    assert score_pitch(72, _ctx(chord_change=True), 72) == score_pitch(72, _ctx(), 72)
```

Rewrite `test_one_note_final_bar_stays_within_an_octave_when_a_floor_bites`
as `test_one_note_final_bar_is_approached_within_a_fifth`. Keep the same
200-seed loop, and assert `abs(notes[1].pitch - notes[0].pitch) <= 7`,
`!= 6`, and `notes[1].pitch % 12 == 7`. Delete the `low_starts`
precondition.

Generator-level tests in `test_compose_melody.py`: `_leap_invariants`,
`test_generated_melodies_obey_the_leap_rules`,
`test_legacy_grid_and_no_floors_obey_the_leap_rules` (Review Focus 3) and
`test_short_pieces_obey_the_leap_rules` (Review Focus 4):

```python
def _leap_invariants(notes, labels, bars=1, tempo=120.0):
    p = [ChordSymbol.parse(c) for c in labels]
    assert metrics.forbidden_interval_count(notes) == 0
    assert metrics.unjustified_large_leap_count(notes, p, bars, tempo) == 0
    leaps, recovered = metrics.large_leap_recovery_counts(notes)
    assert recovered == leaps
    assert max(metrics.chord_change_moves(notes, p, bars, tempo), default=0) <= 7
    if len(notes) > 1:
        assert abs(notes[-1].pitch - notes[-2].pitch) <= 7


@pytest.mark.parametrize(("key", "mode", "labels"), [
    ("C", "major", ("C", "Am", "F", "G")), ("A", "minor", ("Am", "Dm", "E7", "Am")),
    ("C", "major", ("Cmaj7", "Am7", "Dm7", "G7")), ("C", "major", ("C", "B7", "Em", "G7", "C")),
])
def test_generated_melodies_obey_the_leap_rules(key, mode, labels):
    for seed in range(100):
        _leap_invariants(_gen(Scale(key, mode), labels, seed=seed, chord_floors=[None] * len(labels)), labels)


def test_legacy_grid_and_no_floors_obey_the_leap_rules():  # Review Focus 3
    for seed in range(100):
        for npb in (1, 4):
            labels = ("C", "F", "G7", "C")
            _leap_invariants(_gen(Scale("C", "major"), labels, seed=seed, notes_per_bar=npb), labels)


def test_short_pieces_obey_the_leap_rules():  # Review Focus 4
    for seed in range(100):
        for labels in (("D",), ("C", "G7"), ("A", "Dsus4")):
            notes = _gen(Scale(labels[0][0], "major"), labels, seed=seed, chord_floors=[None, 68][: len(labels)])
            _leap_invariants(notes, labels)
```

Composer-level tests in `test_compose_api.py` (review r1 F1), using the
real floors from `arrange()`:

```python
def _api_leap_invariants(arr, labels, tempo=100.0):
    mel, p = arr.melody, _chords(*labels)
    assert metrics.forbidden_interval_count(mel) == 0
    assert metrics.unjustified_large_leap_count(mel, p, 1, tempo) == 0
    leaps, recovered = metrics.large_leap_recovery_counts(mel)
    assert recovered == leaps
    assert max(metrics.chord_change_moves(mel, p, 1, tempo), default=0) <= 7
    assert abs(mel[-1].pitch - mel[-2].pitch) <= 7


def test_review_r1_major_seventh_is_repaired():
    arr = arrange(Scale("C", "major"), _chords("C", "D#m"), seed=19)  # v2 and rev. 1: 64 -> 75
    _api_leap_invariants(arr, ("C", "D#m"))
    assert arr.melody[-1].pitch % 12 == 3  # still ends on D#, the root of D#m


@pytest.mark.parametrize(("key", "mode", "labels", "seed"), [
    ("C", "major", ("C", "D#m"), 20), ("C", "major", ("C", "D#m7"), 20),
    ("A", "minor", ("Am", "D#m"), 20), ("A", "minor", ("Am", "D#m7"), 20),
])
def test_revision_1_sweep_failures_are_repaired(key, mode, labels, seed):
    _api_leap_invariants(arrange(Scale(key, mode), _chords(*labels), seed=seed), labels)


def test_every_root_and_quality_composes_smoothly():  # spec §6.4 bounded sweep
    for key, mode, home in (("C", "major", "C"), ("A", "minor", "Am")):
        for root in ROOTS:
            for quality in QUALITIES:
                x = chord_label(root, quality)
                for labels in ((home, x), (home, x, home)):
                    for seed in range(5):
                        _api_leap_invariants(arrange(Scale(key, mode), _chords(*labels), seed=seed), labels)
```

The bounded sweep runs 1,920 compositions in about 2 s. With default
arguments, `arrange()` uses tempo 100 and EADGBE. The file needs these
imports: `ROOTS`, `QUALITIES` and `chord_label` from `chords.templates`,
and `metrics` from `compose`. Every assertion in this step was run against
the revision-2 planning prototype and passes.

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest -q tests/unit/test_compose_melody.py tests/unit/test_compose_api.py`

Expected: FAIL (`ImportError: admissible`; the sweeps fail on v2 melodies;
C → D#m seed 19 fails on L1).

- [ ] **Step 3: Implement spec §5.3** with the wiring above. Then set
  `COMPOSER_VERSION = 3`, and update `test_compose_api.py`'s `== 2` to
  `== 3`.

- [ ] **Step 4: Remove the three melody xfail markers and add the version test**

In `test_compose_samples.py`, turn the melody params into plain params and
add:

```python
def test_sample_metadata_reports_composer_version_3(tmp_path):
    out = _run(SAMPLES["minor-10-B"], tmp_path)
    assert out["metadata"]["composer_version"] == 3
```

Run: `pytest -q -rxX tests/unit/test_compose_melody.py tests/unit/test_compose_api.py tests/unit/test_compose_samples.py tests/unit/test_compose_musicality.py tests/unit/test_compose_arrangement.py tests/integration/test_compose_e2e.py`

Expected:
- PASS, with no xfail or XPASS left in the samples file
- every existing v2 musicality test still passes

- [ ] **Step 5: Measure and report (no prototype-matching gate)**

Run: `python scripts/compose_musicality_report.py | grep -E "^ *(M1|M2|M4|M4max|L0|L1|L3|L4|L4max|L5) "`

For reference, the spec's prototype values on the development matrices
(validation / edge) are:
- `M1 ≈ 0.977 / 0.984`
- `L0 pooled ≈ 0.0003 / 0`
- `M4max = 8 / 7`
- `L1 = L3 = 0`, `L4max = 7`, `L5 = 7`
- `L4 ≈ 0.884 / 0.848`

A difference is not a failure in itself. Acceptance comes from the targets
in Task 7 and spec §6. Still, explain any mean that differs by more than
0.005 (a wiring difference, or a deliberate correction) in the commit body.
**Do not look at or tune to the holdout numbers** in this task.

- [ ] **Step 6: Run the full gate, then commit**

Run: `pytest -q tests/unit tests/integration && ruff check src tests && ruff format --check src tests && mypy --strict src`

```bash
git add src/music_decoder/compose/melody.py src/music_decoder/compose/api.py \
  tests/unit/test_compose_melody.py tests/unit/test_compose_api.py tests/unit/test_compose_samples.py
git commit -m "fix(compose): singable melodies - no tritones/sevenths, justified recovered leaps, feasible smooth approaches (composer v3)"
```

---

### Task 7: Musicality targets: leaps, shapes, tunings and no regression against v2

**Files:**
- Modify: `tests/unit/test_compose_musicality.py`. Extend the module
  docstring with the spec path, add a `v2` fixture (if Task 4 did not
  already add one), and add the tests below.

**Interfaces:**
- Consumes:
  - `evaluate()` with four matrices (Task 2)
  - `musicality_v2.json` (Task 2)
  - metric keys `L0`, `L1`, `L2`, `L3`, `L4`, `L4max`, `L5`, `S1.*` and `S2`
- Produces: none (tests only).

- [ ] **Step 1: Write the tests**

```python
MATRICES = ("validation", "edge", "tunings", "holdout")
TOL_MEAN, TOL_POOLED = 0.02, 0.05  # spec §6.3: under half a note per piece; about one pooled event


def test_leap_regression_locks_on_development_matrices(report, v2):
    # Development data (spec §6): these are regression locks, not generalisation evidence.
    for matrix, l4_lock in (("validation", 0.87), ("edge", 0.83)):
        o, ref = report[matrix]["overall"], v2[matrix]["overall"]
        assert o["L0"]["pooled"] <= 0.001 and o["L0"]["pooled"] < ref["L0"]["pooled"]
        assert o["M4"]["mean"] <= 0.001 and o["M4"]["mean"] < ref["M4"]["mean"]  # mean of per-piece rates
        assert o["L4"]["mean"] >= l4_lock


@pytest.mark.parametrize("matrix", MATRICES)
def test_chord_changes_are_smoother_than_v2(report, v2, matrix):
    assert report[matrix]["overall"]["L4"]["mean"] >= v2[matrix]["overall"]["L4"]["mean"]
    if matrix != "holdout":  # strict on development data (measured); spec §6.4 asks >= on the holdout
        assert report[matrix]["overall"]["L4"]["mean"] > v2[matrix]["overall"]["L4"]["mean"]


def test_holdout_acceptance(report, v2):  # spec §6.4, frozen before evaluation
    o, ref = report["holdout"]["overall"], v2["holdout"]["overall"]
    assert o["L0"]["pooled"] <= min(0.002, ref["L0"]["pooled"]) or o["L0"]["pooled"] == 0.0
    assert o["M4max"]["max"] <= 12 and o["M6"]["max"] <= 17 and o["T1"]["max"] <= 15
    assert o["M7"]["min"] == 1.0 and o["M8"]["min"] >= 1.5
    assert o["M12"]["min"] == 1.0 and o["M14"]["min"] == 1.0 and o["Tcount"]["min"] == 1.0
    for style in STYLES:
        for key in ("A1", "A4", "H1", "H2"):
            assert o[f"{key}.{style}"]["min"] == 1.0, f"{key}.{style}"


@pytest.mark.parametrize("matrix", MATRICES)
def test_leap_and_shape_invariants(report, matrix):
    o = report[matrix]["overall"]
    assert o["L1"]["max"] == 0  # no tritones, sevenths or leaps over an octave
    assert o["L3"]["max"] == 0  # every leap > a fifth is arpeggiation inside one chord
    assert o["L2"]["n"] == 0 or o["L2"]["pooled"] == 1.0  # ...and is recovered by a step
    assert o["L4max"]["max"] <= 7 and o["L5"]["max"] <= 7
    assert o["S1.missing"]["max"] == 0 and o["S1.foreign"]["max"] == 0 and o["S2"]["max"] == 0


def test_tunings_matrix_keeps_every_v2_hard_invariant(report):
    o = report["tunings"]["overall"]
    assert o["M4max"]["max"] <= 12 and o["M6"]["max"] <= 17
    assert o["M7"]["min"] == 1.0 and o["M8"]["min"] >= 1.5
    assert o["M12"]["min"] == 1.0 and o["M14"]["min"] == 1.0 and o["Tcount"]["min"] == 1.0
    for style in STYLES:
        for key in ("A1", "A4", "H1", "H2"):
            assert o[f"{key}.{style}"]["min"] == 1.0, f"{key}.{style}"
    for name, tuning in PRESETS.items():  # spec D15
        cap = 15 + (64 - tuning.open_pitches[-1])
        assert report["tunings"][f"tuning.{name}"]["T1"]["max"] <= cap, name


@pytest.mark.parametrize(
    ("matrix", "group"),
    [*(("validation", g) for g in GROUPS), ("edge", "overall"), ("holdout", "overall")],
)
def test_no_regression_against_v2(report, v2, matrix, group):
    cand, ref = report[matrix][group], v2[matrix][group]
    for metric in (*IMPROVE_UP, *IMPROVE_DOWN):
        if metric not in cand or metric not in ref:
            continue
        c, r = cand[metric]["mean"], ref[metric]["mean"]
        if metric in IMPROVE_UP:
            assert c >= r - TOL_MEAN, f"{matrix}/{group} {metric}: {r} -> {c}"
        else:
            assert c <= r + TOL_MEAN, f"{matrix}/{group} {metric}: {r} -> {c}"
    for metric in ("M5", "M15"):
        if cand[metric]["n"] and ref[metric]["n"]:
            assert cand[metric]["pooled"] >= ref[metric]["pooled"] - TOL_POOLED, metric
```

Also extend `test_hard_invariants` to `@pytest.mark.parametrize("matrix", ["validation", "edge"])`
(unchanged: T1 ≤ 15 applies to EADGBE only). Leave the v2 pooled thresholds
(`M5 ≥ 0.70`, `M15 ≥ 0.75`) as they are.

- [ ] **Step 2: Run them, then sabotage-check**

Run: `pytest -q tests/unit/test_compose_musicality.py`

Expected: PASS.

Then confirm that the new tests can fail, without committing anything:

1. Temporarily set `FORBIDDEN_INTERVALS = frozenset()` in `melody.py`.
   `test_leap_and_shape_invariants` must fail (L1).
2. Temporarily make the bar-level repair use only the floored option. Both
   `test_review_r1_major_seventh_is_repaired` and
   `test_holdout_acceptance` or the bounded sweep must flag it. If only the
   explicit regression flags it, say so in the commit body.
3. Temporarily set `MAX_FINGERS = 5` in `compose/voicings.py`.
   `test_every_chord_in_every_preset_tuning_is_faithful_and_playable` must
   fail (the Eb D#7 retune `0-7-6-7-5-4` would be accepted again).
4. Temporarily set `MAX_FRETTED_SPAN = 5` in `compose/voicings.py`.
   `test_leap_and_shape_invariants[tunings]` must fail (S2 or S1), or
   `test_every_chord_in_every_preset_tuning_is_faithful_and_playable` must
   fail.

Revert both edits with `git checkout -- src/music_decoder/compose/melody.py src/music_decoder/compose/voicings.py`
and check that `git status --short src` is empty.

- [ ] **Step 3: Run the full gate and time the musicality module**

Run: `pytest -q tests/unit tests/integration && ruff check src tests && ruff format --check src tests && mypy --strict src`

Then run `pytest -q --durations=5 tests/unit/test_compose_musicality.py`.
Expected: the module takes about 6 s or less, of which `evaluate()` is
about 4–4.5 s. Record the figure for Task 8.

**If the holdout acceptance fails:** do not change thresholds or rules to
make it pass. Report the failing numbers in the commit body and in Task 8's
Results, and mark the test with `pytest.mark.xfail(strict=True, reason=...)`
that names the metric. A correction must come from a defect reproduced on
development data, with the spec updated (spec §6.4).

- [ ] **Step 4: Commit**

```bash
git add tests/unit/test_compose_musicality.py
git commit -m "test(compose): v3 musicality targets - leaps, shapes, tunings and holdout matrices, no regression vs v2"
```

---

### Task 8: Refresh the musicality report, docs and results; final gate

**Files:**
- Modify: `docs/usage.md`:
  - `composer_version` becomes `3` in the JSON example (line ~705) and in
    the "Seeds are reproducible" paragraph (line ~1025). Add one sentence on
    version 3: faithful shapes in every tuning, no tritone or seventh leaps,
    and seeds from v2 now give a different take.
  - Mention `musicality_v2.json` next to the baseline (line ~1033).
- Modify: `docs/architecture-technical.md`:
  - The determinism paragraph (line ~689): currently `3`.
  - §5.1 Voicings (lines ~698–722): rewrite steps 2–6 to match spec §5.2,
    covering:
    - the un-mute rule
    - the fretted span ≤ 3 and the four-finger gate
    - the faithfulness gate
    - the search
    - the last-resort fallback
  - §5.2 Melody (lines ~756–790): add to steps 2–5:
    - the admissibility rules R1–R4
    - the feasibility sets and the repair order
    - the ladder
    - the step bonus
  - The S1 note in §5.3 (line ~838): the shapes are now repaired, and S1
    and S2 are asserted 0.
  - "Add a new chord quality" (line ~1312): drop the `_CHORD_PCS` step.
- Modify: `docs/superpowers/specs/2026-11-07-compose-voicings-leaps-design.md`:
  append `## 13. Results`.
- Modify: `src/music_decoder/compose/metrics.py`: check that its docstring
  names both specs (Task 1).

**Interfaces:** none.

- [ ] **Step 1: Refresh the measurements**

Run: `python scripts/compose_musicality_report.py > /tmp/v3_report.txt`.
Do **not** pass `--write`: both committed fixtures stay as they are.

Compare against `musicality_v2.json`, spec §6 (targets) and the prototype
column (for information). Any development-matrix mean that differs from the
prototype by more than 0.005 must be explained in Results: a wiring
difference, or a deliberate correction. The holdout has no prototype
column; report it as measured.

- [ ] **Step 2: Write `## 13. Results` in the spec**

Include:
- a before (v2 snapshot) / after (v3 report) table for every metric in §6.1
  and §6.2, and for the §6.3 rows, on the validation, edge, tunings and
  **holdout** matrices, with the holdout clearly marked as the acceptance
  data
- the bounded-sweep result (1,920 compositions) and the ladder levels
  observed, if any
- the sample sweeps, re-measured with a throwaway script over seeds 0–299
  per sample command, writing outside the repo:
  - forbidden-interval pieces (v2: 22, 15 and 9 of 300; prototype 0)
  - pieces with a leap larger than a fifth (v2: 73, 58 and 79 of 300;
    prototype 20, 22 and 28)
- the gate output and the Task 7 timing
- any deviation from the plan, with its commit

- [ ] **Step 3: Update the docs listed above.**

- [ ] **Step 4: Run the final gate and the literal sample commands**

Run: `pytest -q tests/unit tests/integration && ruff check src tests && ruff format --check src tests && mypy --strict src`

Expected: at least 582 + the new tests passed, 3 skipped, and clean.

Then run each of the three sample commands from spec §2.1 with
`--out /tmp/compose-samples --format json`. Confirm that `composer_version`
is 3 and the command exits 0. The WAV render runs here, so expect a few
seconds each.

- [ ] **Step 5: Commit**

```bash
git add docs/usage.md docs/architecture-technical.md \
  docs/superpowers/specs/2026-11-07-compose-voicings-leaps-design.md src/music_decoder/compose/metrics.py
git commit -m "docs(compose): composer v3 voicing and leap rules; spec results"
```

Run `git status --short` and confirm that only `.synth_marker` files under
`tests/fixtures/synthetic/` are modified. Leave them unstaged.
\n
---

## Review responses (round 1)

Review: `docs/superpowers/reviews/2026-11-07-voicings-leaps-plan-review-r1.md`
(GPT-6 Astra, CHANGES_REQUESTED). Each finding was checked against the code
and the planning prototype before any change. All six were confirmed and
fixed; none is rebutted. The spec is now revision 2, and its §12 summarises
the same changes.

| # | Finding | Verification | Outcome |
|---|---|---|---|
| F1 (Important) | The relaxed fallback reintroduces a major seventh (C → D#m, seed 19) | Reproduced exactly on the revision-1 prototype: `[67, 64, 64, 75, 75]`, floors `[55, 69]`. Every candidate fails R1–R4 at the first slot of the two-note final bar. A wider bounded sweep (every root × quality after C and Am, seeds 0–29, 11,520 compositions) found 25 relaxed notes and 5 pieces with a forbidden interval. | **Fixed.** Spec §5.3 gains feasibility sets for the final and approach bars, a deterministic repair order (floored target → farther floored targets → unfloored targets) and a ladder (strict → R1+R2 only → v2). The ladder never drops R1 or R2 while some candidate satisfies them. `cadence_reach` is dropped and v2's `_candidates` is reused unchanged. On the prototype, the review case becomes `[67, 64, 64, 66, 63]`. The 11,520-piece sweep, all matrices and the sample sweeps never leave ladder level 0. New Task 6 tests: the review case, revision 1's four other sweep failures, a direct feasibility-set test, a direct ladder test (levels 1 and 2) and a 1,920-piece bounded sweep with real floors. Spec §8 now claims no universal guarantee: only level 2 can emit a forbidden interval, and only for a custom window too narrow to avoid one. |
| F2 (Important) | Span ≤ 3 accepts five-finger grips, including the pinned Eb7 `0-7-6-7-5-4` | Reproduced the review's count with a straight-barre finger model: **22 / 576** accepted shapes need 5 fingers, including library C#7 `x43421` and D#7 `x65643` (the open-C7 grip moved up, so no open string replaces a finger). | **Fixed.** Spec §3.1 adds a playability standard: span ≤ 3 and `fingers_needed ≤ 4`, where a barre never crosses an open or muted string. It applies to the library, the retune gate and the search. The search's old "notes above the lowest fret" heuristic is replaced. Library C#7 → `x46464` and D#7 → `x68686` (the A#7 `x13131` family), so seven entries change, not five. Eb D#7 becomes `0-7-9-7-9-7` (3 fingers) and Eb C7 is searched as `x-4-3-1-0-1`. Result: 0 / 576 faulty and 0 / 576 needing five fingers; 29 of 480 alternate-tuning voicings are now searched. Tasks 1, 4 and 5 add `fingers_needed` tests, the negative cases `0-7-6-7-5-4` and `x65643`, a barre blocked by open strings, and a 576-shape fingering audit. The review's example `0-2-0-1-0-0` is also playable. It is not chosen because the retuned library shape passes the gate, and the design only searches when the retune fails. Preferring open-position shapes everywhere would change hundreds of voicings and is out of scope. |
| F3 (Important) | `composer_version == 3` inside the melody xfail masks the musical assertions | Confirmed: the version assertion is first, so on v2 the xfail would always fail there. | **Fixed.** The version check is a separate test added in Task 6. The melody assertions carry labels (`L1`, `L3`, `L2`, `L4max`, `L5`), and Task 3 verifies with `--runxfail` that the first failure is `L3` / `L1` / `L3` for minor-10-B / major-08-Ab / major-09-Eb, and that the voicing failures are the reported chords. Tasks 4 and 5 repeat the melody check. With revision 2's voicing code and v2's melody code, all three flagged leaps were confirmed still present. |
| F4 (Important) | Task 2 commits a test for a fixture that does not exist yet; the snapshot's cleanliness check is incomplete | Confirmed: `-k "not v2_snapshot"` does not make the full gate green. `git status --short src` misses tracked `.synth_marker` changes that make `git describe --dirty` report dirty. | **Fixed.** Task 2 now makes two commits, each with a green full gate: (1) code and runnable tests; (2) snapshot plus snapshot test. Before writing the snapshot it restores the markers (`git checkout -- tests/fixtures/synthetic/`) and requires `git status --porcelain --untracked-files=no` to be empty. The snapshot test asserts `source` is not `-dirty`. The same rule is in Global Constraints. |
| F5 (Important) | "Held-out" matrices were used to choose the bonus and the tolerances | Confirmed: the step bonus (0.5 / 1.0 / 1.5), the L4 thresholds and the 0.02 tolerance were all set after measuring the validation and edge matrices. | **Fixed.** Spec §6 relabels validation, edge and tunings as v3 development data; their thresholds are regression locks. A new 50-melody **holdout** matrix (ten unused progressions across major and minor, all chord qualities, seeds 3000–3004, real floors) gets acceptance bounds frozen in spec §6.4 before it is ever evaluated: hard invariants, L0 ≤ min(v2, 0.002), L4 ≥ v2, and the no-regression tolerances. v2's holdout numbers enter the snapshot before any composer change. The tolerances are now justified in their own units (spec §6.3: under half a note per piece; about one pooled event). "Match the prototype within 0.005" is replaced by reporting and explaining deviations. A failed holdout check is reported, never tuned away (Task 7). |
| F6 (Minor) | `M4 ≤ 0.001` is described as a per-interval frequency but is a mean of per-piece rates | Confirmed in `_aggregate`. | **Fixed.** New pooled `L0` (large / intervals, `n = intervals`, plus `L0.any` for affected pieces) with target pooled ≤ 0.001 on validation and edge. M4 is kept and described correctly. Task 2 adds an unequal-length aggregation test (rows of 1,000 and 10 intervals). |
