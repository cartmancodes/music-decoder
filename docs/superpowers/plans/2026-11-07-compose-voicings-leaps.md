# Faithful Voicings and Singable Leaps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every chord shape the composer plays faithful in every preset
tuning, and make melodies singable: no tritones or sevenths; leaps larger
than a fifth only when justified and always recovered; smooth chord changes
and final approaches. Every v2 musicality metric must stay within its stated
tolerance.

**Architecture:**
- Repair five entries in the shared library `chords/voicings.py`.
- Rebuild the tuning path in `compose/voicings.py`: pitch-preserving retune,
  a narrower un-mute rule, a fretted-only span gate, a faithfulness gate and
  a deterministic shape search, with the v2 root fallback as a last resort.
- Add hard admissibility rules (`admissible`) to the melody sampler, plus
  final-approach lookahead and a step bonus at chord changes. When nothing is
  admissible, fall back to the v2 scoring.
- Measure everything with new metrics, a third "tunings" evaluation matrix
  and a committed v2 snapshot.

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
- `music_decoder.compose` as an *attribute* is the re-exported function, not
  the package. To patch module objects, use dotted strings in `mock.patch`
  or `importlib.import_module("music_decoder.compose.api")`.
- Constants from the spec, verbatim:
  - `SMOOTH_LIMIT = 7`
  - `FORBIDDEN_INTERVALS = frozenset({6, 10, 11})`
  - `CHORD_CHANGE_STEP_BONUS = 1.0`
  - `MAX_FRETTED_SPAN = 3`
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
   first bar). The approach lookahead must not crash or leap. Task 6 pins
   `test_short_pieces_obey_the_leap_rules`.
5. **The searched chords, played by the composer itself** (Drop C G, Drop D
   G7, DADGAD D#dim and the rest). They must compose with H1 and H2 intact
   in every style, not just pass the shape check. Task 5 pins
   `test_searched_voicings_compose_with_their_identity_heard`.

---

## File map

| File | Responsibility | Tasks |
|---|---|---|
| `src/music_decoder/compose/metrics.py` | New leap and shape metrics | 1 |
| `src/music_decoder/compose/voicings.py` | `shape_faults` (Task 1); retune, gate, search, fallback (Task 5) | 1, 5 |
| `src/music_decoder/evaluation/compose_musicality.py` | `Case.tuning`, tunings matrix, new keys, per-tuning groups | 2 |
| `scripts/compose_musicality_report.py` | Docstring only | 2 |
| `tests/fixtures/compose/musicality_v2.json` | New v2 snapshot (generated) | 2 |
| `tests/unit/test_compose_samples.py` | New CLI regression tests for the three samples | 3, 4, 5, 6 |
| `src/music_decoder/chords/voicings.py` | Five repaired shapes and the invariant docstring | 4 |
| `src/music_decoder/compose/melody.py` | `admissible`, context fields, approach and recovery wiring, `cadence_reach` | 6 |
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
from music_decoder.compose.voicings import shape_faults
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest -q tests/unit/test_compose_metrics.py tests/unit/test_compose_voicings.py`

Expected: FAIL with `AttributeError` / `ImportError` for the new names.

- [ ] **Step 3: Implement the five metrics functions and `shape_faults`**

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

### Task 2: Evaluation: tunings matrix, new metric keys and the v2 snapshot

The composer is still unchanged in this task. The snapshot must be written
from it.

**Files:**
- Modify: `src/music_decoder/evaluation/compose_musicality.py`
- Modify: `scripts/compose_musicality_report.py` (docstring: it also writes
  `musicality_v2.json`; the three matrices)
- Create (generated): `tests/fixtures/compose/musicality_v2.json`
- Test: `tests/unit/test_compose_evaluation.py`

**Interfaces:**
- Consumes: Task 1 metrics.
- Produces:
  - `TUNING_PROGRESSIONS: tuple[tuple[str, str, tuple[str, ...]], ...]`,
    holding:
    - `("B", "minor", ("Bm","G","D","A","Em7","C#dim","F#7","Bm"))`
    - `("G#", "major", ("G#","C7","Fm","C#","D#7","G#"))`
    - `("D#", "major", ("D#","Cm","Fm7","A#sus4","A#7","D#"))`

    These are the samples' progressions in sharp spelling.
  - `TUNING_SEEDS = VALIDATION_SEEDS`.
  - `tunings_inputs() -> list[Inputs]`: 15 entries, one bar per chord.
  - `build_cases(inputs, tuning: Tuning = STANDARD_EADGBE) -> list[Case]`.
    `Case` gains a `tuning: Tuning` field. `case_metrics` uses
    `case.tuning` everywhere it now uses `STANDARD_EADGBE` (accompany, S1,
    V1, H1, H2).
  - `case_metrics` adds the keys:
    - `"L1"` (int)
    - `"L2.leaps"` and `"L2.recovered"` (ints)
    - `"L3"` (int)
    - `"L4"` (share of `chord_change_moves` ≤ 4, or `None`)
    - `"L4max"` (max of the moves, or `None`)
    - `"L5"` (abs of the last interval, or `None`)
    - `"S2"` (share of voicings where `not bass_is_root`)
  - `_aggregate` pools `L2` exactly like `M5` (skip the `"L2."` keys; emit
    `"L2": {"pooled", "n"}`).
  - `summarize` adds a group `f"tuning.{name}"` for each distinct
    `case.tuning.name` when the cases span more than one tuning.
  - `evaluate()` returns `{"validation", "edge", "tunings"}`. The tunings
    matrix is `[c for t in PRESETS.values() for c in build_cases(tunings_inputs(), t)]`
    (`PRESETS` from `tabs.tuning`), 90 cases.

- [ ] **Step 1: Write the failing tests** (append to `tests/unit/test_compose_evaluation.py`)

```python
V2 = Path(__file__).parents[1] / "fixtures" / "compose" / "musicality_v2.json"


def test_tunings_matrix_covers_the_flagged_progressions_in_every_tuning():
    inputs = ev.tunings_inputs()
    assert len(inputs) == 15 and {seed for *_, seed in inputs} == set(ev.VALIDATION_SEEDS)
    assert {bars for _, _, bars, _ in inputs} == {1}
    labels = {tuple(c.to_label() for c in prog) for _, prog, _, _ in inputs}
    assert ("G#", "C7", "Fm", "C#", "D#7", "G#") in labels


def test_case_metrics_use_the_case_tuning_and_report_leaps():
    from music_decoder.tabs.tuning import EB_HALF_STEP_DOWN

    (case,) = ev.build_cases(ev.tunings_inputs()[5:6], EB_HALF_STEP_DOWN)
    assert case.tuning == EB_HALF_STEP_DOWN
    row = ev.case_metrics(case)
    assert {"L1", "L2.leaps", "L2.recovered", "L3", "L4", "L4max", "L5", "S2"} <= set(row)


def test_summarize_pools_l2_and_groups_by_tuning():
    from music_decoder.tabs.tuning import DROP_D

    cases = ev.build_cases(ev.tunings_inputs()[:2]) + ev.build_cases(ev.tunings_inputs()[:2], DROP_D)
    summary = ev.summarize(cases)
    assert set(summary["overall"]["L2"]) == {"pooled", "n"}
    assert {"tuning.EADGBE", "tuning.Drop_D"} <= set(summary)
    assert "tuning.EADGBE" not in ev.summarize(cases[:2])


def test_committed_v2_snapshot_has_three_matrices_and_the_new_metrics():
    snap = json.loads(V2.read_text())
    assert {"source", "validation", "edge", "tunings"} <= set(snap)
    v = snap["validation"]["overall"]
    assert v["L1"]["max"] == 1 and v["L4max"]["max"] == 12  # v2's measured defects
    assert snap["tunings"]["overall"]["S1.missing"]["mean"] > 0
```

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest -q tests/unit/test_compose_evaluation.py`

Expected: FAIL with `AttributeError: tunings_inputs` (and the missing
fixture).

- [ ] **Step 3: Implement the interfaces above** in `compose_musicality.py`
  and update the script docstring.

- [ ] **Step 4: Run every test except the snapshot test, plus the musicality suite**

Run: `pytest -q tests/unit/test_compose_evaluation.py tests/unit/test_compose_musicality.py -k "not v2_snapshot"`

Expected: PASS. The existing musicality tests only read the validation and
edge matrices, so they are unaffected.

- [ ] **Step 5: Commit the code (the snapshot comes next, from a clean tree)**

```bash
git add src/music_decoder/evaluation/compose_musicality.py scripts/compose_musicality_report.py tests/unit/test_compose_evaluation.py
git commit -m "feat(eval): tunings matrix, leap/shape metric keys and per-tuning groups"
```

- [ ] **Step 6: Write the v2 snapshot from the unchanged composer**

Run: `git status --short src` (expect no output), then
`python scripts/compose_musicality_report.py --write tests/fixtures/compose/musicality_v2.json`.

Expected values, measured in the spec:
- `"source"` is the Step 5 commit's short hash with no `-dirty`.
- validation `L1.max == 1`, `L4max.max == 12`, `M4.mean ≈ 0.004`,
  `L4.mean ≈ 0.852`
- edge `L4.mean ≈ 0.806`
- tunings `S1.missing.mean ≈ 0.088`, `S2.mean ≈ 0.107`

If `source` ends in `-dirty`, stop: something in the tree is modified.

- [ ] **Step 7: Run the snapshot test**

Run: `pytest -q tests/unit/test_compose_evaluation.py`

Expected: PASS.

- [ ] **Step 8: Commit the snapshot**

```bash
git add tests/fixtures/compose/musicality_v2.json
git commit -m "test(eval): commit composer v2 musicality snapshot (reference for v3)"
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
    out = _run(SAMPLES[name], tmp_path)
    mel, prog, tempo = ...  # as above; bars_per_chord = 1
    assert out["metadata"]["composer_version"] == 3
    assert metrics.forbidden_interval_count(mel) == 0
    assert metrics.unjustified_large_leap_count(mel, prog, 1, tempo) == 0
    leaps, recovered = metrics.large_leap_recovery_counts(mel)
    assert recovered == leaps
    assert max(metrics.chord_change_moves(mel, prog, 1, tempo)) <= 7
    final = abs(mel[-1].pitch - mel[-2].pitch)
    assert final <= 7 and final != 6


@pytest.mark.parametrize("name", list(SAMPLES))
def test_sample_commands_are_deterministic(name, tmp_path):
    a = _run(SAMPLES[name], tmp_path / "a")
    b = _run(SAMPLES[name], tmp_path / "b")
    assert a["melody_notes"] == b["melody_notes"] and a["chord_voicings"] == b["chord_voicings"]
```

- [ ] **Step 2: Run them**

Run: `pytest -q -rxX tests/unit/test_compose_samples.py`

Expected: `3 passed, 6 xfailed`, with no XPASS. Every xfail must fail on an
`AssertionError`, not an error in the test's own code. Check this by running
the file once with `--runxfail`: you should see 6 failures, all
`AssertionError`.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_compose_samples.py
git commit -m "test(compose): the three flagged sample commands as strict-xfail CLI regressions"
```

---

### Task 4: Repair the five library shapes

**Files:**
- Modify: `src/music_decoder/chords/voicings.py`. Change the entries for
  C#dim, Ddim, D#dim, Fmaj7 and G#aug, and add an invariant paragraph to
  the module docstring stating the §3.1 rules.
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

- [ ] **Step 1: Write the failing tests** (in `test_chord_detection_voicings.py`)

```python
from music_decoder.compose.voicings import shape_faults
from music_decoder.tabs.tuning import STANDARD_EADGBE
from music_decoder.types import ChordSymbol


def test_every_library_shape_is_faithful():
    faulty = {
        label: shape_faults(v, STANDARD_EADGBE, ChordSymbol.parse(label))
        for label, v in VOICINGS.items()
        if label != "N"
    }
    assert {k: f for k, f in faulty.items() if f} == {}


@pytest.mark.parametrize(
    ("label", "frets"),
    [
        ("C#dim", (-1, 4, 5, 6, 5, -1)),  # C#3 G3 C#4 E4
        ("Ddim", (-1, -1, 0, 1, 3, 1)),  # D3 G#3 D4 F4
        ("D#dim", (-1, -1, 1, 2, 4, 2)),  # D#3 A3 D#4 F#4
        ("Fmaj7", (-1, -1, 3, 2, 1, 0)),  # F3 A3 C4 E4
        ("G#aug", (-1, -1, 6, 5, 5, 4)),  # G#3 C4 E4 G#4
    ],
)
def test_repaired_shapes(label, frets):
    assert VOICINGS[label] == frets
```

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest -q tests/unit/test_chord_detection_voicings.py`

Expected: FAIL, listing exactly C#dim, Ddim, D#dim, Fmaj7 and G#aug.

- [ ] **Step 3: Change the five entries and the docstring.**

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
   - Replace the V1 line with `cand["V1"]["mean"] <= v2[matrix]["overall"]["V1"]["mean"] + 0.02`.
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

- [ ] **Step 6: Run the full gate, then commit**

Run: `pytest -q tests/unit tests/integration && ruff check src tests && ruff format --check src tests && mypy --strict src`

```bash
git add src/music_decoder/chords/voicings.py tests/unit/test_chord_detection_voicings.py \
  tests/unit/test_compose_metrics.py tests/unit/test_compose_musicality.py tests/unit/test_compose_samples.py
git commit -m "fix(chords): repair C#dim, Ddim, D#dim, Fmaj7 and G#aug library shapes"
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
- **`search_voicing`:**
  - window, bass rule, contiguity, finger rule and selection key as in
    §5.2 item 5
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
                if faults or span > 3:
                    bad.append((name, label, frets, faults))
    assert bad == []  # v2: 98 of 576 faulty (spec §3.3)


def test_standard_tuning_plays_the_library_shapes_unchanged():
    for label, frets in VOICINGS.items():
        if label != "N":
            assert _frets(label, STANDARD_EADGBE) == frets


@pytest.mark.parametrize(
    ("tuning", "label", "frets"),
    [
        (DROP_D, "D", (0, -1, 0, 2, 3, 2)),  # low D added (root); open A2 stays muted
        (DROP_C, "C", (0, 5, 4, 2, 3, 2)),
        (EB_HALF_STEP_DOWN, "D#7", (0, 7, 6, 7, 5, 4)),  # v2: a single D#2
        (EB_HALF_STEP_DOWN, "Cm", (-1, 4, 6, 6, 5, 4)),  # v2: C3 C3
        (D_STANDARD, "A#7", (-1, 3, 5, 3, 5, 3)),  # v2: open D2 under the barre
        (DROP_C, "G", (-1, 0, 2, 2, 2, 0)),  # searched
        (DROP_D, "G7", (5, 2, 3, 0, 3, -1)),  # searched: span 4 rejected
        (DADGAD, "D#dim", (1, 0, 1, 2, 0, 4)),  # searched
    ],
)
def test_retune_and_search_choices(tuning, label, frets):
    assert _frets(label, tuning) == frets


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

- [ ] **Step 5: Run the full gate, then commit**

Run: `pytest -q tests/unit tests/integration && ruff check src tests && ruff format --check src tests && mypy --strict src`

```bash
git add src/music_decoder/compose/voicings.py tests/unit/test_compose_voicings.py \
  tests/unit/test_compose_api.py tests/unit/test_compose_harmony.py tests/unit/test_compose_samples.py
git commit -m "fix(compose): faithful chord voicings in every tuning (retune, span gate, shape search)"
```

---

### Task 6: Singable melodic leaps

**Files:**
- Modify: `src/music_decoder/compose/melody.py`
- Modify: `src/music_decoder/compose/api.py` (`COMPOSER_VERSION = 3`)
- Modify: `tests/unit/test_compose_melody.py`
- Modify: `tests/unit/test_compose_api.py`: the version assertion `== 3`.
- Modify: `tests/unit/test_compose_samples.py`: remove the melody xfails.

**Interfaces:**
- Consumes: Task 1 metrics (in tests).
- Produces, in `compose.melody`:
  - the constants `SMOOTH_LIMIT`, `FORBIDDEN_INTERVALS` and
    `CHORD_CHANGE_STEP_BONUS`
  - `NoteContext` fields `chord_change`, `prev_chord_tone`, `next_fixed`,
    `approach` and `recovery_pool`, with the defaults of spec §5.3
  - `admissible(candidate: int, ctx: NoteContext) -> bool` implementing
    R1–R4
  - `_candidates(..., target_pc, cadence_reach: int = MAX_INTERVAL)`

**Wiring in `generate_melody`.** This must match the prototype exactly so
that §6's numbers reproduce.

- **`final_approach`:**
  - `cadence = cadence_pc(scale, progression[-1])`
  - `in_window = tuple(p for p in range(pitch_low, pitch_high + 1) if p % 12 == cadence)`
  - `floor = chord_floors[-1] if chord_floors is not None else None`
  - `above = tuple(p for p in in_window if floor is None or p >= floor)`
  - `final_approach = above or in_window`
- **`one_note_final = len(bars[-1].slots) == 1`**
- **Per slot `k` of bar `bar_idx`** (`n_slots = len(bar.slots)`):
  - `chord_change = prev_chord is not None and chord != prev_chord`
  - `prev_chord_tone = prev is not None and prev % 12 in tones`
  - `next_fixed = target if target is not None and k == n_slots - 2 else None`
  - `approach_here(bar_idx, k) = one_note_final and bar_idx == len(bars) - 2 and k == n_slots - 1`;
    set `approach = final_approach` when it is true
  - `recovery_pool = frozenset(candidates)` if all of these hold, else `None`:
    - `k + 1 < n_slots`
    - `not (target is not None and k + 1 >= n_slots - 2)`
    - `not approach_here(bar_idx, k + 1)`
- **Scoring:** `scores = [score_pitch(c, ctx, centre) if admissible(c, ctx) else -math.inf for c in candidates]`.
  If none is finite, use `[score_pitch(c, ctx, centre) for c in candidates]`.
- **`_candidates`** is called with
  `cadence_reach = SMOOTH_LIMIT if bar.final and len(bar.slots) == 1 else MAX_INTERVAL`.
  Inside, the final-bar check becomes "some `p` in `above` with
  `p % 12 == target_pc` and (`prev is None` or (`abs(p - prev) <= cadence_reach`
  and `abs(p - prev) != 6`))".
- **`score_pitch`:** inside the `prev_pitch is not None` block, add
  `CHORD_CHANGE_STEP_BONUS` when `ctx.chord_change and 1 <= size <= 2`.
- **R3 recovery check:** the pool must contain some `r` with
  `1 <= (candidate - r) * sign(interval) <= 2`.

- [ ] **Step 1: Write the failing unit tests** (in `test_compose_melody.py`,
  using the existing `_ctx`)

```python
from music_decoder.compose.melody import CHORD_CHANGE_STEP_BONUS, _candidates, admissible


def test_admissible_forbids_tritones_sevenths_and_wide_leaps():
    for c in (78, 82, 83, 85, 66, 62, 61):  # +6 +10 +11 +13, -6 -10 -11 from 72
        assert not admissible(c, _ctx())
    assert admissible(79, _ctx()) and admissible(65, _ctx())  # fifths up and down


def test_admissible_large_leap_needs_arpeggiation_and_a_recovery():
    ok = dict(prev_chord_tone=True, recovery_pool=frozenset({74, 76, 77}))
    assert admissible(76 + 4, _ctx(prev_pitch=72, **ok)) is False  # 80 = G#, not a chord tone
    assert admissible(79, _ctx(prev_pitch=67, prev_chord_tone=True, recovery_pool=frozenset({77}))) is True
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


def test_admissible_keeps_the_final_approach_smooth():
    assert admissible(67, _ctx(next_fixed=72)) and not admissible(64, _ctx(next_fixed=72))
    assert not admissible(66, _ctx(next_fixed=72))  # tritone below the target
    assert admissible(74, _ctx(approach=(67, 79))) and not admissible(73, _ctx(approach=(67, 79)))


def test_chord_change_step_bonus():
    step = score_pitch(74, _ctx(chord_change=True), 72) - score_pitch(74, _ctx(), 72)
    assert step == pytest.approx(CHORD_CHANGE_STEP_BONUS)
    assert score_pitch(72, _ctx(chord_change=True), 72) == score_pitch(72, _ctx(), 72)


def test_one_note_final_bar_floor_yields_unless_a_fifth_reaches_it():
    c, g7 = Scale("C", "major"), ChordSymbol.parse("G7")
    smooth = _candidates(c, g7, 62, 79, 68, 71, 7, cadence_reach=7)  # G5 is 8 above B4
    octave = _candidates(c, g7, 62, 79, 68, 71, 7)
    assert min(smooth) < 68 and min(octave) >= 68
```

Check these against the existing `_ctx` defaults: `chord_pcs={0, 4, 7}`,
`prev_pitch=72`, `prev_interval=0`. The case `79` from `67` is +12 between
G4 and G5, both chord tones, and recovery 77 lies 2 below 79. If an
assertion conflicts with the spec's R-rules, the spec wins: fix the
assertion and note it in the commit.

Rewrite `test_one_note_final_bar_stays_within_an_octave_when_a_floor_bites`
as `test_one_note_final_bar_is_approached_within_a_fifth`, with the same
200-seed loop. Assert `abs(notes[1].pitch - notes[0].pitch) <= 7` and
`!= 6`, and `notes[1].pitch % 12 == 7`. Delete the `low_starts`
precondition. Its branch is now covered by
`test_one_note_final_bar_floor_yields_unless_a_fifth_reaches_it`.

Generator-level tests:

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

Here `"A"` gives key A for `("A", "Dsus4")`. The floors list is cut to the
progression's length (`[None]` for `("D",)`). All of these sweeps, and the
admissibility cases above, were run against the planning prototype and pass.
On these adversarial inputs the relaxed fallback was taken 3 times, and the
invariants still held.

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest -q tests/unit/test_compose_melody.py`

Expected: FAIL (`ImportError: admissible`, and the generator sweeps fail on
v2 melodies).

- [ ] **Step 3: Implement §5.3** with the wiring above. Then set
  `COMPOSER_VERSION = 3` and update `test_compose_api.py`'s `== 2` to
  `== 3`.

- [ ] **Step 4: Turn the three melody params of `test_compose_samples.py`
  into plain params, then run the compose suite**

Run: `pytest -q -rxX tests/unit/test_compose_melody.py tests/unit/test_compose_api.py tests/unit/test_compose_samples.py tests/unit/test_compose_musicality.py tests/unit/test_compose_arrangement.py tests/integration/test_compose_e2e.py`

Expected:
- PASS, with no xfail or XPASS left in the samples file
- every v2 musicality test still passes (spec §6.3: the prototype passes
  them all)

- [ ] **Step 5: Check the numbers against the prototype**

Run: `python scripts/compose_musicality_report.py | grep -E "^ *(M1|M2|M4|M4max|L1|L3|L4|L4max|L5) "`

Expected (validation, then edge):
- `M1 mean=0.977` / `0.9843`
- `M4 mean=0.0002` / `0`
- `M4max max=8` / `7`
- `L1 max=0`, `L3 max=0`, `L4max max=7`, `L5 max=7`
- `L4 mean≈0.885` / `≈0.848`

A deviation of more than 0.005 in any mean means the wiring differs from
spec §5.3. Find the difference before committing.

- [ ] **Step 6: Run the full gate, then commit**

Run: `pytest -q tests/unit tests/integration && ruff check src tests && ruff format --check src tests && mypy --strict src`

```bash
git add src/music_decoder/compose/melody.py src/music_decoder/compose/api.py \
  tests/unit/test_compose_melody.py tests/unit/test_compose_api.py tests/unit/test_compose_samples.py
git commit -m "fix(compose): singable melodies - no tritones/sevenths, justified recovered leaps, smooth approaches (composer v3)"
```

---

### Task 7: Musicality targets: leaps, shapes, tunings and no regression against v2

**Files:**
- Modify: `tests/unit/test_compose_musicality.py`. Extend the module
  docstring with the spec path, add a `v2` fixture (if Task 4 did not
  already add one), and add the tests below.

**Interfaces:**
- Consumes:
  - `evaluate()` with three matrices (Task 2)
  - `musicality_v2.json` (Task 2)
  - metric keys `L1`, `L2`, `L3`, `L4`, `L4max`, `L5`, `S1.*` and `S2`
- Produces: none (tests only).

- [ ] **Step 1: Write the tests**

```python
MATRICES = ("validation", "edge", "tunings")
TOL_MEAN, TOL_POOLED = 0.02, 0.05  # spec §6.3


def test_leap_targets(report, v2):
    for matrix, l4_floor in (("validation", 0.87), ("edge", 0.83)):
        o, ref = report[matrix]["overall"], v2[matrix]["overall"]
        assert o["M4"]["mean"] <= 0.001 and o["M4"]["mean"] < ref["M4"]["mean"]
        assert o["L4"]["mean"] >= l4_floor
    for matrix in MATRICES:
        assert report[matrix]["overall"]["L4"]["mean"] > v2[matrix]["overall"]["L4"]["mean"]


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


@pytest.mark.parametrize(("matrix", "group"), [*(("validation", g) for g in GROUPS), ("edge", "overall")])
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
2. Temporarily set `MAX_FRETTED_SPAN = 5` in `compose/voicings.py`.
   `test_leap_and_shape_invariants[tunings]` must fail (S2 or S1), or
   `test_every_chord_in_every_preset_tuning_is_faithful_and_playable` must
   fail.

Revert both edits with `git checkout -- src/music_decoder/compose/melody.py src/music_decoder/compose/voicings.py`
and check that `git status --short src` is empty.

- [ ] **Step 3: Run the full gate and time the musicality module**

Run: `pytest -q tests/unit tests/integration && ruff check src tests && ruff format --check src tests && mypy --strict src`

Then run `pytest -q --durations=5 tests/unit/test_compose_musicality.py`.
Expected: the module takes about 5 s or less, of which `evaluate()` is
about 3.5 s. Record the figure for Task 8.

- [ ] **Step 4: Commit**

```bash
git add tests/unit/test_compose_musicality.py
git commit -m "test(compose): v3 musicality targets - leaps, shapes, tunings matrix, no regression vs v2"
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
    covering the un-mute rule, the fretted span ≤ 3, the faithfulness gate,
    the search and the last-resort fallback.
  - §5.2 Melody (lines ~756–790): add the admissibility rules R1–R4, the
    relaxed fallback, the step bonus and the one-note-final-bar reach rule
    to steps 2–5.
  - The S1 note in §5.3 (line ~838): the shapes are now repaired, and S1
    and S2 are asserted 0.
  - "Add a new chord quality" (line ~1312): drop the `_CHORD_PCS` step.
- Modify: `docs/superpowers/specs/2026-11-07-compose-voicings-leaps-design.md`:
  append `## 12. Results`.
- Modify: `src/music_decoder/compose/metrics.py`: check that its docstring
  names both specs (Task 1).

**Interfaces:** none.

- [ ] **Step 1: Refresh the measurements**

Run: `python scripts/compose_musicality_report.py > /tmp/v3_report.txt`.
Do **not** pass `--write`: both committed fixtures stay as they are.

Compare against `musicality_v2.json` and the spec's §6 prototype column.
Every number must match the prototype within 0.005, unless a later commit
explains the difference.

- [ ] **Step 2: Write `## 12. Results` in the spec**

Include:
- a before (v2 snapshot) / after (v3 report) table for every metric in §6.1
  and §6.2, and for the §6.3 rows, on the validation, edge and tunings
  matrices
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
