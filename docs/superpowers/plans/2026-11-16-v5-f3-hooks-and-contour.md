# v5 Feature 3: Hooks, Contour, Peaks and Form — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give fingerstyle melodies a recognisable hook, an arch-shaped phrase with one clear high point, and an A A′ B A′ form. Measure each with new metrics.

**Architecture:**
- Three new pure metrics in `compose/metrics.py` define the targets.
- `generate_melody` gains `phrase_shaping: bool = False`, which only fingerstyle passes. When it is on, the generator adds three soft scoring terms (hook, contour arch, peak guard) and a contrasting B-phrase register and rhythm.
- The hard rules (`admissible`), the reach sets and the cadence logic still decide.

**Tech Stack:** Python 3.11, numpy, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-11-16-composer-v5-design.md` (§5.3, §6 items 1, 2 and 5).

**Carried-in ruling (from feature 2, ruling F2a/F2d):** the target `F.M6` mean ≥ v4 + 2 semitones on every matrix, reserve included, belongs to this feature. The v4 values are in `tests/fixtures/compose/musicality_v4_fingerstyle.json`.

## Global Constraints

- Activate the venv first: `source .venv/bin/activate`.
- Strum and arpeggio must not change. `tests/unit/test_compose_frozen.py` must pass untouched. With `phrase_shaping=False`, `generate_melody` output and RNG draws are unchanged.
- `compose()`'s signature and the package exports stay frozen. `COMPOSER_VERSION` stays 5.
- The fingerstyle v4 guarantees hold on every matrix and every preset tuning: P1, P2 and P4 are 1.0, P3t is 1.0, P5 is 0, A4 and H2 hold, and the v3 hard melody rules (`F.L1`, `F.L3`, `F.L2`, `F.L4max`, `F.L5`, `F.M4max`) hold. The seeded random-progression property test passes.
- Feature-2 gates stay: P3 ≥ 0.99, with reserve at ≥ 0.98, and `F.shape_moves` ≤ 0.5.
- Feature-3 targets, all means:
  - on validation: `F.hook` ≥ 0.75, `F.peak` ≥ 0.9, `F.arch` ≥ 0.5;
  - on every matrix: `F.M6` ≥ v4 + 2.
- The weights below are starting values. You may tune these feature-3 weights to reach the targets, but never the targets or any other feature's constants. Record the final values in the report.
- Gates before each commit: `pytest -q tests/unit`, `ruff check src tests`, `ruff format --check src tests`, `mypy --strict src`. Run `pytest -q tests/integration` at the last task.
- If a v4 guarantee fails, stop and report. Do not loosen the assertion.
- End every commit message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Branch: `feat/composer-v5`.

## Review Focus

1. **Short pieces** (1–3 bars, a single phrase): the hook metric is `None` because there is no later phrase. The arch and peak still apply, and nothing crashes. Test: Task 1's `test_metrics_on_a_single_phrase`.
2. **One-note bars** (whole-note rhythms, and the final bar): the hook needs at least 2 notes in bar 0. With fewer, there is no hook, so the term is inactive. Test: Task 2's `test_phrase_shaping_with_a_one_note_first_bar`.
3. **Narrow reach sets** (barre bars, D-family shapes): the arch target may lie outside the allowed set. The term must only pull, never empty the candidate list. Test: the property test plus Task 2's matrix gates.
4. **Odd lengths** (5, 7 and 9 bars): phrase layouts differ, so the form labels must follow `phrase_layout`. Test: Task 2's `test_form_labels_follow_the_phrase_layout`.
5. **The final cadence:** the peak guard and contour must not stop the final approach being within a fifth (`F.L5` ≤ 7). This is covered by the hard gates.

---

### Task 1: Hook, peak and arch metrics plus harness keys

**Files:**
- Modify: `src/music_decoder/compose/metrics.py`
- Modify: `src/music_decoder/evaluation/compose_musicality.py` (`F.hook`, `F.peak` and `F.arch` from the fingerstyle melody)
- Test: `tests/unit/test_compose_metrics.py`, `tests/unit/test_compose_evaluation.py`

**Interfaces:**
- Consumes: `rhythm.phrase_layout(n_bars)` (`phrase_lengths`); `metrics._bar_notes`; `metrics.intervals`.
- Produces:
  - `PEAK_WINDOW = (0.55, 0.75)`, given as fractions of a phrase's duration.
  - `ARCH_PEAK = 0.65`.
  - `hook_cell(melody: Sequence[Note], tempo_bpm: float) -> tuple[int, ...]`: the intervals between the first `min(4, len(bar 0 notes))` notes of bar 0. It has 1–3 intervals, or `()` when bar 0 has fewer than 2 notes.
  - `hook_restatements(melody: Sequence[Note], tempo_bpm: float, n_bars: int) -> float | None`: over every phrase after the first, the share whose first bar starts with intervals each within ±1 semitone of the hook cell's. It is `None` when the hook is empty or there is only one phrase.
  - `peak_uniqueness(melody: Sequence[Note], tempo_bpm: float, n_bars: int) -> float | None`: the share of phrases whose highest pitch occurs exactly once and whose onset falls within `PEAK_WINDOW` of the phrase's time span. It is `None` when there is no phrase with 3 or more notes.
  - `contour_arch(melody: Sequence[Note], tempo_bpm: float, n_bars: int) -> float | None`: the mean, over phrases with 3 or more notes and non-constant pitch, of the Pearson correlation between note pitches and `arch(x)`. Here `x` is the onset as a fraction of the phrase span, `arch(x) = x/ARCH_PEAK` for `x ≤ ARCH_PEAK`, otherwise `(1−x)/(1−ARCH_PEAK)`. It is `None` when no phrase qualifies.

- [ ] **Step 1: Write the failing tests.** `note` is the local builder, and the tempo is 120, so a bar is 2 s and an eighth is 0.25 s.

```python
def _line(pitches, eighths=2, start=0.0):
    step = 0.25 * eighths
    return [note(start + k * step, start + k * step + 0.95 * step, p) for k, p in enumerate(pitches)]

def test_hook_cell_and_restatements():
    # two 4-bar phrases (8 bars), quarter notes, phrase 2 restates the hook a tone higher
    a = _line([60, 62, 64, 62] + [60] * 12)
    b = _line([62, 64, 66, 64] + [62] * 12, start=8.0)
    mel = a + b
    assert metrics.hook_cell(mel, 120.0) == (2, 2, -2)
    assert metrics.hook_restatements(mel, 120.0, 8) == 1.0
    b_bad = _line([62, 55, 66, 64] + [62] * 12, start=8.0)
    assert metrics.hook_restatements(a + b_bad, 120.0, 8) == 0.0

def test_peak_uniqueness_window():
    # 4-bar phrase, 16 quarter notes; peak 72 at note 10 (x = 10/16 = 0.625)
    pitches = [60, 62, 64, 65, 64, 65, 67, 65, 67, 69, 72, 69, 67, 65, 64, 62]
    assert metrics.peak_uniqueness(_line(pitches), 120.0, 4) == 1.0
    tied = list(pitches); tied[3] = 72
    assert metrics.peak_uniqueness(_line(tied), 120.0, 4) == 0.0

def test_contour_arch_prefers_rise_then_fall():
    rise_fall = [60, 62, 64, 65, 67, 69, 71, 72, 71, 72, 74, 72, 69, 67, 64, 62]
    fall_rise = rise_fall[::-1]
    assert metrics.contour_arch(_line(rise_fall), 120.0, 4) > 0.5
    assert metrics.contour_arch(_line(fall_rise), 120.0, 4) < 0.0

def test_metrics_on_a_single_phrase():
    mel = _line([60, 64, 67, 64])        # 2 bars, one phrase
    assert metrics.hook_restatements(mel, 120.0, 2) is None
    assert metrics.peak_uniqueness(mel, 120.0, 2) is not None
    assert metrics.contour_arch(_line([60]), 120.0, 1) is None

# test_compose_evaluation.py
def test_case_metrics_report_hook_peak_arch():
    (case,) = ev.build_cases(ev.validation_inputs()[:1])
    assert {"F.hook", "F.peak", "F.arch"} <= set(ev.case_metrics(case))
```

- [ ] **Step 2: Run** `pytest -q tests/unit/test_compose_metrics.py tests/unit/test_compose_evaluation.py`. Expected: FAIL.
- [ ] **Step 3: Implement** the constants, the four functions and the three harness keys (from `case.fingerstyle.melody`).
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Run the gates and commit**: `feat(compose): hook, peak and arch melody metrics`.

---

### Task 2: Phrase shaping in the generator, wired into fingerstyle

**Files:**
- Modify: `src/music_decoder/compose/melody.py` (`generate_melody(phrase_shaping=...)`, `NoteContext`, `score_pitch`)
- Modify: `src/music_decoder/compose/rhythm.py` (`plan_bars(..., contrast_b=False)`)
- Modify: `src/music_decoder/compose/api.py` (`_arrange_fingerstyle` passes `phrase_shaping=True`)
- Test: `tests/unit/test_compose_melody.py`, `tests/unit/test_compose_musicality.py`

**Interfaces:**
- Consumes: Task 1's `PEAK_WINDOW`, `ARCH_PEAK`, `hook_cell` and the phrase spans (from `phrase_layout`).
- Produces:
  - `generate_melody(..., phrase_shaping: bool = False)`. When it is false, behaviour and RNG draws are identical to today.
  - `rhythm.form_labels(n_bars: int) -> tuple[str, ...]`, one label per phrase:
    - `("A", "A'", "B", "A'")` for 4 phrases;
    - `("A", "A'")` for 2;
    - for any other count, alternate A and A′, with the third phrase B when 3 or more phrases exist.
  - `plan_bars(n_bars, rng, contrast_b: bool = False)`. With `contrast_b`, the B phrase's fresh body rhythms are redrawn, up to 5 times, until they differ from phrase A's. The default stays unchanged.
- Scoring terms, active only when `phrase_shaping` is on. The starting weights are shown; tune them per the Global Constraints.
  - **Hook** (`HOOK_EXACT = 3.0`, `HOOK_NEAR = 1.5`): for the first `len(hook)` intervals of each phrase-start bar after bar 0, add `HOOK_EXACT` if the interval equals the hook interval and `HOOK_NEAR` if it is within 1 semitone. In A′ phrases one interval (index chosen by `rng`) is varied: that slot's target is the hook interval ± 2, so the restatement changes exactly one element. The hook is taken from the melody as generated in bar 0.
  - **Contour** (`CONTOUR_WEIGHT = 0.8`, `CONTOUR_WIDTH = 4.0`): it replaces register gravity inside each phrase.
    - Target `t(x) = base + ARCH_SPAN * arch(x)`.
    - `ARCH_SPAN = 7`.
    - `base = tessitura_center − 3`, plus 2 in a B phrase.
    - The term is `−CONTOUR_WEIGHT * ((cand − t)/CONTOUR_WIDTH)²`.
  - **Peak guard:**
    - outside `PEAK_WINDOW`, add `PEAK_TIE = −1.5` for a candidate equal to or above the phrase's running maximum;
    - inside the window, add `PEAK_BONUS = +1.0` for one that exceeds it;
    - after the window, a candidate at or above the phrase maximum gets `PEAK_AFTER = −2.0`.
- `_arrange_fingerstyle` passes `phrase_shaping=True` to `generate_melody`, and `plan_bars` is called with `contrast_b=True` on that path.

- [ ] **Step 1: Write the failing tests**

```python
# test_compose_melody.py
def test_phrase_shaping_off_is_unchanged():
    kw = dict(scale=Scale("C", "major"), progression=PROG8, tempo_bpm=120.0, seed=5)
    assert generate_melody(**kw) == generate_melody(**kw, phrase_shaping=False)

def test_phrase_shaping_meets_the_melody_targets():
    hook = peak = arch = []
    rows = []
    for seed in range(20):
        mel = generate_melody(scale=Scale("C", "major"), progression=PROG8, tempo_bpm=120.0,
                              seed=seed, phrase_shaping=True, final_whole_note=True)
        rows.append((metrics.hook_restatements(mel, 120.0, 8), metrics.peak_uniqueness(mel, 120.0, 8),
                     metrics.contour_arch(mel, 120.0, 8)))
        assert metrics.forbidden_interval_count(mel) == 0
        assert max(metrics.chord_change_moves(mel, PROG8, 1, 120.0)) <= 7
    assert statistics.fmean(r[0] for r in rows) >= 0.75
    assert statistics.fmean(r[1] for r in rows) >= 0.9
    assert statistics.fmean(r[2] for r in rows) >= 0.5

def test_phrase_shaping_with_a_one_note_first_bar():
    mel = generate_melody(scale=Scale("C", "major"), progression=[ChordSymbol.parse("C")],
                          seed=1, phrase_shaping=True, final_whole_note=True)
    assert len(mel) == 1

def test_form_labels_follow_the_phrase_layout():
    for n in (1, 2, 3, 4, 5, 7, 8, 9, 16):
        labels = form_labels(n)
        assert len(labels) == len(phrase_layout(n).phrase_lengths) and labels[0] == "A"
    assert form_labels(16)[:4] == ("A", "A'", "B", "A'")

# test_compose_musicality.py
@pytest.mark.parametrize("matrix", ALL_MATRICES)
def test_phrase_shaping_targets(all_reports, reserve_report, matrix):
    o = {**all_reports, **reserve_report}[matrix]["overall"]
    assert o["F.M6"]["mean"] >= V4_FS[matrix]["F.M6"] + 2          # rulings F2a/F2d
    if matrix == "validation":
        assert o["F.hook"]["mean"] >= 0.75
        assert o["F.peak"]["mean"] >= 0.9
        assert o["F.arch"]["mean"] >= 0.5
```

(`PROG8 = [ChordSymbol.parse(c) for c in ("C", "Am", "F", "G", "C", "F", "G", "C")]`. Initialise the three lists in the second test separately.)

- [ ] **Step 2: Run** the new tests. Expected: FAIL.
- [ ] **Step 3: Implement** the generator terms, `form_labels`, `plan_bars(contrast_b=)` and the wiring.
- [ ] **Step 4: Run** `pytest -q tests/unit tests/integration`. Expected: PASS.
  - Tune the feature-3 weights if a target is missed.
  - If a target still cannot be met without breaking a hard rule, report DONE_WITH_CONCERNS with the per-matrix values.
  - Feature 2's no-narrowing F.M6 test (F2a) stays as it is.
  - The v3 soft musicality tests for fingerstyle (`F.*` keys and `*.fingerstyle` keys) may move. A failing soft assertion follows the v4 rule: drop only that key from that one assertion, with a comment citing v5 spec §5.3 and both values. List each such change in the report.
- [ ] **Step 5: Run the gates and commit**: `feat(compose): hooks, arch contour, phrase peaks and A A' B A' form for fingerstyle`.
