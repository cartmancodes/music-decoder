# Voicings and leaps plan review — round 2

Reviewed revision 2 of `docs/superpowers/specs/2026-11-07-compose-voicings-leaps-design.md` and `docs/superpowers/plans/2026-11-07-compose-voicings-leaps.md`, the round-one review, the composer, shared shape library and its consumers, evaluation/report code, CLI, and associated tests. Source and tests were not modified. Review probes were written under `/tmp`; no scratch directory was created inside the worktree. No human decision was needed in this headless review.

## Verification and round-one disposition

- The focused compose, shared-voicing, renderer, public-contract and compose integration suite passes **289 tests in 4.56 s**.
- A review-only executable transcription of revision 2 reproduces the repaired C → D#m, seed 19 melody: **[67, 64, 64, 66, 63]**. It also reproduces validation M1 **0.9771**, M4 **0.0002**, M4max **8**, pooled M5 **0.7620**, pooled M15 **0.8485**, and V1 **1.6475**; edge M1 **0.9843**, M4 **0**, and M4max **7**. Neither development matrix uses the relaxation ladder. These are checks of the proposed algorithm, not claims that implementation has landed.
- An additional forbidden-interval probe covers **110,592** generator calls: C major/A minor home chord followed by every ordered pair of the 96 supported chords, seeds 0–2, with phrase rhythms and the one-note legacy grid, using the revised standard-tuning shape floors. It finds no tritone, seventh or interval larger than an octave. This probe checks forbidden intervals, not every musicality invariant or every tuning.
- The revised shape transcription gives **0/576** faithfulness or four-finger failures. The seven replacements contain the intended tones. The diminished-triad repairs, optional fifth in seventh chords, and retained harmonic-minor leading-tone handling are sound. Consumer inspection confirms that recognition uses templates, while the diagram renderer and composer consume the shared shapes; melody tab assignment is separate. The planned API/CLI contract and deterministic ordering are preserved.
- The new holdout was **not evaluated** during this review.

| Round-one finding | Round-two disposition |
|---|---|
| F1: major seventh through the relaxed fallback | Resolved for the reported reproduction; independently reproduced the revised result and development measurements. The feasibility/repair rules and added regressions address the mechanism. |
| F2: five-finger grips | Resolved. The two additional canonical repairs and fingering gate address the reported grips; the 576-shape audit passes. |
| F3: version assertion masks melody failures | Resolved for the melody tests: version checking is now separate. A new, smaller masking problem remains in the voicing test (finding 3 below). |
| F4: failing snapshot commit and dirty provenance | Resolved. Task 2 now separates runnable code from fixture-dependent tests and checks the tracked tree before snapshot generation. |
| F5: development data presented as held out | Partly resolved. The relabeling, frozen new inputs, unit-based tolerance explanations and removal of prototype-matching acceptance are improvements. The actual holdout workflow still has the two Important defects below. |
| F6: M4 aggregation misdescribed as interval frequency | Resolved. L0 supplies pooled counts and an unequal-length test; M4 is described accurately. |

## Findings

### 1. Important — A failed acceptance check can be converted into a green final gate

**Location:** `docs/superpowers/plans/2026-11-07-compose-voicings-leaps.md`, Task 7, “If the holdout acceptance fails” (lines 1294–1298), and Task 8 final gate; `docs/superpowers/specs/2026-11-07-compose-voicings-leaps-design.md`, §6.4.

**Problem:** Task 7 explicitly instructs the implementer to mark a failed holdout test `xfail(strict=True)` and proceed to reporting and committing. A strict xfail still makes an expected failure non-blocking; it only fails the suite if the test unexpectedly passes. Thus a composition can violate the frozen no-regression bound, tab coverage, cadence identity or another acceptance requirement while the pipeline finishes with a green gate. This is materially different from Task 3's temporary xfails, each of which has a named fixing task and mandatory removal. There is no later removal requirement here. Reporting the failure is useful, but does not satisfy the acceptance criteria or the user's no-regression constraint.

**Concrete fix:** Remove the acceptance-to-xfail escape hatch. Keep final acceptance tests active and require zero new acceptance xfails at completion. A failed holdout should produce a recorded failing case and leave the stage incomplete/changes requested, or be corrected through the spec's permitted development reproduction or independently stated musical rule, followed by the full acceptance checks. If a correction uses information from the failed holdout to select behavior, explicitly treat that set as regression data and reserve a new frozen acceptance set. Do not silently redefine a failed acceptance check as an expected failure merely to obtain a green commit.

### 2. Important — Task 6 prints the holdout results while instructing the implementer not to inspect them

**Location:** `docs/superpowers/plans/2026-11-07-compose-voicings-leaps.md`, Task 2 evaluation/report interfaces and opening sequencing promise; Task 6 Step 5 (lines 1138–1153). Relevant code: `scripts/compose_musicality_report.py::main`, `src/music_decoder/evaluation/compose_musicality.py::evaluate`, and the `report` fixture in `tests/unit/test_compose_musicality.py`.

**Problem:** Task 2 makes `evaluate()` return all four matrices and limits the report-script change to its docstring. The actual script iterates and prints every matrix returned by `evaluate()`. Consequently Task 6's prescribed `python scripts/compose_musicality_report.py | grep ...` prints the v3 holdout's selected metric rows too. The grep removes matrix headings, making those holdout rows unlabeled, rather than excluding them. This directly contradicts “Do not look at or tune to the holdout numbers” in the same step. In addition, the existing module-scoped `report` fixture calls `evaluate()`, so the full gates in Tasks 4–6 already compute the holdout, contradicting Task 2's claim that its snapshot is the only evaluation before Task 7. Merely computing unseen results is not itself tuning, but this workflow explicitly displays them during the implementation-measurement step.

**Concrete fix:** Specify a development-only evaluation/report path and use it for Tasks 4–6. For example, add a matrix-selection argument to the internal evaluator and report script, preserving the existing default if desired; the pre-acceptance test fixture and Task 6 command must select only validation, edge and tunings. Task 2's snapshot generation and Task 7's acceptance fixture explicitly select all four. Add a small test that the development path neither builds nor prints holdout cases, and retain matrix labels in diagnostic output. Update the file map and the script's “docstring only” instruction accordingly. This can be done without changing the public compose API or CLI.

### 3. Minor — The Eb sample's prescribed red assertion is masked by the earlier C7 fingering failure

**Location:** `docs/superpowers/plans/2026-11-07-compose-voicings-leaps.md`, Task 3 `test_sample_voicings_are_faithful` and Step 2 (lines 520–532 and 574–580).

**Problem:** The test checks every chord in progression order, asserting both faithfulness and finger count before reaching the sample-specific Eb7 assertion. On the actual v2 code, `major-08-Ab` reaches **C7 in Eb tuning** first: its frets are **`x43421`**, faithful but requiring **five fingers** under the new model. Therefore it fails at `fingers_needed(frets) <= 4`, not at `shape_faults` for the collapsed D#7 as Step 2 requires. I reproduced both shapes: C7 is `(-1, 4, 3, 4, 2, 1)` and D#7 is `(0, -1, -1, -1, -1, -1)`. The plan then says any different failing assertion must be fixed before committing, so its prescribed checkpoint cannot be met as written.

**Concrete fix:** Rebuild the chord data first, assert the named sample defect before the aggregate audit, then check all shapes and finger counts. Alternatively, split the named-defect regression from the all-shapes audit and give each an accurate expected failure and removal task. Retain the C7 fingering assertion; simply accepting it as the sole red result would not demonstrate the Eb7-specific regression. Adjust expected test counts if splitting the tests.

VERDICT: CHANGES_REQUESTED
