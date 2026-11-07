# Voicings and leaps plan review — round 1

Reviewed the design and implementation plan dated 2026-11-07 against the composer, shared voicing library, shape consumers, evaluation/report code, CLI, and their tests. No earlier `2026-11-07-voicings-leaps-*` reviews exist. Source and tests were not modified. Review probes were written under `/tmp`, not in the worktree.

Verification: the focused compose, shared-voicing, renderer, public-contract and compose integration suite passes **289 tests in 4.39 s**. An independent pitch audit reproduces the design's faulty-shape counts: **5/17/22/18/19/17** across EADGBE/Drop D/Eb/D standard/Drop C/DADGAD, totaling **98/576**. A review-only executable transcription of the proposed melody rules reproduces validation M4 **0.0002238176**, three large leaps, L1 **0**, M4max **8**, and edge M4 **0**, L1 **0**, M4max **7**. The counterexample below concerns the proposed rules, not an implemented repository change.

The five proposed pitch-class repairs are correct, and the retained harmonic-minor leading-tone handling is sound. The consumer inventory is accurate: recognition uses templates, the SVG renderer uses the canonical shapes, and the composer uses the shapes for accompaniment, floors, metrics and chord tab. The API/CLI signatures and seed-version policy are otherwise preserved. The following issues prevent approval.

## Findings

### 1. Important — The relaxed fallback reintroduces a major seventh on an ordinary supported input

**Location:** `docs/superpowers/specs/2026-11-07-compose-voicings-leaps-design.md`, §5.3 scoring and `_candidates`, §8; `docs/superpowers/plans/2026-11-07-compose-voicings-leaps.md`, Task 6, generator wiring and tests. Relevant implementation: `src/music_decoder/compose/melody.py`, `_candidates`, `final_target`, and `generate_melody`.

**Problem:** The global promise of no sevenths is not established by the selected matrices. With `Scale("C", "major")`, progression `("C", "D#m")`, standard tuning, one bar per chord, default tempo 100 and seed **19**, the proposed rules produce MIDI pitches **[67, 64, 64, 75, 75]**: E4→D#5, an **11-semitone** leap at the chord change. The unchanged v2 composer produces the same pitches. This uses the public composer's normal register and actual shape-derived floors **[55, 69]**, not an exotic custom range.

At the first slot of the two-note final bar, the retained floor gives candidates `[69, 70, 72, 74, 75, 77, 78]`; `prev=64`, `target=next_fixed=75`. No candidate satisfies R1–R4: A4 (69) is a tritone from the target, Bb4 (70) is a tritone from the previous note, and the remaining pitches leap more than a fifth. The fallback discards every admissibility rule and samples D#5. Merely removing the floor while retaining target 75 also leaves no valid intermediate pitch in this chord-scale pool. The plan explicitly retains the old octave-based floor test for multi-note final bars, causing this conflict.

**Concrete fix:** Specify a deterministic constraint-repair order before sampling: relax the optional register floor and recompute the cadence target/cadence path together when the constrained path is empty. Here the lower D#4 target (63) makes a smooth path possible. Preserve forbidden-interval and recovery constraints whenever a valid musical path exists; do not restore all v2 scores indiscriminately. Add a failing generator/API regression for this exact input, a direct empty-admissible-set test, and a bounded root/quality/cadence sweep with actual composer floors. Define separately what happens for genuinely infeasible custom ranges without claiming a universal guarantee the fallback cannot satisfy.

### 2. Important — “Playable” accepts five-finger grips, including the explicitly pinned Eb7 repair

**Location:** Design §5.2 items 4–6 and §9; plan Task 5, `_is_playable`, `test_retune_and_search_choices`, and the 576-shape audit. Relevant implementation: `src/music_decoder/compose/voicings.py` and `src/music_decoder/chords/voicings.py`.

**Problem:** A four-fret span is not a four-finger fingering. The plan pins Eb-tuned D#7 to **`0-7-6-7-5-4`**, which sounds the right notes but requires five independent fretting fingers under conventional single-finger/straight-barre fingering. The two fret-7 notes cannot share a barre because the intervening string must sound fret 6; frets 6, 5 and 4 need three more fingers. This is the replacement for the broken Eb7 in a human-flagged sample. A simple exhaustive finger-cover check finds **22/576** proposed accepted shapes needing at least five such fingers, including standard C#7 `x43421`, standard D#7 `x65643`, and several DADGAD retunes.

The search's finger heuristic does not protect the retuned-library fast path. In general, its assumption that all lowest-fret notes can share one barre also needs to account for intervening open strings. Declaring library finger counts out of scope does not make the proposed sample repair playable.

**Concrete fix:** Require a feasible four-finger assignment, including legal barre spans that do not cover required open/lower-fret strings, before accepting a retune or search result. Reject failed retunes into the search. Add negative fingering tests for `0-7-6-7-5-4` and separated lowest-fret notes, then audit all 576 results for this property. For example, **`0-2-0-1-0-0` in Eb tuning** is a faithful, readily playable D#7. Repair the additional unplayable canonical entries or explicitly provide feasible canonical alternatives, updating the “only five entries change” and unchanged-library assertions accordingly. A general difficulty model is unnecessary; basic fingering feasibility is enough.

### 3. Important — Sample melody regressions fail on the version before testing the reported defects

**Location:** Plan Task 3, `test_sample_melodies_are_singable`, and Tasks 4–6 xfail removal; design §5.6–5.7.

**Problem:** The first assertion is `composer_version == 3`, while Tasks 3–5 necessarily run version 2. Thus `--runxfail` reports an `AssertionError` without reaching any musical assertion. The proposed red phase never proves that these tests detect the three reported leaps. It also makes the claimed Task 4/5 XPASS safeguard impossible: even if a voicing change removes every leap, the version assertion keeps the test xfailed. Checking only the exception type does not fix this.

**Concrete fix:** Keep the strict-xfail melody tests exclusively about the musical defects. Add the version assertion in a separate test in Task 6 when the version is bumped. In Task 3, verify the actual failing assertion for each sample: B minor must reach the unjustified-leap assertion, Ab must reach the forbidden-interval assertion, and Eb must reach the chord-change/unjustified-leap assertion. Repeat those checks after the voicing tasks before removing the final xfails.

### 4. Important — Task 2 deliberately creates a failing commit and its clean-snapshot procedure is incomplete

**Location:** Plan Global Constraints and Task 2 Steps 1, 4–8; `scripts/compose_musicality_report.py`, `_git_head`.

**Problem:** Step 5 commits `test_committed_v2_snapshot_has_three_matrices_and_the_new_metrics` before Step 6 creates the fixture it reads. Excluding the test with `-k "not v2_snapshot"` does not make that commit green: the mandated full gate raises `FileNotFoundError`. This is a concrete failure point for task-by-task unattended implementation. In addition, `git status --short src` does not establish the cleanliness required by `git describe --always --dirty`; generated tracked fixture markers from the full gate can make the snapshot source dirty despite clean source files.

**Concrete fix:** Commit the evaluation code and its currently runnable tests first. Add the snapshot-reading test only in the subsequent fixture commit, after generating the snapshot from the clean evaluation-code commit. Run the full gate at both boundaries. Before generation, check all tracked changes and explicitly restore only generated fixture markers to their recorded contents, or define and implement source-scoped provenance instead of using whole-tree dirty status. Never stage the markers. State this sequence in the plan so a worker does not have to violate either the green-commit or clean-snapshot requirement.

### 5. Important — The claimed held-out validation has been used to choose weights and acceptance tolerances

**Location:** Design §6.1, §6.3, D6 and D8; plan Task 6 Step 5, Task 7, Task 8 Step 1; `src/music_decoder/evaluation/compose_musicality.py`, validation-seed documentation.

**Problem:** The plan tries bonuses 0.5, 1.0 and 1.5 and chooses the preferred metric tradeoff, puts L4 thresholds approximately 0.015 below the measured prototype, chooses the 0.02 regression tolerance to accommodate its measured 0.013 loss, and demands reproduction of prototype means within 0.005. Those same matrices are then called “held-out.” They are now development data for this change. The baseline numbers establish the defects, but do not independently justify these particular tolerances. This matters because the exact prototype passes every published matrix while retaining the major seventh in finding 1.

**Concrete fix:** Label the existing matrices as regression/development data for v3 and keep their historical comparisons. Freeze the rule choices and musical acceptance bounds before evaluating a small, genuinely unused seed/progression set; capture matching v2 measurements before composer edits and add a bounded final holdout check. Include actual shape-derived floors and cadence/chord-quality diversity. Justify each regression allowance in its metric's units and musical impact, rather than by the largest observed prototype loss. Replace “must match prototype within 0.005” with investigation/reporting of deviations plus independently specified acceptance criteria, allowing a corrected implementation to differ.

### 6. Minor — M4's stated frequency bound does not match its aggregation

**Location:** Design §6.1 M4 row; plan Task 7, `test_leap_targets`; `src/music_decoder/evaluation/compose_musicality.py`, `_aggregate`.

**Problem:** `M4.mean <= 0.001` averages each piece's large-leap proportion equally. It does not mean “at most one leap per thousand intervals,” as the justification says. Unequal lengths in the validation and edge matrices make these different quantities. A long piece with two leaps per thousand intervals and a short leap-free piece can meet the per-piece mean while the pooled interval frequency exceeds the advertised bound.

**Concrete fix:** Preserve M4 for backward compatibility, but either describe the target accurately as a mean of per-piece rates, or additionally expose large-leap and interval counts and enforce the stated pooled bound. Add a hand-constructed unequal-length aggregation test if introducing the pooled measure. Report both the rate and affected-piece counts when explaining rarity.

VERDICT: CHANGES_REQUESTED
