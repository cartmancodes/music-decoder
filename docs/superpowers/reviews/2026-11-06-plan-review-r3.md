# Compose improvements plan review — round 3

Reviewer: GPT-6 Astra (Codex).

Reviewed revision 3 of the design and implementation plan, both earlier reviews, the actual compose modules and tests, and the public API, CLI, voicing and tab-assignment dependencies. Following the pre-approved unattended workflow, I made the review decisions without requesting human input. No source or test files were edited.

Verification:

- The current `pytest -q tests/unit tests/integration` suite, with temporary output directed inside this worktree, completed with **327 passed, 3 skipped**. `ruff check src tests` and `mypy --strict src` passed. `ruff format --check src tests` reports precisely the seven existing files covered by Task 0.
- Executed the plan's implementation fences as in-memory modules against the repository's dependencies. The Task 3 refactor reproduced the original composer's tab, melody, voicings, metadata and MIDI bytes on its three golden cases. The final proposed `compose()` signature matches the existing signature exactly.
- Recomputed the paired baseline and candidate reports on the 400 validation and 110 edge inputs. The published headline values reproduce; **all 30 planned regression cases pass**. Candidate evaluation took **2.64 seconds** in this run.
- Executed **180 planned unit-test cases that require no pytest fixtures**, including the round-2 reproductions, floor sweeps, every-quality/root check and transposed secondary-dominant checks. All passed. A separate 5,760-melody sweep with every supported quality/root in an interior bar also passed the octave limit and existing H1 checks in all three styles.
- Confirmed finding 1 in MIDI written by the proposed `compose()`, with WAV synthesis mocked. These candidate results describe the plan's code, not code already implemented in `src/`.

The cadence reachability and chord-spelling repairs are sound for the tested public inputs. The pure core, separate tab blocks, seeded sampling and staged baseline capture remain appropriate. One Important finding remains.

1. **Important — A brief melody tone can remove an essential tone from the entire ringing accompaniment, leaving the sustained final cadence incomplete.**

   **Location:** `docs/superpowers/plans/2026-11-06-compose-improvements.md`, Task 5 `_thin_below()` and `_accompaniment_notes()` final-bar branch; Task 2 `essential_tone_coverage()`; Task 3 `case_metrics()` assignment of `H1final`; Task 7 floor-fallback tests; Task 8 hard invariants. `docs/superpowers/specs/2026-11-06-compose-improvements-design.md`, §4 H1 and §5.4 identity-preserving thinning/final bar.

   **Problem:** `_thin_below()` treats the union of *all* melody pitch classes overlapping an attack's window as tones supplied by the melody. For the final stroke that window is the entire bar. A short note early in the bar can therefore suppress an essential accompaniment pitch for the whole bar, even though the later held cadence note does not supply it. The newly permitted floor fallbacks expose this interaction. `H1final` is merely the last element of the same whole-bar coverage list, so it cannot detect the loss during the actual sustained ending.

   Two exact reproductions, at 120 BPM, one bar per chord, standard tuning:

   - **C major, `C Am F G7`, seed 5:** final-bar melody is G4/67 at 6.0–6.475 s, F4/65 at 6.5–6.975 s, then G4/67 at 7.0–7.95 s. The ringing accompaniment contains only MIDI **43, 47, 50, 55, 59** (G–B–D–G–B), starting at 6.0 s with the specified stagger and ending at 8.0 s. The shape's F4 is omitted because the middle melody note supplied F briefly. The final two-beat hold consequently sounds **G major, with no seventh**, although the requested chord is G7. This is also the accompaniment in the serialized MIDI.
   - **A major, `A Dsus4`, seed 1:** final melody is A4/69 at 2.0–2.475 s, G4/67 at 2.5–2.7375 s, A4/69 at 2.75–2.9875 s, then A4/69 at 3.0–3.95 s. Accompaniment is **D3–A3–D4** for the whole final bar: the short G removes the suspended fourth from the held ending. A minor, `Am Dsus4`, seed 1 does the same.

   These results occur in every style because all styles share the final-stroke branch. H1 reports every bar covered, H1final is true, and A4 passes. On the plan's own edge matrix, **8 of 110 endings** lack a shape-provided essential pitch class throughout the final held melody note, while all 30 regression cases remain green. Examples already in that matrix include `C Fm C G7`, seed 1006, and `A Dsus4`, seed 1002. The original shapes contain the missing tones, so this is not the inherited shape defect explicitly deferred in round 2.

   **Concrete fix:** Give the ringing final chord a temporal identity policy. A brief earlier melody note must not exempt an essential accompaniment tone needed during the final sustained note. For example, retain every shape-provided essential tone unless the final held melody note itself supplies that pitch class; continue using the whole window to determine register clearance. The documented priority already allows identity to override clearance when a floor yields. Keep ordinary arpeggio coverage at the bar level; this does not require complete simultaneous chords on every arpeggio attack.

   Add a hand-built failing thinning test with a temporary essential melody tone, and failing `arrange()` tests for both examples in all styles. Add a distinct cadence-coverage metric/assertion over the final sustained melody interval instead of aliasing `H1final` to whole-bar coverage. Measure it on baseline and candidate using the same definition, rerun A2/A3 after restoring the tones, and retain A4's single ringing attack. Place the behavioral failures in Task 5 and the floor-coupled failures in Task 7. Round-1 F4 is **partially reopened**: its original D-major case is repaired, but preserving chord identity at the sustained cadence is still not guaranteed.

2. **Minor — The spec overstates which targets run on both matrices and retains the old edge-matrix size.**

   **Location:** `docs/superpowers/specs/2026-11-06-compose-improvements-design.md`, §7 regression-suite bullets and §10 D9; `docs/superpowers/plans/2026-11-06-compose-improvements.md`, Task 8 `test_validation_targets()`, `test_validation_floors()` and `test_pooled_leap_recovery_and_leading_tone_resolution()`.

   **Problem:** §7 says the suite asserts every absolute target and hard invariant on both matrices, but the average targets, M1/M9 floors and pooled M5/M15 thresholds apply only to validation. This distinction is necessary: the edge matrix contains a one-note piece (M9 minimum **1**, below the validation floor of 2) and only **14** eligible M15 transitions, below the validation requirement of 20. D9 still says **80** edge melodies, although the implementation and revised matrix contain **110**. These conflicting instructions could lead an implementer to change correct tests or thresholds to satisfy the prose.

   **Concrete fix:** State that validation receives the absolute averages, validation floors and pooled sample-size targets; both matrices receive the explicitly listed hard invariants and guards; the edge matrix receives its paired improvement comparisons. Update D9 to 400 + 110 and distinguish historical revision-2 timing/count references from the current matrix. Do not weaken a threshold or force single-note pieces to meet a multi-bar rhythm-variety requirement.

Earlier-finding disposition:

| Earlier finding | Round-3 assessment |
|---|---|
| R1 F1 / R2 F1: forced final leap, including a one-note final bar under a floor | **Resolved.** Cadence feasibility now checks the previous pitch before accepting a floor; the original and floor-coupled reproductions pass. Final-bar scoring keeps subsequent choices within reach of the target. |
| R1 F2 / R2 F3: altered-degree errors, tonic deletion and the chromatic metric | **Resolved within the stated scope.** E7/A-minor, D7/C-major, B7 and F#7 produce the intended pools; five applied dominants pass in all twelve major keys. The ambiguous-chord heuristic and tonic-preservation limitations are explicit. M13 is a bounded exposure proxy; M15 separately measures global leading-tone resolution. |
| R1 F3: onset-only overlap metrics | **Resolved.** Full-interval overlap and the staggered/late-entry counterexamples remain present and pass. |
| R1 F4: thinning removes chord identity | **Partially reopened by finding 1.** The original D-major reproduction and shape-relative per-bar preservation pass; the sustained ending still has a temporal blind spot. |
| R1 F5: rendered melody overwritten | **Resolved.** Separate melody/chord blocks remove the collision mechanism; the plan includes decoded-output checks on short and long pieces. |
| R1 F6: YAML/configuration dependency | **Resolved.** The lower-level assigner receives pinned parameters. Config-loader and environment-independence checks are specified; the no-config unit check passes. |
| R1 F7: prototype-calibrated evaluation | **Resolved as an evaluation workflow.** Same-input baseline capture, disjoint tuning seeds, subgroup comparisons, musical rationales and mutation checks are concrete; the paired results reproduce. Finding 1 requires an additional independent cadence measurement. |
| R1 F8: phrase-boundary metrics | **Resolved.** Shared layout and separate motif coverage/agreement handle irregular phrase lengths. |
| R1 F9: proximity mistaken for voice leading | **Convincingly rebutted/narrowed.** V1 is explicitly only a proximity guard; no voice-leading optimization is promised. |
| R1 F10: runtime budget | **Resolved.** The load-dependent expectation is reasonable and reproduced; no brittle wall-clock assertion is planned. |
| R2 F2: valid suspended-chord input raises because a floor removes the target | **Resolved for input acceptance and interval safety.** Dsus4 cases compose; the floor's interaction with final harmony is finding 1. |
| R2 F4: H1 hides defects of the supplied shape | **Resolved by scope and S1.** The C#dim counterexample is explicit and tested; shared shape repair is reasonably deferred. |
| R2 F5: inadequate/out-of-worktree golden check | **Resolved.** The three-case golden comparison now includes complete notes and MIDI bytes, reproduces exactly, and uses worktree-local scratch output. |

Task ordering and failing-test steps are otherwise concrete enough for task-by-task implementation. Baseline capture precedes behavioral changes, component interfaces are specified, and the public signature and CLI flags remain compatible. No new dependency, voicing search or unrelated feature is needed to fix the remaining blocker. Existing fixture-marker changes and pre-existing scratch output are excluded from this commit.

VERDICT: CHANGES_REQUESTED
