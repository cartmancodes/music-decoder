# Compose improvements plan review — round 2

Reviewer: GPT-6 Astra (Codex).

Reviewed revision 2 of the design and plan, the round-1 review, the actual compose modules and tests, and the relevant CLI, public contract, canonical voicing and tab-assignment code. No source or test files were modified. Following the pre-approved unattended workflow, I made the review decisions without requesting human input.

Verification: `pytest -q tests/unit tests/integration` completed with **327 passed, 3 skipped**; `ruff check src tests` and `mypy --strict src` passed. `ruff format --check src tests` still reports precisely the seven pre-existing files covered by Task 0. I loaded the plan's implementation fences as in-memory modules, measured the old composer through the Task 3 refactor, then measured the proposed final implementation. The paired headline baseline/candidate numbers reproduce, and **all 30 parametrized regression cases in Task 8 pass**. The reproductions below execute that proposed implementation against the repository's actual dependencies; they are not claims about changes already present in production. Candidate evaluation took 3.78 seconds in one run and 6.37 seconds while other checks were running. The revised, load-dependent runtime expectation is reasonable. Existing fixture-marker changes are excluded from this commit.

The public signature and CLI flags remain intact, and the proposed random sampling, ordering and direct tab-assignment call preserve seeded determinism without YAML reads. The phrase layout, interval-overlap metrics and separate melody/chord tab blocks are substantial improvements. Three Important findings remain.

1. **Important — A one-note final bar still bypasses the octave limit, now through its interaction with melody floors.**

   **Location:** `docs/superpowers/plans/2026-11-06-compose-improvements.md`, Task 4 `generate_melody()` / `final_target()` / octave-limit tests, and Task 7 `_melody_floor()`; `docs/superpowers/specs/2026-11-06-compose-improvements-design.md`, §5.3 and round-1 F1 response.

   **Problem:** The final-bar scoring restriction only helps when the final bar contains a scored note before its forced target. The `(8,)` cadence template goes directly from the previous bar into `pitch = target`. Floors can remove the nearer target octave. Exact public-core reproduction: `arrange(Scale("C", "major"), [ChordSymbol.parse(c) for c in "C Am F G7".split()], tempo_bpm=120, seed=16)`. The floors are `[55, 63, 60, 68]`; the melody's final two notes are **F4/MIDI 65 at 5.75 seconds and G5/MIDI 79 at 6.0 seconds**, a **14-semitone leap**. G4/67 was excluded by the G7 floor. The same failure occurs with `C G Am G7`, seed 16. This is an ordinary supported progression, not a custom pitch-range case. The claim that the target always satisfies both constraints is false. Task 4's sweep calls `generate_melody()` without the floors that Task 7 introduces, so it misses this coupling.

   **Concrete fix:** Plan reachability across the boundary into a one-note final bar, including the floor-constrained target. Constrain the preceding choices or choose a documented, feasible target/register policy jointly with the floor. Every transition, including a forced target and any fallback, must satisfy the hard interval invariant. Add the two reproductions as failing `arrange()` tests in the integration task, plus a deterministic adversarial test forcing `(8,)` and a distant previous pitch. Retain the standalone-generator tests, but also sweep a small set of final seventh/suspended chords through `arrange()` with floors enabled. Round-1 F1 is **not resolved**.

2. **Important — A valid composition can now fail because the register floor removes every cadence target.**

   **Location:** Plan Task 4 `_candidates()` and `final_target()`, Task 7 `_melody_floor()` / `arrange()`; design §5.3, §5.6 and §6.

   **Problem:** A floor is accepted solely by candidate count, without checking whether the required cadence pitch class survives. `arrange(Scale("A", "major"), [ChordSymbol.parse("A"), ChordSymbol.parse("Dsus4")], tempo_bpm=120, seed=0)` raises **`ValueError: No cadence target pitch in the requested range.`** A minor with `Am Dsus4` fails too. The Dsus4 shape's G4/67 yields floor 70. In A major the surviving pitches are `[71, 73, 74, 76, 78, 79]`, so the floor passes the five-candidate rule. Dsus4 contains A, making A the required final target, but its only octave in 62–79 is MIDI 69, which the floor removed. The current generator succeeds on the A-major input and produces eight notes. This is a new public API/CLI failure for valid inputs, contradicting the design's claim that the default window makes the relevant empty-pool case impossible. The default matrix contains no suspended chords.

   **Concrete fix:** Validate cadence feasibility when applying floors, not merely pool size. Specify deterministic precedence among tonic/root resolution, pitch window, accompaniment clearance and identity coverage. For example, allow a floor exception when the target itself supplies an essential tone and thinning can retain the remaining harmony below it. Do not reject an input the existing composer supports. Add both failing examples, a single-chord variant, and lightweight coverage of every supported chord quality under representative major/minor keys. Verify both successful composition and the resulting pitch/interval/identity invariants after the fallback.

3. **Important — Chord-role guessing still replaces the wrong scale degree for common secondary dominants.**

   **Location:** Plan Task 2 `harmony.chord_scale_pcs()` (especially the foreign perfect-fifth branch) and harmony tests; design §5.1, D7 and round-1 F2 response.

   **Problem:** A pitch's interval above the chord root does not by itself identify the scale degree being altered. In **C major over B7**, the proposed function returns pitch classes **`{0, 3, 4, 5, 6, 9, 11}`**, i.e. C, D-sharp, E, **F, F-sharp**, A, B. It deletes **G**, while retaining F alongside B7's F-sharp. For B7 resolving to Em, the required alterations are D→D-sharp and F→F-sharp; G is the destination minor chord's third and should remain available. The foreign-perfect-fifth branch instead treats F-sharp as a flattened G. Similarly, the C-major/F-sharp7 pool retains F beside F-sharp and deletes G. The E7/A-minor and D7/C-major fixes are correct, but they do not establish the general claim that only the altered degree is replaced or that secondary dominants receive the correct local scale. The broad statement that the global tonic is always retained also conflicts with the plan's own A-major-in-C test, which deliberately replaces C with C-sharp.

   **Concrete fix:** Define a bounded, explicit policy for supported applied dominants and borrowed chords that identifies the actual altered scale degrees; use harmonic context/degree relationships rather than an unconditional upper-neighbour preference for roots and fifths. Define a conservative policy for ambiguous chromatic chords instead of claiming a uniquely correct scale. Add a failing B7→Em example in C major, asserting D-sharp and F-sharp present, D and F absent, and G retained; cover the corresponding transpositions and a multi-alteration dominant. Recompute affected baseline/candidate measurements under any changed metric definitions. Narrow the tonic-preservation documentation to the cases where it is musically intended. Round-1 F2 is **only partially resolved**.

4. **Minor — H1 preserves tones already present in a shape; it cannot establish the advertised chord-identity guarantee.**

   **Location:** Plan Task 2 `essential_tone_coverage()`, Task 8 accompaniment documentation; design §4 H1 and §5.4. Actual dependency: `src/music_decoder/chords/voicings.py`, `C#dim` entry.

   **Problem:** H1 computes `required = essential_pcs(chord) & shape_pcs`. This deliberately hides an essential tone missing from the original shape. The existing C-sharp diminished shape sounds MIDI **52, 58, 61** (E, A-sharp, C-sharp): it lacks the defining diminished fifth G and adds A-sharp. H1 only requires E for that shape, so it can report perfect coverage without a diminished triad. This is a pre-existing voicing defect, not a reason to demand a new voicing-search system, but the new documentation's unconditional promises that chords keep their identity exceed what the metric proves.

   **Concrete fix:** Describe H1 as preservation of the supplied shape's essential tones and separately report missing/foreign shape tones against the requested chord. Add a small hand-built/canonical counterexample so the limitation stays visible. Either explicitly document the inherited shape defect as deferred under the no-voicing-change constraint, or authorize a narrow canonical-data correction with its own test; do not silently count the missing identity tone as a successful guarantee.

5. **Minor — The Task 3 golden check is not byte-for-byte output verification and writes outside the permitted worktree.**

   **Location:** Plan Task 3 Steps 1 and 3; design §3.1.

   **Problem:** The snippet compares only ASCII tab, pitches with rounded onset times, and voicing positions. It does not compare note ends, velocities, accompaniment events or MIDI bytes, despite the claim that the refactor's output is checked byte for byte against old `compose()`. It also writes its text files under `$TMPDIR` and MIDI artifacts through unqualified `tempfile.mkdtemp()`, outside the worktree required by the pipeline's operating instructions.

   **Concrete fix:** Use an explicitly named temporary directory inside this worktree, clean it after verification, and never stage it. Compare normalized complete melody/accompaniment event records and MIDI bytes if claiming byte identity, with timestamps/paths excluded deliberately. Otherwise narrow the design's claim to exactly what is checked. Keep this verification before the baseline capture and before generator changes.

Round-1 disposition:

| Earlier finding | Round-2 assessment |
|---|---|
| F1: final target bypasses octave limit | **Open.** Finding 1 supplies a new reproduction through the public core. |
| F2: tonic deletion / blunt chromatic metric | **Partial.** E7/A-minor and D7/C-major are repaired; chord tones and short weak notes are excluded from M13; M15 is added. The generalized alteration rule still fails (finding 3). |
| F3: onset-only accompaniment metrics | **Addressed.** A2/A3 inspect interval overlap, with staggered/late-entry examples and non-overlap controls. Paired baseline values reproduce. |
| F4: thinning erases final harmony | **Addressed for the original defect.** Identity restoration and melody floors repair the D-major case. Floors introduce findings 1–2; the shape-relative H1 limitation is finding 4. |
| F5: rendered melody overwritten | **Addressed.** Separate tab blocks and one column per melody note remove the collision mechanism; round-trip and long-piece tests are specified. |
| F6: hidden YAML dependency | **Addressed.** The lower-level assigner receives explicit weights/fret cap; config-read and environment-independence tests are present. |
| F7: prototype-calibrated regression strategy | **Substantially addressed.** Same-input baseline capture precedes changes; disjoint tuning/evaluation seeds, subgroup comparisons, bounded allowed regressions and musical threshold rationales are specified. Reproduced paired results support improvement. Passing aggregates still do not replace the boundary/quality tests requested above. |
| F8: phrase metrics ignore actual layout | **Addressed.** Shared layout, separate echo coverage/agreement and irregular-length examples. |
| F9: nearest-pitch metric mistaken for voice leading | **Convincingly resolved by narrowing the claim.** A matching-based voice-leading optimizer/metric is not required for this scope. |
| F10: runtime budget | **Addressed.** Config reads removed, timings recorded with load sensitivity, no brittle wall-clock assertion. |

VERDICT: CHANGES_REQUESTED
