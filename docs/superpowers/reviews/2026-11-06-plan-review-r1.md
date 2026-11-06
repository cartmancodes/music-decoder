# Compose improvements plan review — round 1

Reviewer: GPT-6 Astra (Codex). Verdict: changes requested.

Reviewed the complete design and implementation plan, the four compose modules, their unit and integration tests, the public contract and CLI, and the tab assignment/rendering and canonical voicing dependencies. There were no earlier reviews: `docs/superpowers/reviews/` did not exist. No source or test files were edited. Proposed implementations were extracted from the plan's Python fences and executed as in-memory modules against the existing dependencies; this is evidence about the proposed code, not an assertion that it is already implemented.

Verification: the current `pytest -q tests/unit tests/integration` completed with **327 passed, 3 skipped**. `ruff check src tests` and `mypy --strict src` passed. `ruff format --check src tests` reported exactly the seven files listed in Task 0, so the separate formatting repair is justified. All **23 planned musicality regression assertions passed** when invoked against the in-memory implementation, including the assertions whose blind spots are described below. I also reproduced the full 3,000-melody baseline's M5 counts: 23 eligible leaps, 8 recoveries, ratio 0.347826. The existing fixture-marker changes are excluded from this commit.

The public `compose()` signature, CLI flags, and seeded random-number ordering appear preserved as proposed. Keeping rhythm and pitch generation separate, adding explicit chord-start times, and replacing the irregular arpeggio timing are sound directions. The following findings prevent approval.

1. **Important — The forced final target bypasses the hard octave limit.**

   **Location:** `docs/superpowers/plans/2026-11-06-compose-improvements.md`, Task 3, `generate_melody()` final-note branch and `final_target()`; `docs/superpowers/specs/2026-11-06-compose-improvements-design.md`, M4 and §5.2.

   **Problem:** The target is fixed at the beginning of the final bar, but the preceding notes can wander away from it. `pitch = target` bypasses `score_pitch()` and its exclusion of intervals above 12 semitones. Reproduction with the exact planned implementation: E major, `E A B7 E`, one bar per chord, 120 BPM, seed **283** ends with pitches `68, 64, 63, 76` in the final bar. The last jump is **13 semitones**. D major, `D G A D`, the same timing, seed **5774** ends `66, 78, 62`, a **16-semitone** final jump. Both use the default pitch window and public supported inputs. Seeds 0–4 miss this violation.

   **Concrete fix:** Make final-target reachability a constraint on preceding choices, rather than a soft attraction: at minimum restrict the penultimate pitch to within an octave of the fixed target, with deterministic handling if no path exists. Alternatively choose a reachable target octave at the final note and update the spec accordingly. Add failing tests for both examples before implementing the fix, plus a small adversarial test that forces a distant penultimate choice. Check the final transition through the same invariant as every other transition.

2. **Important — The chord-scale rule deletes the tonic and mistakes legitimate resolution notes for errors.**

   **Location:** Design §5.2 step 1, §10 D7, M13; plan Task 3 `chord_scale_pcs()` and its E7/secondary-dominant tests; Task 1 `nondiatonic_clash_ratio()`; Task 6 architecture documentation.

   **Problem:** Over E7 in A minor the proposed pool is `{C, D, E, F, G#, B}`: it removes **A as well as G**. That is not A harmonic minor, despite D7 describing it as standard harmonic-minor practice. Over D7 in C major it similarly removes G, the local tonic toward which the secondary dominant resolves. A weak or resolving A over E7 is not categorically unmusical merely because G# is in the chord. M13's hard zero and the tests explicitly enforce this over-restriction. M13 can also mark a legitimate chord tone as a clash when another, adjacent chord tone is foreign to the scale, as with the C and C# of C#maj7 in C major. There is no separate test of raised-leading-tone resolution across V–i.

   **Concrete fix:** For the supported dominant cases, replace the altered scale degree (G with G#), preserve the tonic, and assess dissonance by metric placement, duration, actual overlap, and resolution. Add tests that retain A over E7 and G over D7, exclude the conflicting unraised degree where appropriate, and measure leading-tone resolution on V–i/V–I transitions. Redefine M13 so legitimate chord tones and prepared/resolved non-chord tones are not automatically failures; then remeasure its baseline and target. Correct the harmonic-minor explanation and the tests that currently require deleting the tonic.

3. **Important — A2/A3 systematically miss the new strum's sounding notes.**

   **Location:** Design §4 A2/A3 and sounding-note convention; plan Task 1 `_sounding()`, `melody_above_accompaniment_ratio()`, `exposed_semitone_clash_ratio()`; Task 4 `_stroke()`.

   **Problem:** Both metrics probe only `melody.start_s + 0.0001`. Down-strum upper voices start 8–40 ms later; final-strum voices start 10–50 ms later. They are invisible to that probe even if they clash throughout nearly the entire melody note. A hand-built melody C4 from 0 to 1.9 seconds with accompaniment C3 from 0 to 2 and C#4 from 0.008 to 2 scores **A2 = 1.0 and A3 = 0.0**. It plainly has accompaniment above the melody and a sustained semitone clash. On the planned 400-case matrix, strum A3 is **0.0296** using the proposed probe but **0.0675** when an exposed note counts a clash anywhere during its sounding interval. Fingerstyle A2 likewise changes from **0.9817** to **0.9708** with full overlap. The current comparisons partly reward changing attack timing rather than improving harmony.

   **Concrete fix:** Evaluate overlap across sounding intervals, or explicitly duration-weight the violating overlap. Add hand-built failures for staggered attacks and accompaniment entering during a held melody note, as well as non-overlap controls. Apply exactly the same revised definitions to baseline and candidate arrangements before setting targets. Do not repair this merely by moving the single probe later; that would introduce another blind spot.

4. **Important — Thinning can remove chord identity from the entire final cadence.**

   **Location:** Design §5.3 `_thin_below()` and final bar; plan Task 4 `_thin_below()`, `_fingerstyle_bar()`, final-bar branch; Task 6 accompaniment regression tests.

   **Problem:** Keeping the lowest three pitches does not guarantee three distinct pitch classes, a third, or a seventh. For **D major, `D G A D`, seed 9, 120 BPM, one bar per chord**, the planned final melody is a single D4. The final accompaniment is D3–A3–D4: the original voicing's F# disappears, and there is **no third anywhere in the final bar**. The accompaniment also doubles the melody at the same pitch despite the claimed minor-third clearance. This passes every proposed regression assertion. Removing defining tones may be appropriate at particular attacks, but there is no guard against systematically turning major/minor/seventh harmonies into ambiguous root/fifth sonorities. A4 only checks timing; V1 examines the original, unthinned shapes.

   **Concrete fix:** Make thinning aware of harmonic roles and duplicates, and define an explicit policy when register clearance conflicts with retaining a chord's identity. Possible bounded solutions include retaining a defining third supplied by either part, selecting a feasible alternate voicing, or moving a cadence target to a compatible octave. Add the D-major reproduction and representative minor/seventh examples as failing tests. Measure essential-tone coverage over meaningful harmonic windows, especially the final cadence, rather than requiring every individual upstroke to contain a complete chord. Reconcile any necessary voicing change with the current no-voicing-change constraint.

5. **Important — The delivered ASCII tab can erase the improved melody fingering.**

   **Location:** Plan Task 5 `_ascii_tab_for()` and `test_ascii_tab_places_chords_at_their_start_times()`; design §5.4/T1/T2; actual `src/music_decoder/tabs/render.py::render_ascii_tab()`.

   **Problem:** The metrics inspect `Arrangement.melody_tab`, but users receive the merged, rendered tab. Chord anchors are appended after melody notes; the renderer's stable time sort and character writes let chord positions overwrite simultaneous melody positions on the same string. Exact planned reproduction: **C major, `C G Am F`, seed 0, 120 BPM, one bar per chord** assigns the first E4 to string index 3, fret **9**. The C-chord anchor on that string is fret **0**, and the actual rendered first column shows **0**, deleting the melody pitch. The merged positions also describe two different simultaneous frets on one string. The existing mock-based test only verifies anchor timestamps and cannot catch this. The fixed 64-column canvas introduces additional collisions for longer pieces.

   **Concrete fix:** Preserve the melody in the final rendering and present chord symbols/shapes separately unless a combined, collision-free fingering has actually been computed. Separate labeled melody/accompaniment tabs would also fit the two-track model without promising a solo-guitar arrangement. Add an end-to-end rendering assertion for the example above and a longer progression with two-digit frets. If a shared renderer change is necessary, explicitly update the scope; calling the tab an approximation does not justify changing or hiding its notes.

6. **Important — `arrange()` has an unacknowledged filesystem/configuration dependency.**

   **Location:** Design §5.4 pure `arrange()` and seed/version claims; plan Task 5 `_melody_tab()` and purity test; actual `src/music_decoder/tabs/__init__.py::_yaml_tab_params()` / `assign_tabs()` and `src/music_decoder/config/hyperparameters.py::load_hyperparameters()`.

   **Problem:** `assign_tabs(melody, tuning=tuning)` loads YAML on every call, honoring `MUSIC_DECODER_HYPERPARAMETERS`. Thus the proposed pure core reads the filesystem, and identical compose inputs, seed, and `composer_version` can produce different tab positions and costs when analysis configuration changes. T1/T2 and runtime also depend on that external configuration. Changing into an empty directory and checking that no files were created proves neither absence of reads nor independence from configuration. Passing explicit weights to the existing public tab wrapper alone would still perform its unconditional YAML read.

   **Concrete fix:** For the promised pure/versioned core, use explicit, versioned compose tab weights and a fret cap with the existing lower-level assignment function that takes those values directly. Keep any configurable adapter outside the pure core. Alternatively acknowledge configuration as an input and include its identity in reproducibility metadata, but then revise the spec's purity contract. Add a test that rejects config-loader/file-read calls and a test that changing the analysis configuration cannot silently change composer-versioned output.

7. **Important — The regression strategy is calibrated to its prototype rather than independently demonstrating the requested improvement.**

   **Location:** Design §3, §7 and §10 D11; plan Task 1 and Task 6 Steps 1–3.

   **Problem:** The baseline script and prototype are explicitly outside the repository. Most asserted thresholds quote means from a **different, 3,000-case** matrix, while the committed test covers 400 cases. D11 explicitly sets floors below the prototype minimum and means just inside prototype results. Task 6 first runs only after all implementation tasks and proves sensitivity only by changing the repeat weight. There is no retained paired measurement on the same inputs, holdout evaluation, or checks that key/mode/quality subgroups improve. All 23 proposed assertions pass while the concrete cadence, harmony, and tab failures above remain. This is insufficient evidence for the pipeline's requirement that thresholds be justified against the current baseline rather than chosen to pass.

   **Concrete fix:** Before replacing the generator, add a reproducible baseline measurement step using the agreed metric definitions and exactly identified inputs, and retain its script plus compact results or fixtures. Compare baseline/candidate on the same matrix, report useful key/mode/style subgroup results, and reserve disjoint seeds and selected edge progressions for validation rather than weight tuning. Include the adversarial cases in this review. Justify allowable regressions and thresholds musically and quantitatively; do not require a wholesale second legacy composer in production. Move the relevant behavioral failing tests into the implementation tasks, and exercise regressions in cadence, overlap, harmony coverage, and rendered tab, not only repeated-note frequency.

8. **Minor — The phrase metrics do not follow the planner's phrase boundaries.**

   **Location:** Design §4 M11/M12 versus §5.1; plan Task 1 `motif_contour_match()` / `phrase_end_long_ratio()` and Task 6 matrix.

   **Problem:** The planner deliberately splits five bars into 3+2 and nine bars into 4+3+2, but M12 always inspects bars 4, 8, etc. In-memory reproduction with five C-major C-chord bars, seed 0, 120 BPM gives **M12 = 0.0**, even though the actual phrase ends in bars 3 and 5 both last 1.9 beats. M11 similarly assumes fixed bar pairs and silently skips unequal note counts, so removing eligible motif pairs can evade evaluation. The regression matrix only contains four- and eight-bar pieces.

   **Concrete fix:** Supply phrase boundaries and intended echo pairs to the metrics, or derive them from the documented phrase layout. Assert motif-pair coverage as well as agreement. Add 3-, 5-, 7-, and 9-bar metric examples with known answers, including missing/unequal echo pairs. Keep undefined results distinct from successful structure.

9. **Minor — V1 is a pitch-proximity statistic, not evidence of satisfactory voice leading.**

   **Location:** Design §3 V1 and §9/D10; plan Task 1 `voice_leading_distance()` and Task 6 `test_v1_voice_leading_not_worse()`.

   **Problem:** Each destination pitch independently chooses its nearest source pitch. Several voices may reuse the same source, source voices can disappear without penalty, and crossings are not tracked. The statistic also ignores the actual thinned accompaniment. The unchanged canonical voicings therefore pass the guard even when the sounding harmony deteriorates. The spec uses 1.61 semitones to conclude that voice leading is already fine, which the metric does not establish. Deferring a voicing search is reasonable; this numerical justification is too strong.

   **Concrete fix:** Rename/document V1 as nearest-pitch proximity and stop treating it as a voice-leading quality proof. If retaining a voice-leading claim, add an order-preserving or one-to-one matching metric with explicit birth/drop costs and evaluate representative sounding voicings. At minimum include a hand-built counterexample with duplicated destination voices and clarify the limited purpose of the unchanged-shape guard.

10. **Minor — The stated five-second regression budget is not met in this environment.**

    **Location:** Design §7; plan global constraints and Task 6 Step 2.

    **Problem:** A standalone run constructing the exact planned matrix took **6.33 seconds**, and invoking all 23 regression assertions took **6.55 seconds total**, without pytest collection overhead. An earlier run took 5.4 seconds for setup. The claimed 2–3 second setup and sub-five-second total are not reproduced here. Repeated YAML parsing and A* work are real dependencies, not merely test-framework overhead.

    **Concrete fix:** Profile the fixture after removing the implicit config reads, compute reusable deterministic inputs once, and record timings on the pipeline environment. Preserve musical coverage while reducing unnecessary work. If the budget is revised, document a reasoned portable performance expectation; do not add a brittle wall-clock assertion to ordinary unit tests.

VERDICT: CHANGES_REQUESTED
