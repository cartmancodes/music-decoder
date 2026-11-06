Compose improvements code review — round 1

Reviewer: GPT-6 Astra (Codex). Reviewed `master...c5e9ef0` and `git log master..HEAD` against the final design, implementation plan, and all three earlier plan reviews. Completed the checks before the usage-limit interruption; on resuming on November 7, confirmed that HEAD and the source/test working tree were unchanged. No Critical or Important findings remain. The two Minor findings below do not invalidate the measured improvement or block approval.

Following the pre-approved unattended workflow, I made review decisions without requesting human input. No source or tests were edited, no packages or models were installed, and no branches were switched. Composition artifacts were written only under `/tmp/compose-review-r1/`. I created no scratch directories inside the worktree. The two pre-existing/generated synthetic-fixture marker changes are excluded from this commit.

1. **Minor — Cadence coverage can become negative when an accompaniment note lies entirely after the measured interval.**

   **Location:** `src/music_decoder/compose/metrics.py:381` and `src/music_decoder/compose/metrics.py:386`, `_covered_fraction()`.

   **Problem:** Clipping a non-overlapping future note produces a reversed span, but the union calculation does not discard it. For the interval `[0, 1)`, a note sounding from `0` to `0.95` correctly gives coverage `0.95`. Adding a disjoint note from `2` to `3` changes the result to approximately `-0.05`; the future note alone gives `-1.0`. This propagates through `cadence_identity()`: with a final melody C5 from `0` to `1`, a C-major shape, and accompaniment E3 from `0` to `0.95`, H2 is true; adding E3 from `2` to `3` incorrectly makes H2 false. A later note must not change coverage of an earlier interval.

   **Concrete fix:** Discard clipped spans with `end <= start` before merging them, and retain the existing overlap-union calculation for the remaining spans. Add hand-built tests for entirely preceding/following notes, partial coverage plus a future note, and the H2 example above. Verify that coverage stays in `[0, 1]`. This is Minor because the current composer's final ringing chord does not introduce these future attacks; the committed baseline and current evaluation results reproduce unchanged.

2. **Minor — The chord-quality extension instructions point to a removed table and omit the new harmony tables.**

   **Location:** `docs/architecture-technical.md:1312`.

   **Problem:** “Add a new chord quality” still tells maintainers to update `_CHORD_PCS` in `compose/melody.py`. That table has moved out of the melody module, and following these instructions omits the quality mappings now used by pitch selection, essential-tone preservation, and chord spelling.

   **Concrete fix:** Keep the `compose/voicings.py::_CHORD_PCS` instruction, replace the obsolete melody reference with `compose/harmony.py`, and list `CHORD_INTERVALS`, `ESSENTIAL_INTERVALS`, and `_DEGREE_STEPS`. Include harmony and compose coverage when documenting validation of an added quality.

The gate I ran on the reviewed HEAD completed as follows:

| Command | Observed result |
|---|---|
| `pytest -q tests/unit tests/integration` | **571 passed, 3 skipped, 60 warnings in 37.95 s** |
| `ruff check src tests` | **Passed** — all checks passed |
| `ruff format --check src tests` | **Passed** — 159 files already formatted |
| `mypy --strict src` | **Passed** — no issues in 80 source files |
| `git diff --check master...HEAD` | **Passed** |

The skips were one unavailable GuitarSet cache and two fixtures without tab ground truth. Warnings came from dependency compatibility/deprecations, synthetic rendering, and short-input spectral analysis; the gate had no failures. The actual executable environment differed from the operating-context description of a project venv, so I explicitly verified that Python imported `compose/api.py` from this worktree and that `PYTHONPATH` pointed at this worktree's `src`.

Additional verification:

- **Public API, actual synthesis:** Generated six cases, each twice, covering all six tuning presets and all three styles: Db major with strum/Eb tuning; F# minor with arpeggio/Drop C; a single Bbm chord with fingerstyle/DADGAD; E major with fingerstyle/Drop D and two bars per chord; a five-chord Eb-major progression with strum/D standard; and a 32-bar C-major arpeggio in standard tuning. Used seed 2026 and 137 BPM. MIDI files reopened with two valid tracks, WAV files decoded at 22,050 Hz with nonzero audio, and repeated calls produced identical melody records, tab strings, and MIDI bytes. Both MIDI tracks matched `arrange()` in pitch, velocity, and timing within MIDI quantization. Decoding every delivered melody tab recovered every assigned position and the corresponding melody pitch, including all 140 notes of the 32-bar case. This was structural/audio-data verification, not a subjective listening assessment.
- **CLI:** Ran actual `music-decoder compose` commands for Gb major/strum/Drop-D, Eb minor/arpeggio/Drop C, and single-chord Db major/fingerstyle/DADGAD at 93 BPM, seed 2222. Both pretty and JSON output paths succeeded. JSON retained the existing six top-level fields, referenced real MIDI/WAV files, normalized the flat spellings correctly, and included `composer_version: 2`.
- **Independent input coverage:** 5,184 short arrangements covered every supported root and quality, every style, all six presets, and seeds 0, 4, and 42. Another 6,000 deterministic random cases, selected with harness seed 30010, varied major/minor keys, arbitrary supported chord sequences of 1–12 chords, 1–3 bars per chord, all presets/styles, tempos 40/93/137/220, and composer seeds from 10000–99999. There were no exceptions or failures of octave-limit, final-resolution, complete-tab, H1, or H2 checks in either sweep. A custom six-string Open G tuning also produced a complete, pitch-correct tab.
- **Long pieces:** Arrangements and rendered tabs completed at 64, 256, and 1,024 bars, preserving respectively 272, 1,061, and 4,166 melody notes. Observed times were approximately 0.71, 9.45, and 156.29 seconds under concurrent load. Very long pieces are therefore substantially slower; the existing A* implementation repeatedly scans remaining groups, and the new thinning also scans the melody. This does not contradict the documented short regression-suite expectation, but these results should not be read as a claim of linear scaling. Shared A* optimization was explicitly deferred by the plan.
- **Determinism:** An Ab-minor arrangement had the same complete-arrangement digest in fresh processes using `PYTHONHASHSEED=1`, `37`, and `random`.
- **Independent baseline and test sensitivity:** Loaded the recorded pre-change melody, arrangement, and API modules from commit `69c2fa7` into memory, without checking out or editing files. Re-running the common evaluation reproduced the entire committed baseline exactly. The metric/harmony/rhythm/evaluation sources are unchanged since that commit. A fresh current report reproduced the documented results, including validation M1 `0.9802`, M2 `0.5543`, M3 `0.0948`, M11b `0.6946`, and H2 `1.0` in every style on both matrices. An in-memory repeat-weight sabotage changed M3 to `0.8084` and M2 to `0.1205`, violating their regression thresholds. The comparisons are not merely assertions that the implementation returns its own constants.

Earlier plan findings are resolved or convincingly scoped as follows:

| Earlier finding | Assessment of the implemented branch |
|---|---|
| R1 F1 / R2 F1: forced final leaps | Resolved by cadence-feasibility checks before accepting floors and target reachability during final-bar scoring; original, one-note-final-bar, and floor-coupled tests pass. |
| R2 F2: suspended-chord cadence target disappears | Resolved by letting the register floor yield; Dsus4 reproductions and every-quality/root coverage pass. |
| R1 F2 / R2 F3: altered-degree errors and tonic deletion | Resolved within the stated heuristic scope by degree spelling; transposed applied-dominant tests and exposed-note metrics are present and pass. |
| R1 F3: staggered accompaniment invisible to metrics | Resolved by interval-overlap checks with independent late-entry/non-overlap examples. |
| R1 F4 / R3 F1: lost chord identity at the sustained cadence | Resolved by counting the final overlapping melody note rather than every passing tone when restoring identity tones, plus distinct H2 checks. The original and reopened reproductions pass in all styles. Finding 1 above is a separate non-overlap edge case in the metric helper. |
| R1 F5: rendered melody overwritten | Resolved by separate melody/chord blocks and collision-free note columns; delivered tabs round-trip successfully. |
| R1 F6: configuration-dependent core | Resolved by the lower-level tab assigner with pinned parameters; config-loader and environment-independence tests pass. |
| R1 F7: prototype-calibrated evaluation | Addressed by the reproducible same-input baseline, disjoint tuning seeds, subgroup comparisons, musical threshold rationales, and demonstrated sensitivity. |
| R1 F8: phrase metrics use wrong boundaries | Resolved by shared layout and separate echo coverage/agreement, including irregular-length tests. |
| R1 F9 / R2 F4: overclaimed voice leading and shape identity | Convincingly narrowed: V1 is a proximity guard, H1 is shape-relative, and S1 exposes inherited canonical-shape defects. Shared shape repairs remain explicitly out of scope. |
| R1 F10: regression runtime | Addressed by the documented load-dependent expectation and absence of brittle timing assertions; the full gate completed in 37.95 seconds. |
| R2 F5: inadequate golden verification | The revised comparison covers complete records and MIDI bytes, and the previous review reproduced it. The independently regenerated baseline also matches exactly. |
| R3 F2: matrix/target documentation mismatch | Resolved: the spec distinguishes validation-only statistical targets from invariants/guards on both matrices and identifies 110 edge melodies. |

The public signature and CLI flags remain intact. The tab layout change and composer-version metadata follow the approved design. The implementation's extra fingerstyle-pinch repair is justified: it prevents dropping an alternate-bass string when it is the only essential tone in the pinch, is tested, and its small A3 change is disclosed. No new dependency, unexplained scope expansion, or remaining blocking correctness issue was found.

VERDICT: APPROVED
