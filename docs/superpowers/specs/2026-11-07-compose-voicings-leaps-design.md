# Compose fix: faithful chord voicings and singable melodic leaps

**Date:** 2026-11-07 (revision 1)
**Branch:** `fix/compose-voicings-leaps` (cut from `master` at `59d2d68`, composer v2)
**Status:** approved by the planner (Claude Opus 5.5) in an unattended Octet
pipeline. The human partner pre-approved the workflow (Opus plans, GPT-6 Astra
reviews, Opus implements). Every point where brainstorming would normally wait
for the user is listed in §10 with the decision taken and the reason.
**Plan:** `docs/superpowers/plans/2026-11-07-compose-voicings-leaps.md`.

## 1. Intent

The human partner listened to 20 samples from the v2 composer
(`music-decoder compose`) and flagged two defects.

1. **Faulty chord voicing.** In `minor-10-B` the C#dim chord is played with
   the library grip `x-x-2-3-2-x`, which sounds E–A#–C#. It has no G (the
   diminished fifth) and adds a foreign A#, so it sounds wrong. The v2 spec
   deferred repairing shared shape data (its metric S1 and §9). **That
   deferral is lifted.** Every shape the composer can play must be repaired:
   all roots, all qualities and every tuning path. The shape data is shared
   with the analysis UI, so every consumer has to be accounted for.
2. **Overly wide melodic leaps.** `major-08-Ab` has a minor-seventh leap
   G4→F5 at the Fm→Db bar line. `major-09-Eb` has a major sixth Ab4→F5 at
   Cm→Fm7, and `minor-10-B` a minor sixth G5→B4. v2 allows leaps up to an
   octave. Melodies must be singable:
   - leaps larger than a perfect fifth are rare and justified, and are
     recovered by a step in the opposite direction
   - sevenths and tritones are avoided
   - bar-line chord changes are reached by smooth motion

   None of this may regress the v2 musicality metrics.

**Hard constraints (from the request):**
- `compose()` and the CLI stay backward compatible: no new parameters or
  flags, and the same output kinds.
- Output stays deterministic by seed.
- No new dependencies.
- Every change is measurable. Every fix has a test that fails before it, and
  the three sample commands become regression tests.
- The gate stays green and the suite stays fast:
  `pytest -q tests/unit tests/integration`, `ruff check src tests`,
  `ruff format --check src tests` and `mypy --strict src`. Baseline on
  `master`: 582 passed, 3 skipped; ruff, ruff format and mypy clean.
- Out of scope unless needed: flat-key enharmonic spelling, long-piece
  performance.

**Success criteria:**
- No shape the composer can produce (96 chords × 6 preset tunings) misses a
  chord's identity tones, contains a foreign tone, has a non-root bass or
  collapses to fewer than three pitches.
- No melody contains a tritone, a seventh or a leap over an octave.
- Every leap larger than a fifth is justified and recovered (definitions in
  §4).
- Chord changes are approached more smoothly than in v2.
- Every v2 target, hard invariant and paired comparison with the pre-v2
  baseline still holds.
- Every v2 metric stays within a stated tolerance of v2 on the same inputs.

## 2. Reproduction (measured 2026-11-07 on `59d2d68`)

All numbers in this document come from throwaway scripts in the session
scratchpad that call `compose.api.arrange()` and
`evaluation.compose_musicality` directly. No source file was changed.

### 2.1 The three samples

| Sample | Command (abridged) | What v2 produces |
|---|---|---|
| `minor-10-B` | `--scale B:minor --progression "Bm G D A Em7 C#dim F#7 Bm" --style strum --tempo 124 --seed 210 --tuning EADGBE` | C#dim `x-x-2-3-2-x` = E3 A#3 C#4: G missing, A# foreign, bass E. Melody bar 3 (D chord): F#5 G5 **B4**, a minor sixth down from G5. Neither G5 nor B4 is a tone of D, and the leap is not followed by a step back. |
| `major-08-Ab` | `--scale Ab:major --progression "Ab C7 Fm Db Eb7 Ab" --style strum --tempo 110 --seed 108 --tuning Eb` | G4 → **F5** at Fm→Db (minor seventh, +10). **Eb7 sounds one note, D#2**, because in Eb tuning its retuned shape failed the span check and fell back to "root on the low strings". |
| `major-09-Eb` | `--scale Eb:major --progression "Eb Cm Fm7 Bbsus4 Bb7 Eb" --style fingerstyle --tempo 72 --seed 109 --tuning "D standard"` | G#4 → **F5** at Cm→Fm7 (major sixth, +9). Bb7 has its third D2 as the bass: in D standard the low open D was un-muted under the barre. |

Instrumenting the sampler shows that all three leaps were **tail draws** from
the softmax, not forced moves. Their probabilities were 0.3% (G4→F5), 0.8%
(G#4→F5) and 3.7% (G5→B4). The scorer penalises wide leaps but never
excludes them, so over many notes and pieces they occur.

### 2.2 Who consumes the shape data

`src/music_decoder/chords/voicings.py` (`VOICINGS`, `voicing_for`,
`canonical_voicings_for`) is read by:

| Consumer | Use | Effect of a shape repair |
|---|---|---|
| `compose/voicings.py` `voicings_for` (via `canonical_voicings_for`) | Every chord the composer plays: accompaniment pitches, melody floors (`api._melody_floor`), H1/H2/S1/V1 metrics, the chord-shape block of the ASCII tab (`compose/tabtext.py`). | Intended. Changes the output for the repaired chords, so the composer version is bumped (D3). |
| `ui/components/chord_progression.py` `render_chord_diagram_svg` (via `voicing_for`) | SVG chord diagram. No page calls it today (`streamlit_app.py` imports only `render_chord_timeline`). It is covered by `tests/unit/test_chord_progression_renderers.py` (C, Am, N). | A repaired shape draws correctly. Frets above 5 already render with a "+N" label, as for the existing `D#m` and `Bsus4` shapes. |
| `tests/unit/test_chord_detection_voicings.py` | Pins the C, Am, G, G7 and Cmaj7 shapes, checks coverage and the fret range. | Unaffected; none of these shapes change. |

Chord *recognition* (`chords/templates.py`, `recognize.py`, the madmom
backends) uses pitch-class templates, not shapes, so `analyze()` results do
not change. The tab assigner (`tabs/`) never reads `VOICINGS`.

## 3. Shape audit

### 3.1 Method

Every one of the 96 chords (12 roots × 8 qualities) was voiced by the
composer's own path, `compose.voicings.voicings_for(chord, tuning)`, in all 6
preset tunings, giving 576 voicings. Each was checked against one definition
of a **faithful shape**:

1. **At least three sounding strings.**
2. **No foreign tone:** every sounding pitch class is a chord tone.
3. **Root in the bass:** the lowest sounding pitch is the chord root.
4. **Complete:** every chord tone is present, except that a four-note chord
   (7, maj7, min7) may omit its perfect fifth, as the standard open C7
   `x32310` does. For all 8 qualities this implies every essential tone
   (`harmony.ESSENTIAL_INTERVALS`) is present.

### 3.2 Library shapes (EADGBE, the data in `chords/voicings.py`)

| Chord | v2 grip | Sounds | Faults | Repair | Sounds after |
|---|---|---|---|---|---|
| C#dim | `x-x-2-3-2-x` | E3 A#3 C#4 | G missing, A# foreign, bass E | `x-4-5-6-5-x` (the A-string dim shape of Cdim/Bdim, up one fret) | C#3 G3 C#4 E4 |
| Ddim | `x-x-0-1-0-1` | D3 G#3 B3 F4 | B foreign (a dim7 tone) | `x-x-0-1-3-1` (standard open Ddim) | D3 G#3 D4 F4 |
| D#dim | `x-x-1-2-1-2` | D#3 A3 C4 F#4 | C foreign (a dim7 tone) | `x-x-1-2-4-2` (the Ddim shape up one fret) | D#3 A3 D#4 F#4 |
| Fmaj7 | `x-3-3-2-1-0` | C3 F3 A3 C4 E4 | bass C (second inversion) | `x-x-3-2-1-0` (standard Fmaj7) | F3 A3 C4 E4 |
| G#aug | `x-x-2-1-1-0` | E3 G#3 C4 E4 | bass E (it is the Eaug grip) | `x-x-6-5-5-4` | G#3 C4 E4 G#4 |

The other 91 shapes are faithful. C7, C#7 and D#7 omit the fifth, which the
definition allows. Every repaired shape keeps a fretted span of at most 2
frets.

### 3.3 Tuning paths (what the composer actually plays)

`compose/voicings.py` makes a shape for another tuning in three steps:

1. **Retune.** Shift every fretted string so it sounds the **same pitch**
   (in Eb and D standard every fret moves up by 1 or 2). Then un-mute any
   muted string whose open note is a chord tone.
2. **Span gate.** Reject the shape if `max - min > 5` over all non-muted
   frets, **open strings included**.
3. **Fallback.** If rejected, play the root pitch class on whichever strings
   reach it at frets 0–12.

Three faults in that path account for most of the defects:
- **F1.** Un-muting a low open string puts a non-root chord tone in the bass:
  D2 under Bm in Drop D, DADGAD and D standard; D#2 under B in Eb.
- **F2.** The span gate counts that open string (fret 0) against a barre at
  fret 4–9, so the span exceeds 5 and the shape is rejected.
- **F3.** The fallback "chord" is a single root, 1–2 notes with no third.

Counts of the 576 v2 voicings (a voicing can have several faults):

| Tuning | Any fault | Missing essential | Foreign tone | Non-root bass | Incomplete triad | < 3 distinct pitches |
|---|---|---|---|---|---|---|
| EADGBE | 5 | 1 | 3 | 3 | 1 | 0 |
| Drop D | 17 | 3 | 3 | 14 | 2 | 2 |
| Eb | 22 | 14 | 3 | 8 | 10 | 13 |
| D standard | 18 | 9 | 3 | 7 | 7 | 9 |
| Drop C | 19 | 8 | 3 | 9 | 6 | 8 |
| DADGAD | 17 | 3 | 3 | 14 | 2 | 2 |
| **Total** | **98 / 576** | 38 | 18 | 55 | 28 | 34 |

Examples:
- In Eb, D#, D#m, D#7, D#maj7, D#sus4, D#aug, Cm, Cm7 and Cdim each sound
  one or two notes.
- In D standard, Bm, Bm7, Bdim and F#dim each sound one or two notes.

Measured in the composer itself (the tunings matrix of §5.5: the three
flagged progressions × 6 tunings × 5 seeds), **105 of the 600 shapes v2
played (17.5%) were faulty**.

### 3.4 After the fix (prototype, §7)

- **0 / 576** voicings have any fault.
- EADGBE output changes only for the five repaired chords.
- In the five other tunings, 465 of the 480 voicings are the retuned
  library shape. The other **15** need the new shape search: Drop D G7;
  Drop C G, G7 and Gmaj7; and DADGAD A#, B, A#7, A#maj7, Bmaj7, Ddim,
  D#dim, A#sus4, Bsus4, A#aug and Baug.
- The shape search over all 576 takes about 20 ms in total.

## 4. Leap audit and metric definitions

### 4.1 Interval histogram, v2 (counts of melodic intervals, by size in semitones)

| Matrix | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | > 5th total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Validation (400 melodies, 9040 intervals) | 867 | 1371 | 3739 | 1484 | 691 | 671 | **4** | 173 | 20 | 19 | 0 | 0 | 1 | **40** (0.44%) in 39 pieces |
| Edge (110 melodies, 1718 intervals) | 160 | 281 | 692 | 247 | 143 | 129 | **1** | 56 | 3 | 2 | **2** | 0 | 2 | **9** (0.52%) in 9 pieces |
| Sample sweeps (each sample command, seeds 0–299) | | | | | | | | | | | | | | 0.9–1.4% of intervals; 58–79 of 300 pieces have one, and 9–22 of 300 have a tritone or seventh |

### 4.2 Where the leaps larger than a fifth occur (v2, validation; edge in brackets)

| Context | Count | Cause |
|---|---|---|
| **Into the final note** | 16 [4] | The cadence target is fixed at the start of the final bar, from the note before that bar. The bar's other notes then drift (`CADENCE_PULL` is only 0.25 per semitone), or the chord floor lifts the target an octave. Examples: G7's F4 gives floor G#4, so the piece ends on G5 after B4; a one-note final D bar has floor A4 and ends on D5 after F#4. |
| At a chord-change bar line | 10 [4] | Tail draws (the samples' case). |
| Within a bar | 12 [1] | Tail draws. 25 of the 40 move between two tones of the same chord (arpeggiation); the rest touch a non-chord tone. |
| Bar line, same chord | 2 [0] | Tail draws. |

Recovery of the 24 v2 leaps larger than a fifth that are followed by another
note:
- 14 are followed by a step back
- 6 by a skip back
- 4 continue in the same direction

At chord changes, v2 moves by at most a major third 84.4% of the time
(validation) and 82.7% (edge). Its largest move at a chord change is 12.

### 4.3 New metrics (pure functions in `compose/metrics.py`; all on the melody)

| ID | Name | Definition |
|---|---|---|
| **L1** | Forbidden intervals | Count of melodic intervals of 6, 10 or 11 semitones, or more than 12. |
| **L2** | Large-leap recovery (pooled) | Over leaps of **more than 7** semitones that are followed by another note: the share whose next interval moves the opposite way by 1–2 semitones. This is M5 with a "larger than a fifth" threshold. The count is `n`. Pooled like M5; `-1.0` when `n = 0`. |
| **L3** | Unjustified large leaps | Count of leaps of more than 7 semitones where either note is not a chord tone of the chord sounding under it, the two notes sound over different chords, or the leap lands on the final note. |
| **L4** | Smooth chord changes | Over consecutive melody notes that sound over different chords (`chord_at`), the share that move by at most 4 semitones. `None` when the piece has no chord change. |
| **L4max** | Largest chord-change move | The maximum of those moves; `None` when there are none. |
| **L5** | Final approach | `abs` of the interval into the last note; `None` for a one-note melody. |
| **S2** | Non-root bass | The share of a piece's shapes whose lowest sounding pitch is not the chord root. This complements S1. |

Existing **S1** (missing essential / foreign tone) and **M4 / M4max**
(share and size of leaps larger than a fifth) are kept unchanged.

## 5. Design

### 5.0 Approaches considered

**Voicings:**
- **(a) Repair only the five library shapes.**
  - Fixes the flagged C#dim.
  - Leaves 93 of the 98 faulty voicings (the tuning paths, including the
    one-note Eb7 in a flagged sample).
  - **Rejected.**
- **(b) Replace retuning with a full shape search for every tuning.**
  - Changes every EADGBE chord, so V1 and every floor move.
  - Discards the hand-curated library.
  - **Rejected** (larger blast radius, no musical gain).
- **(c) Chosen:**
  - repair the five library shapes
  - fix the three retune faults
  - gate every voicing on the §3.1 faithfulness definition
  - when the retuned library shape fails that gate, use a small
    deterministic shape search
  - keep the v2 root fallback only as a last resort, for custom tunings

**Melody:**
- **(a) Re-weight the softmax** (make 8–12 more negative).
  - Still samples tails.
  - Cannot guarantee "no sevenths" or "always recovered".
  - Cannot fix the final-note leaps, which are forced, not sampled.
  - **Rejected.**
- **(b) Truncated sampling** (top-p or min-p).
  - Removes unlikely picks of every kind, which changes every metric.
  - Still cannot see the forced final note.
  - **Rejected** for this scope.
- **(c) Chosen:** hard admissibility rules applied before sampling, plus
  lookahead for the final approach and a soft bonus for steps into a new
  chord.
  - If no candidate is admissible, the note falls back to the v2 scoring,
    so no input the composer accepted before can fail.
  - That fallback was never used by the prototype on 33,682 sampled notes:
    9,040 validation, 1,718 edge and 22,924 in the sample sweeps.

### 5.1 Library repair: `chords/voicings.py`

- Replace the five entries of §3.2.
- Add one module-docstring paragraph stating the faithfulness invariant of
  §3.1.

No other entry changes, and neither do the function signatures.

### 5.2 Tuning path: `compose/voicings.py`

The module keeps `voicings_for(chord, tuning) -> list[VoicedChord]`, and its
first element is what the composer plays. The changes:

1. **One source of chord tones.** Delete the duplicate `_CHORD_PCS` table
   and use `harmony.chord_pcs` and `harmony.essential_pcs`.
   `test_compose_harmony.py::test_every_chord_quality_has_a_row_in_each_compose_table`
   drops its `voicings._CHORD_PCS` entry, and
   `docs/architecture-technical.md` "Add a new chord quality" drops that
   step.
2. **`shape_faults(frets, tuning, chord) -> tuple[str, ...]`** (public).
   - Returns the §3.1 faults, in a fixed order, as labels from
     `"too few strings"`, `"foreign tone"`, `"root not in bass"` and
     `"missing tone"`.
   - An empty result means the shape is faithful.
   - Tests and the sample regression tests use it.
3. **Retune** (`_retune(frets, tuning, chord) -> tuple[int, ...] | None`):
   - Shift each fretted string so it sounds the same pitch (unchanged).
   - Return `None` if any shifted fret would be below 0. That is only
     possible for custom tunings above standard. v2 silently turned such a
     string into a muted one.
   - Un-mute a muted string only if its open pitch class is a chord tone
     **and** it is either the root or higher than the shape's lowest
     sounding pitch. This fixes F1. "Lowest sounding pitch" means the
     shifted shape **before any un-muting**: in Drop D, D keeps its A2
     muted even after the low D2 is added.
   - When the target tuning equals EADGBE, return the shape unchanged
     (v2's early return). Otherwise the un-mute rule would alter Edim, Fdim
     and G#dim in standard tuning. The Drop D low D under a D chord and
     the Drop C low C under a C chord are still added.
4. **Playability** (`_is_playable`):
   - at least one sounding string
   - every fret ≤ 22
   - the span of **fretted** strings (fret > 0) is at most
     `MAX_FRETTED_SPAN = 3`, a four-fret hand frame, one finger per fret

   Open strings no longer count (this fixes F2). All 96 library shapes
   already satisfy the rule (their largest fretted span is 3), so EADGBE is
   unaffected.
5. **Shape search** (`search_voicing(chord, tuning)`, `functools.lru_cache`,
   both arguments are frozen dataclasses):
   - **Positions.** For each base fret 0–9 the window is the open string
     plus frets `base … base+3`. The bass is on string 0, 1 or 2 and must
     be the root. Strings below it are muted. Every higher string is
     either muted or plays a chord tone inside the window.
   - **Accepted only if all of these hold:**
     - at least 4 sounding strings, contiguous (no muted string between two
       sounding ones)
     - fretted span ≤ 3
     - at most 3 fretted notes above the lowest fretted fret (that fret
       can be barred)
     - faithful by §3.1
   - **Choice.** The smallest key wins: (omitted chord tones, lowest
     fretted fret or 0, minus the number of sounding strings, the fret
     tuple). The last element makes the choice deterministic.
6. **`voicings_for`:**
   - Retune the canonical shape.
   - If the result is playable and faithful, return it.
   - Otherwise return the search result.
   - If the search finds nothing (only possible with an exotic custom
     `Tuning`; the test uses six strings all tuned to C2), return the v2
     root fallback. It never raises where v2 did not.
   - The `InvalidProgressionError` for a chord missing from the library is
     unchanged.

### 5.3 Melody: `compose/melody.py`

New constants:
- `SMOOTH_LIMIT = 7`: the largest interval allowed without justification,
  a perfect fifth.
- `FORBIDDEN_INTERVALS = frozenset({6, 10, 11})`.
- `CHORD_CHANGE_STEP_BONUS = 1.0`.

`MAX_INTERVAL = 12` and `INTERVAL_WEIGHT` are unchanged. The weights still
govern the relaxed fallback.

New `NoteContext` fields. All have defaults, so existing callers and tests
keep working.

| Field | Meaning |
|---|---|
| `chord_change: bool = False` | This note is the first over a new chord. |
| `prev_chord_tone: bool = False` | The previous pitch is a tone of this note's chord. |
| `next_fixed: int \| None = None` | The pitch the next note is forced to: the final target, for the penultimate note of the final bar. |
| `approach: tuple[int, ...] \| None = None` | For the last note before a **one-note** final bar, the cadence pitches it must be able to reach. |
| `recovery_pool: frozenset[int] \| None = None` | The next note's candidate pitches, set only when the next note is in the same bar and has no `next_fixed` or `approach` constraint. `None` forbids a large leap here. |

New pure function **`admissible(candidate, ctx) -> bool`**. It is true
unless one of these rules fails:

- **R1 Forbidden intervals.** The interval from the previous note is not in
  `FORBIDDEN_INTERVALS` and is at most `MAX_INTERVAL`.
- **R2 Recovery.** If the previous interval was larger than `SMOOTH_LIMIT`,
  this note moves 1–2 semitones in the opposite direction.
- **R3 Justified large leaps.** An interval larger than `SMOOTH_LIMIT` is
  allowed only if all of these hold:
  - this is not a chord change
  - this is not the final bar (`target is None`)
  - neither `next_fixed` nor `approach` is set
  - the previous pitch and the candidate are both tones of this chord
    (arpeggiation)
  - the previous interval was a step or a repeat (at most 2), so leaps
    never chain
  - `recovery_pool` contains a pitch 1–2 semitones back toward where the
    leap came from

  As a consequence, **every chord change moves by at most a fifth**.
- **R4 Final approach.**
  - With `next_fixed`, the candidate is within `SMOOTH_LIMIT` of it and not
    a tritone away.
  - With `approach`, some pitch in it is within `SMOOTH_LIMIT` of the
    candidate and not a tritone away.

`score_pitch` gains one term: `+CHORD_CHANGE_STEP_BONUS` when `chord_change`
and the move is 1–2 semitones. Its other terms and `-inf` rules are
unchanged.

**`generate_melody` changes:**

1. **Final approach set.** Before the loop, compute the cadence pitches of
   the final chord (`cadence_pc`) inside `[pitch_low, pitch_high]`. Keep
   only those at or above the final chord's floor, unless none remain. The
   result is `final_approach`. Aiming at the floor lets the floor hold
   instead of yielding: without this step the prototype dropped one
   validation cadence note under the D chord's F#4, so A2 fell to 0.9998.
2. **Per slot,** fill in the new context fields:
   - `chord_change`: the previous note's chord differs.
   - `prev_chord_tone`.
   - `next_fixed`: the target, when this is the penultimate slot of the
     final bar.
   - `approach`: `final_approach`, when this is the last slot of the
     penultimate bar and the final bar has one slot.
   - `recovery_pool`: as defined in the table above.
3. **Scoring.** `scores = [score_pitch(c) if admissible(c) else -inf]`. If
   no score is finite, fall back to the v2 scores (`score_pitch` only).
   This "relaxed" path keeps every v2 input working; it was never taken in
   the prototype.
4. **`_candidates`** gains `cadence_reach: int = MAX_INTERVAL`.
   - In the final bar the floor yields unless some cadence pitch at or
     above it is reachable from the previous note within `cadence_reach`
     and not a tritone away.
   - `generate_melody` passes `SMOOTH_LIMIT` for a one-note final bar and
     `MAX_INTERVAL` otherwise.
   - The other floor-yield conditions are unchanged (D17 of v2; extended
     in D9).

**Determinism.** The rhythm plan is untouched, and every sampled note still
makes exactly one `rng.choice`, so M8, M9, M10, M12 and M14 are identical to
v2. Pitches change, so `COMPOSER_VERSION` becomes **3** (D3).

### 5.4 Metrics: `compose/metrics.py`

Add the §4.3 functions:
- `forbidden_interval_count(melody) -> int`
- `large_leap_recovery_counts(melody) -> tuple[int, int]`
- `unjustified_large_leap_count(melody, progression, bars_per_chord, tempo_bpm) -> int`
- `chord_change_moves(melody, progression, bars_per_chord, tempo_bpm) -> list[int]`
- `bass_is_root(voicing, tuning) -> bool`

L4, L4max and L5 are derived in `case_metrics`. The module docstring's spec
reference gains this document.

### 5.5 Evaluation: `evaluation/compose_musicality.py`, script and fixtures

- **Case carries its tuning.** `Case` gains `tuning: Tuning` and
  `build_cases(inputs, tuning=STANDARD_EADGBE)` uses it for arrange,
  accompany and every tuning-dependent metric. Today these are hard-coded to
  `STANDARD_EADGBE`.
- **New "tunings" matrix.** It holds the three flagged progressions, each in
  its own key: B minor `Bm G D A Em7 C#dim F#7 Bm`, Ab major
  `Ab C7 Fm Db Eb7 Ab` and Eb major `Eb Cm Fm7 Bbsus4 Bb7 Eb`.
  - Each runs × all 6 `PRESETS` × seeds 1000–1004, one bar per chord: 90
    melodies × 3 styles.
  - `evaluate()` returns `validation`, `edge` and `tunings`.
  - Every hard invariant of v2's `test_hard_invariants` applies to this
    matrix: M4max, M6, M7, M8, M12, M14, Tcount, A1, A4, H1 and H2. So do
    the new hard targets: L1, L2, L3, L4max, L5, S1 and S2. Statistical
    targets do not (D7).
  - **T1 is capped per tuning:** `T1 ≤ 15 + (64 − open pitch of the top
    string)`. That gives 15 for EADGBE and Drop D, 16 for Eb, and 17 for
    D standard, Drop C and DADGAD.
    - The melody window tops out at G5 whatever the tuning, and v2 already
      measured exactly these maxima (15/15/16/17/17/17) on this matrix.
    - The melody tab is out of scope (§9), so this restates v2's behaviour
      rather than relaxing it.
    - `summarize` adds one group per tuning (`tuning.<name>`) when a matrix
      has more than one tuning, so the cap can be checked per group.
- **`case_metrics` adds:** `L1`, `L2.leaps`, `L2.recovered`, `L3`, `L4`,
  `L4max`, `L5` and `S2`. `_aggregate` pools `L2` like M5 and M15.
- **New fixture `tests/fixtures/compose/musicality_v2.json`.** The script
  writes it from the **unchanged v2 composer** with the new metric code, as
  a committed plan task before any composer change. Its `source` is that
  commit.
  - It is the reference for "no regression versus v2".
  - The pre-v2 `musicality_baseline.json` (source `69c2fa7`) stays as it
    is, and every v2 paired comparison against it keeps running.
- **Runtime.** The `evaluate()` fixture grows from about 2.8 s to an
  estimated 3.5 s.

### 5.6 Regression tests for the three samples

`tests/unit/test_compose_samples.py` runs each human-flagged command
**literally through the CLI**:
- `click.testing.CliRunner` with the exact flags plus
  `--out <tmp> --format json`.
- `render_wav` is stubbed by monkeypatching the attribute on the module
  object `importlib.import_module("music_decoder.compose.api")`. Note that
  `music_decoder.compose` as an attribute is the re-exported function, not
  the package.
- Each test parses `chord_voicings` and `melody_notes` from the JSON.

Assertions:

| Group | What is checked |
|---|---|
| **Voicing** | Every voicing has `shape_faults == ()`.<br>`minor-10-B`: C#dim sounds {C#, E, G}.<br>`major-08-Ab`: Eb7 sounds ≥ 4 pitches including G and C#.<br>`major-09-Eb`: Bb7's bass is Bb. |
| **Melody** | L1 = 0, L3 = 0, L4max ≤ 7 and L5 ≤ 7 and not 6.<br>Every leap larger than a fifth is recovered.<br>`metadata.composer_version == 3`. |

On v2 the groups fail as follows:
- **Voicing:** minor-10-B (C#dim), major-08-Ab (Eb7 is a single note) and
  major-09-Eb (Bb7's bass is D).
- **Melody:**
  - minor-10-B: the 8-semitone leap G5→B4 has no chord tone at either end
    (L3).
  - major-08-Ab: the 10-semitone leap (L1, L3, L4max).
  - major-09-Eb: the 9-semitone leap at a chord change (L3, L4max).

Each group is added first with `pytest.mark.xfail(strict=True)`. The task
that fixes it removes the marker, and `strict=True` makes a forgotten marker
fail (D11).

### 5.7 Task order (measured)

The voicing fixes land before the melody fix. The test was a build with only
the voicing changes and the v2 melody code. In it all three flagged leaps are
still present (G5→B4, G4→F5, G#4→F5): the repaired chords sit after the
leaps in each sample, so the earlier bars draw identically. The melody
`xfail(strict=True)` markers therefore stay correct until the melody task
removes them. If an implementation ever finds otherwise, it must stop and
report rather than reorder silently.

## 6. Targets

**Where the targets apply.**
- *Validation* is the 400-melody held-out matrix; statistical targets are
  checked here.
- *Hard* targets are checked per composition, as a min/max over every
  matrix listed.
- *vs v2* compares with `musicality_v2.json` on identical inputs.

### 6.1 Leap targets

| ID | v2 (val / edge / tunings) | Prototype (val / edge / tunings) | **Target** | Why this threshold |
|---|---|---|---|---|
| L1 | 4 / 3 / 3 intervals (max 1 per piece) | 0 / 0 / 0 | **max = 0, hard, all three matrices** | The request says tritones and sevenths are avoided. The prototype never needed the relaxed path, so zero is achievable rather than aspirational. |
| L3 | 30 / 8 / 9 unjustified (of 40 / 9 / 9 large leaps) | 0 / 0 / 0 | **max = 0, hard, all three** | "Justified" is defined operationally (§4.3). Only arpeggiation inside one chord qualifies. |
| L2 | 14 / 24 = 0.58 · 4 / 5 · n = 0 (all 9 were final notes) | 3 / 3 = 1.0 · n = 0 · n = 0 | **pooled = 1.0 when n > 0, hard, all three** | Every large leap is recovered by a contrary step, as the request asks. |
| M4 (mean share > 5th) | 0.0040 / 0.0064 | 0.0002 / 0.0 | **≤ 0.001 on validation and edge**, and < v2 | "Rare": at most one leap per thousand intervals, a 4× (validation) and 6× (edge) cut. The 5× headroom over the prototype absorbs implementation noise. Pieces with any such leap fall from 39 / 400 to 3 / 400. |
| M4max | 12 / 12 | 8 / 7 | **≤ 12, hard** (unchanged) | Octaves stay legal when justified. The rules, not the cap, keep them rare. |
| L4 (chord change ≤ a third; per-piece mean) | 0.852 / 0.806 / 0.903 | 0.885 / 0.848 / 0.935 | **mean ≥ 0.87 on validation, ≥ 0.83 on edge, and > v2 on every matrix** | "Smooth motion" into a new chord. Each threshold sits above v2 on its matrix and about 0.015 under the prototype. (Pooled over all chord changes, the shares are 0.844 → 0.878 on validation and 0.827 → 0.878 on edge.) |
| L4max | max 12 / 12 / 7 | max 7 / 7 / 5 | **≤ 7, hard, all three** | Follows from R3: no leap larger than a fifth at a chord change. |
| L5 | max 9 / 10 / 9 | max 7 / 7 / 7 | **≤ 7, hard, all three** (the tritone is caught by L1) | The forced final note is approached smoothly. |

### 6.2 Voicing targets

| ID | v2 | Prototype | **Target** | Why |
|---|---|---|---|---|
| Library audit (unit test) | 5 / 96 faulty | 0 | **0 / 96** | The shared data is correct for every consumer. |
| Composer voicing audit (unit test, 96 chords × 6 tunings) | 98 / 576 faulty | 0 | **0 / 576** | Covers every root, quality and tuning path. |
| S1.missing / S1.foreign (mean) | val 0 / 0.0063; tunings 0.088 / 0.042 | 0 / 0 everywhere | **max = 0, hard, all three matrices** | Replaces the v2 guard "S1 = baseline". |
| S2 (mean) | val and edge 0; tunings 0.107 (max 0.375) | 0 | **max = 0, hard, all three** | Root position, as the accompaniment assumes: the arpeggio and fingerstyle bass is `pitches[0]`. |
| H1, H2, A4 | 1.0 | 1.0 (tunings matrix too) | **min = 1.0, hard, all three** | Unchanged invariants, now also asserted in every tuning. |
| V1 | 1.621 / 1.620 | 1.627 / 1.620 | **≤ v2 + 0.02** | V1 moves only because Ddim changed (it appears as iidim in C minor). It remains a proximity guard, not a voice-leading score. |

### 6.3 No regression versus v2 (and every v2 target kept)

- **Every v2 absolute target, floor, pooled threshold and hard invariant**
  in `test_compose_musicality.py` is kept, and every paired comparison
  against the pre-v2 baseline keeps running.
  - The prototype passes all of them.
  - The only change is replacing the two guards of §6.2: S1 equal to the
    baseline, and V1 at or below it.
- **Paired versus v2, every subgroup** (overall, major, minor, bars1, bars2
  of validation; edge overall):
  - Every v2 "IMPROVE_UP" mean is ≥ v2 − 0.02.
  - Every "IMPROVE_DOWN" mean is ≤ v2 + 0.02.
  - The pooled M5 and M15 are ≥ v2 − 0.05 and still meet their v2
    thresholds.
- **Why 0.02.** The worst prototype deviation in any subgroup is 0.013 (M1
  in edge minor, 40 melodies: 0.986 → 0.973). Its causes:
  - the hard rules occasionally remove the only chord tone reachable by a
    fifth on a downbeat
  - the step bonus slightly favours a non-chord-tone step

  Every M1 still exceeds 0.97, against a target of 0.95. The 0.05 for the
  pooled metrics covers their small `n` (M15: 13–40).

| Metric (validation overall) | v2 | Prototype |
|---|---|---|
| M1 | 0.980 | 0.977 |
| M2 | 0.554 | 0.568 |
| M3 | 0.095 | 0.093 |
| M5 (pooled) | 0.761 (n = 771) | 0.765 (n = 667) |
| M11b | 0.695 | 0.704 |
| M13 | 0.024 | 0.028 |
| M15 (pooled) | 0.825 (n = 40) | 0.853 (n = 34) |
| T2 | 0.124 | 0.122 |
| A2 (every style) | 1.0 | 1.0 |
| A3 arpeggio / strum / fingerstyle | 0.062 / 0.087 / 0.069 | 0.058 / 0.085 / 0.067 |
| M8, M9, M10, M12, M14, A1, A4, H1, H2 | — | identical to v2 |

On the edge matrix:
- A2 rises from 0.987 to 0.988 in every style.
- M13 goes from 0.0196 to 0.0206.
- M11b goes from 0.774 to 0.768.
- M1 goes from 0.991 to 0.984.

## 7. Prototype

All the numbers in §3.4, §6 and §2 come from a throwaway prototype. It was a
copy of `src/` in the session scratchpad with the §5 changes applied, run
with `PYTHONPATH` pointed at it. Nothing in the worktree was modified.

Against the existing suite (compose, voicing, renderer, public-contract and
compose e2e tests) it gave 285 passed and 3 failed, all of them expected:

1. `test_compose_melody.py::test_one_note_final_bar_stays_within_an_octave_when_a_floor_bites`.
   Its "the adversarial low start really occurred" precondition no longer
   occurs, because the approach rule now steers bar 1 toward G5. The plan
   rewrites it as follows:
   - the sweep asserts the final move is at most a fifth
   - the floor-yield branch is covered by a direct `_candidates` test with
     `cadence_reach`
2. `test_compose_metrics.py::test_shape_tone_mismatch_reports_inherited_shape_defects`
   pins the old C#dim defect. The plan rewrites it on a hand-built defective
   shape.
3. `test_compose_musicality.py::test_guards_do_not_regress` pins S1 and V1
   to the baseline. The plan replaces it as in §6.2.

`test_compose_harmony` needs its `_CHORD_PCS` row removed (§5.2).

## 8. Error handling and edge cases

- **No admissible candidate.** The note uses the v2 scores (§5.3). This was
  never observed, and it cannot reject an input the composer accepted
  before.
- **`chord_floors=None`** (direct `generate_melody` callers). The approach
  set is every in-window cadence pitch.
- **Legacy grid (`notes_per_bar=N`).** The same rules apply. With
  `notes_per_bar=1` every bar has one slot, so `recovery_pool` is always
  `None` and large leaps never happen.
- **One-chord piece.** There are no chord changes, so L4 and L4max are
  `None` (skipped). There is no penultimate bar; with one bar the final bar
  is also the first.
- **Custom tunings.**
  - A shift below fret 0, or no faithful retuned shape, falls through to the
    search.
  - If the search also fails, the v2 root fallback is used (§5.2). This is
    tested with a six-C2 tuning.
- **Determinism.** Same seed, same version, same output. The existing
  `test_same_seed_same_music` is kept, and a CLI repeat check is added to
  the sample tests.

## 9. Non-goals

- No voicing optimisation between chords (voice leading) and no Viterbi
  shape selection. V1 stays a guard.
- No finger-count or difficulty model for the library shapes beyond the span
  rule. The search's finger heuristic only filters its own candidates.
- No change to chord recognition, `analyze()` or the SVG diagram renderer
  (including its "+N" display of frets above 5). Flat-key spelling is
  unchanged: sharps, as before.
- No new `compose()` parameters, CLI flags or tunings.
- No change to rhythm, phrase layout, accompaniment patterns, thinning,
  floors (apart from the final-bar reach rule) or the melody tab.
- No performance work on long pieces.
- The v2 spec and the pre-v2 baseline are historical records and are not
  edited.

## 10. Decisions made in place of user approval

| # | Question | Decision | Reason |
|---|---|---|---|
| D1 | Scope of "every library shape" | **Every voicing the composer can emit: 96 chords × 6 preset tunings**, plus the raw library | The flagged sample in Eb tuning plays a one-note Eb7. Fixing only `VOICINGS` would leave 93 of 98 faults. |
| D2 | What "faulty" means | **§3.1:** ≥ 3 strings; no foreign tone; root in the bass; complete except an optional fifth in four-note chords | It matches the essential-tone model v2 already uses. Root position is what the accompaniment's "bass = lowest pitch" logic assumes. Omitting the fifth of a seventh chord is standard guitar practice (open C7). |
| D3 | May seeds change? | **Yes; `COMPOSER_VERSION = 3`** | The fixes change pitches. Determinism holds within a version, the same policy as v2 D3. |
| D4 | Melody mechanism | **Hard admissibility rules + v2 relaxed fallback + step bonus at chord changes** (§5.0) | Only hard rules can guarantee "no sevenths or tritones" and "always recovered". The fallback keeps v2's accept-everything behaviour. |
| D5 | Are sixths and octaves still allowed? | **Yes, only as arpeggiation inside one chord, approached by step, never at a chord change or in the final bar, with a guaranteed contrary-step recovery** | The request says "rare and justified", not "never". These are the idiomatic cases. |
| D6 | Chord-change smoothness | **Hard ≤ a fifth (via R3); soft bonus 1.0 for steps; targets L4 ≥ 0.87 (validation) and ≥ 0.83 (edge)** | Bonus 0.5 and 1.5 and a repeat-inclusive variant were measured. The repeat-inclusive bonus raised M3 (more repeated notes); 1.5 cost more M1 for little extra L4. |
| D7 | Tunings coverage in the evaluation | **A 90-melody "tunings" matrix with hard invariants only** | It keeps the statistical targets on the held-out matrices. The other tunings are exercised through the composer, not just through `voicings_for`. |
| D8 | Reference for "no regression" | **A new committed `musicality_v2.json` written from the unchanged v2 code**, tolerance 0.02 (means) and 0.05 (pooled) | Paired, identical inputs. Committed before any change, like v2's baseline. The tolerances are justified in §6.3. |
| D9 | Floor versus a smooth final approach | **Aim the approach at cadence pitches above the final floor; the floor yields only if none is reachable within a fifth (one-note final bar)** | v2 precedence was octave limit > cadence > register clearance. "Approached within a fifth" is inserted above register clearance. Aiming at the floor first means it almost never has to yield: A2 stays 1.0 on validation. |
| D10 | Span rule | **Fretted span ≤ 3 (open strings excluded)** | Open strings need no finger. The four-fret frame matches the library, whose largest span is 3, and sends the awkward retunes to the search (for example Drop D G7 at frets 1–5). |
| D11 | How "tests fail before the fix" is kept across several commits | **Sample tests land first as `xfail(strict=True)`; each fixing task removes its marker** | Every commit stays green, the failure is demonstrated, and a forgotten marker fails the suite. |
| D12 | CLI or API for the sample tests | **CLI via `CliRunner`, WAV rendering stubbed** | The human reported CLI commands. Stubbing synthesis keeps the suite fast; MIDI is still written. |
| D13 | Shape choices for the five repairs | **The standard grips in §3.2, chosen to match neighbouring library families** (A-string dim, D-string dim, xx3210, xx6554) | Consistency with the existing families. All are root position with spans of at most 2. |
| D14 | Duplicate chord-tone table in `compose/voicings.py` | **Remove it and use `harmony`** | One source of truth. A test already forces the tables to agree. |
| D15 | Tab fret cap in other tunings | **T1 ≤ 15 + (64 − top-string open pitch), per tuning, on the tunings matrix only** | The builder check showed a flat 15 cannot hold once a tuning's top string is lower. v2 measured exactly these maxima, and the tab is out of scope. |

## 11. Builder check (fresh reviewer, 2026-11-07)

| Question | Pile | Answer (now in the text) |
|---|---|---|
| Which hard invariants apply on the tunings matrix, given that T1 depends on the tuning? | Minor (my call) | All of v2's, with a per-tuning T1 cap (§5.5, D15). |
| Is the un-mute reference the shape before or during un-muting? | Answered by the prototype | Before (§5.2). The early return for an unchanged tuning is kept. |
| Commit order: could the voicing fixes erase the flagged leaps before the melody task? | Measured | No (§5.7). |

Nothing needed the human partner.
