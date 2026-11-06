# Compose fix: faithful chord voicings and singable melodic leaps

**Date:** 2026-11-07 (revision 2, after plan review round 1)
**Branch:** `fix/compose-voicings-leaps` (cut from `master` at `59d2d68`, composer v2)
**Status:** approved by the planner (Claude Opus 5.5) in an unattended Octet
pipeline. The human partner pre-approved the workflow (Opus plans, GPT-6 Astra
reviews, Opus implements). Every point where brainstorming would normally wait
for the user is listed in §10 with the decision taken and the reason. Review
round 1 (`docs/superpowers/reviews/2026-11-07-voicings-leaps-plan-review-r1.md`)
is answered in §12.
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
- Every such shape can be fretted with at most four fingers (§3.1).
- No melody contains a tritone, a seventh or a leap over an octave. This is
  asserted on every evaluation matrix and on a bounded sweep of every chord
  root and quality after a major and a minor home chord, with the composer's
  real floors. Only a genuinely infeasible custom pitch window may fall back
  to weaker rules (§5.3, §8).
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

Separately, a shape is **playable** (added in revision 2, review r1 F2) when:

- its fretted span (frets > 0) is at most 3, and
- it needs at most **four fretting fingers** under a straight-barre model:
  - At each fret `f`, take every maximal run of adjacent strings that are
    all fretted at `f` or higher.
  - A run that contains a note at exactly `f` costs one finger: a barre
    may lie under higher frets.
  - A barre can never cross an open or muted string.

For example, `0-7-6-7-5-4` needs 5 fingers (the two fret-7 notes are
separated by a fret-6 string), `x65643` needs 5, a barred F `133211`
needs 3, and open G `320003` needs 3.

### 3.2 Library shapes (EADGBE, the data in `chords/voicings.py`)

| Chord | v2 grip | Sounds | Faults | Repair | Sounds after |
|---|---|---|---|---|---|
| C#dim | `x-x-2-3-2-x` | E3 A#3 C#4 | G missing, A# foreign, bass E | `x-4-5-6-5-x` (the A-string dim shape of Cdim/Bdim, up one fret) | C#3 G3 C#4 E4 |
| Ddim | `x-x-0-1-0-1` | D3 G#3 B3 F4 | B foreign (a dim7 tone) | `x-x-0-1-3-1` (standard open Ddim) | D3 G#3 D4 F4 |
| D#dim | `x-x-1-2-1-2` | D#3 A3 C4 F#4 | C foreign (a dim7 tone) | `x-x-1-2-4-2` (the Ddim shape up one fret) | D#3 A3 D#4 F#4 |
| Fmaj7 | `x-3-3-2-1-0` | C3 F3 A3 C4 E4 | bass C (second inversion) | `x-x-3-2-1-0` (standard Fmaj7) | F3 A3 C4 E4 |
| G#aug | `x-x-2-1-1-0` | E3 G#3 C4 E4 | bass E (it is the Eaug grip) | `x-x-6-5-5-4` | G#3 C4 E4 G#4 |
| C#7 | `x-4-3-4-2-1` | C#3 F3 B3 C#4 F4 | needs 5 fingers (the open C7 grip moved up, with no open string left) | `x-4-6-4-6-4` (the A#7 `x13131` barre family) | C#3 G#3 B3 F4 G#4 |
| D#7 | `x-6-5-6-4-3` | D#3 G3 C#4 D#4 G4 | needs 5 fingers (same reason) | `x-6-8-6-8-6` | D#3 A#3 C#4 G4 A#4 |

The other 89 shapes are faithful and playable. C7 omits the fifth, which the
definition allows. Every repaired shape keeps a fretted span of at most 2
frets and needs at most 3 fingers.

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

**Fingering (review r1 F2).** Of the voicings that revision 1's design
accepted, **22 of 576 need five fingers**:
- EADGBE and Drop D C#7 and D#7
- Eb C7, C#7, D#7 (`0-7-6-7-5-4`) and B7
- D standard and Drop C C7, C#7, D#7 and B7
- six DADGAD retunes (C#, D#, Caug, C#aug, D#aug and Eaug)

All of them are the moved-up open-C7, B7 or C-shape grips, which only work
when an open string replaces a finger.

### 3.4 After the fix (prototype, §7)

- **0 / 576** voicings have any fault, and **0 / 576** need more than
  four fingers.
- EADGBE output changes only for the seven repaired chords.
- In the five other tunings, 451 of the 480 voicings are the retuned
  library shape. The other **29** fail the faithfulness or playability gate
  and come from the shape search:
  - Drop D G7
  - Eb C7 and B7
  - D standard C7 and B7
  - Drop C G, C7, G7, B7 and Gmaj7
  - DADGAD C#, D#, A#, B, C#7, D#7, A#7, A#maj7, Bmaj7, Ddim, D#dim,
    A#sus4, Bsus4, Caug, C#aug, D#aug, Eaug, A#aug and Baug
- Examples:
  - Eb D#7 is now the retuned `x68686` plus the open low Eb,
    `0-7-9-7-9-7` (3 fingers).
  - Eb C7 is searched: `x-4-3-1-0-1` (C3 E3 G3 A#3 E4, 4 fingers).
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

| **L0** | Pooled large-leap frequency (revision 2, review r1 F6) | Per piece, the counts `L0.large` (intervals of more than 7 semitones) and `L0.intervals` (all intervals). Pooled over a matrix as `large / intervals`, with `n = intervals`. A third key, `L0.any` (whether the piece has at least one such leap), averages to the share of affected pieces. |

Existing **S1** (missing essential / foreign tone) and **M4 / M4max** are
kept unchanged. **M4 is the mean over pieces of each piece's share of leaps
larger than a fifth.** It weights a short piece like a long one, so it is
not a per-interval frequency. That frequency is L0.

## 5. Design

### 5.0 Approaches considered

**Voicings:**
- **(a) Repair only the five tone-faulty library shapes.**
  - Fixes the flagged C#dim.
  - Leaves 93 of the 98 faulty voicings (the tuning paths, including the
    one-note Eb7 in a flagged sample).
  - **Rejected.**
- **(b) Replace retuning with a full shape search for every tuning.**
  - Changes every EADGBE chord, so V1 and every floor move.
  - Discards the hand-curated library.
  - **Rejected** (larger blast radius, no musical gain).
- **(c) Chosen:**
  - repair the seven library shapes (five tone faults, two five-finger
    grips)
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
- **(c) Chosen** (revision 2):
  - hard admissibility rules applied before sampling
  - **per-bar feasibility lookahead** for the constrained bars (the final
    bar and the bar before a one-note final bar)
  - a **deterministic repair order** for the floor and the cadence target
  - a soft bonus for steps into a new chord
  - a **graded relaxation ladder**, used only when no candidate survives
    (§5.3)

  Revision 1 dropped straight to v2 scoring when its rules found no
  candidate. Review r1 F1 showed that path reintroducing a major seventh:
  C → D#m, seed 19, gave E4→D#5. On the bounded sweep of §6.4 (11,520
  pieces) revision 1 needed it 25 times, and 5 pieces kept a forbidden
  interval. Revision 2 needs it 0 times on that sweep, on all three
  matrices and on the sample sweeps.

### 5.1 Library repair: `chords/voicings.py`

- Replace the seven entries of §3.2.
- Add one module-docstring paragraph stating the faithfulness and
  playability invariants of §3.1.

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
     `MAX_FRETTED_SPAN = 3`, a four-fret hand frame
   - `fingers_needed(frets) <= MAX_FINGERS = 4`, using the straight-barre
     model of §3.1. `fingers_needed(frets: Sequence[int]) -> int` is public.

   Open strings no longer count toward the span (this fixes F2). After the
   §3.2 repairs, all 96 library shapes pass this gate, so EADGBE plays the
   library unchanged.
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
     - `fingers_needed ≤ 4` (this replaces revision 1's "at most 3 notes
       above the lowest fret" heuristic, which wrongly assumed the
       lowest-fret notes could always share one barre)
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
| `recovery_pool: frozenset[int] \| None = None` | The next note's candidate pitches. Set only when the next note is in the same bar and that bar is not constrained (see below). `None` forbids a large leap here. |
| `allowed: frozenset[int] \| None = None` | In a constrained bar, the pitches from which the bar can still be finished smoothly (the feasibility set for this slot). `None` means no constraint. |

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
  - the previous pitch and the candidate are both tones of this chord
    (arpeggiation)
  - the previous interval was a step or a repeat (at most 2), so leaps
    never chain
  - `recovery_pool` contains a pitch 1–2 semitones back toward where the
    leap came from

  As a consequence, **every chord change moves by at most a fifth**.
- **R4 Feasibility.** If `allowed` is set, the candidate is in it.

`score_pitch` gains one term: `+CHORD_CHANGE_STEP_BONUS` when `chord_change`
and the move is 1–2 semitones. Its other terms and `-inf` rules are
unchanged.

**Smooth moves and feasibility sets.**

- `smooth(a, b)` means `abs(b - a) <= SMOOTH_LIMIT` and `abs(b - a) != 6`.
- `feasible_sets(candidates, n_free, goal) -> list[frozenset[int]]` works
  backward over a bar's free slots:
  - `F[n_free-1] = {c : smooth(c, g) for some g in goal}`
  - `F[k] = {c : smooth(c, x) for some x in F[k+1]}`
- A bar can be **entered** from `prev` when `prev` is `None`, or when some
  `c` in `F[0]` satisfies `smooth(prev, c)`.

**Constrained bars:**
- **The final bar.** Its last slot is the forced target, so it has
  `n_free = n_slots - 1` and `goal = {target}`. With one slot, entering
  means `smooth(prev, target)`.
- **The approach bar:** the bar before a final bar that has exactly one
  slot. It has `n_free = n_slots` and `goal = final_approach`.
  - `final_approach` is the final chord's cadence pitches (`cadence_pc`)
    inside `[pitch_low, pitch_high]`, kept only at or above the final
    chord's floor unless none remain.
  - Aiming at the floor lets the floor hold instead of yielding. In
    revision 1, aiming at every cadence pitch dropped one validation
    cadence note under the D chord's F#4, so A2 fell to 0.9998.

In a constrained bar, `recovery_pool` is always `None`: no large leaps.

**Repair order, decided once at the start of each bar (deterministic).**

- **Options.** `floored = _candidates(...)` is v2's function, **unchanged**:
  it keeps its octave-reach and cadence-reach floor-yield rules.
  `unfloored = _candidates(..., floor=None)`. Try the options in order:
  `floored`, then `unfloored` if it differs.
- **For each option:**
  - *Final bar:* try the cadence pitches in the option, sorted by
    `(abs(p - ref), p)` where `ref = prev if prev is not None else
    tessitura_center`. Take the first one from which the bar can be
    entered. The first try is exactly v2's `final_target`, so nothing
    changes unless v2's target is a dead end.
  - *Approach bar:* take the option if the bar can be entered (`goal =
    final_approach`).
  - *Any other bar:* take the option if `prev` is `None` or some pitch in
    it is `smooth` from `prev`.
- **If no option works,** use v2's floored candidates and target with
  `allowed = None` for the bar. The ladder below then covers each note.

In the review's case (C → D#m, seed 19), the floored final bar `[69 … 78]`
with target D#5 cannot be entered from E4. The unfloored option with
target D#4 can, and the piece now ends `… 64, 66, 63`.

**Scoring ladder, per sampled note:**
`ranked_scores(candidates, ctx, tessitura_center) -> tuple[list[float], int]`
returns the scores and the level used.

- **Level 0:** `score_pitch` where `admissible`, else `-inf`.
- **Level 1:** only R1 and R2 (no forbidden interval, and a pending
  recovery is honoured).
- **Level 2:** v2's plain `score_pitch`.

Use the lowest level with any finite score. On every matrix, the bounded
sweep and the sample sweeps, revision 2 never leaves level 0.

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
- **`case_metrics` adds:** `L0.large`, `L0.intervals`, `L0.any`, `L1`,
  `L2.leaps`, `L2.recovered`, `L3`, `L4`, `L4max`, `L5` and `S2`.
  `_aggregate` pools `L2` like M5 and M15. It pools `L0` as
  `{"pooled": large / intervals, "n": intervals}`, and averages `L0.any`
  as an ordinary mean.
- **Matrix roles (revision 2, review r1 F5).**
  - **Development matrices (for v3).** `validation`, `edge` and `tunings`
    were used to design and tune v3. The v3 statistical thresholds on them
    are **regression locks**. They keep every historical v2 comparison.
  - **Holdout matrix (new).** `holdout` uses inputs no prototype has been
    run on: seeds 3000–3004, EADGBE, one bar per chord. Its ten
    progressions are:
    - F major `F Dm7 Gm7 C7 F`
    - D major `D F#m G A7 Bm Em7 A7 D`
    - G# major `G# C#maj7 A#m7 D#sus4 D#7 G#`
    - B major `B D#m E C#m7 F#7 B`
    - F# major `F# D#m7 G#m7 C#sus4 C#7 F#`
    - C major `C Caug Am F Fm C`
    - D minor `Dm A# Gm A7 Dm`
    - G minor `Gm D#maj7 Cm7 Adim D7 Gm`
    - C# minor `C#m F#m G#7 C#m A G#7 C#m`
    - B minor `Bm Em C#dim F#7 Bm`

    That is 50 melodies × 3 styles. The v3 rules, weights and acceptance
    bounds of §6.4 are **frozen in this revision, before the holdout is
    ever evaluated**. Its v2 numbers are captured in the v2 snapshot,
    before any composer change.
- **New fixture `tests/fixtures/compose/musicality_v2.json`.** The script
  writes it from the **unchanged v2 composer** with the new metric code, as
  a committed plan task before any composer change. It covers all four
  matrices, and its `source` is that commit. The tracked tree must be
  clean: generated `.synth_marker` files are restored with `git checkout`
  before writing, and are never staged.
  - It is the reference for "no regression versus v2".
  - The pre-v2 `musicality_baseline.json` (source `69c2fa7`) stays as it
    is, and every v2 paired comparison against it keeps running.
- **Runtime.** The `evaluate()` fixture grows from about 2.8 s to an
  estimated 4–4.5 s.

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
| **Melody** | L1 = 0, L3 = 0, L4max ≤ 7 and L5 ≤ 7 and not 6.<br>Every leap larger than a fifth is recovered.<br>Musical assertions only. |
| **Version** (separate test, added with the bump in the melody task) | `metadata.composer_version == 3` |

On v2 the groups fail as follows:
- **Voicing:** minor-10-B (C#dim), major-08-Ab (Eb7 is a single note) and
  major-09-Eb (Bb7's bass is D).
- **Melody:**
  - minor-10-B: the 8-semitone leap G5→B4 has no chord tone at either end
    (L3).
  - major-08-Ab: the 10-semitone leap (L1, L3, L4max).
  - major-09-Eb: the 9-semitone leap at a chord change (L3, L4max).

Revision 2 (review r1 F3): the melody test asserts in a fixed order: L1,
then L3, then recovery, then L4max, then L5. On v2 the **first failing
assertion** must be:
- minor-10-B: L3
- major-08-Ab: L1
- major-09-Eb: L3

The plan checks this with `--runxfail` when the tests are added, and again
after each voicing task.

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
- *Validation* (400 melodies) and *edge* (110) were held out for v2.
  - For v3 they are development data: the step bonus and these
    thresholds were chosen on them.
  - Their statistical thresholds below are **regression locks**, not
    evidence of generalisation.
  - The independent acceptance check is the holdout (§6.4).
- *Hard* targets are checked per composition, as a min/max over every
  matrix listed.
- *vs v2* compares with `musicality_v2.json` on identical inputs.

### 6.1 Leap targets

| ID | v2 (val / edge / tunings) | Prototype (val / edge / tunings) | **Target** | Why this threshold |
|---|---|---|---|---|
| L1 | 4 / 3 / 3 intervals (max 1 per piece) | 0 / 0 / 0 | **max = 0, hard, all three matrices** | The request says tritones and sevenths are avoided. The prototype never needed the relaxed path, so zero is achievable rather than aspirational. |
| L3 | 30 / 8 / 9 unjustified (of 40 / 9 / 9 large leaps) | 0 / 0 / 0 | **max = 0, hard, all three** | "Justified" is defined operationally (§4.3). Only arpeggiation inside one chord qualifies. |
| L2 | 14 / 24 = 0.58 · 4 / 5 · n = 0 (all 9 were final notes) | 3 / 3 = 1.0 · n = 0 · n = 0 | **pooled = 1.0 when n > 0, hard, all three** | Every large leap is recovered by a contrary step, as the request asks. |
| L0 (pooled frequency of leaps > 5th) | 40 / 9040 = 0.0044; 9 / 1718 = 0.0052 | 3 / 9040 = 0.0003; 0 | **pooled ≤ 0.001 on validation and edge, and < v2** | "Rare" means at most one leap larger than a fifth per thousand melodic intervals, a cut of more than 4× on both matrices. It is stated in the unit it is measured in (review r1 F6). Affected pieces (`L0.any`): v2 39 / 400 and 9 / 110; prototype 3 / 400 and 0 / 110. |
| M4 (mean of per-piece shares) | 0.0040 / 0.0064 | 0.0002 / 0.0 | **mean ≤ 0.001 on validation and edge, and < v2** | Kept for continuity with v2. It is a mean of per-piece rates, not a frequency (see L0). |
| M4max | 12 / 12 | 8 / 7 | **≤ 12, hard** (unchanged) | Octaves stay legal when justified. The rules, not the cap, keep them rare. |
| L4 (chord change ≤ a third; per-piece mean) | 0.852 / 0.806 / 0.903 | 0.884 / 0.848 / 0.924 | **> v2 on every matrix, including the holdout** (the acceptance criterion). Regression locks: mean ≥ 0.87 on validation and ≥ 0.83 on edge. | The acceptance criterion is that chord changes are smoother than in v2. The locks sit between v2 and the measured value. They are labelled as development-data locks because they were set after measuring (review r1 F5). Pooled over all chord changes: 0.844 → 0.878 on validation and 0.827 → 0.878 on edge. |
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
| Fingering audit (unit test, 576) | 22 / 576 need ≥ 5 fingers (revision 1 design); v2 library: C#7, D#7 | 0 | **0 / 576 need more than 4 fingers; fretted span ≤ 3** | Playable as written (review r1 F2). |
| V1 | 1.621 / 1.620 | 1.648 / 1.620 | **≤ v2 + 0.1 semitone** | V1 is a mean distance in semitones per voice. +0.1 means one voice in ten moving one more semitone. That is below audibility as a change of hand position, and it is the price of the Ddim and C#7 repairs (iidim in C minor; V7 in F# minor). V1 remains a proximity guard, not a voice-leading score. |

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
- **Why these tolerances** (revision 2: justified in each metric's own
  units, not by the largest loss observed; review r1 F5):
  - **Note-share means** (M1, M2, M3, M13, M11b, A2, A3, T2): 0.02.
    - A typical piece has about 22 melody notes and 8–16 strong beats, so
      0.02 is under half a note per piece on average. No single piece can
      hide a systematic change within it.
    - The allowance is an order of magnitude smaller than the v2 gains it
      protects: M1 +0.12, M2 +0.20, M3 −0.18, A2 +0.09–0.18 and
      A3 −0.04–0.06.
    - Every v2 absolute target, such as M1 ≥ 0.95, still applies on
      validation.
  - **Event-pooled rates** (M5, M15): 0.05.
    - M15 pools 13–40 resolution events, so one event is worth 0.025–0.08.
      0.05 allows about one event.
    - M5 pools about 150–770 leaps, so 0.05 is 7–38 recoveries. That is
      generous, but the v2 floor M5 ≥ 0.70 still binds.
  - **Exact-equality rows** (M8, M9, M10, M12, M14, A1, A4, H1, H2): no
    allowance is needed, because the rhythm and accompaniment code is
    unchanged. Their hard invariants stay.
- **For information**, the worst measured subgroup deviation:
  - M1 in edge minor (40 melodies): 0.986 → 0.973, or −0.013.
  - Its causes: the hard rules occasionally remove the only chord tone
    within a fifth on a downbeat, and the step bonus slightly favours a
    non-chord-tone step.
  - M5 in validation major is −0.012.

| Metric (validation overall) | v2 | Prototype (revision 2) |
|---|---|---|
| M1 | 0.980 | 0.977 |
| M2 | 0.554 | 0.569 |
| M3 | 0.095 | 0.094 |
| M5 (pooled) | 0.761 (n = 771) | 0.762 (n = 668) |
| M11b | 0.695 | 0.704 |
| M13 | 0.024 | 0.026 |
| M15 (pooled) | 0.825 (n = 40) | 0.849 (n = 33) |
| T2 | 0.124 | 0.122 |
| A2 (every style) | 1.0 | 1.0 |
| A3 arpeggio / strum / fingerstyle | 0.062 / 0.087 / 0.069 | 0.058 / 0.084 / 0.066 |
| M8, M9, M10, M12, M14, A1, A4, H1, H2 | — | identical to v2 |

On the edge matrix:
- A2 rises from 0.987 to 0.988 in every style.
- M13 goes from 0.0196 to 0.0206.
- M11b goes from 0.774 to 0.768.
- M1 goes from 0.991 to 0.984.

### 6.4 Holdout acceptance and the bounded sweep (frozen before evaluation)

**Holdout matrix (§5.5).** The prototype has not been run on it. Every
bound is fixed here, independently of any measurement:

- **Hard, per composition:**
  - L1 = 0 and L3 = 0
  - L2 pooled = 1.0, or n = 0
  - L4max ≤ 7 and L5 ≤ 7
  - S1 and S2 = 0
  - every v2 hard invariant: M4max ≤ 12, M6 ≤ 17, M7 = 1, M8 ≥ 1.5,
    M12 = 1, M14 = 1, T1 ≤ 15, Tcount = 1, and A1, A4, H1 and H2 = 1 in
    every style
- **Paired with v2's holdout numbers** (from the snapshot):
  - L0 pooled ≤ v2's and ≤ 0.002
  - L4 mean ≥ v2's
  - every note-share mean within 0.02 of v2, and every pooled rate within
    0.05, in the direction of §6.3
- **Not applied to the holdout:** v2's absolute statistical targets (such
  as M1 ≥ 0.95). They were set on v2's validation data, and v2 itself was
  never checked against them on these inputs.
- **If a holdout check fails,** the implementer reports it with the
  numbers. The fix must come from a defect reproduced on development data,
  or from a stated musical rule, with this spec updated. Thresholds are
  never retuned to the holdout.

**Bounded sweep.** It runs every root × quality (96 chords) after a home
chord, through `arrange()` with real floors and default settings:
- in C major (`C X`, `C X C`) and A minor (`Am X`, `Am X Am`)
- with seeds 0–4

That is 1,920 compositions (about 2 s), and it runs as a unit test. It
asserts L1 = 0, L3 = 0, every large leap recovered, L4max ≤ 7 and L5 ≤ 7.

Revision 1's rules pass seeds 0–4. Its failures appeared only in the wider
sweep over seeds 0–29 (11,520 pieces): 25 relaxed-fallback notes and 5
pieces with a forbidden interval. Those five compositions become explicit
regression cases:
- C → D#m, seeds 19 and 20
- C → D#m7, seed 20
- Am → D#m, seed 20
- Am → D#m7, seed 20

On revision 2 the full 0–29 sweep gives 0 forbidden intervals and 0 ladder
fallbacks.

## 7. Prototype

All the numbers in §3.4, §6 and §2 come from a throwaway prototype. It was a
copy of `src/` in the session scratchpad with the §5 changes applied, run
with `PYTHONPATH` pointed at it. Nothing in the worktree was modified.
Revision 2 re-measured everything on a prototype with the repair order,
ladder, fingering gate and seven library repairs. That prototype was never
run on the holdout inputs.

Against the existing suite (compose, voicing, renderer, public-contract and
compose e2e tests) it gave 285 passed and 3 failed, all of them expected:

1. `test_compose_melody.py::test_one_note_final_bar_stays_within_an_octave_when_a_floor_bites`.
   Its "the adversarial low start really occurred" precondition no longer
   occurs, because the approach rule now steers bar 1 toward G5. The plan
   rewrites it as follows:
   - the sweep asserts the final move is at most a fifth
   - the repair branch is covered by the review's C → D#m case and a
     direct feasibility-set test (revision 2 drops `cadence_reach`)
2. `test_compose_metrics.py::test_shape_tone_mismatch_reports_inherited_shape_defects`
   pins the old C#dim defect. The plan rewrites it on a hand-built defective
   shape.
3. `test_compose_musicality.py::test_guards_do_not_regress` pins S1 and V1
   to the baseline. The plan replaces it as in §6.2.

`test_compose_harmony` needs its `_CHORD_PCS` row removed (§5.2).

## 8. Error handling and edge cases

- **No admissible candidate.** First the bar-level repair order is tried,
  then the per-note ladder (§5.3). Level 1 still forbids tritones and
  sevenths, and still honours a pending recovery. Only level 2 is v2's
  scoring. Level 2 is reachable only with custom pitch windows too narrow
  to hold a non-forbidden move. Revision 2 never left level 0 on any matrix
  or sweep. Nothing raises that v2 accepted.
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
- No difficulty model beyond the span ≤ 3 and four-finger feasibility of
  §3.1. Stretch comfort, thumb use and string-muting technique are not
  modelled.
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
| D4 | Melody mechanism | **Hard admissibility rules + feasibility lookahead + bar-level repair order + graded ladder + step bonus** (§5.3; revision 2) | Only hard rules can guarantee "no sevenths or tritones" and "always recovered". Lookahead and repair keep the rules satisfiable, and the ladder never drops R1 or R2 before it has to (review r1 F1). |
| D5 | Are sixths and octaves still allowed? | **Yes, only as arpeggiation inside one chord, approached by step, never at a chord change or in the final bar, with a guaranteed contrary-step recovery** | The request says "rare and justified", not "never". These are the idiomatic cases. |
| D6 | Chord-change smoothness | **Hard ≤ a fifth (via R3); soft bonus 1.0 for steps; targets L4 ≥ 0.87 (validation) and ≥ 0.83 (edge)** | Bonus 0.5 and 1.5 and a repeat-inclusive variant were measured. The repeat-inclusive bonus raised M3 (more repeated notes); 1.5 cost more M1 for little extra L4. |
| D7 | Tunings coverage in the evaluation | **A 90-melody "tunings" matrix with hard invariants only** | It keeps the statistical targets on the held-out matrices. The other tunings are exercised through the composer, not just through `voicings_for`. |
| D8 | Reference for "no regression" | **A new committed `musicality_v2.json` written from the unchanged v2 code**, tolerance 0.02 (means) and 0.05 (pooled) | Paired, identical inputs. Committed before any change, like v2's baseline. The tolerances are justified in §6.3. |
| D9 | Floor versus a smooth final approach | **The floor and target are repaired in a fixed order: the floored nearest target (v2's), then farther floored targets, then unfloored targets. The approach bar aims at cadence pitches above the final floor.** | v2 precedence was octave limit > cadence > register clearance. "A smooth path exists" is inserted above register clearance. Because v2's choice is tried first, outputs change only where it was a dead end: A2 stays 1.0 on validation. |
| D10 | Span rule | **Fretted span ≤ 3 (open strings excluded)** | Open strings need no finger. The four-fret frame matches the library, whose largest span is 3, and sends the awkward retunes to the search (for example Drop D G7 at frets 1–5). |
| D11 | How "tests fail before the fix" is kept across several commits | **Sample tests land first as `xfail(strict=True)`; each fixing task removes its marker** | Every commit stays green, the failure is demonstrated, and a forgotten marker fails the suite. |
| D12 | CLI or API for the sample tests | **CLI via `CliRunner`, WAV rendering stubbed** | The human reported CLI commands. Stubbing synthesis keeps the suite fast; MIDI is still written. |
| D13 | Shape choices for the seven repairs | **The standard grips in §3.2, chosen to match neighbouring library families** (A-string dim, D-string dim, xx3210, xx6554, and the A#7 `x13131` barre family for C#7 and D#7) | Consistency with the existing families. All are root position, with spans of at most 2 and at most 3 fingers. |
| D14 | Duplicate chord-tone table in `compose/voicings.py` | **Remove it and use `harmony`** | One source of truth. A test already forces the tables to agree. |
| D16 | Playability standard | **Fretted span ≤ 3 and ≤ 4 fingers under the straight-barre model (§3.1), applied to the library, the retune gate and the search** | Review r1 F2: span alone accepted five-finger grips, including the flagged sample's Eb7. A full difficulty model is unnecessary. |
| D17 | Development versus holdout data | **Relabel validation, edge and tunings as v3 development matrices (their thresholds are regression locks); add a frozen 50-melody holdout with bounds fixed in §6.4** | Review r1 F5: the bonus and the tolerances were chosen on those matrices. |
| D18 | How tolerances are justified | **In each metric's own units (§6.3)** | Review r1 F5. |
| D19 | Large-leap frequency | **New pooled L0; M4 kept and described as a mean of per-piece rates** | Review r1 F6. |
| D15 | Tab fret cap in other tunings | **T1 ≤ 15 + (64 − top-string open pitch), per tuning, on the tunings matrix only** | The builder check showed a flat 15 cannot hold once a tuning's top string is lower. v2 measured exactly these maxima, and the tab is out of scope. |

## 11. Builder check (fresh reviewer, 2026-11-07)

| Question | Pile | Answer (now in the text) |
|---|---|---|
| Which hard invariants apply on the tunings matrix, given that T1 depends on the tuning? | Minor (my call) | All of v2's, with a per-tuning T1 cap (§5.5, D15). |
| Is the un-mute reference the shape before or during un-muting? | Answered by the prototype | Before (§5.2). The early return for an unchanged tuning is kept. |
| Commit order: could the voicing fixes erase the flagged leaps before the melody task? | Measured | No (§5.7). |

Nothing needed the human partner.

## 12. Review round 1: what changed

The finding-by-finding responses are in the plan, section "Review responses
(round 1)". In summary:

| Finding | Outcome | Where |
|---|---|---|
| F1: the relaxed fallback could reintroduce a major seventh (C → D#m, seed 19) | **Fixed.** Feasibility lookahead, a bar-level floor/target repair order and a graded ladder. The case is now a regression test, plus a direct ladder test and a 1,920-piece bounded sweep. | §5.0, §5.3, §6.4, §8 |
| F2: five-finger grips accepted, including the Eb7 repair | **Fixed.** Four-finger feasibility on the library, the retune gate and the search. C#7 and D#7 were repaired in the library, and 22 → 0 five-finger voicings. | §3.1–§3.4, §5.2, D16 |
| F3: the version assertion masked the melody xfails | **Fixed.** A separate version test; the first-failing assertion is checked per sample. | §5.6 |
| F4: Task 2 committed a failing test; snapshot cleanliness | **Fixed in the plan** (two commits; markers restored). | §5.5, plan Task 2 |
| F5: held-out data used for tuning | **Fixed.** Matrices relabelled, a frozen holdout added, tolerances justified in units, and "match the prototype within 0.005" dropped. | §5.5, §6, §6.4, D17–D18 |
| F6: M4's bound described as a frequency | **Fixed.** Pooled L0 added; M4 described correctly. | §4.3, §6.1, D19 |
