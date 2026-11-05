# Compose improvements: a more musical, measurable composer

**Date:** 2026-11-06
**Branch:** `feat/compose-improve`
**Status:** approved by the planner (Claude Opus 5.5) in an unattended Octet
pipeline. The human partner pre-approved the workflow (Opus plans, GPT-6 Astra
reviews, Opus implements). Every point where brainstorming would normally wait
for the user is listed in §10 with the decision taken and the reason.

## 1. Intent

**What was asked:** make the output of `music_decoder.compose` (melody plus
accompaniment over a scale and chord progression, written as MIDI, WAV and
ASCII tab) noticeably more musical, and back every improvement with objective
musicality metrics asserted in tests across many seeds, keys, modes,
progressions and styles.

**Hard constraints (from the request):**

- `compose()` keeps its public signature and the CLI keeps its behaviour. New
  optional parameters would be allowed, but
  `tests/unit/test_public_contract.py::test_compose_signature_is_frozen`
  asserts the exact parameter list, so this design adds **none** (§10, D1).
- Output stays deterministic for a given seed.
- No new dependencies and no trained models: rule-based and statistical methods
  over numpy and pretty_midi only.
- Gate commands stay green (`pytest -q tests/unit tests/integration`,
  `ruff check src tests`, `ruff format --check src tests`, `mypy --strict src`)
  and the suite stays fast (the new metrics test has a 5 s budget).

**Assumptions (mine, not stated in the request):**

- "More musical" means the conventions of Western tonal folk and pop melody:
  stepwise motion, chord tones on strong beats, leaps recovered by a step in
  the opposite direction, phrases of two or four bars that end on long notes,
  a final cadence on the tonic, and rhythmic variety bound together by
  repetition.
- 4/4 time is kept. Every existing caller assumes it.
- A melody that changes for a given seed is acceptable. Determinism means
  "same version + same inputs + same seed gives the same output", not
  "byte-identical to the previous release" (§10, D3).

**Success criteria:** every metric in §4 meets its target on the regression
matrix (§7). Each target is a clear improvement on the measured baseline in §3,
or a guard that holds it where the baseline is already fine. All gates stay
green.

## 2. Current composer (what exists)

| File | What it does today |
|---|---|
| `compose/melody.py` | `generate_melody`: one note per beat (`notes_per_bar=4`, every note a quarter note). On beats 1 and 3 it uses a chord tone with probability 0.7, otherwise 0.3. The pool is limited to pitches within ±4 semitones of the previous note, and the pick is uniform. Range C4–C6 (60–84). No phrase structure, no cadence, no motifs. |
| `compose/voicings.py` | Looks up the single canonical EADGBE shape for the chord in `chords/voicings.py` and retunes it. `voicings_for` returns at most one candidate in practice. |
| `compose/arrangement.py` | Two-track MIDI. `strum`: a block chord on beats 1 and 3. `arpeggio`: the voicing played up then down, spread evenly over the bar (2·n−1 notes, e.g. 11 per bar for six strings, so the rhythm is an **11-tuplet**, off any metric grid). `fingerstyle`: the same bass note on 1 and 3, the treble cluster on 2 and 4. The accompaniment ignores the melody. |
| `compose/api.py` | Orchestration. The melody tab puts each note at its own lowest fret, so the line sits on the high-e string up to fret 20. Chord tab anchors are spread evenly over the melody span. |

## 3. Baseline measurement

A throwaway script (kept outside the repo, in the session scratchpad) called
the current internals directly (`generate_melody`, `voicings_for`,
`_accompaniment_notes`, the lowest-fret tab logic and `tabs.assign_tabs`) over
this matrix:

- **Major:** keys C, G, D, A, E, F, A#, D# × progressions I–V–vi–IV,
  Imaj7–vi7–ii7–V7, I–IV–V–I, ii7–V7–Imaj7–Imaj7, I–vi–IV–V.
- **Minor:** keys A, E, D, B, F#, C, G × progressions i–iv–V7–i,
  i–VI–III–VII, i–iv–v–i, i–VII–VI–V, ii°–V7–i–i.
- `bars_per_chord` ∈ {1, 2}, seeds 0–19, styles strum / arpeggio / fingerstyle,
  120 BPM, EADGBE.
- 75 progressions × 2 × 20 = **3000 melodies**, each accompanied in 3 styles
  (9000 arrangements).

A prototype of the design in §5, also throwaway and in the scratchpad, was
measured on the same matrix so that the targets are known to be reachable.
Its numbers are in the "Prototype" column. The prototype was also run on the
400-melody regression matrix of §7 (2.2 s including A\* and three accompaniment
styles), and every target held there too.

| ID | Metric (definition in §4) | Baseline mean (p10 / max) | Prototype mean | **Target** |
|---|---|---|---|---|
| M1 | Strong-beat chord-tone ratio | 0.841 (p10 0.688) | 0.980 (min 0.625) | **mean ≥ 0.95; every composition ≥ 0.60** |
| M2 | Stepwise ratio (1–2 semitones) | 0.372 | 0.548 | **mean ≥ 0.50** |
| M3 | Repeated-note ratio | 0.229 | 0.095 | **mean ≤ 0.15** |
| M4 | Large-leap ratio (> 7 semitones) | 0.000 (max 0.067) | 0.017 | **mean ≤ 0.04; no interval > 12 (hard)** |
| M5 | Leap recovery (pooled) | 0.348 (only 23 leaps) | 0.755 | **≥ 0.70** |
| M6 | Ambitus (semitones) | 12.4 (max 24) | 11.5 (max 17) | **≤ 17 for every composition** |
| M7 | Final-note resolution | tonic 0.152, chord tone 0.669 | 1.000 | **1.0 (hard)** |
| M8 | Final note ≥ 1.5 beats | 0.0 (always 0.95 beats) | 1.0 | **1.0 (hard)** |
| M9 | Distinct note durations per composition | 1.0 | 3.88 (min 3) | **mean ≥ 3.0; min ≥ 2** |
| M10 | Inter-onset-interval entropy (bits) | 0.000 | 1.450 | **mean ≥ 1.0** |
| M11 | Motif contour match (bar 1 vs bar 3 of each 4-bar group) | 0.300 | 0.672 | **mean ≥ 0.60** |
| M12 | Phrase-end long note (bar 4 of each group) | 0.000 | 1.000 | **≥ 0.95** |
| M13 | Non-diatonic chord clash (progressions with a non-diatonic chord) | 0.032 (max 0.219) | 0.000 | **0.0 (hard)** |
| M14 | Melody onsets on the 8th-note grid | 1.0 | 1.0 | **1.0 (hard guard)** |
| A1 | Accompaniment stroke heads on the 16th grid | arpeggio **0.106**, strum 1.0, fingerstyle 1.0 | 1.0 / 1.0 / 1.0 | **1.0 every style (hard)** |
| A2 | Melody above sounding accompaniment | arpeggio 0.955, strum **0.842** (p10 0.41), fingerstyle 0.923 | 0.996 / 0.996 / 0.978 | **mean ≥ 0.97 each style** |
| A3 | Exposed semitone clash, melody vs accompaniment | arpeggio 0.034, strum **0.169**, fingerstyle **0.121** | 0.024 / 0.032 / 0.050 | **mean ≤ 0.08 each style** |
| A4 | Final chord rings through the last bar, single attack | no (ends at 3.9–3.95 beats, re-struck) | yes | **true (hard)** |
| T1 | Melody tab max fret (EADGBE) | 15.3 (max **20**) | 12.4 (max 15) | **≤ 15 every composition** |
| T2 | Hand-position shifts per fretted transition | 0.182 (lowest fret) / 0.135 (A\* at the old register) | 0.111 | **mean ≤ 0.13** |
| V1 | Mean nearest-voice motion between successive distinct voicings (semitones) | 1.61 | 1.61 (unchanged) | **≤ 1.61 (guard)** |

What the baseline shows, in order of how much it hurts the music:

1. **The rhythm is completely uniform.** Every note is a quarter note
   (1 distinct duration, IOI entropy 0). This is the single most "robotic"
   property of the output.
2. **There are no cadences or phrases.** The last note is the tonic 15% of the
   time and a chord tone 67% of the time, and it lasts 0.95 beats like every
   other note. Nothing marks phrase ends.
3. **The arpeggio style is off the grid.** It plays 11-tuplets: 89% of its
   onsets miss the 16th grid.
4. **The pitch line is a random walk.** 23% repeated notes, only 37% steps,
   leap recovery 35%, contour repetition at chance level (0.30, no motifs), and
   ambitus up to 24 semitones.
5. **The melody and accompaniment collide.** Strum puts accompaniment notes
   above the melody 16% of the time (p10: 59%) and makes semitone clashes on
   17% of exposed melody notes. Fingerstyle: 8% and 12%.
6. **Non-diatonic chords clash.** In minor keys with V/V7 (for example E7 in
   A minor) the melody uses natural-minor G against the chord's G#. That
   affects 3% of notes on average and up to 22% in one composition.
7. **The tab is hard to play.** It goes up to fret 20, nearly all on one
   string, and averages 0.18 hand shifts per note.
8. **Voicings are already fine.** Voice leading between successive open-
   position shapes moves 1.6 semitones per voice on average. There is little
   to gain without a larger shape library (§9).

## 4. Metric definitions

All metrics are pure functions in the new `src/music_decoder/compose/metrics.py`
(§5.5). Shared conventions:

- beat = 60 / tempo; bar = 4 beats; eighth = beat / 2.
- A note's **metric position** is `round(start_s / eighth) mod 8`. Positions 0
  and 4 are **strong** (beats 1 and 3).
- The **chord at time t** is `progression[min(floor(t / bar) // bars_per_chord, len − 1)]`.
- **Chord tones** are the chord's pitch-class set from the quality table
  (maj, min, maj7, min7, 7, dim, sus4, aug).
- **Intervals** are the differences between consecutive melody pitches.
- **Sounding accompaniment at t** is every accompaniment note with
  `start ≤ t + 1e-4 < end`.

| ID | Definition |
|---|---|
| M1 | Among melody notes at strong positions: the fraction whose pitch class is a chord tone of the chord sounding at their onset. |
| M2 | Fraction of intervals with \|i\| ∈ {1, 2}. |
| M3 | Fraction of intervals equal to 0. |
| M4 | Fraction of intervals with \|i\| > 7. The hard part: max \|i\| ≤ 12. |
| M5 | Over all leaps \|i\| ≥ 5 that have a following interval j: the fraction where `sign(j) = −sign(i)` and 1 ≤ \|j\| ≤ 2. Pooled across the matrix (count leaps, count recoveries), because a single composition has few leaps. |
| M6 | max(pitch) − min(pitch). |
| M7 | The last note's pitch class is the tonic if the final chord contains the tonic; otherwise it is the final chord's root. |
| M8 | Duration of the last melody note in beats. |
| M9 | Number of distinct melody durations, each quantised to the nearest eighth by onset-to-onset spacing (last note: its own length, also quantised). |
| M10 | Shannon entropy (bits) of the inter-onset intervals quantised to eighths. |
| M11 | For each 4-bar group g (bars 4g and 4g+2, both present and holding the same number of notes, at least 2): the fraction of intra-bar interval signs that agree. Averaged over pairs; `None` when no pair qualifies. |
| M12 | For each complete 4-bar group, whether the last note starting in bar 4g+3 lasts ≥ 1.5 beats. Fraction over groups. |
| M13 | For progressions where some chord has a pitch class outside the scale: the fraction of melody notes whose pitch class is a semitone from one of the current chord's out-of-scale tones. |
| M14 | Fraction of melody onsets within 1 ms of the eighth-note grid. |
| A1 | Group accompaniment onsets that fall within 60 ms of the group's first onset (that is, one stroke). The fraction of group heads within 1 ms of the 16th grid. |
| A2 | Fraction of melody notes whose pitch is above every accompaniment pitch sounding at their onset (true when nothing is sounding). |
| A3 | Among melody notes that are at a strong position or last ≥ 0.9 beats: the fraction where some sounding accompaniment pitch is 1 or 11 semitones away by pitch class. |
| A4 | Accompaniment notes in the final bar all start within 60 ms of the bar start and end at or after the bar end minus 1 ms. |
| T1 | Max fret of the melody tab positions returned by `arrange()` (§5.4). |
| T2 | Walk the fretted (> 0) melody tab frets with a hand window spanning at most 4 frets (max − min ≤ 4). A note outside the window starts a new window and counts as a shift. Shifts ÷ (fretted notes − 1). |
| V1 | For successive voicings with different chords: the mean over the second chord's pitches of the distance to the nearest pitch of the first. Averaged over the progression. |

## 5. Design

### 5.0 Approaches considered

1. **Retune the existing Markov walk** (change probabilities, narrow the
   interval cap). Cheap, but it cannot create rhythm, phrases, cadences or
   motifs, which are weaknesses 1, 2 and 4. Low ceiling. Rejected.
2. **Rule-based phrase planner plus scored stochastic pitch choice**
   (**chosen**). Rhythm comes from curated one-bar templates arranged in
   phrase plans. Pitches are sampled from a softmax over hand-set musical
   scores: interval size, chord-tone fit by metric weight, leap recovery,
   register gravity, motif echo and cadence pull. Every rule maps onto a
   metric, it is deterministic under a seeded `numpy` generator, it needs no
   data, and the prototype took about 0.3 ms per 8-bar melody.
3. **Statistical n-gram model learnt from the music21 corpus.** It would sound
   idiomatic, but it means shipping or deriving transition tables, it is hard
   to steer toward cadences and chord fit, and it is harder to test. Deferred.
   The scoring in option 2 can later absorb corpus-derived interval weights
   without any structural change.

The **accompaniment** follows the same reasoning: fix the patterns and make
them melody-aware, rather than adding a voicing search (§9).

### 5.1 Rhythm and phrase planner: new `compose/rhythm.py`

Pure functions with no I/O. The only randomness is an `np.random.Generator`
passed in by the caller.

- **Units:** eighth notes. Every bar template sums to 8.
- `BODY_RHYTHMS` (11 templates): `(2,2,2,2)`, `(2,1,1,2,2)`, `(1,1,2,2,2)`,
  `(2,2,1,1,2)`, `(3,1,2,2)`, `(2,2,3,1)`, `(4,2,2)`, `(1,1,1,1,2,2)`,
  `(2,1,1,1,1,2)`, `(3,1,3,1)`, `(4,1,1,2)`.
- `CADENCE_RHYTHMS` (each ends on a note of ≥ 4 eighths): `(2,2,4)`, `(4,4)`,
  `(3,1,4)`, `(2,1,1,4)`, `(8,)`.
- `split_phrases(n_bars) -> list[int]`. Phrases of 4 bars. A remainder of 2 or
  3 becomes a final shorter phrase. A remainder of 1 (with n > 4) turns the
  last 4-bar phrase into 3 + 2. If n ≤ 4 there is one phrase of n bars.
- `plan_bars(n_bars, rng) -> list[BarPlan]`, where
  `BarPlan(rhythm: tuple[int, ...], cadence: bool, final: bool, motif_from: int | None)`
  and `motif_from` is the absolute index of the bar whose interval pattern
  this bar should echo:
  - **Antecedent** phrases (even phrase index): draw rhythms A and B from
    `BODY_RHYTHMS`. A 4-bar phrase plays `[A, B, A, cadence]`, with bar 3
    echoing bar 1. A 3-bar phrase plays `[A, B, cadence]`, a 2-bar phrase
    `[A, cadence]`, and a 1-bar phrase `[cadence]`.
  - **Consequent** phrases (odd phrase index) reuse the previous antecedent's
    body rhythms. Each body bar echoes the matching antecedent bar's
    intervals. If the consequent is longer than the antecedent body, it is
    padded with fresh `BODY_RHYTHMS` draws that have no echo.
  - Each phrase ends in a `CADENCE_RHYTHMS` bar. In the **final** phrase only
    cadences whose last value is ≥ 4 qualify (all of them do today; the check
    keeps future templates honest).

### 5.2 Melody pitch selection: rewrite of `compose/melody.py`

`generate_melody` keeps its keyword interface except as follows (§10, D4):

- `notes_per_bar: int | None = None`. `None` (the new default) uses the phrase
  planner. An integer uses a uniform grid of that many notes per bar (the old
  rhythm), still with the new pitch scorer, so the existing statistical test
  and any old callers keep working. On this legacy grid the old timing stays
  (`bar / notes_per_bar` per note) and so does the old strong-note rule
  (`index % max(notes_per_bar // 2, 1) == 0`). Each note counts as
  `8 / notes_per_bar` eighths for the duration rules. There is no cadence
  planning, but the last note of the piece still takes the final-note target
  (step 2).
- `pitch_low` / `pitch_high` defaults change from 60 / 84 to **62 / 79**
  (D4–G5). This lowers the tessitura into a playable guitar register
  (T1 ≤ 15 on the high e).
- `strong_beat_chord_tone_prob`, `weak_beat_chord_tone_prob` and
  `markov_max_interval_semitones` are **removed**. They have no meaning under
  scored selection, and no caller in `src/` or `tests/` passes them.
- New keyword `tessitura_center: int = 70`.

**Algorithm**, for each bar from `plan_bars` (or the uniform grid):

1. **Chord-scale pool.** `chord_scale_pcs(scale_pcs, chord_pcs)` starts from
   the scale. For every chord tone outside the scale (G# in E7 over A minor),
   it removes the scale pitch classes a semitone away from it that are not
   chord tones, then adds the chord tone. Over diatonic chords the pool is the
   scale, so `test_all_notes_in_scale` and the e2e "all melody pitch classes
   in C major" check still hold. Candidates are the pitches in
   `[pitch_low, pitch_high]` whose pitch class is in the pool.
2. **Final-note target.** In the final bar: the tonic if the final chord
   contains it, otherwise the final chord's root. Take the candidate with that
   pitch class nearest to the last pitch before the final bar starts
   (`tessitura_center` if there is none; ties go to the lower pitch). The
   target is fixed once, when the final bar starts. The
   last note of the piece is exactly this target.
3. **Score each candidate c.** p is the previous pitch, iv = c − p,
   prev_iv is the previous interval, and the score is a sum of these terms:
   - Interval: `INTERVAL_WEIGHT[|iv|]` = {0: −0.8, 1: +0.8, 2: +0.8, 3: −0.2,
     4: −0.6, 5: −1.2, 6: −4.0, 7: −1.8, 8: −3.5, 9: −3.5, 10: −4.0,
     11: −4.5, 12: −3.0}. |iv| > 12 is excluded.
   - Chord fit, by metric weight, taking the first rule that applies:
     a strong position gets chord tone +2.0, non-chord tone −3.0. A note of
     ≥ 3 eighths gets +1.0 / −1.5. A note of ≥ 2 eighths that is a non-chord
     tone a semitone from a chord tone gets −1.0 (this cut the fingerstyle A3
     clash from 0.076 to 0.048 in the prototype). Otherwise a chord tone gets
     +0.3.
   - Leap recovery: if |prev_iv| ≥ 5, +2.5 for an opposite-direction step
     (1–2 semitones), −2.5 for a same-direction or repeated move.
   - Motif echo: when the bar plan names `motif_from` and the source bar has
     an interval at this note index (index > 0), +1.5 if
     `sign(iv) == sign(src)` and another +1.0 if `|iv − src| ≤ 1`.
   - Register gravity: −0.5 · ((c − tessitura_center) / 6)².
   - Cadence pull, final bar only: −0.25 · |c − target|.
   - The last note of any cadence bar: a non-chord tone gets −3.0.
   - First note of the piece: no interval terms.
4. **Sample** with `rng.choice(len(cands), p=softmax(scores))`.
5. **Note timing:** start = (bar·8 + position) · eighth. Duration =
   value · eighth · 0.95, the same articulation gap as today.
6. **Velocity:** position 0 gets `velocity_strong` (80), position 4 the
   midpoint (75), others `velocity_weak` (70). The final note gets
   `velocity_strong`.

The weights are module-level constants with a one-line comment each. They come
from the prototype tuning in §3.

### 5.3 Accompaniment: `compose/arrangement.py`

`_accompaniment_notes` gains `melody: Sequence[Note] = ()`, and `build_midi`
passes its `melody` through. Signatures otherwise stay the same.

- **Melody-aware thinning** `_thin_below(pitches, melody, t0, t1)`. Let `low`
  be the lowest melody pitch sounding anywhere in `[t0, t1)`. Keep the
  accompaniment pitches ≤ low − 3, so the accompaniment sits at least a minor
  third under the melody. If that leaves fewer than 3 pitches, keep the lowest
  `min(3, len(pitches))` instead. With no melody, nothing is thinned.
- **arpeggio:** 8 straight eighths per bar. The bass (lowest pitch) goes on
  eighth 0. The other seven cycle up and back down over the treble pitches
  `P[1:]` (cycle = `P[1:] + P[-2:0:-1]`). A treble note is dropped when
  thinning over its own eighth would remove it. Each note lasts 0.95 eighth.
  Velocity: bass 70, treble 62.
- **strum:** strokes at eighths 0, 2, 3, 5, 6, 7 (D · D U · U D U). Each
  stroke rings until the next one. Down strokes play the thinned voicing low
  to high with 8 ms between strings. Up strokes play the top 3 thinned pitches
  high to low. Velocity: beat 1 down 80, other downs 72, ups 60.
- **fingerstyle:** alternating bass. Beat 1 plays `P[0]`. Beat 3 plays `P[1]`
  when the voicing has ≥ 4 pitches, otherwise `P[0]`. On beats 2 and 4 the
  treble cluster is `P[1:]`, thinned, minus the alternate-bass pitch. Each
  note lasts 0.95 beat.
- **Final bar (all styles):** a single down stroke of the thinned voicing
  (10 ms per string, so at most 50 ms for six strings, inside A1's and A4's
  60 ms windows) that rings to the end of the bar. Velocity 80.
- Accompaniment notes are returned as `pretty_midi.Note`, as today.
  `arrange()` (§5.4) converts them to `Note`.

### 5.4 Tab and orchestration: `compose/api.py`

- New pure function `arrange(scale, progression, *, bars_per_chord, tempo_bpm, style, tuning, seed) -> Arrangement`.
  `Arrangement` is a frozen dataclass in `compose/api.py` with these fields:
  `melody: tuple[Note, ...]`, `voicings: tuple[VoicedChord, ...]`,
  `accompaniment: tuple[Note, ...]`, `melody_tab: tuple[TabbedNote, ...]`.
  It validates, builds the voicings, generates the melody, generates the
  accompaniment, and assigns the melody tab. `compose()` becomes: validate →
  `arrange()` → write files → render → ASCII tab → `Composition`. The
  metrics and regression tests call `arrange()` without file I/O.
- **Melody tab** comes from `music_decoder.tabs.assign_tabs` (the existing
  A\* with GuitarSet-tuned weights) in place of the per-note lowest fret. The
  prototype measured ≈ 3.5 ms per melody. Any note A\* drops falls back to the
  old lowest-fret position, so the tab never loses notes.
- **Chord tab anchors** go at the real chord start times
  (`i · bars_per_chord · bar`), not spread evenly over the melody span.
- `Composition.metadata` gains `"composer_version": 2`. This is additive, so
  JSON consumers keep working, and it lets anyone comparing seeds tell the old
  generator from the new one.

### 5.5 Metrics module: new `compose/metrics.py`

The pure functions behind §4, typed for `mypy --strict`. Each takes plain
sequences (`Note`, `ChordSymbol`, `Scale`, frets) and a tempo. One function per
metric, plus `summarize(arrangement, scale, progression, bars_per_chord, tempo_bpm) -> dict[str, float | None]`.
It lives in `src/` (not `tests/`) so a future `--report` flag or benchmark
script can reuse it. It does not import pretty_midi.

### 5.6 Data flow after the change

```text
compose(scale, progression, …)            # signature unchanged
  └─ arrange(...)                          # pure
       ├─ _validate
       ├─ voicings_for(c, tuning)[0]       # unchanged
       ├─ generate_melody(...)             # rhythm.plan_bars + scored pitches
       ├─ _accompaniment_notes(..., melody=melody)
       └─ assign_tabs(melody) (+ lowest-fret fallback)
  ├─ build_midi(...) → composition.mid     # two tracks, unchanged layout
  ├─ render_wav(...) → composition.wav
  └─ _ascii_tab_for(arrangement, ...)      # A* melody + real chord anchor times
```

## 6. Error handling and edge cases

- `bars_per_chord ≥ 1` and a non-empty progression are validated as today.
  `split_phrases` handles 1, 2, 3, 5, 6, 7, 9 … bars (unit-tested).
- A progression of a single 1-bar chord gives one cadence bar ending on the
  target.
- If the chord-scale pool has no candidate in range (impossible with a
  17-semitone window, but guarded), raise `ValueError` as today.
- Non-standard tunings: the melody window is in absolute MIDI pitch. A\*
  places it on whatever strings exist, and the T1 threshold is asserted for
  EADGBE only.
- `seed=None` still draws fresh entropy (`default_rng(None)`).

## 7. Testing strategy

- **TDD per unit:** `tests/unit/test_compose_rhythm.py` (templates sum to 8,
  phrase splitting, plan structure, echo indices, final cadence length),
  `test_compose_melody.py` (chord-scale pool, final target, scorer
  properties, determinism, the legacy `notes_per_bar` path),
  `test_compose_arrangement.py` (grid alignment, thinning, strum stroke
  layout, alternating bass, final-bar ring), `test_compose_metrics.py` (each
  metric on a hand-built example with a known answer),
  `test_compose_api.py` (`arrange()` purity, A\* tab, chord anchor times,
  `composer_version`).
- **Metrics regression test:** `tests/unit/test_compose_musicality.py`. Its
  matrix: major keys C, G, A#, E and minor keys A, E, C, F#, each with the
  five progressions of §3 = 40 progressions, × `bars_per_chord` {1, 2} ×
  seeds 0–4 = **400 melodies**, × 3 styles for the accompaniment metrics,
  at 120 BPM in EADGBE. It asserts every target in §3: aggregates over the
  matrix, plus the per-composition hard invariants (M4 max interval, M6, M7,
  M8, M13, M14, A1, A4, T1). Budget < 5 s. The matrix is built once in a
  module-scoped fixture.
- Existing tests: `test_strong_beats_favor_chord_tones_statistically` keeps
  `notes_per_bar=4` (the legacy grid) and should now pass by a wider margin.
  `test_all_notes_in_scale` and the e2e C-major pitch-class check stay valid
  because their progressions are diatonic. The public contract test stays
  unchanged.

## 8. Ranked improvements (impact against effort)

| Rank | Improvement | Weakness it fixes | Impact | Effort | Metrics |
|---|---|---|---|---|---|
| 1 | Phrase-structured rhythm with motif repetition (§5.1) | uniform quarters, no structure | high | medium | M9, M10, M11, M12 |
| 2 | Cadences: final tonic, long phrase-end notes (§5.1–5.2) | no resolution | high | low | M7, M8, M12 |
| 3 | Scored pitch selection with register gravity (§5.2) | random walk, repeats, unrecovered leaps | high | low–medium | M1–M6 |
| 4 | On-grid arpeggio (§5.3) | 11-tuplets | high for that style | low | A1 |
| 5 | Melody-aware thinning plus a final ringing chord (§5.3) | accompaniment above the melody, clashes, no ending | medium | low–medium | A2, A3, A4 |
| 6 | Chord-scale adaptation for non-diatonic chords (§5.2) | G vs G# over V7 in minor | medium | low | M13 |
| 7 | Lower tessitura plus A\* melody tab (§5.2, §5.4) | fret 20, one-string lines | medium | low | T1, T2 |
| 8 | Strum pattern and alternating-bass fingerstyle (§5.3) | static two-hit patterns | medium | low | structural unit tests, A1–A3 |
| 9 | Real chord anchor times in the ASCII tab, `composer_version` metadata | cosmetic, reproducibility | low | trivial | unit tests |

## 9. Non-goals

- **No voicing search or voice-leading optimisation.** That would need a
  larger shape library (CAGED / moveable barres) plus Viterbi selection. The
  baseline voice leading is already small (V1 = 1.61 semitones), and barre
  shapes make the result harder to play. V1 is only guarded.
- No new `compose()` parameters, CLI flags or UI controls (D1).
- No modes beyond major and natural minor, no time signatures other than 4/4,
  no swing, no tempo-dependent density.
- No humanised (random) timing or velocity, beyond the deterministic strum
  stagger.
- No let-ring sustain modelling for arpeggios, and no phrase-level dynamics
  (crescendo arcs).
- No corpus-trained or neural models, and no new dependencies.
- No change to `voicings.py`, `chords/voicings.py`, `tabs/` or `synth/`.
- No change to the analyze pipeline.

## 10. Decisions made in place of user approval

| # | Question brainstorming would have asked | Decision | Reason |
|---|---|---|---|
| D0 | Path classification | **Architectural** (full spec → plan) | It rewrites the melody generator, adds two modules and changes how the components fit together (pure `arrange()`, melody-aware accompaniment). |
| D1 | Add an optional `compose()` parameter (for example `melody_style`)? | **No** | The contract test freezes the exact parameter list. Improving the default output serves everyone without changing the API. |
| D2 | Approach | **Rule-based phrase planner + scored softmax sampling** (§5.0 option 2) | It covers the most metrics, is deterministic, needs no data, and was prototyped against the targets. |
| D3 | May a given seed produce a different melody than before? | **Yes**, recorded through `metadata["composer_version"] = 2` | Improving the music requires it. Determinism within a version is kept and tested. |
| D4 | `generate_melody` back-compat | Keep the function and its keyword style. `notes_per_bar` becomes `Optional` (legacy grid when set). Change the pitch window defaults. Remove the three probability/interval knobs. | Internal API with no external callers. Ignored-but-accepted parameters would mislead readers. |
| D5 | Melody register | **62–79 (D4–G5), centre 70** | Measured: T1 drops from a 20 max to 15, T2 from 0.18 to 0.11. Thinning keeps the melody on top. |
| D6 | How to stop the accompaniment masking the melody | **Thin accompaniment pitches above melody − 3, keeping at least 3** | Directly targets A2 and A3. It keeps the bass and chord body and is deterministic. Raising the melody instead would undo D5. |
| D7 | Melody over a non-diatonic chord tone | **Adapt the scale to the chord** (swap the clashing degree) | The standard chord-scale practice (harmonic minor over V7). Diatonic progressions are untouched, so the existing in-scale tests hold. |
| D8 | Where the metrics live | **`src/music_decoder/compose/metrics.py`** | Reusable and type-checked, and it keeps test files small. |
| D9 | Size of the regression matrix | **400 melodies, < 5 s** | Covers 8 keys, both modes, 10 progressions, 2 bar lengths and 3 styles while keeping `pytest -q tests/unit` fast. The full 3000-melody sweep was run once here for the baseline. |
| D10 | Voicing improvements | **Deferred** (§9) | The best impact for the effort is in melody and rhythm. The baseline voice leading is already fine. |
| D11 | How tight to set the targets | **Means sit 1–5 points inside the prototype's values; per-composition floors sit below the prototype's minimum** (for example M1 ≥ 0.60 against a prototype minimum of 0.625) | The implementation will not match the prototype bit for bit. The targets must catch real regressions without flaking when a constant is retuned. |
