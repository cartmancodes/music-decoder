# Compose improvements: a more musical, measurable composer

**Date:** 2026-11-06 (revision 2, after review round 1)
**Branch:** `feat/compose-improve`
**Status:** approved by the planner (Claude Opus 5.5) in an unattended Octet
pipeline. The human partner pre-approved the workflow (Opus plans, GPT-6 Astra
reviews, Opus implements). Every point where brainstorming would normally wait
for the user is listed in §10 with the decision taken and the reason. Review
round 1 (`docs/superpowers/reviews/2026-11-06-plan-review-r1.md`) is answered
finding by finding in §11 and at the end of the plan.

## 1. Intent

**What was asked:** make the output of `music_decoder.compose` (melody plus
accompaniment over a scale and chord progression, written as MIDI, WAV and
ASCII tab) noticeably more musical, and back every improvement with objective
musicality metrics asserted in tests across many seeds, keys, modes,
progressions and styles, compared against the current baseline.

**Hard constraints (from the request):**

- `compose()` keeps its public signature and the CLI keeps its flags and
  output kinds. New optional parameters would be allowed, but
  `tests/unit/test_public_contract.py::test_compose_signature_is_frozen`
  asserts the exact parameter list, so this design adds **none** (D1).
- Output stays deterministic for a given seed.
- No new dependencies and no trained models: rule-based and statistical
  methods over numpy and pretty_midi only.
- Gate commands stay green (`pytest -q tests/unit tests/integration`,
  `ruff check src tests`, `ruff format --check src tests`, `mypy --strict src`)
  and the suite stays fast. **Measured on 2026-11-06:** `ruff format --check`
  already fails on the base commit (7 files of drift). A format-only commit
  restores it first (D12).
- Test runs rewrite fixture markers (`tests/fixtures/synthetic/.*.synth_marker`).
  These must never be committed; stage explicit paths only.

**Assumptions (mine, not stated in the request):**

- "More musical" means the conventions of Western tonal folk and pop melody:
  - stepwise motion, with chord tones on strong beats
  - leaps recovered by a step in the opposite direction
  - leading tones that resolve to the tonic
  - phrases of two to four bars that end on long notes, and a final cadence
    on the tonic
  - rhythmic variety bound together by repetition (motifs)
  - an accompaniment that supports the tune: it sits under the melody, keeps
    each chord's identity and stays in time
- 4/4 time is kept. Every existing caller assumes it.
- A melody that changes for a given seed is acceptable. Determinism means
  "same composer version + same inputs + same seed gives the same output", not
  "byte-identical to the previous release" (D3).

**Success criteria:**

- The committed regression test compares the new composer with the
  pre-change composer **on identical, held-out inputs** (§3, §7).
- Every primary metric improves in every subgroup (major and minor; 1 and 2
  bars per chord; the edge matrix).
- Every absolute target in §3 is met, and every hard invariant holds.
- All gates stay green.

## 2. Current composer (what exists)

| File | What it does today |
|---|---|
| `compose/melody.py` | `generate_melody`: one note per beat (`notes_per_bar=4`, every note a quarter note). On beats 1 and 3 it uses a chord tone with probability 0.7, otherwise 0.3. The pool is limited to pitches within ±4 semitones of the previous note, falling back to the whole pool, which allows jumps of up to 23 semitones. The pick is uniform. Range C4–C6 (60–84). Natural minor only, so G clashes with E7's G# in A minor. No phrase structure, no cadence, no motifs. |
| `compose/voicings.py` | Looks up the single canonical EADGBE shape for the chord in `chords/voicings.py` and retunes it. `voicings_for` returns at most one candidate in practice. |
| `compose/arrangement.py` | Two-track MIDI. `strum`: a block chord on beats 1 and 3. `arpeggio`: the voicing played up then down, spread evenly over the bar (2·n−1 notes, e.g. 11 for six strings, so an **11-tuplet** off any grid). `fingerstyle`: the same bass note on 1 and 3, the treble on 2 and 4. The accompaniment ignores the melody. |
| `compose/api.py` | Orchestration. The melody tab puts each note at its own lowest fret, so the line sits on the high-e string up to fret 20. Chord tab anchors are spread evenly over the melody span and written into the same 64-column grid, where they can overwrite melody notes. |
| `config/hyperparameters.yaml` | Has a `composition:` block (notes per bar, chord-tone probabilities, Markov cap, velocities) that **no code reads**. |

## 3. Baseline and targets

### 3.1 How the numbers were produced

The measurement is reproducible and committed:

- `src/music_decoder/evaluation/compose_musicality.py` defines two fixed input
  matrices, the summaries and the subgroups.
- `scripts/compose_musicality_report.py` prints a summary, or writes one with
  `--write`.
- `tests/fixtures/compose/musicality_baseline.json` is that summary for the
  **pre-change composer**. Plan Task 3 writes it after a behaviour-preserving
  refactor and before any generator change. The refactor's output is checked
  byte for byte against the old `compose()`.

The two matrices:

- **Validation matrix** (400 melodies × 3 styles):
  - major keys C, G, A#, E × I–V–vi–IV, Imaj7–vi7–ii7–V7, I–IV–V–I,
    ii7–V7–Imaj7–Imaj7, I–vi–IV–V
  - minor keys A, E, C, F# × i–iv–V7–i, i–VI–III–VII, i–iv–v–i, i–VII–VI–V,
    ii°–V7–i–i
  - `bars_per_chord` ∈ {1, 2}, seeds 1000–1004, 120 BPM, EADGBE
- **Edge matrix** (80 melodies × 3 styles): eight progressions × seeds
  1000–1009, one bar per chord:
  - 3, 5, 7 and 9 bars
  - a single chord
  - the secondary dominant C–D7–G–C
  - the borrowed iv in C–Fm–C–G7
  - E minor with B7

**Held-out seeds.** The generator's weights were tuned on seeds 0–19 of an
exploratory 3000-melody sweep (15 keys, 10 progressions, 2 bar lengths). That
sweep was run once in the planning scratchpad and is not used for any
assertion. The matrices above use seeds ≥ 1000, so no assertion is made on
tuning data.

The candidate numbers below come from the revision-2 design, built in full in
a scratch copy of the repo and run through the same evaluation.

### 3.2 Paired results and targets (validation matrix, overall)

"Pooled" means hits ÷ eligible over the whole matrix. *Hard* means the
condition is checked as a per-composition min/max over **both** matrices.

| ID | Metric (§4) | Baseline | Candidate | **Target** | Why this threshold |
|---|---|---|---|---|---|
| M1 | Strong-beat chord-tone ratio | 0.863 (min 0.50) | 0.980 (min 0.75) | **mean ≥ 0.95; min ≥ 0.60** | Strong beats carry the harmony in tonal melody; a few accented non-chord tones (appoggiaturas) are idiomatic, hence < 1. |
| M2 | Stepwise ratio | 0.352 | 0.554 | **≥ 0.50** | Conjunct motion predominates in folk and pop melody; half the intervals as steps is a conservative floor. |
| M3 | Repeated-note ratio | 0.279 | 0.095 | **≤ 0.15** | Repeats are allowed but should not dominate; the baseline repeats more than a quarter of the time. |
| M4 | Large-leap ratio (> a fifth) | 0.0005 | 0.004 | **≤ 0.02** (allowed rise) | The baseline avoided sixths and octaves only through its ±4 cap. Its fallback still jumped up to 23 semitones (M4max). Occasional sixths and octaves are idiomatic; 2% bounds them. |
| M4max | Largest interval | max **23** | max 12 | **≤ 12, hard** | An octave is the largest singable melodic leap. This includes the move into the final note (review F1). |
| M5 | Leap recovery (pooled) | 0.0 (n = 5) | 0.761 (n = 771) | **≥ 0.70, n ≥ 100** | "Gap-fill": a leap is normally followed by a step back. |
| M6 | Ambitus | max 24 | max 17 | **≤ 17, hard** | An octave and a fourth is comfortable for a lead line. |
| M7 | Final resolution | 0.238 | 1.0 | **1.0, hard** | Every piece ends on the tonic, or on the last chord's root when that chord lacks the tonic. |
| M8 | Final note length (beats) | 0.95 | min 1.9 | **≥ 1.5, hard** | A final cadence needs a long note. |
| M9 | Distinct durations | 1.0 | 4.1 (min 3) | **mean ≥ 3.0; min ≥ 2** | Uniform rhythm is the most robotic trait of the baseline. |
| M10 | IOI entropy (bits) | 0.0 | 1.47 | **≥ 1.0** | Equivalent to at least two equally likely spacings. |
| M11a | Echo-pair coverage | 1.0 | 1.0 | **≥ 0.95** | The motif needs equal note counts in echoing bars (trivial for uniform rhythm, earned for varied rhythm). |
| M11b | Echo contour agreement | 0.336 | 0.695 | **≥ 0.60** | Chance is about a third; 0.6 means audible motif repetition. |
| M12 | Phrase-final long notes (layout-aware) | 0.0 | 1.0 | **mean ≥ 0.95; min 1.0, hard** | Phrases must end on a held note. |
| M13 | Exposed chromatic clashes | 0.094 | 0.024 | **≤ 0.04 and ≤ 0.7 × baseline** | Over altered chords, accented or long non-chord tones a semitone from the altered tone should be rare. They are still allowed (appoggiaturas), so the target is not zero (review F2). |
| M14 | Melody on the 8th grid | 1.0 | 1.0 | **1.0, hard** | Guard. |
| M15 | Leading-tone resolution (pooled) | 0.385 (n = 39) | 0.825 (n = 40) | **≥ 0.75, n ≥ 20** | The leading tone rises to the tonic at V–I / V–i (review F2). |
| A1 | Stroke heads on the 16th grid | arpeggio **0.104**; strum and fingerstyle 1.0 | 1.0 / 1.0 / 1.0 | **1.0 every style, hard** | Accompaniment in time. |
| A2 | Melody above overlapping accompaniment | arp 0.912 / strum **0.821** / fing 0.906 | 1.0 / 1.0 / 1.0 | **≥ 0.97 each** | The tune must not be masked. Judged over each note's whole sounding interval (review F3). |
| A3 | Exposed semitone clashes vs overlapping accompaniment | arp 0.117 / strum **0.145** / fing 0.110 | 0.062 / 0.087 / 0.067 | **≤ 0.10 each and ≤ 0.7 × baseline** | Accented minor seconds against the chord should be the exception; some passing dissonance on long notes is idiomatic. |
| A4 | Final chord rings, one attack | 0 | 1.0 | **1.0, hard** | A cadence needs a held final chord. |
| H1 | Essential tones heard in every bar (each style) | 1.0 | 1.0 | **1.0 every bar, hard** | Thinning must never remove a chord's third or seventh (review F4). The baseline never thinned. |
| T1 | Melody tab max fret | mean 15.9, max **20** | mean 13.0, max 15 | **≤ 15, hard** | Playable without high-position acrobatics. |
| T2 | Hand-position shifts per note | 0.171 | 0.124 | **≤ 0.14** | Fewer position shifts make the line easier to play. |
| V1 | Nearest-pitch proximity of shapes | 1.621 | 1.621 | **≤ baseline (guard)** | The shapes are unchanged. This is a proximity statistic only, not a voice-leading score (review F9). |

**Paired, per subgroup.** In each of overall, major, minor, 1 bar per chord,
2 bars per chord and the edge matrix, the candidate must beat the baseline
on M1, M2, M3, M7, M8, M9, M10, M11b, M12, M13, T2, A1-arpeggio and A2/A3/A4
in every style. Where a metric is undefined it is skipped; for example, M13
has no altered chords to judge in the all-diatonic major group. Every one of
these comparisons held in the scratch build.

**Edge matrix highlights** (baseline → candidate):

- M7: 0.10 → 1.0
- M15: 0.30 (n = 10) → 1.0 (n = 11)
- M11b: 0.33 → 0.76
- M13: 0.103 → 0.015
- A2 strum: 0.92 → 1.0
- T1 max: 20 → 15

### 3.3 What the baseline shows, in order of musical harm

1. **The rhythm is completely uniform.** Every note is a quarter note
   (1 distinct duration, IOI entropy 0).
2. **There are no cadences or phrases.** The tonic ends only 24% of pieces,
   every final note lasts 0.95 beats, and no phrase ends long.
3. **The arpeggio style is off the grid.** It plays 11-tuplets: 90% of its
   onsets miss the 16th grid.
4. **The pitch line is a random walk.** 28% repeated notes and 35% steps.
   Leaps are almost never recovered, the 4-semitone cap's fallback jumps up
   to 23 semitones, and motif repetition is at chance level.
5. **The melody and accompaniment collide.** Strum has accompaniment above
   the melody for 18% of notes and semitone clashes on 15% of exposed notes.
6. **Altered chords clash.** In minor keys natural-minor G sounds against
   V7's G#; the leading tone resolves only 39% of the time.
7. **The tab is hard to play.** It goes up to fret 20, nearly all on one
   string, and the merged renderer can overwrite melody notes with chord
   frets (review F5).

## 4. Metric definitions

All metrics are pure functions in `src/music_decoder/compose/metrics.py`.
Shared conventions:

- beat = 60 / tempo; bar = 4 beats; eighth = beat / 2.
- A note's **metric position** is `round(start_s / eighth) mod 8`. Positions 0
  and 4 are **strong** (beats 1 and 3).
- A melody note is **exposed** if it is at a strong position or lasts
  ≥ 0.9 beats.
- The **chord at time t** is
  `progression[min(floor(t / bar) // bars_per_chord, len − 1)]`.
- Chord tones, essential tones and scales come from `compose/harmony.py`.
- An accompaniment note **overlaps** a melody note when they share more than
  0.1 ms (`a.start < m.end − ε` and `a.end > m.start + ε`). Staggered and
  late-entering accompaniment voices therefore count (review F3).
- The **phrase layout** (`rhythm.phrase_layout(n_bars)`) depends only on the
  bar count. It gives the phrase lengths, each phrase's cadence bar and the
  motif echo pairs, and every melody, old or new, is judged against it
  (review F8).
- Ratios return `None` when nothing qualifies, which is distinct from 0.

| ID | Definition |
|---|---|
| M1 | Among melody notes at strong positions: the fraction whose pitch class is a chord tone of the chord at their onset. |
| M2 / M3 / M4 | Fraction of intervals with \|i\| ∈ {1, 2} / = 0 / > 7. M4max = max \|i\|. |
| M5 | Counts (leaps \|i\| ≥ 5 followed by another interval j; of those, j opposite in sign with 1 ≤ \|j\| ≤ 2), pooled over a matrix. |
| M6 | max(pitch) − min(pitch). |
| M7 | The last note's pitch class is the tonic if the final chord contains it, otherwise the final chord's root. |
| M8 | Duration of the last note, in beats. |
| M9 | Number of distinct durations, using onset-to-onset spacing quantised to eighths (the last note uses its own length). |
| M10 | Shannon entropy (bits) of inter-onset intervals quantised to eighths. |
| M11a | Fraction of the layout's echo pairs whose two bars hold the same number (≥ 2) of notes. `None` if the layout has no echo pairs. |
| M11b | Over the covered pairs, the fraction of intra-bar interval signs that agree, averaged. `None` if no pair is covered. |
| M12 | Fraction of the layout's cadence bars whose last note lasts ≥ 1.5 beats. |
| M13 | Over melody notes that sound over a chord with an out-of-scale tone and are exposed: the fraction that are **non-chord tones** a semitone from one of those foreign tones. Chord tones and short unaccented notes never count. `None` if no such note exists. |
| M14 | Fraction of melody onsets within 1 ms of the eighth grid. |
| M15 | Counts (consecutive pairs (a, b) where a's pitch class is the leading tone (tonic − 1) and a chord tone of a's chord, and b starts in a different chord that contains the tonic; of those, b = a + 1). |
| A1 | Group accompaniment onsets within 60 ms of a group's first onset (one stroke). The fraction of group heads within 1 ms of the 16th grid. |
| A2 | Fraction of melody notes higher than every overlapping accompaniment pitch (true when none overlaps). |
| A3 | Among exposed melody notes: the fraction overlapping an accompaniment pitch at pitch-class distance 1 or 11. |
| A4 | Final-bar accompaniment notes all start within 60 ms of the bar start and end at or after the bar end minus 1 ms. |
| H1 | Per bar: the essential pitch classes of the chord that its shape contains must all be heard (accompaniment or melody, overlapping the bar). Reported as the fraction of bars, plus the final bar alone. |
| T1 | Max fret of `Arrangement.melody_tab`. The rendered tab shows exactly these positions (review F5; enforced by a round-trip test). |
| T2 | Walk the fretted (> 0) melody frets with a hand window of at most 4 frets. A note outside the window starts a new window and counts as a shift. Shifts ÷ (fretted notes − 1). |
| V1 | For successive shapes with different chords: the mean, over the second shape's pitches, of the distance to the nearest pitch of the first. A **proximity guard only**: voices may share a nearest source, dropped voices cost nothing, and crossings are ignored (review F9). |

## 5. Design

### 5.0 Approaches considered

1. **Retune the existing Markov walk.** Cheap, but it cannot create rhythm,
   phrases, cadences or motifs. Rejected.
2. **Rule-based phrase planner plus scored stochastic pitch choice**
   (**chosen**). Rhythm comes from curated one-bar templates arranged by a
   phrase layout. Pitches are sampled from a softmax over hand-set musical
   scores. It is deterministic under a seeded `numpy` generator, needs no data,
   and every rule maps onto a metric.
3. **An n-gram model learnt from the music21 corpus.** Idiomatic, but it means
   shipping tables, it is hard to steer toward cadences and chord fit, and it
   is harder to test. Deferred. The scorer could later absorb corpus-derived
   interval weights.

The accompaniment follows the same reasoning: fix the patterns and make them
melody-aware and identity-preserving, rather than adding a voicing search (§9).

### 5.1 Pitch-class helpers: new `compose/harmony.py`

- `scale_pcs`, `chord_pcs`, `tonic_pc` and `leading_tone_pc`.
- `essential_pcs(chord)` returns the identity tones:
  - maj / min: the third
  - 7 / maj7 / min7: the third and the seventh
  - dim / aug: the third and the altered fifth
  - sus4: the fourth
- `chord_scale_pcs(scale, chord)` replaces, for each out-of-scale chord tone,
  **only the one scale degree it chromatically alters**. The tone's interval
  above the chord root identifies that degree:
  - a major third, major seventh or augmented fifth is raised, so it replaces
    the degree below it
  - a minor third, minor seventh, diminished fifth or fourth is lowered, so
    it replaces the degree above it
  - for a foreign root or perfect fifth, it replaces the neighbour that is in
    the scale, preferring the upper one (flat-side borrowing)

  Every other degree, including the tonic, is kept (review F2). Examples:
  - E7 in A minor gives A harmonic minor (G → G#, A kept).
  - D7 in C major replaces F with F# and keeps G.
  - C7 replaces B with B♭; Fm replaces A with A♭; A major replaces C with C#.
  - Diatonic chords leave the scale unchanged.

### 5.2 Rhythm and phrase planner: new `compose/rhythm.py`

- `BODY_RHYTHMS` (11 one-bar templates in eighths) and `CADENCE_RHYTHMS` (5;
  each ends on ≥ 4 eighths). All sum to 8.
- `split_phrases(n_bars)`: phrases of 4 bars. A remainder of 2 or 3 bars
  becomes a shorter last phrase; a remainder of 1 (with n > 4) turns the last
  4 into 3 + 2.
- `phrase_layout(n_bars) -> PhraseLayout(phrase_lengths, cadence_bars, echo_pairs)`.
  It depends only on the bar count:
  - A 4-bar antecedent echoes bar 1 in bar 3.
  - A consequent phrase echoes the preceding antecedent bar for bar
    (`min(body, antecedent body)` pairs).
  - Phrases alternate antecedent / consequent by index.
- `plan_bars(n_bars, rng) -> list[BarPlan]` follows the layout:
  - Echoing bars reuse their source bar's rhythm (and get
    `motif_from = source`).
  - Antecedent bodies draw A and B.
  - Unmatched consequent bars draw fresh templates.
  - Every phrase ends on a cadence template, and the final phrase's cadence
    must end on ≥ 4 eighths.

### 5.3 Melody: rewrite of `compose/melody.py`

`generate_melody` keeps its keyword interface except as follows (D4):

- `notes_per_bar: int | None = None`. `None` uses the phrase planner; an
  integer uses the legacy uniform grid, with the old timing and old
  strong-note rule, plus the new scorer and final target.
- `pitch_low` / `pitch_high` defaults are 62 / 79 (D4–G5).
- New keyword `tessitura_center = 70`.
- New keyword `chord_floors: Sequence[int | None] | None`, one entry per
  chord. A floor applies only if it leaves at least `MIN_FLOOR_CANDIDATES = 5`
  pitches.
- `strong_beat_chord_tone_prob`, `weak_beat_chord_tone_prob` and
  `markov_max_interval_semitones` are removed. Nothing passes them, and the
  dead `composition:` YAML block that mirrored them is deleted (D15).

Per bar:

1. **Candidates:** the chord-scale pitches (§5.1) in the window, at or above
   the chord's floor.
2. **Final target** (last bar only): the tonic if the final chord contains it,
   otherwise its root, taking the candidate nearest the previous pitch.
3. **Score** each candidate (all weights are module constants). Exclusions:
   - interval > 12: `-inf`
   - final bar only: |c − target| > 12 is `-inf`. Every note of the final
     bar stays within an octave of the target, so the forced final note can
     never be more than an octave away (review F1). The target itself always
     satisfies both constraints, so a path always exists. `_sample` still
     falls back to the nearest pitch if every score is `-inf`.

   Scored terms:
   - **interval:** {0: −0.8, 1: +0.8, 2: +0.8, 3: −0.2, 4: −0.6, 5: −1.2,
     6: −4.0, 7: −1.8, 8: −3.5, 9: −3.5, 10: −4.0, 11: −4.5, 12: −3.0}
   - **chord fit**, first rule that applies:
     - strong position: chord tone +2.0, non-chord tone −3.0
     - ≥ 3 eighths: +1.0 / −1.5
     - ≥ 2 eighths, non-chord tone a semitone from a chord tone: −1.0
     - otherwise, chord tone: +0.3
   - **leap recovery:** after a leap of |prev| ≥ 5, an opposite step gets +2.5
     and a same-direction or repeated move −2.5
   - **leading-tone resolution:** +3.0 for the step up a semitone when the
     previous note is the leading tone, it was a chord tone, and the chord
     has changed to one containing the tonic (review F2)
   - **motif echo:** +1.5 for the same direction as the echoed interval, and
     +1.0 more within a semitone of it
   - **register gravity:** −0.5 · ((c − 70) / 6)²
   - **cadence pull** (final bar): −0.25 · |c − target|
   - **phrase-ending non-chord tone:** −3.0
4. **Sample** with `rng.choice` over the softmax.
5. **Timing and velocity:**
   - start = bar · bar_s + slot offset; duration = 0.95 × slot.
   - Velocity is 80 on beat 1, 75 on beat 3 and 70 elsewhere; the final note
     is 80.

### 5.4 Accompaniment: `compose/arrangement.py`

`_accompaniment_notes(..., melody=())` and `build_midi` keep their
signatures; `build_midi` passes its `melody` through.

- **Thinning policy** (review F4): `_thin_below(pitches, essential, melody, t0, t1)`.
  1. Find the lowest melody pitch sounding in the window; nothing is thinned
     if the melody is silent.
  2. Keep the pitches at least a minor third below it.
  3. **Always keep the bass.**
  4. Put back each essential pitch class that neither the kept notes nor the
     sounding melody supply, at its lowest voicing pitch.

  The chord therefore never loses its third or seventh, and a doubled
  melody unison is removed. The old "keep the lowest three" rule is gone
  (it could keep D3 A3 D4 and drop F#).
- **Melody floors** (in `arrange()`, §5.6) are the other half of the policy.
  For each chord shape the melody stays a minor third above the highest of
  the shape's lowest essential tones (open D: F#4 + 3 = A4). So the restored
  identity tones still sit under the tune; A2 reached 1.0 in every style.
  When a floor cannot apply (too few candidates), identity wins over
  register, and A2 measures any such cases.
- **arpeggio:** 8 straight eighths: the bass on the downbeat, then seven notes
  cycling up and down the treble. A note is skipped when thinning would drop
  it.
- **strum:** D · D U · U D U on eighths 0, 2, 3, 5, 6, 7. Down strokes stagger
  8 ms per string low→high; up strokes take the top 3 strings high→low. Each
  stroke rings until the next.
- **fingerstyle:** the root on beat 1, the second-lowest pitch on beat 3
  (with ≥ 4 pitches), and the thinned treble on beats 2 and 4.
- **Final bar (every style):** one stroke (10 ms per string) ringing to the
  bar end.

### 5.5 Tab rendering: new `compose/tabtext.py`

`render_composition_tab(melody_tab, voicings, *, bars_per_chord, tempo_bpm, tuning)`
returns two labelled blocks (review F5):

- **`Melody`:** one column per note in playing order, bars separated by `|`,
  four bars per line. Every melody position is printed and none can be
  overwritten.
- **`Chords`:** each shape under its chord name (`x` = muted).

String labels follow the tuning (`E A D G B e`; `D A D G A d`; `D# G# C# F#
A# d#`). Rhythm is not drawn; the MIDI is the source of truth for timing. The
shared `tabs/render.render_ascii_tab` (used by `analyze`) is not changed.
`Composition.ascii_tab` stays a `str` and the CLI and UI print it exactly as
before; only its content layout changes (D13).

### 5.6 Orchestration: `compose/api.py`

- **`arrange(scale, progression, *, bars_per_chord, tempo_bpm, style, tuning, seed) -> Arrangement`**
  is pure. It reads no files or configuration (review F6). `Arrangement`
  holds `melody`, `voicings`, `accompaniment` and `melody_tab`, all as
  tuples. `compose()` = `arrange()` + MIDI + WAV + `render_composition_tab`
  + `Composition`.
- **`accompany(voicings, melody, *, bars_per_chord, tempo_bpm, style, tuning)`**
  returns the accompaniment as `Note`s. The evaluation uses it to accompany
  one melody in the other two styles.
- **Melody tab:** `tabs.assigner.assign_tab` (the lower-level A\* that takes
  its parameters directly), called with `COMPOSE_TAB_WEIGHTS` and
  `COMPOSE_TAB_MAX_FRET = 22`. These are pinned in `compose/api.py` with the
  values tuned for analysis in 2026-10 (D14). A test pins them equal to
  `tabs._DEFAULT_WEIGHTS`, and other tests prove that neither
  `load_hyperparameters` nor `assign_tabs` is called and that
  `MUSIC_DECODER_HYPERPARAMETERS` cannot change the output. Notes A\* drops
  fall back to the lowest fret.
- **Floors:** `_melody_floor(voicing, tuning)`, as described in §5.4.
- **Validation:** `bars_per_chord < 1` raises `InvalidProgressionError`
  (it used to give an empty melody silently).
- **Metadata:** `metadata["composer_version"] = 2`.

### 5.7 Evaluation: new `evaluation/compose_musicality.py`, script and baseline fixture

- `realize(scale, numeral)`, `validation_inputs()`, `edge_inputs()`,
  `build_cases(inputs)` (one `arrange()` per input, plus `accompany()` for
  the other styles) and `case_metrics(case)`.
- `summarize(cases)` gives mean/min/max for every metric, and pooled
  ratio + n for M5 and M15. The groups are overall, major, minor, bars1 and
  bars2. `evaluate()` returns both matrices.
- `scripts/compose_musicality_report.py [--write PATH]` prints or saves the
  summary, with `"source"` set to the git HEAD.
- `tests/fixtures/compose/musicality_baseline.json` (about 25 KB) is written
  at plan Task 3, on the old composer.

### 5.8 Data flow after the change

```text
compose(scale, progression, …)            # signature unchanged
  └─ arrange(...)                          # pure: no files, no YAML
       ├─ _validate (mode, non-empty, bars_per_chord ≥ 1)
       ├─ voicings_for(c, tuning)[0]       # unchanged
       ├─ _melody_floor(voicing) per chord
       ├─ generate_melody(..., chord_floors)   # phrase layout + scored pitches
       ├─ accompany(voicings, melody, ...)     # thinning keeps bass + identity tones
       └─ assign_tab(melody, pinned weights)   # + lowest-fret fallback
  ├─ build_midi(...) → composition.mid     # two tracks, unchanged layout
  ├─ render_wav(...) → composition.wav
  └─ render_composition_tab(...)           # "Melody" block + "Chords" block
```

## 6. Error handling and edge cases

- `bars_per_chord < 1` raises `InvalidProgressionError` (new). An empty
  progression and a bad mode raise as before. The CLI turns these into a
  clean `ClickException`.
- Phrase layout and planning handle every bar count ≥ 1 (unit-tested for 1–13
  bars, with layouts for 1, 3, 4, 5, 7, 8 and 9 spelled out).
- A single 1-bar chord gives one final cadence bar on the target, with a
  ringing final chord.
- A chord floor that leaves fewer than 5 candidates is ignored. An empty pool
  raises `ValueError` (impossible with the default 17-semitone window).
- Non-standard tunings: the melody window is in absolute MIDI pitch. A\*
  places notes on whatever strings exist, with a lowest-fret fallback, and a
  test checks every position's pitch for Drop D, DADGAD and D standard. T1 is
  asserted for EADGBE only.
- Extreme tempos (40 and 220 BPM, the UI slider's limits) keep the grid and
  the total length (unit-tested).
- `seed=None` still draws fresh entropy.

## 7. Testing strategy

- **TDD per unit**, with each behavioural failing test placed in the task
  that implements it:
  - `test_compose_rhythm.py`: templates, splitting, layout, plan follows
    layout, determinism
  - `test_compose_harmony.py`: essential tones, the chord-scale rule, the
    E7, D7, C7, Fm, A, B♭ and C#maj7 cases
  - `test_compose_metrics.py`: every metric on hand-built examples,
    including the review's staggered-strum counterexample, non-overlap
    controls, 5-bar layout cases, unequal echo bars, the chord-tone-next-to-a-
    foreign-tone case, and the V1 blind-spot counterexample
  - `test_compose_evaluation.py`: numerals, matrix sizes, held-out seeds,
    summary shape, the committed baseline
  - `test_compose_melody.py`:
    - the review's seeds 283 and 5774
    - a 400-seed max-interval sweep over three progressions
    - an adversarial final-target score
    - E7 and D7 keeping A and G
    - leading-tone resolution ≥ 0.8
    - floors, and a floor ignored when it leaves too few pitches
    - tempos 40 and 220
  - `test_compose_arrangement.py`:
    - thinning keeps the bass and restores missing thirds, but not ones the
      melody supplies
    - minor and seventh-chord identity
    - the review's D-major seed-9 cadence in every style
  - `test_compose_tabtext.py`: labels, every note kept, two-digit frets,
    wrapping, empty bars, chord block, the review's overwrite case
  - `test_compose_api.py`:
    - determinism and no files written
    - no config reads, and the env var has no effect
    - pinned weights, fallback, `bars_per_chord` validation, three tunings
    - floor values and the D-major cadence
    - a rendered-tab round trip for the review's C–G–Am–F seed 0 and a
      16-bar piece
- **Regression suite** `tests/unit/test_compose_musicality.py`:
  - computes `evaluate()` once per module
  - asserts every absolute target and hard invariant in §3.2 on both
    matrices
  - checks per-subgroup improvement over the committed baseline
  - checks a relative clash reduction (≤ 0.7 × baseline for M13 and A3)
  - checks the guards
- **Proof that the suite bites:** each of the following deliberate breaks was
  applied alone in the scratch build. Every one turned the suite red; each
  item names the test that catches it.
  - repeat weight +3.0 → M2/M3/M5 targets
  - final-target reachability removed → the 400-seed sweep
  - thinning disabled → A2/A3
  - identity restore disabled → the thinning and D-cadence unit tests
  - floors removed → the D-cadence API test
  - lowest-fret tab → T2 and the paired comparisons
  - plain scale → M13
  - no leading-tone term → M15
  - inverted motif term → M11b
  - a short cadence template → hard invariants
- **Runtime:** in the scratch build, `evaluate()` takes 2.3–3.0 s
  standalone, and the regression module takes 2.3–4.7 s under pytest,
  depending on machine load (A\* is about 85% of it; removing the per-call
  YAML read halved A\*). The plan records this expectation and does **not**
  assert wall-clock time (review F10). On a slower runner the module may take
  longer; reducing the matrix is preferred over weakening coverage only if it
  exceeds about 10 s there.
- The plan was replayed task by task on a fresh copy: all four gates were
  green after every task, and the Task 3 baseline regenerated identically.

## 8. Ranked improvements (impact against effort)

| Rank | Improvement | Weakness | Impact | Effort | Metrics |
|---|---|---|---|---|---|
| 1 | Phrase layout + rhythm templates + motif echo (§5.2) | uniform quarters, no structure | high | medium | M9–M12 |
| 2 | Cadences: final tonic within reach, long phrase ends (§5.2–5.3) | no resolution | high | low | M4max, M7, M8, M12 |
| 3 | Scored pitch choice (§5.3) | random walk, repeats, unrecovered leaps | high | low–medium | M1–M6 |
| 4 | On-grid arpeggio (§5.4) | 11-tuplets | high for that style | low | A1 |
| 5 | Identity-preserving thinning + melody floors + ringing final chord (§5.4, §5.6) | masking, clashes, ambiguous cadences | medium–high | medium | A2–A4, H1 |
| 6 | Role-based chord scale + leading-tone resolution (§5.1, §5.3) | G vs G#, unresolved leading tones | medium | low | M13, M15 |
| 7 | Lower tessitura + A\* melody tab with pinned weights (§5.3, §5.6) | fret 20, one-string lines, config coupling | medium | low | T1, T2 |
| 8 | Collision-free tab text (§5.5) | chord frets overwriting the melody | medium | low | round-trip tests |
| 9 | Strum pattern + alternating bass (§5.4) | static two-hit patterns | medium | low | structural tests, A1–A3 |
| 10 | Paired evaluation, baseline fixture, report script (§5.7) | unverifiable claims | enabling | medium | all |

## 9. Non-goals

- **No voicing search or voice-leading optimisation**, and no new shapes. This
  would need a larger shape library plus Viterbi selection, and barre shapes
  make the result harder to play. V1 is only a proximity guard. It is **not**
  evidence that voice leading is good, so no such claim is made (review F9).
- No new `compose()` parameters, CLI flags or UI controls (D1).
- No modes beyond major and natural minor (with chord-driven alterations), no
  time signatures other than 4/4, no swing, no tempo-dependent density.
- No humanised (random) timing or velocity beyond the deterministic strum
  stagger. No let-ring modelling for arpeggios; no phrase-level dynamics.
- No corpus-trained or neural models; no new dependencies.
- No change to `compose/voicings.py`, `chords/`, `tabs/` (including the shared
  ASCII renderer), `synth/` or the analyze pipeline. The only config change is
  deleting the unread `composition:` block (D15).
- No rhythm notation in the ASCII tab.

## 10. Decisions made in place of user approval

| # | Question | Decision | Reason |
|---|---|---|---|
| D0 | Path classification | **Architectural** | New modules, a rewritten generator, and changed component boundaries (pure `arrange()`, melody-aware accompaniment). |
| D1 | Add a `compose()` parameter? | **No** | The contract test freezes the parameter list; better defaults serve everyone. |
| D2 | Approach | **Rule-based phrase planner + scored softmax sampling** | Covers the most metrics, deterministic, needs no data. |
| D3 | May a seed's melody change? | **Yes**, with `metadata["composer_version"] = 2` | Required to improve the music; determinism within a version is tested. |
| D4 | `generate_melody` back-compat | Keep the function; `notes_per_bar` becomes optional; new window defaults; remove the three dead knobs; add `tessitura_center` and `chord_floors` | Internal API with no external callers. |
| D5 | Melody register | **62–79, centre 70**, with per-chord floors | T1 max falls from 20 to 15. The floors keep identity tones under the tune. |
| D6 | Masking vs chord identity | **Thin above melody − 3, always keep the bass, restore missing essential tones; floors keep the restored tones under the melody** | Review F4. Identity is never traded away, and register clearance is achieved by moving the melody, not by deleting the third. |
| D7 | Melody over an altered chord tone | **Replace only the altered degree (role-based)**, keep the tonic, add leading-tone resolution | Review F2. This is harmonic minor over V7 and the correct local scale for secondary dominants and borrowed chords. |
| D8 | Where metrics live | `compose/metrics.py` (+ `compose/harmony.py` helpers) | Reusable, type-checked, one definition shared by baseline and candidate. |
| D9 | Evaluation matrices | **400 validation + 80 edge melodies, held-out seeds ≥ 1000, committed baseline JSON, paired comparisons per subgroup** | Review F7. The comparison is on identical inputs, with no assertions on tuning seeds. |
| D10 | Voicing improvements | **Deferred** (§9) | Best impact for the effort is in melody, rhythm and accompaniment behaviour. |
| D11 | How targets are set | **Absolute floors justified musically (§3.2 "Why"), plus paired improvement over the baseline in every subgroup, plus relative clash reductions.** Allowed regressions are named (M4) and bounded. | Review F7 replaced the earlier "just inside the prototype" rule. |
| D12 | Pre-existing `ruff format` drift | **A separate `style:` commit first** | A green gate and reviewable diffs. |
| D13 | Tab text format | **Two labelled blocks (melody columns + chord shapes) from a compose-specific renderer** | Review F5. Collision-free by construction; the shared analyze renderer is untouched; `ascii_tab` stays a string. |
| D14 | Melody-tab A\* parameters | **Pinned in `compose/api.py` and passed to `tabs.assigner.assign_tab`** | Review F6. Compose output depends only on its inputs, seed and version, and A\* time halves. |
| D15 | The unread `composition:` YAML block | **Delete it** | It documents removed knobs and suggests a configurability that does not exist. Nothing reads it (verified by grep and by the loader, which does not model it). |
| D16 | Regression-test runtime | **Document the measured 2–5 s; no wall-clock assertion** | Review F10. Runtime varies with load; A\* in `tabs/` is out of scope to optimise. |

## 11. Review round 1: what changed

| Finding | Response |
|---|---|
| F1: the forced final note could leap > 12 | **Fixed.** Final-bar candidates must lie within 12 of the target (§5.3). The review seeds are tested, plus a 400-seed sweep and an adversarial score test. |
| F2: the chord scale deleted the tonic; M13 too blunt | **Fixed.** A role-based altered-degree rule (§5.1). M13 was redefined around exposed non-chord tones and is no longer a hard zero. M15 (leading-tone resolution) was added and the scorer resolves the leading tone. The D7 text was corrected. |
| F3: A2/A3 probed only the onset | **Fixed.** Overlap over the whole note; staggered and late-entry counterexamples plus controls; the baseline was re-measured with the same definitions. |
| F4: thinning could erase chord identity | **Fixed.** The bass and identity tones are kept, the melody floors keep them under the tune, and H1 coverage is asserted every bar. The review's D-major case is tested. |
| F5: chord frets overwrote melody frets | **Fixed.** A collision-free two-block renderer, with round-trip tests on the review's case and a 16-bar piece. |
| F6: `arrange()` read YAML | **Fixed.** Pinned weights through `assign_tab`; tests reject config reads and env-var influence. |
| F7: thresholds calibrated to the prototype | **Fixed.** A committed paired baseline on identical held-out inputs, subgroup comparisons, an edge matrix, justified thresholds, and a ten-point sabotage check. |
| F8: phrase metrics ignored the planner's layout | **Fixed.** A shared `phrase_layout`; coverage reported separately from agreement; 5/7/9-bar cases. |
| F9: V1 is not voice leading | **Fixed by renaming and documenting** (`nearest_pitch_proximity`, with a blind-spot test). **Not done:** a new one-to-one voice-leading metric, because shapes are out of scope (YAGNI). |
| F10: the 5 s budget was not met | **Addressed.** YAML reads removed and the evaluation profiled; the expectation is documented without a wall-clock assertion. **Not done:** cutting coverage to hit a number. |
