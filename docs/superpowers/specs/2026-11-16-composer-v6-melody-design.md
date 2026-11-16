# Composer v6: melodies built from what real tunes do

Date: 2026-11-16. Status: design, awaiting approval. Follows `2026-11-16-composer-v5-design.md`.

## 1. Why

The user listened to the v5 samples and did not like the melody of any of them. The guitar part, the sound and the
playability work are not in question. The goal of v6 is a fingerstyle melody that a broad audience would hear as a tune,
in every key, major and minor.

v5's melody is written by hand-made rules that were never checked against real melodies. This design replaces the
guesses with measurements, taken from research papers and from two bodies of real melodies.

## 2. What the research found

### 2.1 Sources

- **POP909, cleaned by Dai et al.** (MIT licence, `github.com/Dsqvival/hierarchical-structure-analysis`): 908 pop songs
  with the melody on a sixteenth grid with rests, chords, key and human phrase labels. 543 major, 365 minor. Measured
  here on whole songs and on 5,070 eight-bar windows that start at a labelled phrase.
- **Our pilot dataset** (`spikes/preference-pilot/`, 488 YouTube songs, half "liked"): the vocal-melody features
  already extracted, and a test showing that note-by-note vocal lines can be recovered from the audio (Let It Be came
  out recognisable, with short ghost notes to clean).
- **Papers** (read this session unless marked): Temperley 2006 (probabilistic melody model, Essen corpus); Temperley
  and de Clercq 2013 (rock corpus); Tan, Lustig and Temperley 2019 (anticipatory syncopation); von Hippel and Huron
  2000 (skip reversal is regression to the mean); Dai, Zhang and Dannenberg 2020 (phrase structure in POP909);
  Van Balen and Burgoyne (Hooked on Music); Jakubowski et al. 2017 (earworms); Gold et al. 2019 (liking peaks at
  medium surprise); Zhang et al. 2023 (WuYun, skeleton-first generation).

The numbers in 2.2 were measured with throwaway scripts on the dataset's sixteenth grid, over every window. They show
the size of the gap. The ranges the engine is gated against are recomputed by the statistics script on the eighth-grid
view of section 5.1, so they will differ a little. The dataset is not in the repo; it is fetched with
`git clone --depth 1 https://github.com/Dsqvival/hierarchical-structure-analysis ~/.music-decoder/datasets/pop909-hsa`.

### 2.2 The composer against real melodies

"Real" is the middle half (25th to 75th percentile) of POP909 eight-bar windows unless noted. "v5" is the median of 72
fresh fingerstyle pieces, 12 keys, major and minor, three tempos.

| Measure | v5 | Real | Verdict |
|---|---|---|---|
| Same note repeated (share of moves) | 0.06 | 0.14–0.31 (pilot vocals 0.28–0.42) | far too low |
| Steps of 1–2 semitones | 0.64 | 0.36–0.53 | far too high |
| Thirds (3–4 semitones) | 0.19 | 0.12–0.24 | fine |
| Leaps of a fourth or more | 0.10 | 0.09–0.19 | low side |
| Notes per bar | 3.75 | 4.5–6.5 at a median tempo of 74 (about 1.7 notes per second) | too sparse at slow tempos |
| Distinct bar rhythms in 8 bars | about 3 | 5–7 | too few |
| Bar rhythm equal to the bar 4 earlier | 0.82 | 0.00–0.50 (median 0.25) | far too repetitive |
| Bar rhythm equal to the bar 2 earlier | 0.50 | 0.00–0.20 | far too repetitive |
| Three-interval figures that recur | near 0 | 0.18–0.55 | missing |
| Onsets off the quarter-note beat | 0.23 | 0.45–0.65 | too square |
| Bars that do not attack beat 1 | 0.00 | 0.00–0.38 (median 0.12) | missing |
| Phrases entered by pickup notes | 0.00 | 0.59 of phrases | missing |
| Chord tone on beats 1 and 3 | 0.93–1.00 | 0.62–0.92 | too safe |
| Highest note's position in 8 bars | forced to 55–75% | median 30% (9–59%) | wrong rule |
| Range (semitones) | 13 | 12–16 | fine |
| Pentatonic share of notes | 0.79 | 0.82–0.98 (Chinese pop; Western check pending) | low side |

Liked and not-liked songs in the pilot set do not differ on these measures, so they describe what tunes are, not what
makes one tune more popular than another.

### 2.3 What v5 does that the data contradicts

1. `INTERVAL_WEIGHT` rewards steps (+0.8) and punishes repeated notes (−0.8); the best-of-12 score does the same
   (`W_STEPS = +1`, `W_REPEATS = −1.5`). Real melodies repeat a note in about a quarter of their moves.
2. Rhythm is one of 11 hand-written one-bar templates, drawn at random and then reused as `[A, B, A, cadence]`, with
   the next phrase copying it. Real tunes use five to seven different bar rhythms in eight bars and rarely repeat a
   bar's rhythm exactly.
3. Every bar attacks beat 1, no phrase has a pickup, and there is no syncopation. In real melodies more than half the
   onsets are off the beat and most phrases start with a pickup.
4. Strong beats almost always take a chord tone (`+2.0` against `−3.0`). Real melodies do so about three times in four.
5. The phrase peak is forced into the 55–75% window and the contour onto an arch. In real eight-bar tunes the highest
   note comes early as often as late.
6. Each note is picked one at a time by a local score, so nothing a listener could call a figure ever comes back. The
   "motif echo" only rewards the same direction.

Two suspicions the data did not support: the range (13 semitones against 12–16) and the low register are not the main
problem, so the hand-shape model stays as it is.

## 3. Goals, non-goals, anti-goals

**Goals**

- G1. A new melody engine for the fingerstyle style whose output sits inside the real-melody ranges of section 2.2,
  in all 12 keys, major and minor, at tempos 66–120.
- G2. The statistics the engine uses are learned from data by a script, kept in one small file in the package, and
  can be rebuilt.
- G3. Automatic gates compare generated melodies with the real-melody ranges, so "sounds like a tune" is measured
  and cannot regress unnoticed.
- G4. The user hears samples as soon as the engine works, before the remaining work is finished.
- G5. Every v4 guitar guarantee holds: playable moments (P1), melody on top (P2), chord identity (P3), tab equals
  MIDI (P4), final fallback (P5).

**Non-goals (this plan)**

- A sixteenth-note grid. The arranger, the tab and the performance layer all assume eighths; 56% of real bars use
  eighths only, and those are the bars the rhythm tables are learned from.
- Notes tied across a bar line (the hand shape can change there).
- A melody-first hand model that lifts the register up the neck.
- Strum and arpeggio styles; they keep the v5 rules engine.
- More scale types than major and minor (section 9).
- A neural melody model. No chord-conditioned model with released weights, a usable licence and CPU speed was found.
  The existing `--melody-model amt` option keeps working.

**Anti-goals**

- Copying a recognisable melody. Only statistics leave the datasets: bar rhythm patterns, interval counts, degree
  counts. No pitch sequence longer than a three-interval figure count is stored, and no song is redistributed.
- Passing the gates by tuning the gates. The ranges come from the stats file, not from what the engine happens to do.

## 4. Approaches considered

1. **Retune the v5 sampler.** Allow repeated notes, add rhythm templates, relax the chord-tone and peak rules.
   Smallest change. The melody would still be a note-by-note walk over hand-written templates, which is the design
   the user rejected, and the rhythm vocabulary would still be guessed.
2. **Learned tables with an explicit phrase plan (chosen).** Rhythm comes from real bar patterns and how they follow
   each other. Pitch comes from real interval behaviour, chord-tone rates and degree use. A cell memory brings short
   figures back, and the phrase plan gives pickups, cadences and restatement. Pure Python, table lookups, milliseconds
   per melody. Quality ceiling is "statistically like a real pop tune"; it cannot promise a great tune, which is why
   best-of-N and the listening check stay.
3. **A neural model writes the tune.** Highest ceiling in principle. Rejected for now: nothing found that takes chords,
   has weights, a licence we can use and runs on a laptop CPU, and v5's attempt (AMT) had to be filtered so hard by
   the rules that it sounded like the rules.
4. **Reuse real phrases as templates.** Closest to real tunes, and rejected: it reproduces other people's melodies.

The user asked for research-driven improvement and left the method open, so this design goes with approach 2.

## 5. Design

### 5.1 Statistics file

`scripts/build_tune_stats.py` reads the cleaned POP909 folder (not vendored; the script takes its path) and writes
`src/music_decoder/compose/data/tune_stats.json`, which is committed and shipped. `compose/tune_stats.py` loads it into
frozen dataclasses once per process.

All rhythm statistics use the **eighth-grid view**: a bar counts only if every onset is on an eighth (even sixteenth
slot); its pattern is the tuple of eighth slots (0–7) that have an onset.

Contents:

- `rhythm`: per bar role (`open` = first bar of a labelled phrase, `cadence` = its last bar, `body` = the rest):
  pattern counts, and counts of pattern pairs (previous bar's pattern, this bar's pattern). Patterns seen fewer than
  5 times are dropped.
- `steps`: counts of the next melodic move in **scale steps** (−7..+7, 0 = repeat), keyed by the previous move and the
  metric class of the target slot (`strong` = slots 0 and 4, `beat` = 2 and 6, `off` = odd slots), with the
  unconditioned counts per metric class for backoff. Notes outside the key's seven-note scale break the chain.
- `chord_tone`: share of notes that are chord tones, per metric class.
- `degrees`: share of each scale degree, per mode; and of phrase-final notes, per mode.
- `density`: notes per second (median and quartiles).
- `bands`: for every feature of section 5.4, the 25th, 50th and 75th percentile over eight-bar windows, per mode.
  A window starts at a labelled 8-bar phrase or at two consecutive 4-bar phrases, and counts only if every bar in it
  that has notes is an eighths-only bar and it has at least 12 notes. Windows with any other bar are skipped whole.
- `pilot` (optional): interval-class shares, pentatonic share, degree shares and range from the pilot dataset's
  vocal lines (section 5.6). When present, `degrees` is the mean of the two sources and each pitch band is widened to
  cover both.
- `meta`: source names, licences, counts, the date built.

### 5.2 The engine: `compose/tune.py`

```
generate_tune(*, scale, progression, bars_per_chord, tempo_bpm, seed, tessitura_center,
              allowed_pitches, final_whole_note=True, pitch_prior=None, stats=None) -> list[Note]
```

Same contract as the fingerstyle call of `generate_melody`: notes on the eighth grid, every note of bar `b` in
`allowed_pitches[b]`, no note crossing a bar line, the last bar one held cadence note, deterministic for a seed
(`numpy.random.default_rng(seed)` only).

**Phrases.** `rhythm.split_phrases` and `rhythm.form_labels` are reused, so a piece is A A′ (8 bars), A A′ A′
(12), A A′ B A′ (16). Each bar has a role: `open`, `body` or `cadence`.

**Rhythm plan.** For each bar in order:

1. Target onsets per bar = stats notes per second × the bar's length in seconds, clamped to 3–6.5.
2. If the bar is in an A′ phrase and the matching bar of phrase A exists, copy A's pattern with probability
   `P_RESTATE_RHYTHM`.
3. Otherwise draw a pattern with weight `P(pattern | role, previous pattern) × exp(−(onsets − target)² / (2 × 1.5²))`,
   where the conditional probability is interpolated half and half with `P(pattern | role)` when the previous
   pattern has at least 20 following bars in the table, and is `P(pattern | role)` otherwise.
4. The last bar is one whole note. The first bar of the piece must attack slot 0.

Pickups need no rule of their own: real cadence bars often carry onsets on slots 5–7 that lead into the next phrase,
and those patterns are in the `cadence` table.

**Pitch plan.** Notes are chosen left to right. The candidates of a note are the bar's allowed pitches whose pitch
class is in `chord_scale_pcs(scale, chord)` (all allowed pitches if that leaves none). Each candidate's weight is the
product of:

- **Move**: `P(scale steps | previous move, metric class)`, backed off to `P(scale steps | metric class)` when the
  context has fewer than 20 observations. Scale steps are counted on the ladder of the bar's chord scale. For a chord
  in the key that is the key's scale, which is what the statistics count on; for an altered chord it is the key's
  scale with the altered degree in place of the plain one, so a step still means one degree.
- **Harmony**: the stats chord-tone share for the slot's metric class if the candidate is a chord tone, one minus it
  otherwise, each divided by the number of candidates in its class.
- **Degree**: `(share of this degree in the mode / mean share) ** DEGREE_EXPONENT`.
- **Register**: a normal curve around `tessitura_center`, `REGISTER_SIGMA` semitones wide. This is the
  regression-to-the-mean that the literature finds behind "leaps turn back"; there is no separate leap-recovery rule.
- **Prior** (only with `pitch_prior`): as in v5, `exp(PRIOR_WEIGHT × exp(−d / PRIOR_WIDTH))`.

Hard rules, applied before weighting (if they leave nothing, they are dropped in this order: degree pool, then the
chord-change limit; the interval limits and `allowed_pitches` are never dropped):

- no move larger than an octave, no tritone, no seventh;
- the move into a new chord and the move into the final note are at most a fifth (the v3 invariants L4 and L5);
- the final note is the cadence pitch chosen as in v5 (`final_bar_pitches`, `final_target`), and the notes of the bar
  before it stay inside `feasible_sets` so it can be reached.

**Cell memory.** After each bar the engine remembers its pattern and its moves in scale steps. When a later bar has a
pattern already used, then with probability `P_RESTATE_CELL` (always, for a bar copied by rule 2 of the rhythm plan in
its first two bars) it replays those moves from the starting pitch that gives the most chord tones on the bar's
`strong` slots. Any replayed note that breaks a hard rule or leaves `allowed_pitches` is replaced by a normally sampled
one, and replay stops for that bar after two replacements. This is what makes figures return; the gate on recurring
three-interval figures measures it.

**Phrase ends.** The last note of an A or B phrase is drawn with the stats phrase-final degree shares in place of the
general ones, restricted to chord tones, and excluding the tonic when another chord tone is available (an open
ending). The last note of the piece is the cadence pitch.

**Note lengths.** A note lasts until the next onset in its bar, or the bar's end, times 0.95.

Tunable constants (`P_RESTATE_RHYTHM`, `P_RESTATE_CELL`, `DEGREE_EXPONENT`, `REGISTER_SIGMA`) are pinned in
`tune.py`. They are tuned only to bring the features into the stats bands.

### 5.3 Wiring

- `compose_with_options(..., melody_engine: Literal["tune", "rules"] = "tune")`, and the CLI, default to the new
  engine. `arrange` and `arrange_candidates` gain the same parameter with the default `"rules"`, so code and tests that
  call them are unchanged (they already differ from `compose` in their default number of candidates). `"rules"` is
  the v5 generator, note for note. Strum and arpeggio ignore the parameter and use the rules engine.
- Existing tests that assert v5 melody behaviour through `compose` or `compose_with_options` are changed to ask for
  the rules engine; nothing else about them changes.
- `compose()` keeps its signature and now produces the tune engine's output. `COMPOSER_VERSION = 6`.
- Metadata gains `melody_engine`.
- CLI: `music-decoder compose --melody-engine [tune|rules]`.
- `--melody-model amt` works with both engines.

### 5.4 Features and the candidate score

`compose/tune_metrics.py`: `tune_features(melody, scale, progression, bars_per_chord, tempo_bpm) -> dict[str, float]`
on the eighth grid, computed per 8-bar window (bars 0–7, 8–15) and averaged over the piece's windows, so a 16-bar
piece is compared with the 8-bar ranges like any other:

`repeat`, `step`, `third`, `leap` (interval-class shares); `notes_per_second`; `rhythm_variety` (distinct bar patterns ÷ bars with notes);
`rhythm_lag2`, `rhythm_lag4` (share of bars whose pattern equals the bar 2 or 4 earlier); `figure_recurrence` (share of
three-interval figures that occur more than once); `offbeat` (share of onsets on odd slots); `late_entry` (share of
bars with no onset on slot 0); `chord_tone_strong`; `pentatonic`; `range`.

`band_distance(features, bands) -> float`: the sum over features of how far the value lies outside its 25–75% band,
in units of the band's width (0 inside the band).

`score_arrangement` gains `melody_engine`. For `"tune"` the melody terms of v5 (chord tones, steps, rhythm entropy,
repeats, leap recovery, motif, hook, peak, arch, ambitus) are replaced by `−W_CORPUS × band_distance`. The identity,
melody-above, shape-move and fallback terms are unchanged. For `"rules"` the score is v5's.

### 5.5 Gates

A new harness, `evaluation/compose_tune.py`, builds fingerstyle pieces with the tune engine: 6 major and 6 minor keys ×
3 progressions each × tempos 72, 96, 120 × 2 seeds, 8 bars, plus 16-bar pieces in 4 keys; and a holdout set with other
seeds and keys, evaluated once at the end.

- **T1 (real-melody fit).** For every feature of 5.4, per mode, the median over the matrix lies inside the stats
  25–75% band. T1 and T2 are measured on **single candidates** (candidate 0 of each piece, before any selection).
  The best-of-12 score picks by the same ranges, so gating its output would pass by construction; the engine has to
  land in the ranges on its own.
- **T2 (not all alike).** For `repeat`, `notes_per_second`, `rhythm_variety` and `figure_recurrence`, the matrix's own
  25–75% spread is at least half as wide as the band.
- **T3 (guitar guarantees).** On the pieces as `compose()` writes them (best of 12): P1 = 1, P2 = 1, P4 true, mean P3 ≥ 0.99 for triad bars, no
  forbidden interval, no move over an octave, chord-change and final-approach moves at most a fifth, final note at
  least 1.5 beats and on the cadence pitch.
- **T4 (same seed, same music)** and **T5** (every key and mode produces a melody; no exceptions on the edge and
  tunings inputs of the existing harness).
- **Existing gates.** The existing musicality harness builds its fingerstyle cases with `melody_engine="rules"`, so
  every v3–v5 gate keeps guarding the rules engine unchanged.

If a band cannot be met together with a guitar guarantee, the guarantee wins. The miss is recorded as a ruling with
its measured value and cause, listed by name in the gate test, and reported to the user. Work then continues; the
melody-aware shape choice of section 3 is not started under this plan. More than two missed features in a mode means
the design is not working, and work stops for a decision.

### 5.6 Cross-check on our own dataset

A throwaway script in `spikes/preference-pilot/` reuses the pilot's pipeline (beat tracking, key, chords, vocal stem,
transcription) on the cached audio of the 488 songs and writes cleaned vocal lines: notes shorter than 0.12 s dropped,
the highest note per onset kept, onsets snapped to the tracked eighth grid. Only pitch-side statistics are taken from
it (interval classes, pentatonic share, degree shares, range), because its timing is too noisy for rhythm tables. Key
and mode come from the pipeline's own key estimate, as in the pilot. Its purpose is to stop the degree profile from
being purely Chinese-pop pentatonic. It runs in the background (about four hours) while the engine is built. The gates
are tuned against one final statistics file: tuning starts when the run has finished, or, if it has not finished six
hours after it started, with the POP909-only file, and the merge becomes a recorded follow-up. The holdout set is
evaluated once, against that same file.

### 5.7 Listening checkpoint

As soon as the engine is wired in (before the gates are tuned), ten pieces are generated with the same commands and
seeds as the v5 sample set, in both engines, and sent to the user as MP3s. Work continues meanwhile; the user's
comments steer the tuning.

## 6. What is left to the builder

Internal structure of `tune.py`, the JSON layout beyond the fields named in 5.1, the exact matrix of progressions,
how the 0.95 articulation interacts with let-ring, and the starting values of the four constants.

## 7. Risks

- **Statistically right, still not liked.** Matching real-melody statistics is necessary, not sufficient. The
  listening checkpoint comes early for this reason, and best-of-12 stays.
- **The hand shape allows only 8–9 pitches per bar.** Replayed figures and learned moves will often be bent by it. If
  the `figure_recurrence` band cannot be met, the next step is the melody-aware shape choice listed under non-goals.
- **POP909 is Chinese pop.** Its tunes are more pentatonic than Western pop. Section 5.6 measures the difference.
- **Bars that do not attack beat 1** are new to the arranger; T3 runs on them.

## 8. Success criteria

1. T1–T5 pass on the development matrix and on the holdout set; any missed band has a recorded ruling (section 5.5).
2. Every existing test passes with the rules engine.
3. The user has received the comparison samples. Whether the user likes them is the real test and is outside what
   this plan can guarantee; their comments start the next round of work.
4. `ruff`, `ruff format --check` and `mypy --strict` are clean.

## 9. Later

- **More scales.** The engine works on scale steps and degree shares, so a scale type is a pitch-class set plus a
  degree profile. Dorian, mixolydian, harmonic minor and the pentatonic and blues scales can be added once the user
  likes the major and minor output; modes without data of their own would start from the parent scale's profile with
  the characteristic degree raised.
- Melody-aware shape choice for a wider, higher register; sixteenth notes; ties across bar lines; a liking term for
  medium surprise (Gold et al. 2019).
