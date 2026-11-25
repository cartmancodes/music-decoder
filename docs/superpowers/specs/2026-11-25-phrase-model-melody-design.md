# Phrase model: melodies learned from whole phrases of real tunes

Date: 2026-11-25. Status: awaiting the user's approval. Follows `2026-11-16-composer-v6-melody-design.md`.

## 1. Intent

The user writes complete songs themselves. What they want from the composer is **very good melodies** ("I can write
complete songs, our job is to generate very good melodies", 2026-11-25). Song form, sections and chord progressions
are theirs and are out of scope.

Where the melody stands today (composer version 7, the `tune` engine):

- The user's blind ratings of five current pieces were 4, 7, 9, 7, 9 out of 10 (mean 7.2, 2026-11-23). The engine can
  write a 9 and a 4 on the same settings.
- The score that picks the best of 12 candidates (distance from the real-melody bands) does not track the user's ear.
  Neither does Audiobox Aesthetics; its top clip got the user's lowest rating.
- Hand-written theory rules scored lower than the current engine (6.4 against 7.2). Learned note endings and lead-in
  bars made no audible difference. A judge trained on YouTube likes is at chance at the note level (AUC 0.487).

Why the engine is stuck: it picks each note from tables that see only the previous move and the current bar. It cannot
state a short figure and develop it over a phrase. The research found this session points the same way. Dai et al.
2021 (arXiv 2109.00663) used the same 909 songs, split the job into rhythm then pitch as we do, but learned each part
with a small network that sees the phrase structure; listeners rated its tunes as good as or better than the real
songs about half the time. Yin-Yang (arXiv 2501.17759) names note-by-note writing that cannot return to its motif as
the core obstacle.

So this design adds a small model, trained on real melodies with their chords, that reads whole phrases. It is used
two ways: to **pick** the best of many candidates from today's engine, and to **write** melodies itself.

## 2. Goals, non-goals, anti-goals

**Goals**

- G1. In a blind listening round on the same chords and tempos, the model's pick or the model's own tune is rated at
  least 1.0 point higher on average than today's pick (section 4). This is the only test that decides what ships.
- G2. A phrase model trained here, on this Mac, from data whose licence lets the weights be committed.
- G3. Two opt-in uses in `compose`: a pick rule (`--melody-pick phrase`) and a melody engine
  (`--melody-engine phrase`). Defaults change only as the listening verdict says.
- G4. Every existing guarantee holds for the phrase engine: the guitar guarantees (playable moments, melody on top,
  chord identity, tab equals MIDI, final fallback), the eighth grid, no note across a bar line, no move over a fifth,
  same seed gives the same music.
- G5. The phrase engine never reproduces a training melody (section 5.6).

**Non-goals**

- Song form, sections, verse and chorus contrast, chord progression generation. The user does these.
- Sixteenth-note onsets in the written melody. The arranger, tab and performance layers assume eighths.
- Western chart corpora. CoCoPops (414 songs) is CC BY-NC 4.0 and the Hooktheory set is CC BY-NC-SA 3.0, so weights
  trained on them could not be committed to this MIT repo. Adding them is a later decision for the user (section 8).
- Lyrics, or melodies fitted to words. Melodies stay instrument leads, as today.
- A taste judge from YouTube labels, manual rating rounds of 100 clips, or tuning the model against Audiobox.
- The strum and arpeggio styles, and the boom-bap and pop-rap productions. Productions read `composition.mid`, so
  they get the new melodies without change.

**Anti-goals**

- Changing what `compose` writes by default before the user's verdict.
- Copying a recognisable melody.
- Passing a gate by moving the gate. Bands and thresholds come from data by the rules stated here.
- Shipping weights trained on non-commercial data.

## 3. Approaches considered

1. **A phrase model as picker and writer (chosen by the user, 2026-11-25).** The only option that can learn how a
   figure develops. Largest build. Risks: 909 songs is little data, so the model may be bland or may copy; and
   "likely under the model" is not the same as "the user likes it", which is why G1 is a listening test.
2. **Better tables.** Rebuild the tables from Western chart songs and add sixteenth notes. Small and safe, but the
   same kind of change as the last two rounds, which levelled off near 7 out of 10.
3. **Hand-written figure rules.** State, repeat, vary, answer. Already tried in the 2026-11-23 theory spike, which
   scored below the current engine.

## 4. Stage 0: throwaway spike and blind round (before anything is built into the package)

The spike lives in `spikes/phrase-model/` and its clips in `compose-samples/2026-11-25-phrase-model-round/`. Neither
folder is ever staged. Spike code has no tests and is not kept; what it learns goes back into this document.

Steps:

1. Train the model of section 5 on POP909.
2. **Premise check.** Train the same network with its context cut to one bar (16 slots). On the held-out songs the
   full-context model must need at least 10% fewer bits per bar than the one-bar model. If it does not, long context
   is not buying anything on this data: stop and report, with no listening round.
3. **Read-out against the ear data we have.** Score the ten clips the user rated on 2026-11-23
   (`compose-samples/2026-11-23-theory-melody-spike/ratings.json`) and report the rank correlation of the ratings
   with: the phrase score of section 5.4, raw likelihood, and today's band score. Ten clips is weak evidence and is
   reported as such. One rule depends on it: if the phrase score correlates negatively with the ratings while raw
   likelihood correlates positively, the round uses raw likelihood as the pick rule, and the write-up says so.
4. **Blind round.** Four settings (scale, progression, tempo): those of the current-engine pieces in `ratings.json`,
   the two major ones and the first two minor ones by clip number. Seeds are new (the rated seeds are not reused), so
   the user has heard none of the clips. For each setting, three versions from the same base seed, 12 clips in all:
   - today: `tune` engine, best of 12 by the band score;
   - model's pick: `tune` engine, best of 200 by the phrase score;
   - model's tune: `phrase` engine, best of 12 by the band score.
   Clips are rendered the way the 2026-11-23 round was, numbered blind, with the key in `blind_key.json`. The user
   rates each out of 10, as they did then. The round needs only the hard guarantees of G4, which the engine keeps by
   construction; any band the phrase engine misses (section 7) is listed beside the clips, and does not stop the
   round.
5. **Verdict.** "Today" is today's mean in this round, not the earlier 7.2. A version wins if its mean rating is at
   least 1.0 above it. With four clips a version, a gap of about a point is a signal, not proof: if a version lands
   within 1.0 either side of today, a second round of 12 on new settings is offered before deciding. The user can
   overrule the arithmetic either way.
   - Model's pick wins: build sections 5.1 to 5.4 and 6.1.
   - Model's tune wins: build sections 5.1 to 5.6 and 6.2.
   - Both win: build all of it.
   - Neither wins: build nothing. Write up what was learned and stop.

After the round the user also decides the defaults (section 6.3).

## 5. The phrase model

### 5.1 Training data

POP909 in the cleaned release already used for the tune statistics (`~/.music-decoder/datasets/pop909-hsa`, MIT
licence, Shuqi Dai 2020): 909 songs, 307,683 melody notes, 70,112 bars, 544 major and 365 minor, on a sixteenth grid,
with chords, keys, tempo and human phrase labels. `compose/tune_corpus.read_song` already reads it.

Songs are split by song into train, validation and held-out, 80/10/10, with a fixed seed recorded in the weights
file. Every band, threshold and check in this document that says "real" or "held-out" uses the held-out songs only.

### 5.2 What the model reads and predicts

One token per sixteenth slot, 16 per bar. A token is either an onset at a pitch or "no onset".

- **The model sees onsets, not note lengths or rests.** The composer decides where notes start and at what pitch;
  how long each rings belongs to the existing performance layer (the tune engine holds every note for 0.95 of the gap
  to the next onset, never past the bar line, and the user's 2026-11-23 blind test found learned note lengths made
  no audible difference). POP909's rests (about a quarter of its slots) are therefore read as "no onset", and real
  and composed melodies are tokenised identically.
- **Pitch is relative to the key.** The anchor is the tonic at or below the melody's middle (the song's median melody
  pitch in training; `tessitura_center` when writing). An onset token is the pitch minus the anchor, in semitones.
  The token range is the smallest that covers at least 99.5% of training notes; spans with a note outside it are
  left out of training.
- **Each slot also carries what the composer knows in advance**, added to the input for that slot:
  mode (major or minor); the bar's chord as root relative to the tonic and quality; the next bar's chord the same
  way; slot in the bar; bars since the phrase began and bars until it ends (each capped at 15); whether the phrase
  is new or a restatement of an earlier one; and a tempo bucket.
  - Chord qualities are the composer's own eight (`harmony.CHORD_INTERVALS`) plus "other" and "no chord". POP909
    chord names map onto them by their pitch classes.
  - Phrases come from POP909's human labels in training: a letter seen earlier in the song is a restatement. When
    writing, they come from the composer's phrase plan (`rhythm.split_phrases`, `rhythm.form_labels`): A and B are
    new, A' is a restatement.
  - A song with several keys uses the key of each labelled segment, as `tune_corpus` already does.

### 5.3 The network and its training

A small causal transformer (each slot sees only earlier slots, plus the advance information for its own slot). It has
at most 5 million parameters and a context of at least 32 bars (512 slots). Width, depth and the remaining settings
are chosen in the spike by validation loss among a few sizes. Training uses torch on the Mac's GPU or CPU; a run is
expected to take minutes, not hours. Training spans start at a labelled phrase start with nothing before them, so a
melody that begins cold, as every composed piece does, is something the model has seen.

`scripts/train_phrase_model.py POP909_DIR` rebuilds the weights. Training is not bit-for-bit repeatable across
machines, so the committed weights file is the artefact, and its hash identifies the model everywhere (section 6.4).

### 5.4 The phrase score (the pick rule)

A melody's **surprise** `S` is its mean bits per bar under the model, given the piece's chords, tempo and phrase
plan. Real and composed melodies are measured on the same footing:

- Both are cut into the eight-bar windows `tune_metrics.tune_features` already uses, each window scored cold (no
  earlier context), and a piece's `S` is the mean over its windows.
- A final bar that is the forced whole cadence note is left out, since it is the same for every candidate.
- The real reference is the held-out songs' eight-bar windows that start at a labelled phrase and that the composer
  could write (every onset on an eighth), per mode.

Lowest surprise is not the pick rule. Ranking by likelihood favours bland, repetitive lines, and listeners prefer
medium predictability (Gold et al. 2019). The rule is typicality:

`phrase_distance = |S - median| / IQR`

with `median` and `IQR` taken from the real reference. It has no flat zone, so two candidates rarely tie and the
model, not the candidate's index, decides between them. If every candidate of a piece sits on one side of the real
median, the rule picks the one nearest real tunes on that side, which is intended; the spike reports where the
candidates fall against the real band.

With `melody_pick="phrase"` the candidate score is today's `tune` score with the term
`-W_CORPUS x band_distance` replaced by `-W_PHRASE x phrase_distance`, `W_PHRASE = 2.0` (equal to `W_CORPUS`). The
guitar terms (chord identity, melody above, shape moves, final fallback) stay as they are.

### 5.5 The phrase engine (the writer)

`generate_phrase_tune` takes the same keyword arguments as `tune.generate_tune` and returns the same list of notes,
so `_arrange_fingerstyle` calls it in the same place. It samples tokens left to right, and at each slot removes what
the composer forbids before drawing:

1. onsets only on even slots (the eighth grid);
2. an onset's pitch must be in `allowed_pitches` for its bar;
3. no move of more than 7 semitones from the previous note;
4. with `final_whole_note`, the last bar is one held note on the cadence pitch class, as the tune engine does.

Note lengths are the tune engine's: each note rings for 0.95 of the gap to the next onset and never past its bar
line. The phrase engine writes no rests of its own.

Two outcomes are required and their mechanism is left to the builder: each phrase ends on a tone of its chord, and
no phrase is empty.

**Density.** Most POP909 songs are slow (median 74 BPM) and the rated pieces run at 110 to 126, so the model alone
may write too many or too few notes at the composer's tempos. The phrase engine therefore steers density to the
tune engine's own tempo target (`tune.py`'s onsets-per-bar target and its clamp) with one bias on the "no onset"
choice per piece. How the bias is found is left to the builder; the outcome is checked by the notes-per-second band
of section 7.

The draw itself is made with the composer's numpy generator from the model's probabilities, so the seed decides the
melody. Temperature and nucleus are fixed constants, chosen once in the spike from temperature {0.8, 0.9, 1.0} and
nucleus {0.9, 0.95, 1.0}: the pair whose pieces put the most of the 14 melody features inside the real bands on the
existing development matrix (`tests/unit/test_compose_tune_gates.py`); ties go to the higher temperature.

`--melody-model amt` (the pitch prior) is not supported with the phrase engine and is refused with a clear error.

### 5.6 Copy check

A figure is a run of consecutive notes described by its intervals and its gaps between onsets, so it is the same in
any key. The length `L` is the smallest for which at most 1% of held-out real eight-bar windows share a run of `L`
notes with a different song in the training set. All training runs of length `L` are stored as hashes beside the
weights; no melody can be read back from them.

A written melody that contains a stored run is thrown away and written again from the next sub-seed, up to 8 times.
After that the candidate falls back to the tune engine and the piece's metadata records `phrase_fallback`.

## 6. How it fits into the composer

### 6.1 Pick rule

`arrange_candidates` and `compose_with_options` gain `melody_pick: "bands" | "phrase"`, default `"bands"`; the CLI
gains `--melody-pick`. `"phrase"` is valid with the `tune` and `phrase` engines and refused with `rules`. More
candidates cost time in a straight line (about 0.13 s each today), so `--candidates` stays the user's choice.

### 6.2 Engine

`MelodyEngine` becomes `"tune" | "rules" | "phrase"`; the CLI's `--melody-engine` gains `phrase`. Fingerstyle only,
as `tune` is. `compose()`'s frozen signature is untouched.

### 6.3 Defaults and the composer version

Adding the two options changes no default and keeps composer version 7. After the round the user chooses the default
engine, pick rule and candidate count from a table of the measured options and their compose times. The task that
changes a default sets the composer version to 8 and updates the tests that pin it.

### 6.4 Files, dependencies and metadata

- Weights: one file in `src/music_decoder/compose/data/`, numpy format (no pickle), half-precision, at most 6 MB,
  loaded once per process. Beside the arrays it records its sources and licence, the split seed, the token range,
  the network settings, the held-out bits per bar, the surprise median and IQR per mode, `L`, and the figure hashes.
- `torch` is already installed through `demucs`; it becomes an explicit core dependency and is imported only when
  the phrase model is used. The model runs on the CPU when composing.
- Take metadata and the take manifest gain `melody_pick`, `phrase_model` (the weights hash) and `phrase_fallback`.
  The output folder hash covers the new options and the weights hash.

## 7. Automatic gates (beside the listening round)

- T1 to T5 of `test_compose_tune_gates.py` run for the phrase engine too: medians of the 14 features inside the real
  bands in both modes, pieces not all alike, guitar guarantees, same seed same music, every key and mode composes.
  The bands are the existing ones in `tune_stats.json`, unchanged; "held-out" in this document applies only to the
  phrase model's own numbers (the premise check, the surprise median and IQR, and `L`).
- **If the phrase engine wins the round but still misses a T1 or T2 band:** the user's ear outranks the bands, which
  exist only as a stand-in for it. The bands are not moved. Each miss is shown to the user with its numbers, and one
  they accept is recorded in the test's known-misses list with the reason "user's blind verdict" and the date. A
  miss they do not accept blocks making the phrase engine a default, not shipping it as an option.
- The copy check holds on every piece of the development and hold-out matrices.
- The shipped weights load, and score a fixed melody to a pinned value (to 3 decimals).
- Unit tests use a tiny network with fixed random weights, so they need neither the dataset nor the shipped file.
- Composing with the phrase engine at 12 candidates takes no more than 3 times as long as with the tune engine,
  after the model is loaded.

## 8. Decisions

**The user's**

- Approach 1 (2026-11-25).
- Melody quality is the job; song form and chords are theirs.
- Open, for after the round: the defaults (6.3), and whether to add the non-commercial Western corpora for a model
  kept only on this machine.

**Calls made by the designer (Claude), each open to correction**

- Train on POP909 only, because its MIT licence lets the weights be committed.
- Typicality, not lowest surprise, as the pick rule, measured as plain distance from the real median; `W_PHRASE = 2.0`.
- The premise check and its 10% threshold; the 1.0 point winning margin; 12 clips over 4 settings on new seeds, with
  a second round offered when the result is within a point.
- The model reads onsets only. Note lengths and rests stay with the existing performance layer.
- Surprise is measured on cold eight-bar windows for real and composed melodies alike.
- The phrase engine steers note density to the tune engine's tempo target.
- A band the winning engine misses is the user's to accept or refuse; the bands themselves are never changed.
- A sixteenth-grid model that is only allowed to write eighths, so the data is used whole and sixteenths can be
  opened later without retraining.
- The copy-check length is set from data at the 1% level; 8 rewrites before the fallback.
- Weights committed in the package (at most 6 MB) so `compose` works without a download.
- The phrase engine refuses the AMT pitch prior.
- Changes stay uncommitted until the user asks, as on the piano-lead branch.

## 9. Left to the builder

Module and function names, how the code is split into files, the network's exact size within the cap, the position
scheme, optimiser and schedule, how a phrase is made to end on a chord tone, how float noise is kept from flipping a
draw, the hash and file layout of the figure index, and test shapes.

## 10. Risks

- **Too little data.** 909 songs may give a model that is bland or that memorises. The premise check, the copy check
  and the round are the answers; if it is bland, the round will say so.
- **Likely is not liked.** The score measures closeness to real tunes, not taste. Only G1 tests taste.
- **One tradition.** POP909 is Chinese pop and leans pentatonic. The tune statistics already carry the same lean.
- **Repeatability.** Model output can differ between torch versions or machines. Seeds repeat on one setup, as with
  the two existing optional models.
