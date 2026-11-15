# Composer v5: five improvements to the fingerstyle composer

**Date:** 2026-11-16
**Base:** `master` at `6b9cf8a` (composer v4, `COMPOSER_VERSION = 4`)
**How this was decided:** the human partner asked for all five improvements from the 2026-11-16 research summary to be built. They set a goal to build them without pausing for questions, so the question round was skipped. Every choice below that the partner did not state is marked **(my call)**.

## 1. Intent and why

The partner wants a local composer whose tunes a broad audience likes. Composer v4 makes fingerstyle one playable guitar part, but the research summary found five weak spots:

1. It sounds programmed. Every note in a role has the same velocity, every note stops at 95% of its slot, and the audio is a General MIDI soundfont.
2. Each chord has one fixed shape. That caps the melody range near one octave, and 2–6% of seventh-chord bars lose their seventh in the accompaniment (v4 spec §10.4).
3. The melody has no deliberate hook and no phrase shape.
4. Each request yields one random piece. Nothing keeps the best of several.
5. The melody comes only from hand-tuned rules, with no learned model.

This project builds one improvement for each. The order of §5 is the build order.

## 2. What exists (verified on `6b9cf8a`)

| Part | Where it lives |
|---|---|
| Fingerstyle part | `compose/fingerstyle.py`: `hand_shape`, `body_bar`, `final_bar`, `arrange_fingerstyle` |
| Wiring | `api._arrange_fingerstyle` |
| Melody | `compose/melody.py` (`generate_melody` with `allowed_pitches`, `final_whole_note`, `tessitura_center`) and `compose/rhythm.py` (phrase planner) |
| Shapes | `compose/voicings.py`: `voicings_for` gives one shape per chord; `ranked_shapes` gives every faithful, playable shape |
| Audio | `synth.render_wav`, through FluidSynth with GeneralUser GS (or the bundled TimGM6mb) |
| Metrics and harness | `compose/metrics.py` (incl. P1–P4) and `evaluation/compose_musicality.py`, which reports P1–P5, P3t and `F.*` |
| Strum/arpeggio guard | `tests/unit/test_compose_frozen.py` |
| Public contract | `compose()`'s signature, the package exports and the CLI flags are frozen by `tests/unit/test_public_contract.py`. New CLI flags may be added. |

Environment: torch 2.11 and torchaudio are installed, `transformers` is not. About 30 GB of disk is free. The machine is an Apple M4 with 24 GB of RAM.

## 3. Goals, non-goals, anti-goals

**Goals**
1. Fingerstyle sounds performed: dynamics, phrasing, notes that ring, small timing variation in the audio. A string-aware plucked-guitar sound is available.
2. Each bar may use a different shape of its chord. This fills in the missing sevenths and widens the melody.
3. Fingerstyle melodies have a recognisable hook, an arch-shaped phrase and one clear high point per phrase.
4. Fingerstyle `compose()` writes several candidates and keeps the best by an objective score. An optional pretrained audio-taste model can re-rank them.
5. An optional pretrained melody model proposes melodies. The rules still decide, so playability and the hard melody rules hold.

**Non-goals**
- Changing strum or arpeggio in any way (my call). Their output stays identical to v3, and `test_compose_frozen.py` keeps guarding that.
- Changing `compose()`'s signature or the package exports. Both are frozen.
- Training models of our own, or pairwise preference learning (out of reach, see the research summary).
- Making the ML features part of the default install.

**Anti-goals** (a failure even if the tests pass)
- Losing any v4 guarantee on the evaluation matrices: P1 (playable), P2 (melody on top at onsets), P4 (tab = MIDI), P3t (triads keep identity), A4 and H2 (the final chord rings with full harmony), and the v3 hard melody rules.
- Tests or the default `compose()` path needing the network or the ML models.
- Output that is no longer reproducible from a seed, with or without the models.

## 4. Approaches considered (all my calls)

| Feature | Chosen | Rejected, with the reason |
|---|---|---|
| 1. Sound | Expressive velocities and let-ring lengths written into the MIDI. Micro-timing applied only to the rendered WAV, so the MIDI and tab stay on the grid. A new numpy plucked-string renderer, opt-in. | A learned performance model: piano-only and mostly closed. A downloaded guitar sample library: licences unverified, and FluidSynth can't play per-string SFZ. |
| 2. Shapes | Choose a shape per bar with dynamic programming (Viterbi) over `ranked_shapes`, scored on identity safety, melody reach and hand movement. Then a repair pass that re-voices any bar that still misses identity. | A learned fingering model: our parts are placed by construction already. |
| 3. Melody | Explicit rules: hook cell, contour plan, phrase peak, A–A′–B–A′ form. Measurable with new metrics. | Learning hooks from data: no suitable fingerstyle corpus. |
| 4. Best-of-N | The default ranks N candidates with a symbolic score built from our own metrics. The opt-in re-rank renders the top K and scores them with Audiobox Aesthetics. | The audio model as the default: an extra 415 MB download, slow, and it may reward production over tune. |
| 5. Model | The Anticipatory Music Transformer (`stanford-crfm/music-small-800k`, Apache-2.0, 512 MB) proposes a melody over our accompaniment. Its pitches become a **soft prior** in `score_pitch`, so every hard rule and the reach sets still apply. | Taking the model's notes as they are: they would break playability and the hard rules. The medium model (1.4 GB): too large for the free disk. |

## 5. Design

### 5.1 Feature 1: expressive performance and a plucked-guitar sound

**Dynamics** (written into the MIDI velocities, fingerstyle only):
- Melody notes are at least 8 velocity units louder than the accompaniment notes sounding with them.
- Strong-beat melody notes get +6.
- Each phrase swells: velocity rises about 10 toward its peak note (feature 3's peak, or the phrase's highest note before feature 3 lands), then falls to the cadence.
- Thumb notes on beats 1 and 3 are louder than those on beats 2 and 4.
- Fills are the softest notes.
- Every velocity is clamped to 30–118.
- Exact values are the builder's call.

**Let-ring** (written into the MIDI lengths):
- An accompaniment note rings until its string is plucked again, or until the end of its bar.
- It is cut short just before any onset where ringing on would break playability (P1), or would put it at or above a starting melody note (P2).
- The final roll still rings to the end of the piece.
- Melody notes keep their v4 lengths.
- The tab is unchanged, because it draws onsets only. P1 and P2 are measured with the longer notes.

**Micro-timing** (audio only):
- When the WAV is rendered, each onset gets a seeded offset of at most ±6 ms. The final roll's stagger is kept.
- The MIDI file and the tab stay exactly on the grid.
- Rendering uses a temporary performance MIDI, removed afterwards. A seed gives the same WAV every time.

**Plucked-string renderer** (new `synth/plucked.py`, opt-in):
- An extended Karplus–Strong voice per note: a pluck-position comb filter, loss that depends on the string's frequency and fret, a slight stiffness all-pass, and a short body-resonance filter bank.
- It renders from the tab (string and fret) when the tab is available, otherwise from the MIDI.
- The output is peak-normalised like `synthesize_midi_to_wav`.
- It is selected by the new CLI flag `--sound [soundfont|plucked]`. The default is `soundfont`, so the default sound does not change until the partner has listened to both (my call).

### 5.2 Feature 2: per-bar chord shapes

**Candidates.** For each bar, the candidates are the chord's v4 shape plus its `ranked_shapes(chord, tuning)` in order. Barre shapes are capped at base fret 9, the `search_voicing` range. All are faithful and playable.

**Cost per bar and shape** (the weights are the builder's call; the sign of each term is pinned):
- (+) each essential tone the shape can supply only on its top string, since the melody holds that string;
- (−) bigger `allowed` reach;
- (+) lower library rank, preferring open and library shapes;
- (+) distance from a melody range centred on G4.

**Transition cost** between bars: (+) movement of the hand window's lowest fret. It is zero when the shape is unchanged, so a chord held for several bars tends to keep its shape.

**Search.** Viterbi over bars picks the shape sequence. Then `arrange()` runs as in v4 with those hands.

**Repair pass.** After `arrange_fingerstyle`, for each bar in `identity_misses`, try the other candidate shapes in cost order. Use the first whose hand reaches every melody note of that bar (same pitches), and arrange that bar again with it. Keep the change only if P1, P2 and identity all hold for the bar. The melody's pitches never change.

**Outputs.**
- `Arrangement` gains `bar_voicings: tuple[VoicedChord, ...]`, one per bar.
- `voicings` stays one per chord: the first bar's shape. So `Composition.chord_voicings` keeps its type.
- The tab's chord list shows every distinct shape used, in order of first use.

### 5.3 Feature 3: hooks, contour, peaks and form (fingerstyle melodies)

`generate_melody` gains an optional `phrase_shaping: bool = False`, which only fingerstyle passes. With it on:

- **Hook.**
  - The first bar's first two to four notes form the hook cell: their intervals and rhythm.
  - Every antecedent phrase restates it at its start, transposed to fit the chord.
  - The consequent phrase restates it with exactly one element changed: one interval, or the rhythm of one note.
  - Implementation: extend the existing motif echo, which today only echoes inside a phrase pair.
- **Contour.** Each phrase has a target arch: it rises to a peak at 55–75% of the phrase, then falls to the cadence. `score_pitch` gets a contour term pulling toward that arch.
- **Peak.** One note per phrase is its unique highest note, and it falls in the peak window. The scoring term discourages ties and any second peak.
- **Form.**
  - Pieces of 8 bars or more follow A A′ B A′ at phrase level.
  - B has a contrasting rhythm draw, and its register target sits at least 2 semitones higher.
  - Shorter pieces use A A′.
- All v3 hard rules (`admissible`) and the reach sets still apply.

New metrics in `compose/metrics.py`:
- `hook_restatements(melody, tempo, n_bars)`: the share of phrase starts that restate the hook's intervals, give or take a semitone.
- `peak_uniqueness`: the share of phrases with exactly one top note, inside the peak window.
- `contour_arch`: the correlation of each phrase's pitch contour with its arch target.

### 5.4 Feature 4: generate many, keep the best

**Candidates.** Fingerstyle `arrange()` builds N candidates, with **N = 12** (my call). The seed for candidate k is `hash((seed, k))`, which is deterministic. When `seed is None`, a random base seed is drawn once.

**Symbolic score.** Each candidate is scored by `compose/scoring.py: score_arrangement(arr, ...) -> float`. It is a weighted sum of normalised terms:
- accompaniment-only identity share (P3);
- A2 (melody above everything at every instant);
- F.M1 (chord tones on strong beats);
- F.M2 (step ratio);
- F.M10 (rhythm entropy);
- hook restatements, peak uniqueness and contour arch (from §5.3);
- a penalty for repeated notes;
- a penalty for any final-voicing fallback;
- hand movement.

The weights are pinned in the module and documented. The best candidate wins, with ties going to the lowest k.

**Audio re-rank (opt-in).**
- The CLI flag `--rank-audio` renders the top K = 4 candidates and scores them with Audiobox Aesthetics (`facebook/audiobox-aesthetics`, CC-BY-4.0, about 415 MB).
- The audio score is the mean of its "content enjoyment" and "production quality" axes, plus the symbolic score scaled into the same range.
- It needs the `ml` extra and the weights from `fetch-models --composer-models`. Without them, the command stops with a clear error naming both.
- It runs on the CPU, so results repeat.

**Metadata.** `Composition.metadata` gains:
- `candidates`;
- `chosen_candidate`;
- `score`;
- `final_fallback`;
- `identity_misses`.

These make the counted outcomes visible to users (v4 final-review minor).

### 5.5 Feature 5: a pretrained melody model proposes the tune (opt-in)

**How the model is used.**
- The CLI flag `--melody-model amt` turns it on.
- For each candidate, we build a MIDI context from the progression, the tempo and the hands' thumb notes. The Anticipatory Music Transformer (AMT) then generates a melody over that accompaniment.
- AMT samples from its own generator, seeded with `torch.Generator(seed)` on the CPU, so results repeat.

**From model output to a soft prior.** The model's output becomes a per-bar list of target pitches:
- quantise its onsets to the eighth grid;
- take the highest note at each onset;
- move it to the nearest pitch in the bar's allowed set.

`generate_melody` then takes `pitch_prior: Sequence[Sequence[int | None]] | None`: one target per rhythm slot, filled by nearest onset. `score_pitch` adds a bonus that falls off with the distance from the target. The weight is pinned. It is small enough that the hard rules, the reach sets and the cadence still decide.

**Installation.**
- A new optional dependency group `[project.optional-dependencies] ml`: `transformers`, `huggingface_hub`, `audiobox_aesthetics`, and `anticipation` from its GitHub repository at a pinned commit, since it is not on PyPI.
- Every ML import is lazy and lives only in `compose/ml/`.
- `fetch-models --composer-models` downloads both weights into the Hugging Face cache, about 930 MB in all.

**Fallback.** If the model's output is empty or unusable for a bar, that bar gets no prior.

### 5.6 Shared

**Public API.**
- New function `compose.api.compose_with_options(scale, progression, *, ..., sound="soundfont", rank_audio=False, melody_model=None, candidates=12)`. `compose()` calls it with the defaults. It is not exported from the package, which stays frozen.
- The CLI `compose` command gains `--sound`, `--rank-audio`, `--melody-model` and `--candidates`.

**Version.** `COMPOSER_VERSION = 5`, set in feature 1. Each later feature changes output further within v5 before the branch merges.

**Harness.**
- `evaluate()` reports fingerstyle with the defaults: features 1–4, without the audio re-rank and without the model.
- New keys:
  - `F.hook`, `F.peak` and `F.arch` (feature 3);
  - `F.score` (feature 4);
  - `F.vel_std` and `F.ring` (feature 1: velocity spread and mean accompaniment length in beats);
  - `F.shape_moves` (feature 2: window moves per bar).

**Tests.** Unit tests use fake models. Opt-in integration tests with the real weights run only when `MUSIC_DECODER_ML_E2E=1`.

## 6. Success criteria

1. **v4 guarantees hold.** On every matrix and every preset tuning, with the defaults:
   - P1, P2 and P4 are 1.0;
   - P3t is 1.0;
   - P5 is 0;
   - A4 and H2 hold for fingerstyle;
   - the v3 hard melody rules (`F.L1`, `F.L3`, `F.L2`, `F.L4max`, `F.L5`, `F.M4max`) hold.

   The seeded random-progression property test still passes.
2. **Strum and arpeggio are unchanged:** `test_compose_frozen.py` passes untouched.
3. **Feature 1 (sound):**
   - `F.vel_std` is at least 8 (v4 is about 3);
   - the melody is at least 8 velocity units over the accompaniment notes sounding with it;
   - `F.ring` is at least 1.5× the v4 value;
   - the MIDI and the tab stay on the grid;
   - `--sound plucked` renders deterministically, with no clipping, for every preset tuning.
4. **Feature 2 (shapes):**
   - P3 is at least 0.99 on every matrix (v4: 0.94–0.98);
   - the mean `F.M6` (ambitus) rises by at least 2 semitones over v4;
   - `F.shape_moves` stays at or below 0.5 window moves per bar.
5. **Feature 3 (melody):** on the validation matrix, `F.hook` ≥ 0.75, `F.peak` ≥ 0.9 and `F.arch` mean ≥ 0.5. All the hard rules in item 1 still hold.
6. **Feature 4 (best of N):**
   - the chosen candidate's score is at least every other candidate's, by construction and tested;
   - `F.score` mean rises over N = 1 on every matrix;
   - two runs with the same seed give identical output.
7. **Feature 5 (model):**
   - with a fake model, the prior measurably pulls pitches toward its targets and every hard check holds;
   - with the real weights (opt-in test), one 8-bar piece takes ≤ 60 s on the CPU, passes every item-1 check, and repeats from its seed.
8. **Gates:** unit and integration tests, `ruff check`, `ruff format --check` and `mypy --strict src` are green. The opt-in productionize e2e passes.
9. **Listening:** samples are rendered in `compose-samples/2026-11-16-v5/`:
   - 10 pieces with the defaults;
   - the same 10 with `--sound plucked`;
   - 4 with `--melody-model amt`;
   - 4 with `--rank-audio`.

   Each comes with a README table of the metrics.

## 7. Risks and what we do about them

- **Feature 2 may push shapes far up the neck.** The movement cost and the G4-centred range term limit it, and `F.shape_moves` is gated.
- **Contour and hook terms may fight the hard rules.** The hard rules always win; the new terms are soft.
- **Hooks repeat on purpose,** so the existing repeated-note metric (M3) may rise. That is allowed if it stays within the v3 tolerance. Otherwise it is listed in Results.
- **The `anticipation` package may not install on Python 3.11 or numpy < 2.** If so, I vendor the minimum of its tokenisation (Apache-2.0, credited) into `compose/ml/amt_tokens.py`.
- **Disk:** about 930 MB of weights on a disk with 30 GB free. No other large downloads.

## 8. Left to the builder

All weights, velocities and scoring constants; internal names; how candidates share work; and the voices of the plucked-string renderer. Every such choice is pinned in code and documented in the Results.
