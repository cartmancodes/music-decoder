# Phrase Model Melody Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A small model trained on POP909 phrases that picks the best of many tune-engine melodies and writes melodies itself, built only as far as the user's blind listening verdict says.

**Architecture:** Tasks 1–2 are a throwaway spike that trains the model, checks its premise, and produces a 12-clip blind round; execution then stops for the user's ratings. Tasks 3–5 build the model into the package (tokens, runtime, training and shipped weights). Task 6 adds the pick rule, Task 7 the engine, each only if its version won. Task 8 is docs, defaults and the final run.

**Tech Stack:** Python 3.11, numpy, torch (CPU when composing; GPU or CPU when training), click, pytest.

**Spec:** `docs/superpowers/specs/2026-11-25-phrase-model-melody-design.md`

## Global Constraints

- **Where the work happens:**
  - Package work (Tasks 3–8) happens in a git worktree on a new branch `feat/phrase-model` cut from `master`, never on `feat/poprap-piano-lead`, which holds the user's uncommitted piano work.
  - Run tests there as `PYTHONPATH=$PWD/src /Users/shubhojeet/Projects/music-decoder/.venv/bin/pytest`, so the worktree's code is imported rather than the main checkout's.
  - The spike (Tasks 1–2) runs from the main checkout and changes no package file.
- **No commits.** Leave all changes uncommitted, as on the piano-lead branch, until the user asks. Never stage `spikes/`, `out/` or `compose-samples/`.
- **Spike and samples folders:** spike code goes in `spikes/phrase-model/`, the round's files in `compose-samples/2026-11-25-phrase-model-round/`. Spike code gets no tests and is never copied into the package. Tasks 3–7 re-implement it test-first.
- **Data:** POP909 only, from `~/.music-decoder/datasets/pop909-hsa/POP909` (MIT, Shuqi Dai 2020), read with `tune_corpus.read_song`. Song split is 80/10/10 train/validation/held-out with split seed `909`. Every "held-out" number uses held-out songs only.
- **Tokens:**
  - 16 slots per bar. Token 0 means no onset; tokens 1..R are onsets at relative pitch `pitch_lo..pitch_hi`. That range is the smallest covering at least 99.5% of training notes.
  - The context is 512 slots (32 bars).
  - Bars since the phrase start and bars until its end are each capped at 15.
  - Tempo buckets have edges `(80.0, 100.0, 120.0)` in quarter notes per minute.
  - Chord qualities are `("maj", "min", "maj7", "min7", "7", "dim", "sus4", "aug", "other", "none")`.
- **The pitch anchor** is the pitch with the key's tonic pitch class at or below a middle pitch:
  - in training, the song's median melody pitch;
  - when writing, `tessitura_center`;
  - when scoring a finished melody, that melody's median pitch (my ruling: it matches how training measured).
- **Network:**
  - A causal transformer of at most 2,000,000 parameters. The spec caps it at 5 million but also caps the file at 6 MB, which wins: weights are float16 at 2 bytes each, plus the figure index.
  - The weights file is `src/music_decoder/compose/data/phrase_model.npz`, saved with `np.savez` and loaded with `allow_pickle=False`, at most 6 MB.
  - Its hash is the first 12 hex digits of the file's sha256.
- **Pick rule:**
  - `phrase_distance = |S - median| / IQR`, with `S` = mean bits per bar over cold 8-bar windows, leaving out a forced final whole-note bar.
  - The weight is `W_PHRASE = 2.0`.
  - A melody with a note outside the token range gets `PHRASE_OUT_OF_RANGE = 10.0` (my ruling).
- **Engine:**
  - Onsets only on even slots, pitches only from `allowed_pitches`, no move over 7 semitones, notes held `tune.LENGTH_FACTOR` (0.95) of the gap to the next onset, none crossing a bar line.
  - Density is steered to `tune._target_onsets(tempo, stats)`.
  - The copy check allows at most `MAX_PHRASE_REWRITES = 8` rewrites, then falls back to the tune engine.
- **Options:**
  - `--melody-pick [bands|phrase]`, default `bands`. `phrase` is refused with the `rules` engine.
  - `--melody-engine` gains `phrase`.
  - `phrase` plus `--melody-model amt` is refused.
  - Strum and arpeggio ignore both options.
- **Metadata:** `melody_pick`, `phrase_model` (the weights hash, or `null` when no phrase option is used) and `phrase_fallback` (bool).
- **Take-folder hash:** it gains `melody_pick` and the weights hash only when a phrase option is used, so the default options keep their current folder names.
- **Composer version:** stays 7 until the user changes a default (Task 8).
- **Dependency:** `torch>=2.2` becomes a core dependency. It is imported only inside phrase-model code paths, never by importing `music_decoder.compose.api` or the CLI.

## Review Focus

- Strum or arpeggio with `melody_pick="phrase"`: ignored, as `melody_engine` is, with no error. The Task 6 test is `test_phrase_pick_is_ignored_by_strum_and_arpeggio`.
- Default options keep the same take-folder name. The new fields enter the folder hash only when they are not the defaults. The Task 6 test is `test_default_take_folder_hash_is_unchanged`.
- A scored melody with a note outside the model's pitch range gets distance 10.0 and no crash. The Task 6 test is `test_out_of_range_melody_scores_the_fixed_distance`.
- A bar whose allowed pitches are all outside the token range makes the phrase engine fall back to the tune engine with `phrase_fallback` set, and no crash. The Task 7 test is `test_unreachable_bar_falls_back_to_the_tune_engine`.
- Weights missing or unreadable: a `PhraseModelError` naming the path, raised only when a phrase option is used; default compose never touches the file or torch. The Task 4 tests are `test_missing_weights_raise_with_the_path` and `test_compose_api_import_does_not_import_torch`.

---

### Task 1: Spike — train the model and check its premise

**Files:** create `spikes/phrase-model/{data.py,model.py,train.py}`, and `spikes/phrase-model/findings.json`. No tests (spike).

- [ ] **Step 1:** Write `data.py`:
  - encode every POP909 song by the Global Constraints' token rules: per-slot tokens plus the ten conditioning streams `mode, root, quality, next_root, next_quality, slot, since, until, restate, tempo`;
  - the split;
  - training spans of up to 32 bars, starting at a capital-letter phrase's first bar with nothing before them.

  The rules for chords and phrases:
  - Root and quality are the unique pair, over 12 roots × the 8 qualities, whose pitch-class set equals the chord's. No match or several matches give `other`; no chord gives `none`.
  - A phrase is any label in `human_label1.txt`. It is a restatement when the same letter, case-sensitive, appeared earlier in the song.
  - A span with a note outside the token range is left out.
- [ ] **Step 2:** Write `model.py` (the causal transformer) and `train.py`.
  - Train three sizes under 2,000,000 parameters; pick the lowest validation bits per bar.
  - Then train the same chosen size with attention limited to the last 16 slots (the one-bar model).
- [ ] **Step 3:** Run `.venv/bin/python spikes/phrase-model/train.py`.
  - Expected: `findings.json` holds `split_seed, pitch_lo, pitch_hi, arch, params, heldout_bits_full, heldout_bits_onebar, premise_ratio` (the share of bits saved: `1 - full/onebar`) and the weights path.
- [ ] **Step 4: Premise gate.**
  - If `premise_ratio < 0.10`: stop. Write `compose-samples/2026-11-25-phrase-model-round/README.md` with the numbers, tell the user long context buys nothing on this data, and end the plan here.
  - Otherwise go to Task 2.

### Task 2: Spike — scores, read-out, prototype engine and the blind round

**Files:** create `spikes/phrase-model/{score.py,engine.py,round.py}`; extend `findings.json`. Write to `compose-samples/2026-11-25-phrase-model-round/`. No tests (spike).

- [ ] **Step 1:** In `score.py`:
  - compute `S` and the held-out reference windows (8 bars, starting at a labelled phrase, every onset on an even slot);
  - record `surprise[mode] = (median, iqr)`;
  - print where 200 tune-engine candidates of each round setting fall against the real band.
- [ ] **Step 2: Read-out.**
  - Take the ten rated clips in `compose-samples/2026-11-23-theory-melody-spike/ratings.json` (melodies from `takes/<piece>/composition.mid`).
  - Compute the Spearman correlation of the ratings with `-phrase_distance`, with `-S` and with `-band_distance`. Record all three and the clip count in `findings.json`.
  - If the first is negative while the second is positive, the round's pick rule is `-S` (lowest surprise). Record the rule as `pick_rule`.
- [ ] **Step 3:** In `engine.py`, write `generate_phrase_tune` with the same keyword arguments as `tune.generate_tune`.
  - It samples with the composer's numpy generator, under the Engine constraints, with density steering and the copy check.
  - Set the copy length `L`: the smallest at which at most 1% of held-out 8-bar windows share an `L`-note run (intervals plus onset gaps) with a different training song.
- [ ] **Step 4:** Choose temperature from {0.8, 0.9, 1.0} and nucleus from {0.9, 0.95, 1.0}.
  - Monkeypatch `music_decoder.compose.api.generate_tune` with the spike engine and run `evaluation.compose_tune.tune_cases("dev")`.
  - Pick the pair whose `tune_report` puts the most (mode, feature) medians inside the `tune_stats.json` bands; on a tie, the higher temperature, then the higher nucleus.
  - Record the pair, its in-band count and the list of missed bands.
- [ ] **Step 5:** `round.py` builds 12 clips with standard tuning and base seeds `26000 + i`. The four settings:

  | i | Scale | Progression | Tempo | Bars per chord |
  |---|---|---|---|---|
  | 0 | C:major | C Am F G C Am Dm G | 112 | 2 |
  | 1 | E:major | E B C#m G#m A E A B | 110 | 2 |
  | 2 | F#:minor | F#m Bm C#m F#m | 124 | 2 |
  | 3 | D#:minor | D#m B F# C# | 110 | 2 |

  Three versions of each, through `compose_with_options(..., style="fingerstyle")`:
  - **today:** `candidates=12` as shipped;
  - **model's pick:** `candidates=200`, with `api.score_arrangement` monkeypatched to the `pick_rule` score;
  - **model's tune:** `candidates=12`, with `api.generate_tune` monkeypatched to the spike engine.

  Convert each `composition.wav` to mp3. Shuffle the 12 with `numpy.random.default_rng(2611)`, name them `blind/01.mp3` to `blind/12.mp3`, and write `blind_key.json` (clip → version, setting, seed) and a `README.md` asking for a rating out of 10 per clip, which names no versions.
- [ ] **Step 6:** Send the 12 clips to the user with SendUserFile. Report:
  - the premise ratio;
  - the read-out correlations, with "10 clips, weak evidence";
  - the missed bands;
  - each version's compose time.

  Ask for the ratings, and say the defaults choice (spec 6.3) comes after them.
- [ ] **Step 7: STOP.** Execution waits for the user's ratings. On receipt, write `ratings.json` beside the clips and work out the verdict per spec section 4, step 5:
  - model's pick wins → Tasks 3, 4, 5, 6, 8;
  - model's tune wins → Tasks 3, 4, 5, 7, 8;
  - both win → all tasks;
  - neither wins → write up the findings in the round's `README.md`, update the memory `composer-job-is-melodies`, end;
  - within 1.0 either way → offer the second round of 12 on new settings (rerun Step 5 with base seeds `26100 + i` and the next four rated settings in `ratings.json`).

  Ledger the verdict and the user's default choices.

### Task 3: Phrase tokens

**Files:** create `src/music_decoder/compose/phrase_tokens.py` and test `tests/unit/test_compose_phrase_tokens.py`, using the existing fixture `tests/fixtures/compose/pop909_mini`.

**Interfaces — Produces:**
- Constants: `SLOTS_PER_BAR = 16`, `NO_ONSET = 0`, `CONTEXT_SLOTS = 512`, `MAX_PHRASE_BARS = 15`, `TEMPO_EDGES`, `QUALITIES`, `CONDITION_KEYS` (the ten stream names, in the order above) and `CONDITION_SIZES: dict[str, int]`.
- `@dataclass(frozen=True) class Encoded`, with fields:
  - `tokens: np.ndarray` (int64, one per slot);
  - `cond: dict[str, np.ndarray]` (int64, one per slot per key);
  - `out_of_range_bars: frozenset[int]`.
- `anchor_pitch(tonic_pc: int, middle: int) -> int`.
- `piece_conditioning(*, scale: Scale, progression: Sequence[ChordSymbol], bars_per_chord: int, tempo_bpm: float) -> dict[str, np.ndarray]`. Phrases come from `rhythm.split_phrases` and `rhythm.form_labels` (`A'` is a restatement).
- `encode_piece(melody: Sequence[Note], *, scale, progression, bars_per_chord, tempo_bpm, middle: int, pitch_lo: int, pitch_hi: int) -> Encoded`. The onset slot is `round(start_s / (15 / tempo_bpm))`.
- `encode_song(song: Song, pitch_lo: int, pitch_hi: int) -> Encoded`, `song_relative_pitches(song: Song) -> np.ndarray`, `phrase_starts(song: Song) -> list[int]` (first bars of capital-letter phrases) and `eighth_only(enc: Encoded, first_bar: int, bars: int) -> bool`.
- `figure_hashes(onsets: Sequence[int], pitches: Sequence[int], length: int) -> np.ndarray` (uint32). One hash per run of `length` consecutive notes, over its intervals and its onset gaps in sixteenths, taken from the first 4 bytes of `hashlib.blake2b`. It is independent of `PYTHONHASHSEED`.

- [ ] **Step 1:** Write the tests:
  - `test_anchor_is_the_tonic_at_or_below_the_middle`: `anchor_pitch(0, 64) == 60` and `anchor_pitch(0, 60) == 60`.
  - `test_piece_conditioning_marks_restatements`: an 8-bar C-major piece has `restate` 0 in phrase A and 1 in A'; `slot` cycles 0..15; `until` counts down to 0 at each phrase's last bar.
  - `test_chord_quality_maps_by_pitch_class_set`: `{9, 0, 4, 7}` gives root 9, `min7`; `{0, 4, 8}` gives `other`; no chord gives `none`.
  - `test_encode_piece_round_trips_onsets`: two notes at 0 s and 0.25 s at 120 BPM give onsets at slots 0 and 2, zeros elsewhere.
  - `test_out_of_range_note_is_reported`: a note above `pitch_hi` puts its bar in `out_of_range_bars`.
  - `test_encode_song_reads_the_mini_corpus`: `phrase_starts(read_song(mini / "001")) == [0, 4]`; tokens have 16 slots per bar.
  - `test_figure_hashes_ignore_transposition`: a run moved up 5 semitones hashes the same; changing one gap changes the hash.
- [ ] **Step 2:** Run `PYTHONPATH=$PWD/src …/.venv/bin/pytest tests/unit/test_compose_phrase_tokens.py -q`. Expected: FAIL (module missing).
- [ ] **Step 3:** Implement `phrase_tokens.py`.
- [ ] **Step 4:** Rerun. Expected: PASS.

### Task 4: Phrase model runtime

**Files:** create `src/music_decoder/compose/phrase_model.py` and `tests/compose_phrase_helpers.py`; test `tests/unit/test_compose_phrase_model.py`.

**Interfaces:**
- Consumes: Task 3's `Encoded`, `CONDITION_KEYS` and `CONDITION_SIZES`.
- Produces:
  - `class PhraseModelError(RuntimeError)` and `WEIGHTS_PATH: Path`.
  - `@dataclass(frozen=True) class PhraseMeta`, with fields `sources`, `licence`, `split_seed`, `pitch_lo`, `pitch_hi`, `arch: dict[str, int]`, `heldout_bits: float`, `surprise: dict[str, tuple[float, float]]`, `figure_length: int`, `temperature: float` and `nucleus: float`.
  - `init_arrays(arch: Mapping[str, int], vocab: int, seed: int) -> dict[str, np.ndarray]`.
  - `save_phrase_model(path: Path, arrays, meta: PhraseMeta, figures: np.ndarray) -> None` (float16 arrays, meta as one JSON string array, figures as a sorted uint32 array).
  - `load_phrase_model(path: Path | None = None) -> PhraseModel`, cached per path. It raises `PhraseModelError` naming the path when the file is missing or unreadable, and when torch is not installed.
  - `class PhraseModel`, with attributes `meta` and `weights_hash` and these methods:
    - `slot_bits(enc: Encoded) -> np.ndarray`: per-slot bits from a cold start;
    - `sampler(cond: dict[str, np.ndarray]) -> PhraseSampler`, where `PhraseSampler` has `probs() -> np.ndarray` (float64 over the vocabulary, for the next slot) and `push(token: int) -> None`;
    - `has_figure(onsets: Sequence[int], pitches: Sequence[int]) -> bool`.
  - Test helper: `tiny_model(tmp_path: Path) -> PhraseModel`, a model of fewer than 20,000 parameters from `init_arrays(..., seed=0)`.

- [ ] **Step 1:** Write the tests:
  - `test_saved_model_loads_without_pickle`: `np.load(path, allow_pickle=False)` reads every key.
  - `test_slot_bits_match_the_sampler`: for a fixed `Encoded`, `slot_bits` equals `-log2` of `sampler.probs()[token]` slot by slot, within 1e-6.
  - `test_probs_sum_to_one`.
  - `test_same_inputs_same_bits`.
  - `test_has_figure_finds_a_stored_run`.
  - `test_missing_weights_raise_with_the_path`.
  - `test_compose_api_import_does_not_import_torch`: in a subprocess, `import music_decoder.compose.api, music_decoder.cli.main` leaves `"torch"` out of `sys.modules`.
- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement. Import torch inside the functions only. Sampling steps reuse cached keys and values so each slot costs one forward step.
- [ ] **Step 4:** Rerun. Expected: PASS.

### Task 5: Training script and the shipped weights

**Files:**
- create `src/music_decoder/compose/phrase_train.py` and `scripts/train_phrase_model.py`;
- create `src/music_decoder/compose/data/phrase_model.npz`;
- modify `pyproject.toml` (add `"torch>=2.2"` to `dependencies`);
- test `tests/unit/test_compose_phrase_train.py`;
- extend `tests/unit/test_compose_phrase_model.py`.

**Interfaces:**
- Consumes: Tasks 3 and 4.
- Produces: `build(root: Path, out: Path, *, arch: Mapping[str, int], steps: int, seed: int, temperature: float, nucleus: float) -> PhraseMeta`. It does the split, pitch range, training, held-out bits, surprise reference, `L` at the 1% level and the figure index from training songs, then saves. The script's usage is `train_phrase_model.py POP909_DIR [--out PATH] [--steps N]`; the other arguments come from Task 2's `findings.json`, written into the script as constants.

- [ ] **Step 1:** Write `test_build_on_the_mini_corpus_writes_a_loadable_model` (a few steps on `pop909_mini`; the meta has all fields; `surprise` has both modes or the mode present) and `test_split_is_by_song_and_seeded`.
- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement `phrase_train.py` and the script.
- [ ] **Step 4:** Rerun. Expected: PASS.
- [ ] **Step 5:** Run `PYTHONPATH=$PWD/src …/.venv/bin/python scripts/train_phrase_model.py ~/.music-decoder/datasets/pop909-hsa/POP909`.
  - Expected: the file is 6 MB or less, and `heldout_bits` is within 3% of the spike's `heldout_bits_full`.
  - If it is outside that, use systematic-debugging. The trainer is meant to match the spike, so a gap is a bug until shown otherwise.
- [ ] **Step 6:** Add `test_shipped_weights_score_a_fixed_melody`. It pins `slot_bits(...).sum()` of a fixed 8-bar C-major melody to 3 decimals and asserts the file size is at most 6 MB. Run it. Expected: PASS.

### Task 6: The pick rule (only if "model's pick" won)

**Files:** modify `compose/scoring.py`, `compose/api.py`, `compose/ml/aesthetics.py` (scale choice only) and `cli/main.py`. Tests: `tests/unit/test_compose_scoring.py`, `test_compose_api.py` and `test_cli_compose.py` (or the file holding the compose CLI tests).

**Interfaces:**
- `scoring.py`:
  - `MelodyPick = Literal["bands", "phrase"]` (`MelodyEngine` is Task 7's to change);
  - `W_PHRASE = 2.0` and `PHRASE_OUT_OF_RANGE = 10.0`;
  - `phrase_surprise(melody, *, scale, progression, bars_per_chord, tempo_bpm, final_whole_note=True, model=None) -> float | None` (`None` when out of range);
  - `phrase_distance(...) -> float`;
  - `score_arrangement(..., melody_pick: MelodyPick = "bands")`. `"phrase"` swaps the band term for `-W_PHRASE × phrase_distance`.
- `api.py`: `arrange_candidates` and `compose_with_options` gain `melody_pick: MelodyPick = "bands"`. Metadata and the manifest gain `melody_pick` and `phrase_model`. If the round's `pick_rule` was `-S`, the term is `-W_PHRASE × (S - median) / IQR` instead; ledger it.
- CLI: `--melody-pick [bands|phrase]`.

- [ ] **Step 1:** Write the tests, using `tiny_model` with a monkeypatched loader:
  - `test_phrase_pick_reorders_candidates` (the candidate nearest the median wins);
  - `test_phrase_pick_with_rules_is_refused` (`ValueError`);
  - `test_phrase_pick_is_ignored_by_strum_and_arpeggio`;
  - `test_out_of_range_melody_scores_the_fixed_distance`;
  - `test_default_take_folder_hash_is_unchanged` (the hash of default options equals the current formula's string);
  - `test_metadata_records_the_pick_and_the_weights_hash` (`phrase_model` is null for defaults);
  - `test_cli_melody_pick_reaches_the_options`;
  - `test_rank_audio_with_phrase_pick_runs` (fake scorer; the scale stays `SYMBOLIC_SCALE`).
- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run `…pytest tests/unit/test_compose_scoring.py tests/unit/test_compose_api.py tests/unit/test_compose_ml_aesthetics.py -q` and the CLI test. Expected: PASS.

### Task 7: The phrase engine (only if "model's tune" won)

**Files:**
- create `src/music_decoder/compose/phrase_tune.py`;
- modify `compose/api.py` (`_arrange_fingerstyle`, `Arrangement.phrase_fallback: bool = False`, validation, metadata), `evaluation/compose_tune.py` (`build_tune_case(..., engine="tune")` and `tune_cases(matrix, engine="tune")`) and `cli/main.py`;
- tests: `tests/unit/test_compose_phrase_tune.py`, and parametrize `tests/unit/test_compose_tune_gates.py` over `("tune", "phrase")`.

**Interfaces:**
- `scoring.MelodyEngine = Literal["tune", "rules", "phrase"]`. The `phrase` engine is scored like `tune` (its band term, or Task 6's pick rule when chosen). `_validate_engine`'s message lists all three.
- Produces `generate_phrase_tune(*, scale, progression, bars_per_chord=1, tempo_bpm=100.0, seed=None, tessitura_center, allowed_pitches, final_whole_note=True, pitch_prior=None, model=None) -> list[Note]` and `class PhraseFallback(Exception)`. The exception is raised after `MAX_PHRASE_REWRITES = 8` copy hits, or when a bar has no allowed pitch in the token range.
- `TEMPERATURE` and `NUCLEUS` are module constants, equal to the shipped meta's values.
- `_arrange_fingerstyle` catches `PhraseFallback`, calls `generate_tune` with the same arguments, and sets `phrase_fallback=True`.

- [ ] **Step 1:** Write the tests, using `tiny_model`:
  - `test_onsets_are_on_eighths_and_inside_allowed_pitches`;
  - `test_no_move_over_a_fifth`;
  - `test_final_bar_is_one_held_cadence_note`;
  - `test_each_phrase_ends_on_a_chord_tone_and_none_is_empty`;
  - `test_same_seed_same_melody`;
  - `test_density_follows_the_tempo_target` (mean onsets per bar within 1.0 of `_target_onsets` over 20 seeds);
  - `test_copied_figure_is_rewritten_then_falls_back` (an index holding every run);
  - `test_unreachable_bar_falls_back_to_the_tune_engine`;
  - `test_phrase_engine_with_amt_is_refused`;
  - `test_cli_melody_engine_phrase`.
- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement. For T1/T2, in `test_compose_tune_gates.py` run every gate for both engines with the shipped weights. Key `KNOWN_MISSES` by `(engine, mode, feature)`; add a phrase entry only for a miss the user accepted in Task 2, with the reason `"user's blind verdict 2026-11-??"` (the date of their ratings).
- [ ] **Step 4:** Add `test_phrase_engine_copy_check_holds_on_both_matrices` and `test_phrase_engine_is_at_most_3x_slower` (12 candidates, model preloaded; mark it `slow`).
- [ ] **Step 5:** Run `…pytest tests/unit/test_compose_phrase_tune.py tests/unit/test_compose_tune_gates.py -q`. Expected: PASS, with xfails only for accepted misses.

### Task 8: Docs, defaults and the final check

**Files:** modify `docs/usage.md` (compose options, the phrase model, its licence and credit), and `compose/api.py` (`COMPOSER_VERSION`) only if the user changed a default; also the tests that pin the version.

- [ ] **Step 1:** If the user chose a new default in Task 2, Step 7, apply it, set `COMPOSER_VERSION = 8`, and update the version tests. Otherwise leave version 7.
- [ ] **Step 2:** Update `docs/usage.md`: the new options, what they do, the compose times measured, the training script, and the POP909 MIT credit.
- [ ] **Step 3:** Run the full suite, `…pytest -q`. Expected: all pass, with the same skips as `master`.
- [ ] **Step 4:** Run `music-decoder compose --scale A:minor --progression "Am F C G" --bars 2` with each new option (via the worktree's code), and confirm the metadata fields and timings.
