# Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix three reproduced bugs:
1. A take folder can end on a different home chord from its `Composition`.
2. A dominated candidate can flip the audio re-rank.
3. Candidate scoring ignores the last bars of 9–15-bar passes.

**Architecture:**
- **Bug 1:** `compose` writes a `composition.json` manifest of its metadata. `productionize` reads it for folder and `.mid` sources.
- **Bug 2:** `choose_by_audio` uses fixed, measured scales instead of a per-shortlist min-max.
- **Bug 3:** `grid_features` can add an 8-bar window anchored at the last bar. Candidate scoring uses it; corpus statistics do not.

**Tech Stack:** Python 3.11, pretty_midi, pytest, mypy --strict, ruff.

**Spec:** `docs/superpowers/specs/2026-11-22-review-fixes-design.md`

## Global Constraints

- **Boom-bap and pop-rap output for a `Composition` source must not change.**
- **Strum and arpeggio stay frozen at v3:** `tests/unit/test_compose_frozen.py` must pass untouched.
- **A bare MIDI with no manifest behaves exactly as today.**
- **Corpus bands (`tune_stats`) and `tune_corpus.py` extraction are unchanged.**
- **Existing tests stay unedited, with three exceptions:**
  - Task 2 may update the `choose_by_audio` tests in `tests/unit/test_compose_ml_aesthetics.py` that pin the old min-max. The floor test's intent must survive: a gap under `AUDIO_SPAN_FLOOR = 0.3` does not by itself overturn the symbolic order.
  - Task 3 updates the two `COMPOSER_VERSION == 6` asserts, to 7.
- **Every task ends green:** `.venv/bin/pytest -q tests/unit`, `.venv/bin/mypy --strict src`, `.venv/bin/ruff check src tests` and `.venv/bin/ruff format --check src tests`.
- **Commits** end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Stage files by name. Never stage `spikes/`, `out/` or `compose-samples/`.

## Review Focus

1. **A take folder plus an explicit `--key` or `--progression`.** The explicit value wins over the manifest. Pinned in Task 1.
2. **A manifest beside a hand-edited `.mid` that no longer matches it,** for example a progression that doesn't divide the bars. The error is a clear `ProductionError`, raised before anything is written. Pinned in Task 1.
3. **A corrupt or future `composition.json`.** It raises a `ProductionError` that names the file, never a raw `JSONDecodeError` or `KeyError`. Pinned in Task 1.
4. **All candidates have equal scores.** `choose_by_audio` still returns index 0, and never divides by zero. Pinned in Task 2.
5. **Exactly 8 or 16 bars.** The features are byte-identical to today's, so most pieces don't change. Pinned in Task 3.

---

### Task 1: Take manifest

**Files:**
- Modify: `src/music_decoder/compose/api.py` (`compose_with_options`, where it builds `Composition(...)` near line 765)
- Modify: `src/music_decoder/productionize/song.py` (`composition_kwargs`)
- Modify: `src/music_decoder/productionize/api.py` (`resolve_song`, `resolve_poprap_song`)
- Modify: `docs/usage.md` ("Composition output" near line 889; the `productionize` SOURCE bullet)
- Test: `tests/unit/test_compose_api.py`, `tests/unit/test_productionize_api.py`

**Interfaces.** These are produced in `productionize/song.py`:

```python
MANIFEST_NAME = "composition.json"
MANIFEST_VERSION = 1
def manifest_path(midi_path: Path) -> Path                         # midi_path.with_suffix(".json")
def read_manifest(midi_path: Path) -> dict[str, Any] | None        # None if absent; ProductionError if unreadable or version != 1
def manifest_kwargs(metadata: Mapping[str, Any]) -> dict[str, Any] # tempo_bpm, progression, scale, title, key, exactly as composition_kwargs derives them today
# composition_kwargs(composition) becomes manifest_kwargs(composition.metadata)
```

- **`compose_with_options`** writes `work / "composition.json"` as `json.dumps({"manifest_version": 1, **metadata}, indent=2)`. `metadata` is the exact dict passed to `Composition`.
- **`resolve_song` and `resolve_poprap_song`, folder and `.mid` paths:**
  - `kw = manifest_kwargs(m) if (m := read_manifest(path)) else {}`.
  - An explicit `progression` or `key` replaces the manifest's. The explicit `title` and `key` overrides after building are unchanged.
  - For boom-bap, pass `tempo_bpm`, `progression`, `scale`, `title` and `key` from `kw` into `song_from_midi`. For pop-rap, pass them into `poprap_song`.

- [ ] **Step 1: Write the failing tests.** Compose with `Scale("C","major")`, `Dm G C Am`, 96 bpm, seed 7, through `compose()` into `tmp_path`.
  - `test_compose_writes_a_manifest`:
    - `composition.json` exists beside the MIDI;
    - `json["manifest_version"] == 1`;
    - every key of `result.metadata` equals the JSON value (compare after a `json` round trip).
  - `test_take_folder_matches_its_composition[boom-bap|pop-rap]`:
    - `resolve_song(folder) == resolve_song(comp)`, and `resolve_song(comp.midi_path) == resolve_song(comp)`;
    - for pop-rap, the same holds with `resolve_poprap_song` and the same `cfg`;
    - the pop-rap `parts.home.root == 0` (C).
  - `test_explicit_key_and_progression_beat_the_manifest`: `resolve_song(folder, key="Am")` gives `key == "Am"` and an A-rooted home chord.
  - `test_bare_midi_without_a_manifest_is_unchanged`: delete the JSON. `resolve_song(folder)` then equals `song_from_midi(midi)` exactly, as today.
  - `test_bad_manifest_is_a_clear_error`:
    - writing `"{"` raises `ProductionError` matching `composition.json`;
    - writing `{"manifest_version": 2}` raises `ProductionError` matching `version`;
    - in both cases `out_dir` does not exist after calling `productionize`.
  - `test_manifest_progression_that_does_not_divide_the_bars_is_an_error`: rewrite the manifest progression to 3 chords. `productionize(folder, ...)` raises `ProductionError`, and `out_dir` is absent.
- [ ] **Step 2: Run** `.venv/bin/pytest -q tests/unit/test_compose_api.py tests/unit/test_productionize_api.py`. Expected: the new tests fail.
- [ ] **Step 3: Implement** the interfaces above.
- [ ] **Step 4: Update the docs.**
  - Add `composition.json` to "Composition output": "the take's settings: scale, progression, tempo, bars per chord, seed and the rest of the metadata; `productionize` reads it".
  - Add one sentence to the `productionize` `SOURCE` bullet: a take folder's manifest supplies the key, progression and tempo, and `--key` / `--progression` override it.
- [ ] **Step 5: Run** the global check commands. Expected: PASS.
- [ ] **Step 6: Commit.** Message: `fix(compose): write a take manifest; productionize reads it`.

### Task 2: Fixed audio re-rank scales

**Files:**
- Modify: `src/music_decoder/compose/ml/aesthetics.py` (`choose_by_audio`, constants near lines 37-41)
- Modify: `docs/architecture-technical.md` (the re-rank formula near line 1142)
- Test: `tests/unit/test_compose_ml_aesthetics.py`

**Interfaces.** These are produced in `aesthetics.py`:

```python
AUDIO_SCALE: float      # measured (Step 1), and at least AUDIO_SPAN_FLOOR
SYMBOLIC_SCALE: float   # measured (Step 1)
def choose_by_audio(arrangements, audio_scores) -> int
# total_i = x_i / AUDIO_SCALE + SYMBOLIC_WEIGHT * s_i / SYMBOLIC_SCALE, with x = (CE + PQ) / 2 and s = arrangement.score.
# The first index wins ties. Same ValueErrors as today.
```

- [ ] **Step 1: Measure the scales.**
  - Write a script in `/private/tmp/claude-501/-Users-shubhojeet-Projects-music-decoder/87b150ed-dd08-4235-8266-9155ac0b57cf/scratchpad/rerank-scales/`. Never put it in the repo.
  - Draw 30 prompts the way `/private/tmp/claude-501/-Users-shubhojeet-Projects-music-decoder/87b150ed-dd08-4235-8266-9155ac0b57cf/scratchpad/batch_generate.py` does, for ids 0-29.
  - For each prompt:
    - take `arrange_candidates(..., style="fingerstyle", candidates=12)[:RANK_AUDIO_TOP_K]`;
    - render each candidate with `_render_fingerstyle_wav`, using the soundfont sound;
    - score the renders with `load_scorer()`;
    - record the span of `x` and the span of `s`.
  - `AUDIO_SCALE = max(median(max(span_x, 0.3)), AUDIO_SPAN_FLOOR)` and `SYMBOLIC_SCALE = median(span_s)`. Round both to 3 significant figures.
  - Put the two numbers, the prompt rule and the median spans in a comment above the constants.
- [ ] **Step 2: Write the failing tests.**
  - `test_a_dominated_candidate_does_not_flip_the_order`: with symbolic scores (1.0, 0.0) and x of (6.1, 6.2), A wins. Appending C (symbolic -10.0, x 6.0) still makes A win.
  - `test_appending_a_dominated_candidate_never_reorders`: draw 200 random shortlists of 2-4 with `random.Random(0)`, x in [4, 8], s in [-2, 2]. Append a candidate whose x and s are both below every existing one, and assert that `choose_by_audio` returns the same index as it did without it.
  - `test_equal_scores_pick_the_first`: all candidates equal gives 0.
  - **Floor intent:** a 0.1 audio gap against a symbolic lead of `SYMBOLIC_SCALE` keeps the symbolic winner. Adapt the existing floor test to this, rather than deleting its intent.
- [ ] **Step 3: Run** `.venv/bin/pytest -q tests/unit/test_compose_ml_aesthetics.py`. Expected: the dominated-candidate test fails.
- [ ] **Step 4: Implement** the fixed-scale `choose_by_audio`.
  - Remove `_min_max` if nothing else uses it.
  - Keep `AUDIO_SPAN_FLOOR` as the lower bound for `AUDIO_SCALE`.
  - Update the docstring and the architecture-technical formula.
- [ ] **Step 5: Run** the global check commands. Expected: PASS.
- [ ] **Step 6: Commit.** Message: `fix(compose): audio re-rank on fixed scales, not per-shortlist min-max`.

### Task 3: Score the last bars; composer v7

**Files:**
- Modify: `src/music_decoder/compose/tune_metrics.py` (`grid_features` near line 116, `tune_features` near line 156)
- Modify: `src/music_decoder/compose/api.py` (`COMPOSER_VERSION = 7`)
- Modify: `tests/unit/test_compose_tune_wiring.py:69` and `tests/unit/test_compose_api.py:125` (6 to 7 only)
- Test: `tests/unit/test_compose_tune_metrics.py`

**Interfaces:**
- `grid_features(..., *, tail_window: bool = False)`. When `tail_window and n_bars > WINDOW_BARS and n_bars % WINDOW_BARS`, it appends start `n_bars - WINDOW_BARS` to `starts`. Everything else is unchanged.
- `tune_features(...)` calls it with `tail_window=True`. `tune_corpus.py` is untouched.

- [ ] **Step 1: Write the failing tests.**
  - `test_tail_window_sees_the_ending[9|12|15]`:
    - two melodies share the first 8 bars; one ends with 4 notes and the other with 32 in the remaining bars;
    - with `tail_window=True` their features differ;
    - with False they are equal, which is today's behaviour.
  - `test_tail_window_is_a_no_op_on_whole_windows[8|16]`: the features with True equal the features with False, compared exactly.
  - `test_tune_features_scores_the_ending`: a 12-bar `tune_features` case (3 chords × 4 bars) where only the last 4 bars differ gives different features.
- [ ] **Step 2: Run** `.venv/bin/pytest -q tests/unit/test_compose_tune_metrics.py`. Expected: the new tests fail.
- [ ] **Step 3: Implement** the change. Bump `COMPOSER_VERSION` to 7 and update the two version asserts.
- [ ] **Step 4: Run** the global check commands. Expected: PASS, and `test_compose_frozen.py` untouched and passing.
- [ ] **Step 5: Commit.** Message: `fix(compose): score the last bars of passes that aren't whole 8-bar windows; composer v7`.
