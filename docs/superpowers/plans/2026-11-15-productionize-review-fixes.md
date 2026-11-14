# Productionize Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the four approved review fixes on PR #3: click-free limiter, tonic ending for composer songs, opt-in render tests, samples prefetched by setup.

**Architecture:** Each fix is local to one module (`master.py`, `song.py` + `verse.py`, the e2e test + Makefile, `setup.sh`). No interface used outside `productionize` changes.

**Tech Stack:** Python 3.11, numpy/scipy, pytest, bash.

**Spec:** `docs/superpowers/specs/2026-11-15-productionize-review-fixes-design.md`

## Global Constraints

- Work in the worktree `.worktrees/pr-3` (branch `pr-3` = PR head); run with `PYTHONPATH=$PWD/src` and the repo's `.venv`.
- `ruff check src tests`, `ruff format --check src tests`, `mypy --strict src` stay clean; full `pytest tests/unit` stays green.
- The Hisaab reference song's notes and checks must not change.
- Nothing is pushed until the user says so.

## Review Focus

- A composer progression with no chord on the tonic (e.g. `F G Am F` in C major) → tail is the scale's tonic triad (Task 2 test).
- No scale and no key tag → tail falls back to the first bar's harmony (Task 2 test).
- Minor key tag `"F#m"` → tonic F# minor (Task 2 test).
- Limiter on a signal needing no reduction → gain stays exactly 1 (Task 1 test).
- `--no-models` must skip the sample fetch too (Task 4 test).

---

### Task 1: Click-free limiter

**Files:** Modify `src/music_decoder/productionize/master.py` (`limiter_gain`); Test `tests/unit/test_productionize_master.py`

- [ ] **Step 1: Failing tests** — `test_limiter_gain_has_no_sample_to_sample_jumps`: the 2 s bursty signal (0.5·sin 220 Hz + eight 30 ms noise bursts ×0.6, `default_rng(0)`, stereo); assert `np.abs(np.diff(g)).max() < 0.01` and `true_peak(x * g) <= ceiling_gain(cfg) * 1.0001`. `test_limiter_gain_is_unity_when_under_the_ceiling`: `0.1·sin` → `np.all(g == 1.0)`.
- [ ] **Step 2: Run** `pytest tests/unit/test_productionize_master.py -k limiter -q` → first FAILS (max step ≈ 0.048).
- [ ] **Step 3: Implement** — `target = uniform_filter1d(held, size=ahead, mode="nearest")` (drop the `np.minimum(held, …)`); comment why it holds the ceiling.
- [ ] **Step 4: Run** the master tests → PASS; full unit suite green.
- [ ] **Step 5: Commit** `fix(productionize): limiter gain without sample-to-sample jumps`.

### Task 2: Composer songs end on the tonic

**Files:** Modify `src/music_decoder/productionize/verse.py`, `src/music_decoder/productionize/song.py` (`song_from_midi` tail block), `src/music_decoder/productionize/api.py` (`resolve_song` passes `key`); Test `tests/unit/test_productionize_song.py`, `tests/unit/test_productionize_verse.py`

**Interfaces (produces):**
- `verse.tonic_harmony(harmonies: Sequence[BarHarmony], tonic_pc: int, minor: bool, scale_pcs: frozenset[int]) -> BarHarmony` — first bar harmony whose root is `tonic_pc`, else a triad (`{0,4,7}` / `{0,3,7}` on the tonic, essential = the third).
- `verse.resolve_voicing(h: BarHarmony, prev: tuple[int, ...], cfg: ComposerArrangeConfig) -> tuple[int, ...]` — the `right_hand_options(h, cfg)` option with the lowest `_voicing_cost(o, h, prev, centre)` among options the same size as `prev` (else all).
- `song._tonic(scale: Scale | None, key: str, first_root_pc: int) -> tuple[int, bool]` — `(pitch class, minor)` from the scale, else a key tag like `"F#m"`/`"Bb"` (flats mapped to sharps), else `(first_root_pc, False)`.

- [ ] **Step 1: Failing tests** (song): with the `composer_midi` fixture (Am F C G, A minor):
  - `test_tail_resolves_to_the_tonic`: `song_from_midi(composer_midi, progression=PROGRESSION, scale=Scale("A","minor"))` → tail piano pitch classes `== {9, 0, 4}` and lowest tail pitch % 12 == 9; final strings pitch classes ⊆ `{9, 0, 4}`.
  - `test_tail_uses_the_key_tag_without_a_scale`: `song_from_midi(composer_midi, key="Am")` → same pitch-class set.
  - `test_tail_falls_back_to_the_first_bar`: `song_from_midi(composer_midi)` → lowest tail pitch % 12 == 9 (first bar Am).
  - `test_tail_is_voiced_like_the_verses`: tail notes above E2 only (`min >= cfg.lh_low`), right-hand tail notes within `[cfg.rh_low, cfg.rh_high]`.
  (verse) `test_tonic_harmony_builds_a_triad_when_the_loop_lacks_it`: harmonies of `F G Am F` in C major, tonic 0 → tones `{0,4,7}`.
- [ ] **Step 2: Run** → FAIL (tail ends on G).
- [ ] **Step 3: Implement** the three functions; in `song_from_midi` build the tail and final strings chord from `left_hand(final_h) + resolve_voicing(final_h, voicings[-1], cfg)`; `resolve_song` passes `key=key or ""` into `song_from_midi`. Update the `--verse held` docs line only if it claims the tail is unchanged.
- [ ] **Step 4: Run** song + verse tests and full unit suite → PASS; re-render the review take and confirm it ends on C.
- [ ] **Step 5: Commit** `fix(productionize): composer songs end on the tonic, voiced like the verses`.

### Task 3: Render tests on request

**Files:** Modify `tests/integration/test_productionize_e2e.py`, `Makefile`, `docs/usage.md` (dev section); Test `tests/unit/test_productionize_e2e_gate.py`

- [ ] **Step 1: Failing test** — load the e2e module; assert a `skipif` mark whose condition is true when `MUSIC_DECODER_PRODUCTION_E2E` is unset, and whose reason names the variable.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** the mark (`os.environ.get("MUSIC_DECODER_PRODUCTION_E2E") != "1"`); `make test-production` = `MUSIC_DECODER_PRODUCTION_E2E=1 pytest -q tests/integration/test_productionize_e2e.py`.
- [ ] **Step 4: Run** `pytest tests/integration/test_productionize_e2e.py -q` → 2 skipped; gate test PASS.
- [ ] **Step 5: Commit** `test(productionize): run the render tests only on request`.

### Task 4: Samples in setup

**Files:** Modify `setup.sh` (models step + header help), `README.md`, `docs/usage.md`; Test `tests/unit/test_setup_script.py`

- [ ] **Step 1: Failing test** — `setup.sh` contains `music-decoder fetch-samples` inside the `SKIP_MODELS` else-branch (assert the substring appears after `--no-models)` handling and before `# --- 7.`), and `--help` output mentions samples.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** — after `fetch-models`, run `music-decoder fetch-samples || fetch_ok=0`; help text: "pre-downloads models and production samples".
- [ ] **Step 4: Run** setup tests; `bash -n setup.sh`, `zsh -n setup.sh` → OK.
- [ ] **Step 5: Commit** `feat(setup): prefetch production samples by default`.
