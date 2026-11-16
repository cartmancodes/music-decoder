# v5 Feature 5: Pretrained Melody Model as a Soft Prior — Implementation Plan (and v5 wrap-up)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An opt-in pretrained melody model, the Anticipatory Music Transformer (AMT) `stanford-crfm/music-small-800k`, proposes a melody over the piece's thumb pattern. Its pitches become a soft prior in the melody generator, so every hard rule and reach set still decides. This plan then finishes v5 with the docs, the listening samples and the spec's Results.

**Architecture:**
- `generate_melody` gains `pitch_prior: Sequence[Note] | None`. For each rhythm slot, the highest prior note sounding at the slot's onset gives a target pitch, and `score_pitch` adds an octave-aware bonus near that target.
- A `MelodyProposer` protocol in `compose/ml/__init__.py` decouples the composer from the model.
- `compose/ml/amt.py` implements it with AMT, loaded lazily and run on the CPU.
- The proposal is made once per piece and shared by all 12 candidates, which keeps an 8-bar piece under 60 s.

**Tech Stack:**
- Python 3.11, numpy, pytest.
- The `ml` extra gains:
  - `anticipation @ git+https://github.com/jthickstun/anticipation@af37397922665a0fb8d474d7988b0f3755a38d45` (Apache-2.0, no declared dependencies);
  - `tqdm`, which `anticipation.sample` imports.
- The weights are `stanford-crfm/music-small-800k` (Apache-2.0, a 512 MB `pytorch_model.bin`). They must be loaded with torch's weights-only loading; check that `transformers` uses `weights_only=True` for `.bin` files, and if it does not, load the state dict yourself with `torch.load(..., weights_only=True)`.

**Spec:** `docs/superpowers/specs/2026-11-16-composer-v5-design.md` (§5.5, §5.6, §6 items 1, 2, 7, 8 and 9).

## Global Constraints

- Activate the venv first: `source .venv/bin/activate`.
- Strum and arpeggio must not change. `tests/unit/test_compose_frozen.py` must pass untouched. With `pitch_prior=None`, `generate_melody` and every arrangement must be byte-identical to now.
- `compose()`'s signature and the package exports stay frozen. `COMPOSER_VERSION` stays 5.
- The fingerstyle v4 guarantees hold, with and without the model: P1, P2 and P4 are 1.0, P3t is 1.0, P5 is 0, A4 and H2 hold, and the v3 hard melody rules hold. All feature 2–4 gates stay as they are. The model is opt-in, so the default harness does not use it.
- The default path and the default tests never import torch, transformers or anticipation through this feature. ML imports live only under `compose/ml/` and are lazy.
- Same seed and same arguments give identical output, with or without the model. The model runs on the CPU under `torch.random.fork_rng()`, with `torch.manual_seed(seed)`.
- Downloads pin a revision and send no token (`token=False`), as feature 4's `fetch_audiobox` does.
- If a v4 guarantee fails, stop and report. Do not loosen the assertion.
- Gates before each commit: `pytest -q tests/unit`, `ruff check src tests`, `ruff format --check src tests`, `mypy --strict src`. Run `pytest -q tests/integration` at the last task.
- End every commit message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Branch: `feat/composer-v5`.

## Review Focus

1. **The model proposes nothing, or only notes outside the reach** (empty output, silence, drums): those bars get no prior, and the piece still composes. Test: Task 1's `test_an_empty_or_unreachable_prior_changes_nothing_or_little`.
2. **The model's notes sit outside the reach sets:** the bonus is octave-aware, so a proposal two octaves away still pulls toward its pitch class. Test: Task 1's `test_prior_is_octave_aware`.
3. **`--melody-model amt` with strum or arpeggio:** it is ignored and recorded as `metadata["melody_model"] = None`. The model is never loaded. Test: Task 1's `test_melody_model_is_fingerstyle_only`.
4. **The ML extra or the AMT weights are missing:** `MLUnavailableError` with the install hint. Test: Task 2's `test_amt_without_weights_explains_how_to_fetch`.
5. **Long pieces** (16 bars and more): generation time grows with length, so the model generates only to the piece's end, and an 8-bar piece takes 60 s or less. Test: Task 2's opt-in e2e.

---

### Task 1: The pitch prior in the generator and the proposer plumbing (fake model)

**Files:**
- Modify: `src/music_decoder/compose/melody.py` (`generate_melody(pitch_prior=)`; `NoteContext.prior_target`; `score_pitch`)
- Modify: `src/music_decoder/compose/ml/__init__.py` (the `MelodyProposer` protocol)
- Modify: `src/music_decoder/compose/api.py` (`arrange(..., melody_proposer=None)`, `arrange_candidates(..., melody_proposer=None)`, `compose_with_options(..., melody_model: Literal["amt"] | None = None)`; `_thumb_context`)
- Modify: `src/music_decoder/cli/main.py` (`--melody-model [none|amt]`)
- Test: `tests/unit/test_compose_melody.py`, `tests/unit/test_compose_ml_prior.py` (new)

**Interfaces:**
- Produces:
  - `PRIOR_WEIGHT = 2.0` and `PRIOR_WIDTH = 2.0`, in semitones.
  - `generate_melody(..., pitch_prior: Sequence[Note] | None = None)`.
    - For each slot, the target is the highest `pitch_prior` note with `start_s ≤ onset < end_s`. If none sounds, it is the nearest prior onset within one eighth after the onset; otherwise there is no target.
    - The bonus is `PRIOR_WEIGHT * exp(−d / PRIOR_WIDTH)`, where `d = min_k |cand − (target + 12k)|`, so it is octave-aware.
    - It is added in `score_pitch` only when a target exists. The hard rules (`admissible`, `ranked_scores`), the reach sets and the cadence target are unchanged.
  - `class MelodyProposer(Protocol)`, with one method: `def propose(self, context: Sequence[Note], *, end_s: float, tempo_bpm: float, seed: int) -> list[Note]: ...`.
  - `api._thumb_context(bar_voicings, tuning, *, tempo_bpm) -> tuple[Note, ...]`: the thumb pattern of each bar's hand (root on beats 1 and 3, alternate on 2 and 4), before any melody exists. It is the model's conditioning.
  - `arrange(..., melody_proposer: MelodyProposer | None = None)` and `arrange_candidates(..., melody_proposer=None)`, fingerstyle only.
    - The proposal is computed once per piece, from `_thumb_context(choose_bar_shapes(...))`, with `seed = seed` (or the drawn base seed when `seed` is None).
    - It is passed as `pitch_prior` to every candidate's `generate_melody`.
  - `compose_with_options(..., melody_model=None)`:
    - `"amt"` loads `compose.ml.amt.load_proposer()`, which Task 2 implements; until then it raises `MLUnavailableError`.
    - For strum and arpeggio it is ignored.
    - `metadata["melody_model"]` records the value.
  - CLI: `compose --melody-model [none|amt]`, default `none`.

- [ ] **Step 1: Write the failing tests** (fake proposers only):

```python
class _Fixed:          # proposes a fixed line regardless of context
    def __init__(self, notes): self.notes = notes
    def propose(self, context, *, end_s, tempo_bpm, seed): return list(self.notes)

def _line(pitches, step=0.5):
    return [Note(start_s=k * step, end_s=(k + 1) * step, pitch=p, velocity=80, confidence=1.0)
            for k, p in enumerate(pitches)]

def test_prior_off_is_unchanged():
    kw = dict(scale=Scale("C", "major"), progression=PROG8, tempo_bpm=120.0, seed=3)
    assert generate_melody(**kw) == generate_melody(**kw, pitch_prior=None)

def test_prior_pulls_pitch_classes_toward_the_proposal():
    target = _line([67] * 32)          # G4 for 16 s
    hits_with = hits_without = 0
    for seed in range(20):
        a = generate_melody(scale=Scale("C", "major"), progression=PROG8, tempo_bpm=120.0, seed=seed)
        b = generate_melody(scale=Scale("C", "major"), progression=PROG8, tempo_bpm=120.0, seed=seed,
                            pitch_prior=target)
        hits_without += sum(n.pitch % 12 == 7 for n in a)
        hits_with += sum(n.pitch % 12 == 7 for n in b)
        assert metrics.forbidden_interval_count(b) == 0
    assert hits_with > 1.3 * hits_without

def test_prior_is_octave_aware():
    far = _line([67 + 24] * 32)        # G6, far above any reach set
    near = _line([67] * 32)
    a = generate_melody(scale=Scale("C", "major"), progression=PROG8, tempo_bpm=120.0, seed=1, pitch_prior=far)
    b = generate_melody(scale=Scale("C", "major"), progression=PROG8, tempo_bpm=120.0, seed=1, pitch_prior=near)
    assert [n.pitch % 12 for n in a] == [n.pitch % 12 for n in b]

def test_an_empty_or_unreachable_prior_changes_nothing_or_little():
    a = arrange(Scale("C", "major"), PROG8, tempo_bpm=120.0, seed=2)
    b = arrange(Scale("C", "major"), PROG8, tempo_bpm=120.0, seed=2, melody_proposer=_Fixed([]))
    assert a.melody == b.melody

def test_fingerstyle_with_a_proposer_keeps_every_guarantee():
    prop = _Fixed(_line([64, 65, 67, 69, 71, 72, 71, 69] * 4))
    for seed in range(10):
        a = arrange(Scale("C", "major"), PROG8, tempo_bpm=120.0, seed=seed, candidates=4, melody_proposer=prop)
        part = [*a.melody_tab, *a.accompaniment_tab]
        assert metrics.playable_moment_ratio(part) == 1.0
        assert metrics.melody_onset_on_top_ratio(a.melody, a.accompaniment) == 1.0
        assert metrics.final_chord_rings(a.accompaniment, 8, 120.0)

def test_the_proposal_is_made_once_per_piece():
    calls = []
    class _Count(_Fixed):
        def propose(self, context, **kw): calls.append(kw["seed"]); return super().propose(context, **kw)
    arrange(Scale("C", "major"), PROG8, seed=5, candidates=12, melody_proposer=_Count([]))
    assert calls == [5]

def test_melody_model_is_fingerstyle_only(tmp_path, monkeypatch): ...
#   monkeypatch compose.ml.amt.load_proposer to fail if called; strum + melody_model="amt" → metadata None
def test_cli_accepts_melody_model_none(tmp_path): ...
```

- [ ] **Step 2: Run** the new tests. Expected: FAIL.
- [ ] **Step 3: Implement** the prior, the protocol, `_thumb_context`, the plumbing and the CLI flag. Create a stub `compose/ml/amt.py` whose `load_proposer()` raises `MLUnavailableError`.
- [ ] **Step 4: Run** `pytest -q tests/unit`. Expected: PASS, with `test_compose_frozen.py` and the musicality gates unchanged.
- [ ] **Step 5: Run the gates and commit**: `feat(compose): a soft pitch prior and a melody-proposer hook for fingerstyle`.

---

### Task 2: The real AMT proposer

**Files:**
- Modify: `pyproject.toml` (the `ml` extra gains `anticipation @ git+https://github.com/jthickstun/anticipation@af37397922665a0fb8d474d7988b0f3755a38d45` and `tqdm`)
- Modify: `src/music_decoder/compose/ml/amt.py`
- Modify: `src/music_decoder/assets.py`:
  - `AMT_REPO = "stanford-crfm/music-small-800k"`;
  - `AMT_REVISION` (pin the commit sha from the Hugging Face API);
  - `fetch_amt() -> Path`;
  - `fetch-models --composer-models` also calls it.
- Test: `tests/unit/test_compose_ml_prior.py`; `tests/integration/test_compose_ml_e2e.py` (opt-in `MUSIC_DECODER_ML_E2E=1`)

**Interfaces:**
- Produces:
  - `amt.load_proposer() -> MelodyProposer`. It loads `transformers.AutoModelForCausalLM` from the pinned snapshot on the CPU, in eval mode. It raises `MLUnavailableError` if `anticipation`, `transformers` or the weights are missing.
  - `AMTProposer.propose(context, *, end_s, tempo_bpm, seed)`:
    - Encode `context` as AMT **controls** with instrument 25 (steel guitar), through `anticipation.convert`/`ops` (events become controls by adding `CONTROL_OFFSET`, as in the AMT README's accompaniment example).
    - Generate events with `anticipation.sample.generate(model, start_time=0, end_time=end_s, controls=controls, top_p=0.95)`, under `torch.random.fork_rng()` with `torch.manual_seed(seed)`.
    - Decode the generated (non-control) events to `Note`s.
    - Keep only non-drum notes with pitch 55–84, then a monophonic top line (the highest note per onset group within 30 ms).
  - The generated line is a soft prior only. Task 1 already handles empty output.

- [ ] **Step 1: Write the failing tests.**
  - Unit test:
    - `test_amt_without_weights_explains_how_to_fetch` monkeypatches the snapshot lookup to be missing, so `load_proposer()` raises `MLUnavailableError` matching `fetch-models --composer-models`.
    - `test_amt_top_line_keeps_the_highest_note_per_onset` is a pure helper test on decoded events.
  - Opt-in e2e (`MUSIC_DECODER_ML_E2E=1`), with the real model:
    - `compose_with_options(C major, C Am F G C F G C, seed=1, melody_model="amt")` takes 60 s or less of wall time;
    - P1, P2 and P4 are 1.0, A4 holds, and `forbidden_interval_count == 0`;
    - a second run with the same seed gives identical `melody_notes`;
    - `metadata["melody_model"] == "amt"`.
- [ ] **Step 2: Run** the unit tests. Expected: FAIL.
- [ ] **Step 3: Install and fetch.**
  - Run `pip install -e ".[ml]"`. Check that numpy stays below 2.
  - Run `music-decoder fetch-models --composer-models` and report the disk used.
  - Implement the proposer, checking the actual `anticipation` API at the pinned commit (`sample.generate`, `convert`, `ops`, `vocab`).
- [ ] **Step 4: Run** `pytest -q tests/unit tests/integration`, then `MUSIC_DECODER_ML_E2E=1 pytest -q tests/integration/test_compose_ml_e2e.py`. Expected: PASS. Report the AMT runtime per 8-bar piece.
- [ ] **Step 5: Run the gates and commit**: `feat(compose): Anticipatory Music Transformer melody proposals (--melody-model amt)`.

---

### Task 3: v5 docs, listening samples and the spec's Results

**Files:**
- Modify: `docs/usage.md`, `docs/architecture-overview.md`, `docs/architecture-technical.md`
- Modify: `docs/superpowers/specs/2026-11-16-composer-v5-design.md` (append `## 9. Results`)
- Create (local only, git-excluded): `compose-samples/2026-11-16-v5/`

- [ ] **Step 1: Update the three docs** for v5. Cover:
  - expressive dynamics and let-ring, audio-only micro-timing, and `--sound plucked`;
  - per-bar shapes and the identity repair;
  - hooks, arch, peaks and form;
  - best of 12 and `--candidates`, `--rank-audio`, `--melody-model amt`;
  - the `ml` extra and `fetch-models --composer-models`, with their sizes and licences (Audiobox Aesthetics is CC-BY-4.0, AMT Apache-2.0);
  - the new `Arrangement` fields and metadata keys;
  - `composer_version` 5, with a sentence on what v5 changes for a seed.
- [ ] **Step 2: Render the samples** through the real CLI with `--format json`, all fingerstyle:
  - 10 pieces with the defaults: 10 different keys, both modes, at least EADGBE, Drop D and DADGAD, at least 2 with seventh chords and 2 with barres, tempos 66–120, seeds 601–610, and 8 bars or more each;
  - the same 10 with `--sound plucked`;
  - 4 of them with `--melody-model amt`;
  - 4 of them with `--rank-audio`.

  Each folder holds `composition.mid`, `composition.wav`, `compose.json`, `tab.txt` and `command.txt`. Add `README.md` with a table: id, key, progression, tuning, tempo, seed, options, P1, P3, final fallback, F.hook, F.peak, chosen candidate, score.

  Also export MP3s (ffmpeg, `-q:a 2`) into the scratchpad folder `/private/tmp/claude-501/-Users-shubhojeet-Projects-music-decoder/87b150ed-dd08-4235-8266-9155ac0b57cf/scratchpad/v5-samples/`, named `<id>[-plucked|-amt|-rank].mp3`, so the controller can send them to the human partner.
- [ ] **Step 3: Write `## 9. Results`** in the v5 spec, with these parts:
  - **Criteria:** a per-criterion table (met / not met) for §6 items 1–9, with per-matrix numbers from the harness. Regenerate it with `scripts/compose_musicality_report.py` into `out/v5-report.txt`.
  - **Design changes made during the build:** one or two plain sentences each, with why. Take every "Ruling" line from the five feature ledgers in `.superpowers/sdd/2026-11-16-v5-*/progress.md`.
  - **Remaining gaps:**
    - holdout F.M6 at +1.88, against the +2 target;
    - reserve P3 at 0.9888;
    - feature-3 margins that are thin;
    - F.M5 and F.M1 below v3's absolute values off the validation matrix.
  - **Performance:**
    - the `compose` time with the defaults;
    - with `--rank-audio`;
    - with `--melody-model amt`;
    - the test-suite runtime.
  - **Disk:** the space used by the `ml` weights.
  - **Samples:** the sample folder.
- [ ] **Step 4: Run all gates**: `pytest -q tests/unit tests/integration`, `ruff check src tests`, `ruff format --check src tests`, `mypy --strict src`, `MUSIC_DECODER_ML_E2E=1 pytest -q tests/integration/test_compose_ml_e2e.py` and `MUSIC_DECODER_PRODUCTION_E2E=1 pytest -q tests/integration/test_productionize_e2e.py`. Expected: all green.
- [ ] **Step 5: Commit**: `docs(compose): composer v5 docs and spec results`.
