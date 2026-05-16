# Aggressive Behavior-Preserving Refactor — Design

**Date:** 2026-05-17
**Status:** Approved (design); pending spec review → implementation plan
**Author:** brainstorming session (Claude + Shubhojeet)

## 1. Goal & Constraints

Refactor the `music_decoder` codebase to senior-Python standards — DRY,
single-responsibility modules, idiomatic OOP where it pays, docstrings on
every module/public class/public function, comments on non-obvious *why* —
**without changing any externally observable behavior**.

Decisions locked during brainstorming:

| Dimension | Decision |
|---|---|
| Scope | Aggressive internal structural rewrite |
| Behavior contract | **Preserve all observable behavior** — public API, CLI surface, JSON schema, chord/key accuracy unchanged |
| Paradigm | Idiomatic Python: OOP (Protocol/ABC, stateful classes) only where it genuinely pays; keep pure functions + frozen dataclasses where already correct |
| Hard acceptance gate | **All 236 unit tests + integration + the 6-song e2e accuracy must stay green** (only blocking criterion) |
| Execution approach | **C** — interface-first Phase 0 (DRY + Protocols, zero logic change), then strangler module-by-module |
| Docstrings/comments/DRY/modularity | Goals applied during every module pass; *not* pass/fail gates |
| mypy-clean / file-splitting | Desirable, done opportunistically per module; *not* blocking gates |

### Frozen contract boundary

Everything behind this line may move; this line may not:

- `music_decoder/__init__.py` exports (`__all__`).
- `analyze()` and `compose()` signatures (params, defaults, kw-only-ness).
- `errors.py` exception hierarchy (names + base classes) and raise-sites.
- CLI commands, flags, and human-readable text (`analyze`, `compose`,
  `ui`, `doctor`).
- `--format json` schema for both `analyze` and `compose`.
- Chord/key accuracy within tolerance of the recorded e2e baseline.

## 2. Baseline (must be reproduced)

Codebase: ~5,521 LOC / 68 files; clean, committed, 236/236 unit tests
green; e2e verified this session.

e2e accuracy baseline (6 songs, full-mix + Demucs, `--format json`):

| Song | Key | Chord-root accuracy |
|---|---|---|
| Three Little Birds | A major ✓ | 100% |
| Stand By Me | A major ✓ | 97% |
| Knockin' on Heaven's Door | G major ✓ | 100% |
| Brown Eyed Girl | G major ✓ | 97% |
| Bad Moon Rising | D major ✓ | 98% |
| Let It Be | C major ✓ | 100% |

Re-run acceptance: key must stay **6/6 correct**; each chord-root figure
within **±3% absolute** of the above (tolerance absorbs basic-pitch stem
nondeterminism without masking a real regression).

Known smells targeted: 4 `mypy --strict` errors
(`config/runtime.py:80`, `chords/voicings.py:132`); 3× duplicated
`Path(__file__).resolve().parents[3] / "config"/…`; 4 `cast()` and
6 `# type: ignore` escape hatches; oversized grab-bag files
(`types.py` 315 L, `evaluation/metrics.py` 288 L, `cli/main.py` 279 L).

## 3. Target Architecture (behavior-preserving)

- **New `music_decoder/paths.py`** — `project_root()`,
  `bundled_config(name)`. Single home for the path resolution currently
  duplicated 3× (the idiom that caused this session's packaging bug).
- **`types.py` → `types/` package** (`music.py`, `audio.py`,
  `analysis.py`, `compose.py`). `types.py`'s public names are
  **re-exported** so every `from music_decoder.types import X` is
  unchanged.
- **`cli/main.py` → `cli/` package**: `commands.py`, `serialize.py`,
  `doctor.py`, `_io.py` (the `_quiet_stdout` fd redirect). Console entry
  point unchanged.
- **`chords/`, `tabs/`, `key/`, `compose/`** keep package boundaries;
  internal cohesion passes + formal backend interfaces.
- **No package renames, no moved public import paths.**

OOP placement: `Protocol`/ABC for pluggable strategies; cohesive config
object; stateful adapters as classes. DSP/transform code stays pure
functions + frozen dataclasses.

## 4. Phase 0 — Interface & DRY Extraction (one green commit, zero logic change)

- **0a. `paths.py`** + replace the 3 call sites
  (`config/runtime.py`, `config/hyperparameters.py`, `synth/__init__.py`).
  Identical resolved paths (existing config tests prove it).
  **Fold the `config/runtime.py:80` mypy fix in here** (trivial
  `int(object)` overload + unused-ignore; that file is already edited).
- **0b. Backend Protocols** (`typing.Protocol`, structural — no forced
  inheritance):
  - `ChordRecognitionBackend` — replaces informal
    `chords/backends/base.py`.
  - `SynthBackend` — fluidsynth vs sine auto-select.
  - `TabAssignmentStrategy` — A* vs heuristic.
  Descriptive of current duck-typed behavior → no runtime change.
- **0c. Config cohesion** — `paths.py`-based loader indirection only; no
  field/schema changes.
- **0d. Contract guard test** (`tests/unit/test_public_contract.py`) —
  see §6. Written here, green, must stay green throughout.

Explicitly NOT in Phase 0: algorithm changes, file splits, docstring
sweep.

## 5. Strangler Sequence (full unit suite green + commit after each module)

Leaf-first so any regression bisects to one module:

1. `dsp/` — pure functions, leaf; sets docstring/comment conventions.
2. `key/` — depends only on `dsp`.
3. `tabs/` — apply `TabAssignmentStrategy`; collapse `n_strings`/
   `num_strings` to one internal name + back-compat shim at the public
   edge.
4. `chords/` — apply `ChordRecognitionBackend`; fix `voicings.py:132`
   mypy; clean the `ChordSegment.quality` `cast()`/no-chord `""` smell
   behind existing public types.
5. `transcription/` + `separation/` + `synth/` — adapter classes
   implementing the Protocols; `SynthBackend` formalized.
6. `compose/` — idiomatic class for the stateful arrangement builder.
7. `ingest/` — keep the `noplaylist`/quiet fixes; cohesion pass.
8. `evaluation/` — split `metrics.py` (288 L); keep both legacy
   index-aligned and time-aligned modes (both are tested contract).
9. `api.py` + `cli/` + `ui/` — orchestration/edge last; `cli/` package
   split per §3.

Per-module conventions: Google-style docstrings (module/public
class/public function); comments only on non-obvious *why*; lift
duplication to shared helpers; Protocol-conforming strategy classes
where they pay; split any file passing ~250 L along its natural seam
(re-exported, imports stable).

**Hard rule:** no test deleted or weakened to make a rewrite pass. A
test asserting moved internal structure is rewritten to assert the same
behavior through the public seam — never removed.

## 6. Behavior-Contract Enforcement & Testing

- After every module: `pytest -q tests/unit` (236 pass, unchanged).
- At phase boundaries: `pytest tests/integration` + `pytest -m
  regression` (thresholds in `config/eval_thresholds.yaml`) + the
  6-song e2e harness vs the §2 baseline.
- **New `tests/unit/test_public_contract.py`** (written in Phase 0,
  always green): asserts `__all__`; `inspect.signature` of `analyze`/
  `compose`; `errors.py` hierarchy; `--format json` key structure on a
  synthetic input; `music-decoder --help` subcommands + per-command
  flags.
- Per-module commit discipline: one green commit per module; no
  squashing until end-to-end verified; any later failure bisects to one
  module.
- Error-handling contract: each rewritten module raises the same
  exception type at the same boundary.

## 7. Risks, Rollback, Out-of-Scope

| Risk | Mitigation |
|---|---|
| Silent accuracy regression | e2e at every phase boundary; key 6/6 + chord-root ±3%; chord/key paths rewritten last & smallest-diff within their module |
| Hidden coupling | Leaf-first order; full suite per module; one-module bisect |
| Scope-creep into behavior change | Contract guard test + frozen `__all__`/signatures |
| `cast()`/`ignore` removal exposes latent type bug | Surfaced as a finding (systematic-debugging), never silenced by re-adding the ignore |
| Half-done stall | Every commit independently green & shippable |

**Rollback:** dedicated feature branch (not `master`); each module is an
isolated green commit → `git revert <module-commit>` restores that
subsystem only. Phase 0 revertible alone.

**Out of scope (YAGNI):** no new features; no algorithm/accuracy
changes; no public API/CLI/JSON-schema changes; no dependency or
Python-version changes; no forced OOP where functions are idiomatic; no
speculative abstraction. mypy-clean and docstring-everywhere are pass
goals, not blocking gates.
