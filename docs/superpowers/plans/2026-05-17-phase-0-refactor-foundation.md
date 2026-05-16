# Phase 0 — Refactor Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the zero-logic-change foundation for the aggressive refactor: a single project-paths helper (killing 3× duplication), the public-API contract-guard test, and the folded `runtime.py` mypy fix — all on a green suite.

**Architecture:** Behavior-preserving extraction only. New `music_decoder/paths.py` centralizes path resolution; three call sites delegate to it with byte-identical resolved paths. A new characterization test snapshots the frozen public contract so every later strangler module is regression-checked against it.

**Tech Stack:** Python 3.11, pytest, mypy --strict, ruff. Branch: `refactor/aggressive-behavior-preserving`.

---

## Spec Reconciliation (read first)

Closer reading of the code changes spec §4 "0b Backend Protocols":

- **`ChordBackend` Protocol already exists** and is well-formed in `src/music_decoder/chords/backends/base.py`. No work needed; do **not** recreate it.
- **`SynthBackend` is a `StrEnum`** selector (`sine`/`fluidsynth`) dispatched by a function in `synth/fluidsynth_wrapper.py` — not a duck-typed plug-point. Adding a Protocol here is speculative abstraction, explicitly **out of scope** per spec §7 (YAGNI). Skip.
- **`tabs/` exposes a single `assign_tab()`** function; A*/heuristic are internal helpers, not pluggable backends. No Protocol — speculative. Skip.

Net Phase 0 scope: **0a (paths + runtime.py mypy fix)** and **0d (contract guard test)** only. 0b/0c collapse to "already satisfied / intentionally not done." Task 6 records this in the spec so it stays honest.

## File Structure

- **Create** `src/music_decoder/paths.py` — sole owner of repo-root / bundled-config path resolution. ~15 LOC, one responsibility.
- **Create** `tests/unit/test_paths.py` — unit tests for the helper.
- **Create** `tests/unit/test_public_contract.py` — frozen-contract characterization test (the refactor's regression net for API drift).
- **Modify** `src/music_decoder/config/runtime.py` — line 12 delegates to `paths`; fold the line-80 mypy fix.
- **Modify** `src/music_decoder/config/hyperparameters.py` — line 132 delegates to `paths`.
- **Modify** `src/music_decoder/synth/__init__.py` — the `parents[3]` fixtures path delegates to `paths`.
- **Modify** `docs/superpowers/specs/2026-05-17-aggressive-refactor-design.md` — append reconciliation note.

---

### Task 1: `paths.py` helper

**Files:**
- Create: `src/music_decoder/paths.py`
- Test: `tests/unit/test_paths.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_paths.py
from pathlib import Path

from music_decoder import paths


def test_project_root_is_repo_root():
    root = paths.project_root()
    assert (root / "pyproject.toml").is_file()
    assert (root / "src" / "music_decoder").is_dir()


def test_bundled_config_points_into_config_dir():
    p = paths.bundled_config("runtime.yaml")
    assert p == paths.project_root() / "config" / "runtime.yaml"
    assert p.is_file()


def test_project_root_matches_legacy_parents3():
    # Byte-identical to the idiom being replaced in config/runtime.py
    legacy = (
        Path(__import__("music_decoder.config.runtime", fromlist=["x"]).__file__)
        .resolve()
        .parents[3]
    )
    assert paths.project_root() == legacy
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_paths.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'music_decoder.paths'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/music_decoder/paths.py
"""Single source of truth for filesystem paths the package needs.

Replaces the ``Path(__file__).resolve().parents[3] / "config" / ...``
idiom that was copy-pasted across config/runtime, config/hyperparameters
and synth, and which previously resolved wrong under a wheel install.
"""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return the repository root (the directory containing ``pyproject.toml``).

    ``paths.py`` lives at ``src/music_decoder/paths.py``; ``parents[2]``
    is therefore the repo root.
    """
    return Path(__file__).resolve().parents[2]


def bundled_config(name: str) -> Path:
    """Return the absolute path to ``config/<name>`` under the repo root."""
    return project_root() / "config" / name
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_paths.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Typecheck + lint the new module**

Run: `.venv/bin/mypy --strict src/music_decoder/paths.py && .venv/bin/ruff check src/music_decoder/paths.py`
Expected: `Success: no issues found in 1 source file` and `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add src/music_decoder/paths.py tests/unit/test_paths.py
git commit -m "refactor(paths): add project_root/bundled_config helper (Phase 0a)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Delegate `config/runtime.py` + fold the line-80 mypy fix

**Files:**
- Modify: `src/music_decoder/config/runtime.py:12` and `:80`

- [ ] **Step 1: Replace the path constant (line 12)**

Old line 12:
```python
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "runtime.yaml"
```
New (add `from music_decoder.paths import bundled_config` to imports, then):
```python
_DEFAULT_CONFIG_PATH = bundled_config("runtime.yaml")
```

- [ ] **Step 2: Fix the line-80 mypy error**

Old line 80:
```python
        sample_rate_hz=int(raw.get("sample_rate_hz", 22050)),  # type: ignore[arg-type]
```
New (cast through `str` so the `int(...)` overload resolves; drop the now-unused ignore):
```python
        sample_rate_hz=int(str(raw.get("sample_rate_hz", 22050))),
```

- [ ] **Step 3: Verify the existing config tests still pass (behavior unchanged)**

Run: `.venv/bin/pytest tests/unit -q -k "runtime or config"`
Expected: PASS, same count as before this task (no test added/removed).

- [ ] **Step 4: Verify the two mypy errors in this file are gone**

Run: `.venv/bin/mypy --strict src/music_decoder/config/runtime.py`
Expected: `Success: no issues found in 1 source file` (previously 2 errors at line 80).

- [ ] **Step 5: Full unit suite green (regression net)**

Run: `.venv/bin/pytest -q tests/unit`
Expected: `236 passed` (unchanged — paths task added 3 tests; expect `239 passed` total. Record the exact number; it must only ever grow by tests this plan adds).

- [ ] **Step 6: Commit**

```bash
git add src/music_decoder/config/runtime.py
git commit -m "refactor(config): runtime.py uses paths helper; fix line-80 mypy (Phase 0a)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Delegate `config/hyperparameters.py`

**Files:**
- Modify: `src/music_decoder/config/hyperparameters.py:132`

- [ ] **Step 1: Replace the path constant (line 132)**

Old line 132:
```python
_PROJECT_ROOT_HP = Path(__file__).resolve().parents[3] / "config" / "hyperparameters.yaml"
```
New (add `from music_decoder.paths import bundled_config` to imports, then):
```python
_PROJECT_ROOT_HP = bundled_config("hyperparameters.yaml")
```
Leave line 144 (`path = _PROJECT_ROOT_HP`) unchanged — the name is reused.

- [ ] **Step 2: Verify hyperparameter loading is unchanged**

Run: `.venv/bin/pytest tests/unit -q -k "hyperparam"`
Expected: PASS, same count as before.

- [ ] **Step 3: Full unit suite green**

Run: `.venv/bin/pytest -q tests/unit`
Expected: same total as end of Task 2 (e.g. `239 passed`).

- [ ] **Step 4: Commit**

```bash
git add src/music_decoder/config/hyperparameters.py
git commit -m "refactor(config): hyperparameters.py uses paths helper (Phase 0a)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Delegate the `synth/__init__.py` fixtures path

**Files:**
- Modify: `src/music_decoder/synth/__init__.py` (the `Path(__file__).resolve().parents[3] / "tests" / ...` block, ~lines 45-52)

- [ ] **Step 1: Replace the parents[3] expression**

Old block (inside `_resolve_soundfont_path`):
```python
        candidates.append(
            Path(__file__).resolve().parents[3]
            / "tests"
            / "fixtures"
            / "synthetic"
            / "soundfont"
            / sf_path
        )
```
New (add `from music_decoder.paths import project_root` to imports, then):
```python
        candidates.append(
            project_root()
            / "tests"
            / "fixtures"
            / "synthetic"
            / "soundfont"
            / sf_path
        )
```
(Use `project_root()`, not `bundled_config` — this is a tests-fixtures path, not a `config/` file.)

- [ ] **Step 2: Verify synth path resolution unchanged**

Run: `.venv/bin/pytest tests/unit -q -k "synth or soundfont"`
Expected: PASS, same count as before.

- [ ] **Step 3: doctor still resolves the soundfont (behavior smoke)**

Run: `.venv/bin/music-decoder doctor`
Expected: `soundfont present.............. OK`, exit 0.

- [ ] **Step 4: Full unit suite + confirm no remaining parents[3]**

Run: `.venv/bin/pytest -q tests/unit && ! grep -rn "parents\[3\]" src/music_decoder --include="*.py"`
Expected: suite green (same total); grep prints nothing and the `!` makes the compound succeed (zero `parents[3]` left in `src/`).

- [ ] **Step 5: Commit**

```bash
git add src/music_decoder/synth/__init__.py
git commit -m "refactor(synth): use paths.project_root for fixtures path (Phase 0a)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Public-contract characterization test

This is a *characterization* test: it must pass against the **current** code (it snapshots today's frozen contract). It then guards every later strangler module.

**Files:**
- Create: `tests/unit/test_public_contract.py`

- [ ] **Step 1: Write the contract test**

```python
# tests/unit/test_public_contract.py
"""Frozen public-contract guard for the aggressive refactor.

Spec: docs/superpowers/specs/2026-05-17-aggressive-refactor-design.md
Any change here means the externally observable contract moved — which
the refactor forbids. If a rewrite trips this, the rewrite is wrong,
not this test.
"""

import inspect

import music_decoder
from music_decoder import api, errors


def test_package_exports_are_frozen():
    assert sorted(music_decoder.__all__) == sorted([
        "DADGAD", "DROP_C", "DROP_D", "D_STANDARD", "EB_HALF_STEP_DOWN",
        "STANDARD_EADGBE", "AnalysisResult", "ChordSegment", "ChordSymbol",
        "Composition", "KeyEstimate", "Note", "Scale", "TabPosition",
        "TabbedNote", "Tuning", "VoicedChord", "analyze", "compose",
    ])


def test_analyze_signature_is_frozen():
    sig = inspect.signature(api.analyze)
    params = list(sig.parameters)
    assert params == [
        "source", "declared_kind", "tuning", "use_separation", "progress",
    ]
    p = sig.parameters
    assert p["source"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert p["declared_kind"].kind is inspect.Parameter.KEYWORD_ONLY
    assert p["declared_kind"].default == "full_mix"
    assert p["use_separation"].default is True
    assert p["progress"].default is None
    assert p["tuning"].default == music_decoder.STANDARD_EADGBE


def test_compose_signature_is_frozen():
    sig = inspect.signature(api.compose)
    params = list(sig.parameters)
    assert params == [
        "scale", "progression", "bars_per_chord", "tempo_bpm",
        "style", "tuning", "seed", "out_dir",
    ]
    p = sig.parameters
    assert p["scale"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert p["bars_per_chord"].kind is inspect.Parameter.KEYWORD_ONLY
    assert p["bars_per_chord"].default == 1
    assert p["tempo_bpm"].default == 100.0
    assert p["style"].default == "fingerstyle"
    assert p["seed"].default is None
    assert p["out_dir"].default is inspect.Parameter.empty
    assert p["tuning"].default == music_decoder.STANDARD_EADGBE


def test_error_hierarchy_is_frozen():
    expected = {
        "MusicDecoderError": Exception,
        "IngestError": errors.MusicDecoderError,
        "YouTubeError": errors.IngestError,
        "CorruptAudioError": errors.IngestError,
        "SilentAudioError": errors.IngestError,
        "ClipTooShortError": errors.IngestError,
        "SeparationError": errors.MusicDecoderError,
        "TranscriptionError": errors.MusicDecoderError,
        "ChordRecognitionError": errors.MusicDecoderError,
        "TabAssignmentError": errors.MusicDecoderError,
        "KeyDetectionError": errors.MusicDecoderError,
        "BeatTrackingError": errors.MusicDecoderError,
        "CompositionError": errors.MusicDecoderError,
        "InvalidScaleError": errors.CompositionError,
        "InvalidProgressionError": errors.CompositionError,
        "SynthesisError": errors.MusicDecoderError,
    }
    for name, base in expected.items():
        cls = getattr(errors, name)
        assert issubclass(cls, base), f"{name} must subclass {base.__name__}"
        assert issubclass(cls, errors.MusicDecoderError) or cls is errors.MusicDecoderError


def test_cli_subcommands_and_flags_are_frozen():
    from click.testing import CliRunner

    from music_decoder.cli.main import main

    r = CliRunner().invoke(main, ["--help"])
    assert r.exit_code == 0
    for cmd in ("analyze", "compose", "ui", "doctor"):
        assert cmd in r.output
    ra = CliRunner().invoke(main, ["analyze", "--help"])
    assert ra.exit_code == 0
    for flag in ("--tuning", "--solo-guitar", "--full-mix",
                 "--no-separation", "--format"):
        assert flag in ra.output
    rc = CliRunner().invoke(main, ["compose", "--help"])
    assert rc.exit_code == 0
    for flag in ("--scale", "--progression", "--style", "--tempo",
                 "--bars", "--seed", "--tuning", "--out", "--format"):
        assert flag in rc.output
```

- [ ] **Step 2: Run it against current code — must PASS now**

Run: `.venv/bin/pytest tests/unit/test_public_contract.py -q`
Expected: PASS (5 passed). If any assertion fails, the snapshot is wrong — correct the *test* to match today's real contract (do not change product code in this task).

- [ ] **Step 3: Typecheck + lint the test**

Run: `.venv/bin/ruff check tests/unit/test_public_contract.py`
Expected: `All checks passed!`

- [ ] **Step 4: Full unit suite green**

Run: `.venv/bin/pytest -q tests/unit`
Expected: previous total + 5 (e.g. `244 passed`). Record this number — it is the new floor for every later module.

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_public_contract.py
git commit -m "test(contract): freeze public API/CLI/errors surface for refactor (Phase 0d)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Record spec reconciliation

**Files:**
- Modify: `docs/superpowers/specs/2026-05-17-aggressive-refactor-design.md`

- [ ] **Step 1: Append a reconciliation note to §4**

Add at the end of section "## 4. Phase 0 …":

```markdown
### Reconciliation (post-code-read, 2026-05-17)

Closer reading reduced 0b/0c: `ChordBackend` Protocol already exists
(`chords/backends/base.py`); `SynthBackend` is a `StrEnum` selector and
`tabs` exposes a single `assign_tab()` — neither is a duck-typed
plug-point, so adding Protocols there is speculative abstraction barred
by §7. Implemented Phase 0 = 0a (paths + runtime.py:80 mypy) + 0d
(contract guard test) only.
```

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-05-17-aggressive-refactor-design.md
git commit -m "docs(spec): record Phase 0 reconciliation (Protocols already exist / YAGNI)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Phase 0 exit verification

- [ ] **Step 1: Full suite + integration + mypy delta**

Run:
```bash
.venv/bin/pytest -q tests/unit && \
.venv/bin/pytest -q tests/integration && \
.venv/bin/mypy --strict src 2>&1 | tail -1
```
Expected: unit green at the recorded floor; integration green; mypy errors **reduced from 4 to 2** (only `chords/voicings.py:132` remains — that one is deferred to the `chords/` strangler pass per spec §4).

- [ ] **Step 2: Confirm clean tree + branch**

Run: `git status --short && git branch --show-current`
Expected: empty status; branch `refactor/aggressive-behavior-preserving`.

Phase 0 complete. The next plan (`dsp/` strangler module) is written against this foundation.

---

## Self-Review

**1. Spec coverage:** Phase 0a (paths + runtime.py:80 mypy) → Tasks 1-4. Phase 0d (contract guard) → Task 5. Spec §4 0b/0c reconciled honestly → Task 6 + the Reconciliation section. Phase 0 exit criteria (spec §6 regression net) → Task 7. The remaining mypy error (`voicings.py:132`) is explicitly deferred to the `chords/` module per spec §4 — not dropped.

**2. Placeholder scan:** No TBD/TODO/"handle edge cases". Every code step shows complete code; every run step has an exact command + expected output. Test counts are written as "previous total + N / record the number" because the absolute floor depends on the live suite — this is a concrete instruction, not a placeholder.

**3. Type/name consistency:** `project_root()` and `bundled_config(name)` are defined in Task 1 and used by exactly that name/signature in Tasks 2-4. Contract test imports (`music_decoder`, `api`, `errors`) match real module paths verified against the codebase. `__all__`, both signatures, and the 16-class error hierarchy were captured from the live code, not invented.

**4. Scope:** Phase 0 only; later strangler modules are separate plans (each independently shippable per spec §5/§7). Correct decomposition — not a placeholder gap.
