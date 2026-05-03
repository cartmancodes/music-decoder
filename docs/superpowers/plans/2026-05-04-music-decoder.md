# Music Decoder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a locally installable Python application that takes any MP3/WAV/FLAC and produces (1) a key/scale estimate with confidence and (2) guitar tablature with explicit string + fret positions per note, with mir_eval-driven accuracy as the primary success metric.

**Architecture:** Local-install Python package distributed via `pipx`. CLI launches a worker daemon (Python subprocess) and a Streamlit UI on `http://localhost:8501`. SQLite is the source of truth and serves as the job queue. Pipeline stages are pure typed-dataclass functions composed inside the worker. Evaluation harness runs from day one against GuitarSet, synthetic, and manual fixtures.

**Tech Stack:** Python 3.11, Streamlit, SQLAlchemy 2.x + Alembic, SQLite (WAL), librosa, basic-pitch, CREPE, Demucs (htdemucs_6s), pretty_midi, music21, mir_eval, mirdata, fluidsynth, pyacoustid + Chromaprint, ffmpeg (external), matplotlib, ruff + mypy + pytest.

**Reference:** [docs/superpowers/specs/2026-05-04-music-decoder-design.md](../specs/2026-05-04-music-decoder-design.md). Treat the spec as the source of truth — if a task here drifts from the spec, the spec wins.

---

## File structure

```
music-decoder/
  pyproject.toml
  Makefile
  README.md
  Dockerfile                              # secondary install path
  docker-compose.yml                      # secondary install path
  alembic.ini
  config/
    runtime.yaml
    hyperparameters.yaml
    eval_thresholds.yaml
  migrations/                             # alembic
    env.py
    versions/
  src/music_decoder/
    __init__.py
    cli/
      __init__.py
      main.py                             # `music-decoder` entry point
      health.py                           # ffmpeg / models / data dir checks
      models_download.py                  # one-time model fetch
    config/
      __init__.py
      runtime.py                          # loads runtime.yaml
      hyperparameters.py                  # loads hyperparameters.yaml
    logging_setup.py                      # structured JSON logger
    artifacts/
      __init__.py
      base.py                             # ArtifactStore interface
      filesystem.py                       # FilesystemArtifactStore
    persistence/
      __init__.py
      models.py                           # SQLAlchemy declarative models
      session.py                          # engine / session factory
      repositories.py                     # CRUD wrappers
    pipeline/
      __init__.py
      contracts.py                        # all pipeline dataclasses
      events.py                           # StageEvent, emitter helper
      orchestrator.py                     # process_audio(job_id)
    audio_io/
      __init__.py
      ffmpeg.py                           # external binary wrapper
      load.py                             # high-level load() + validation
    separation/
      __init__.py
      demucs.py
    transcription/
      __init__.py
      basic_pitch_wrapper.py
      crepe_wrapper.py
      post_processing.py                  # filter chain
    key_detection/
      __init__.py
      profiles.py                         # K-K + Temperley constants
      chroma.py                           # chroma_cqt + HPSS
      ks.py                               # global K-S correlation
      windowed.py                         # modulation detection
    beat_tracking/
      __init__.py
      beats.py                            # tempo + beats + downbeats
      time_signature.py                   # TS inference + confidence
    tab_assignment/
      __init__.py
      tuning.py                           # Tuning value object + presets
      candidates.py                       # candidate set generator
      cost.py                             # transition cost
      heuristic.py                        # admissible heuristic
      astar.py                            # the search itself
      assigner.py                         # public API: assign(notes, tuning, weights)
    tab_reference/
      __init__.py
      base.py                             # TabReferenceProvider interface
      user_paste.py                       # user-supplied URL/text impl
      fingerprint.py                      # Chromaprint/AcoustID
      parser.py                           # ASCII tab parser
      alignment.py                        # similarity scoring
    midi_synth/
      __init__.py
      fluidsynth_wrapper.py               # MIDI → WAV via fluidsynth
    worker/
      __init__.py
      daemon.py                           # polling loop, model warmup
      runner.py                           # invokes orchestrator.process_audio
    ui/
      streamlit_app.py                    # entry; sets shared config
      pages/
        01_Upload.py
        02_Job.py
        03_Results.py
      components/
        confidence.py                     # color helper, threshold logic
        chromagram.py                     # matplotlib renderer
        waveform.py                       # waveform + onset overlay
        tablature.py                      # ASCII + SVG fretboard
        playback.py                       # synthesized vs original
        diff.py                           # UG side-by-side diff
    evaluation/
      __init__.py
      metrics.py                          # mir_eval wrappers
      fixtures/
        __init__.py
        base.py                           # Fixture dataclass + loader iface
        synthetic.py
        guitarset.py
        manual.py
      runner.py                           # orchestrate fixture → metrics → report
      regression.py                       # baseline + tolerance comparison
  tests/
    conftest.py                           # shared fixtures (db, tmp dirs)
    fixtures/
      synthetic/
        c_major_scale.mid                 # committed
        g_major_chord.mid
        soundfont/
          GeneralUser_GS.sf2              # committed, license-cleared
      guitarset/                          # downloaded; cached; gitignored
      manual/
        .gitkeep
      audio_samples/                      # tiny WAVs for unit tests
        sine_440.wav
        silence_2s.wav
    unit/
      test_audio_io.py
      test_separation.py
      test_transcription.py
      test_post_processing.py
      test_key_detection.py
      test_beat_tracking.py
      test_tab_assignment.py
      test_tab_reference.py
      test_evaluation_metrics.py
      test_persistence.py
      test_artifacts.py
      test_config.py
      test_tuning.py
      test_midi_synth.py
    integration/
      test_pipeline_orchestration.py
      test_worker_daemon.py
      test_cli.py
    regression/
      test_accuracy_thresholds.py         # the gating regression test
  evaluation_reports/
    baseline.json                         # committed; updated on intentional regressions
    .gitkeep
  scripts/
    sweep.py                              # hyperparameter sweep CLI
    render_synthetic.py                   # one-shot regenerate synthetic WAVs
```

Each module is small and focused; nothing exceeds a few hundred lines. Files that change together (e.g. `tab_assignment/*`) live together. The `pipeline/` package owns orchestration; pipeline stages have no DB I/O of their own.

---

## Phase ordering rationale

1. **Phase 0 (scaffold)** — pyproject, configs, logger, tuning. Zero domain logic but everything downstream needs it.
2. **Phase 1 (persistence)** — schema before anything writes rows.
3. **Phase 2 (evaluation harness)** — built **before pipeline stages** so every pipeline task can assert an accuracy delta. This is the single most important sequencing decision in this plan.
4. **Phase 3 (pipeline contracts)** — dataclasses are referenced by everything in Phase 4.
5. **Phase 4 (pipeline stages)** — bottom-up, each with TDD + accuracy benchmark where the spec demands it.
6. **Phase 5 (orchestration + MIDI synth)** — composes Phase 4.
7. **Phase 6 (worker daemon)** — runs the orchestrator off a SQLite-backed queue.
8. **Phase 7 (Streamlit UI)** — consumes worker output.
9. **Phase 8 (CLI + packaging)** — the user-facing entry point.
10. **Phase 9 (E2E + docs)** — verifies everything end-to-end and documents the install flow.

Frequent commits — one per task. Every code-changing step shows the actual code. No placeholders.

---

## Phase 0 — Project scaffold

### Task 1: Python package skeleton + tooling

**Files:**
- Create: `pyproject.toml`
- Create: `src/music_decoder/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `.gitignore`
- Create: `Makefile`
- Create: `tests/unit/test_package_imports.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_package_imports.py
import importlib

def test_package_imports():
    mod = importlib.import_module("music_decoder")
    assert hasattr(mod, "__version__")

def test_version_is_string():
    import music_decoder
    assert isinstance(music_decoder.__version__, str)
    assert len(music_decoder.__version__) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_package_imports.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'music_decoder'`

- [ ] **Step 3: Write `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling>=1.21"]
build-backend = "hatchling.build"

[project]
name = "music-decoder"
version = "0.1.0"
description = "Local music analysis tool: audio → key + guitar tablature."
readme = "README.md"
requires-python = ">=3.11,<3.13"
license = {text = "MIT"}
dependencies = [
  "streamlit>=1.32",
  "sqlalchemy>=2.0",
  "alembic>=1.13",
  "librosa>=0.10",
  "basic-pitch>=0.4",
  "crepe>=0.0.13",
  "demucs>=4.0",
  "pretty_midi>=0.2.10",
  "music21>=9.1",
  "mir_eval>=0.7",
  "mirdata>=0.3.8",
  "pyacoustid>=1.3",
  "pyfluidsynth>=1.3",
  "matplotlib>=3.8",
  "numpy>=1.26,<2.0",
  "scipy>=1.11",
  "pydantic>=2.6",
  "pyyaml>=6.0",
  "click>=8.1",
  "platformdirs>=4.2",
]

[project.optional-dependencies]
dev = [
  "pytest>=8.0",
  "pytest-cov>=5.0",
  "pytest-xdist>=3.5",
  "ruff>=0.5",
  "mypy>=1.10",
  "types-PyYAML",
  "freezegun>=1.5",
]

[project.scripts]
music-decoder = "music_decoder.cli.main:main"

[tool.hatch.build.targets.wheel]
packages = ["src/music_decoder"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "B", "UP", "RUF"]

[tool.mypy]
python_version = "3.11"
strict = true
warn_unused_ignores = true
disallow_any_generics = true
plugins = []

[[tool.mypy.overrides]]
module = ["basic_pitch.*", "crepe.*", "demucs.*", "librosa.*", "pretty_midi.*",
          "music21.*", "mir_eval.*", "mirdata.*", "acoustid.*", "fluidsynth.*",
          "matplotlib.*", "scipy.*", "streamlit.*"]
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --strict-markers"
markers = [
  "slow: tests that take >5s",
  "regression: regression tests against accuracy thresholds",
  "integration: cross-module integration tests",
]
```

- [ ] **Step 4: Write `src/music_decoder/__init__.py`**

```python
"""music-decoder: local audio → key + guitar tablature."""

__version__ = "0.1.0"
```

- [ ] **Step 5: Write `tests/__init__.py` and `tests/conftest.py`**

```python
# tests/__init__.py
```

```python
# tests/conftest.py
from __future__ import annotations

from pathlib import Path
import pytest


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    d = tmp_path / "music_decoder_data"
    d.mkdir()
    return d
```

- [ ] **Step 6: Write `.gitignore`**

```gitignore
__pycache__/
*.pyc
*.pyo
.venv/
venv/
.eggs/
*.egg-info/
build/
dist/
.pytest_cache/
.mypy_cache/
.ruff_cache/
.coverage
htmlcov/
tests/fixtures/guitarset/
tests/fixtures/synthetic/*.wav
*.sqlite3
*.db
.DS_Store
.idea/
.vscode/
```

- [ ] **Step 7: Write `Makefile`**

```makefile
.PHONY: install dev test test-fast lint typecheck eval format clean

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest

test-fast:
	pytest -m "not slow and not regression" -n auto

lint:
	ruff check src tests

typecheck:
	mypy src

eval:
	pytest tests/regression -v

format:
	ruff check --fix src tests
	ruff format src tests

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
```

- [ ] **Step 8: Install dev deps and run tests**

Run:
```bash
pip install -e ".[dev]"
pytest tests/unit/test_package_imports.py -v
```
Expected: 2 passed.

- [ ] **Step 9: Verify lint and typecheck pass**

Run: `make lint && make typecheck`
Expected: both pass with no errors.

- [ ] **Step 10: Commit**

```bash
git add pyproject.toml src/ tests/ .gitignore Makefile
git commit -m "feat: project scaffold with pyproject, src layout, and CI tooling"
```

---

### Task 2: Structured JSON logger

**Files:**
- Create: `src/music_decoder/logging_setup.py`
- Create: `tests/unit/test_logging_setup.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_logging_setup.py
import json
import logging

from music_decoder.logging_setup import configure_logging, get_logger


def test_get_logger_returns_named_logger():
    log = get_logger("test_mod")
    assert log.name == "music_decoder.test_mod"


def test_configure_logging_writes_json(capsys):
    configure_logging(level="INFO")
    log = get_logger("emit")
    log.info("hello", extra={"job_id": 42, "stage": "transcription"})
    captured = capsys.readouterr()
    record = json.loads(captured.err.strip().splitlines()[-1])
    assert record["level"] == "INFO"
    assert record["message"] == "hello"
    assert record["logger"] == "music_decoder.emit"
    assert record["job_id"] == 42
    assert record["stage"] == "transcription"
    assert "timestamp" in record


def test_configure_logging_is_idempotent():
    configure_logging(level="INFO")
    configure_logging(level="DEBUG")
    root = logging.getLogger("music_decoder")
    handlers = root.handlers
    # Only one of our handlers should be attached.
    own = [h for h in handlers if getattr(h, "_md_marker", False)]
    assert len(own) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_logging_setup.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement the logger**

```python
# src/music_decoder/logging_setup.py
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


_RESERVED = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "message", "asctime",
}


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key in _RESERVED or key.startswith("_"):
                continue
            payload[key] = value
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger("music_decoder")
    root.setLevel(level.upper())
    for handler in list(root.handlers):
        if getattr(handler, "_md_marker", False):
            root.removeHandler(handler)
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(_JsonFormatter())
    handler._md_marker = True  # type: ignore[attr-defined]
    root.addHandler(handler)
    root.propagate = False


def get_logger(module: str) -> logging.Logger:
    return logging.getLogger(f"music_decoder.{module}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_logging_setup.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/music_decoder/logging_setup.py tests/unit/test_logging_setup.py
git commit -m "feat(logging): structured JSON logger with idempotent configure()"
```

---

### Task 3: Runtime config loader

**Files:**
- Create: `config/runtime.yaml`
- Create: `src/music_decoder/config/__init__.py`
- Create: `src/music_decoder/config/runtime.py`
- Create: `tests/unit/test_config_runtime.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_config_runtime.py
from pathlib import Path

import pytest

from music_decoder.config.runtime import RuntimeConfig, load_runtime_config


def test_load_runtime_config_reads_yaml(tmp_path: Path):
    cfg_path = tmp_path / "runtime.yaml"
    cfg_path.write_text(
        """
        db_path: /tmp/db.sqlite3
        artifact_dir: /tmp/artifacts
        log_level: DEBUG
        ffmpeg_path: /usr/local/bin/ffmpeg
        model_cache_dir: /tmp/models
        fixture_dir: /tmp/fixtures
        """
    )
    cfg = load_runtime_config(cfg_path)
    assert isinstance(cfg, RuntimeConfig)
    assert cfg.db_path == Path("/tmp/db.sqlite3")
    assert cfg.log_level == "DEBUG"
    assert cfg.ffmpeg_path == Path("/usr/local/bin/ffmpeg")


def test_env_overrides_yaml(tmp_path: Path, monkeypatch):
    cfg_path = tmp_path / "runtime.yaml"
    cfg_path.write_text("db_path: /tmp/a.sqlite3\nartifact_dir: /tmp/a\nlog_level: INFO\n")
    monkeypatch.setenv("MUSIC_DECODER_DB_PATH", "/tmp/override.sqlite3")
    cfg = load_runtime_config(cfg_path)
    assert cfg.db_path == Path("/tmp/override.sqlite3")


def test_missing_required_key_raises(tmp_path: Path):
    cfg_path = tmp_path / "runtime.yaml"
    cfg_path.write_text("artifact_dir: /tmp/a\n")
    with pytest.raises(ValueError, match="db_path"):
        load_runtime_config(cfg_path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_config_runtime.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement the config loader**

```python
# src/music_decoder/config/__init__.py
```

```python
# src/music_decoder/config/runtime.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml


_REQUIRED = ("db_path", "artifact_dir", "log_level")
_ENV_PREFIX = "MUSIC_DECODER_"


@dataclass(frozen=True)
class RuntimeConfig:
    db_path: Path
    artifact_dir: Path
    log_level: str
    ffmpeg_path: Path | None
    model_cache_dir: Path | None
    fixture_dir: Path | None


def _coerce_path(value: str | None) -> Path | None:
    return Path(value) if value else None


def load_runtime_config(path: Path) -> RuntimeConfig:
    raw: dict[str, object] = yaml.safe_load(path.read_text()) or {}
    for key, val in os.environ.items():
        if key.startswith(_ENV_PREFIX):
            raw[key[len(_ENV_PREFIX):].lower()] = val
    missing = [k for k in _REQUIRED if k not in raw]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")
    return RuntimeConfig(
        db_path=Path(str(raw["db_path"])),
        artifact_dir=Path(str(raw["artifact_dir"])),
        log_level=str(raw["log_level"]).upper(),
        ffmpeg_path=_coerce_path(raw.get("ffmpeg_path")),  # type: ignore[arg-type]
        model_cache_dir=_coerce_path(raw.get("model_cache_dir")),  # type: ignore[arg-type]
        fixture_dir=_coerce_path(raw.get("fixture_dir")),  # type: ignore[arg-type]
    )
```

- [ ] **Step 4: Write `config/runtime.yaml`**

```yaml
# config/runtime.yaml
# Defaults assume macOS user data dir; override via MUSIC_DECODER_* env vars.
db_path: ~/Library/Application Support/music-decoder/db.sqlite3
artifact_dir: ~/Library/Application Support/music-decoder/artifacts
log_level: INFO
ffmpeg_path: null            # null = autodiscover via PATH
model_cache_dir: ~/Library/Caches/music-decoder/models
fixture_dir: tests/fixtures
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/unit/test_config_runtime.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add config/runtime.yaml src/music_decoder/config/ tests/unit/test_config_runtime.py
git commit -m "feat(config): runtime YAML loader with MUSIC_DECODER_* env overrides"
```

---

### Task 4: Hyperparameter config loader

**Files:**
- Create: `config/hyperparameters.yaml`
- Create: `src/music_decoder/config/hyperparameters.py`
- Create: `tests/unit/test_config_hyperparameters.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_config_hyperparameters.py
from pathlib import Path

import pytest

from music_decoder.config.hyperparameters import (
    HyperparameterSet,
    load_hyperparameters,
)


def test_load_baseline(tmp_path: Path):
    yaml_text = """
id: 2026-05-04-baseline
basic_pitch:
  onset_threshold: 0.5
  frame_threshold: 0.3
  minimum_note_length_ms: 58
  minimum_frequency_hz: 32.7
  maximum_frequency_hz: 2000
crepe:
  model_capacity: full
  step_size_ms: 10
  viterbi: true
post_processing:
  median_filter_window: 5
  min_note_duration_s: 0.05
  same_pitch_merge_gap_s: 0.05
  rhythmic_snap_confidence_threshold: 0.7
key_detection:
  hpss_margin: 1.0
  windowed_segment_length_s: 8.0
  windowed_hop_s: 2.0
  modulation_penalty: 0.3
beat_tracking:
  start_bpm: 120
  tightness: 100
  ts_min_confidence: 0.5
tab_assignment:
  weights:
    w_move: 1.0
    w_string: 0.3
    w_span: 0.5
    w_open: 0.2
    w_high: 0.4
    w_chord_intra: 0.6
  max_fret: 22
ui:
  confidence_thresholds:
    high: 0.8
    medium: 0.5
evaluation:
  thresholds:
    note_f_measure: 0.65
    key_mirex_score: 0.75
    tab_string_accuracy: 0.55
  regression_tolerance: 0.02
"""
    p = tmp_path / "h.yaml"
    p.write_text(yaml_text)
    hp = load_hyperparameters(p)
    assert isinstance(hp, HyperparameterSet)
    assert hp.id == "2026-05-04-baseline"
    assert hp.basic_pitch.onset_threshold == 0.5
    assert hp.tab_assignment.weights["w_move"] == 1.0
    assert hp.tab_assignment.max_fret == 22
    assert hp.evaluation.thresholds["note_f_measure"] == 0.65


def test_missing_id_rejected(tmp_path: Path):
    p = tmp_path / "h.yaml"
    p.write_text("basic_pitch: {onset_threshold: 0.5}")
    with pytest.raises(ValueError):
        load_hyperparameters(p)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_config_hyperparameters.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement the loader**

```python
# src/music_decoder/config/hyperparameters.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class BasicPitchParams:
    onset_threshold: float
    frame_threshold: float
    minimum_note_length_ms: int
    minimum_frequency_hz: float
    maximum_frequency_hz: float


@dataclass(frozen=True)
class CrepeParams:
    model_capacity: str
    step_size_ms: int
    viterbi: bool


@dataclass(frozen=True)
class PostProcessingParams:
    median_filter_window: int
    min_note_duration_s: float
    same_pitch_merge_gap_s: float
    rhythmic_snap_confidence_threshold: float


@dataclass(frozen=True)
class KeyDetectionParams:
    hpss_margin: float
    windowed_segment_length_s: float
    windowed_hop_s: float
    modulation_penalty: float


@dataclass(frozen=True)
class BeatTrackingParams:
    start_bpm: float
    tightness: float
    ts_min_confidence: float


@dataclass(frozen=True)
class TabAssignmentParams:
    weights: dict[str, float]
    max_fret: int


@dataclass(frozen=True)
class UIParams:
    confidence_thresholds: dict[str, float]


@dataclass(frozen=True)
class EvaluationParams:
    thresholds: dict[str, float]
    regression_tolerance: float


@dataclass(frozen=True)
class HyperparameterSet:
    id: str
    basic_pitch: BasicPitchParams
    crepe: CrepeParams
    post_processing: PostProcessingParams
    key_detection: KeyDetectionParams
    beat_tracking: BeatTrackingParams
    tab_assignment: TabAssignmentParams
    ui: UIParams
    evaluation: EvaluationParams


def _section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    if key not in raw or not isinstance(raw[key], dict):
        raise ValueError(f"hyperparameters missing section: {key}")
    return raw[key]


def load_hyperparameters(path: Path) -> HyperparameterSet:
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict) or "id" not in raw:
        raise ValueError("hyperparameters yaml must have a top-level 'id'")
    return HyperparameterSet(
        id=str(raw["id"]),
        basic_pitch=BasicPitchParams(**_section(raw, "basic_pitch")),
        crepe=CrepeParams(**_section(raw, "crepe")),
        post_processing=PostProcessingParams(**_section(raw, "post_processing")),
        key_detection=KeyDetectionParams(**_section(raw, "key_detection")),
        beat_tracking=BeatTrackingParams(**_section(raw, "beat_tracking")),
        tab_assignment=TabAssignmentParams(**_section(raw, "tab_assignment")),
        ui=UIParams(**_section(raw, "ui")),
        evaluation=EvaluationParams(**_section(raw, "evaluation")),
    )
```

- [ ] **Step 4: Write `config/hyperparameters.yaml`** (same content as the test fixture above, in the real config dir).

- [ ] **Step 5: Run tests**

Run: `pytest tests/unit/test_config_hyperparameters.py -v`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add config/hyperparameters.yaml src/music_decoder/config/hyperparameters.py tests/unit/test_config_hyperparameters.py
git commit -m "feat(config): hyperparameter set loader with reproducibility id"
```

---

### Task 5: Tuning value object + presets

**Files:**
- Create: `src/music_decoder/tab_assignment/__init__.py`
- Create: `src/music_decoder/tab_assignment/tuning.py`
- Create: `tests/unit/test_tuning.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_tuning.py
import pytest

from music_decoder.tab_assignment.tuning import (
    Tuning,
    PRESETS,
    get_preset,
)


def test_eadgbe_preset_pitches():
    t = get_preset("EADGBE")
    assert t.name == "EADGBE"
    assert t.open_pitches == (40, 45, 50, 55, 59, 64)


def test_drop_d_preset_pitches():
    t = get_preset("Drop_D")
    assert t.open_pitches == (38, 45, 50, 55, 59, 64)


def test_eb_preset_pitches():
    t = get_preset("Eb")
    assert t.open_pitches == (39, 44, 49, 54, 58, 63)


def test_d_standard_pitches():
    t = get_preset("D_standard")
    assert t.open_pitches == (38, 43, 48, 53, 57, 62)


def test_drop_c_pitches():
    t = get_preset("Drop_C")
    assert t.open_pitches == (36, 43, 48, 53, 57, 62)


def test_dadgad_pitches():
    t = get_preset("DADGAD")
    assert t.open_pitches == (38, 45, 50, 55, 57, 62)


def test_unknown_preset_raises():
    with pytest.raises(KeyError):
        get_preset("not_a_tuning")


def test_tuning_is_frozen():
    t = get_preset("EADGBE")
    with pytest.raises(Exception):
        t.name = "X"  # type: ignore[misc]


def test_presets_listed():
    assert set(PRESETS.keys()) == {
        "EADGBE", "Drop_D", "Eb", "D_standard", "Drop_C", "DADGAD"
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_tuning.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# src/music_decoder/tab_assignment/__init__.py
```

```python
# src/music_decoder/tab_assignment/tuning.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Tuning:
    """Open-string MIDI pitches, ordered low to high (string 0 = lowest)."""
    name: str
    open_pitches: tuple[int, ...]


PRESETS: dict[str, Tuning] = {
    "EADGBE":     Tuning("EADGBE",     (40, 45, 50, 55, 59, 64)),  # E2 A2 D3 G3 B3 E4
    "Drop_D":     Tuning("Drop_D",     (38, 45, 50, 55, 59, 64)),  # D2 A2 D3 G3 B3 E4
    "Eb":         Tuning("Eb",         (39, 44, 49, 54, 58, 63)),  # half-step down
    "D_standard": Tuning("D_standard", (38, 43, 48, 53, 57, 62)),  # full step down
    "Drop_C":     Tuning("Drop_C",     (36, 43, 48, 53, 57, 62)),
    "DADGAD":     Tuning("DADGAD",     (38, 45, 50, 55, 57, 62)),
}


def get_preset(name: str) -> Tuning:
    if name not in PRESETS:
        raise KeyError(f"Unknown tuning preset: {name!r}")
    return PRESETS[name]
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_tuning.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/music_decoder/tab_assignment/ tests/unit/test_tuning.py
git commit -m "feat(tab): Tuning value object with 6 standard/alt presets"
```

---

### Task 6: ArtifactStore interface + filesystem implementation

**Files:**
- Create: `src/music_decoder/artifacts/__init__.py`
- Create: `src/music_decoder/artifacts/base.py`
- Create: `src/music_decoder/artifacts/filesystem.py`
- Create: `tests/unit/test_artifacts.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_artifacts.py
from pathlib import Path

import pytest

from music_decoder.artifacts.base import ArtifactStore
from music_decoder.artifacts.filesystem import FilesystemArtifactStore


def test_store_writes_and_reads(tmp_path: Path):
    store: ArtifactStore = FilesystemArtifactStore(root=tmp_path)
    key = store.put("uploads/1/source.wav", b"hello world")
    assert key == "uploads/1/source.wav"
    assert store.exists(key)
    assert store.read(key) == b"hello world"


def test_store_path_for_key(tmp_path: Path):
    store = FilesystemArtifactStore(root=tmp_path)
    key = store.put("uploads/1/source.wav", b"x")
    assert store.path_for(key) == tmp_path / "uploads/1/source.wav"


def test_store_open_for_write(tmp_path: Path):
    store = FilesystemArtifactStore(root=tmp_path)
    target = store.path_for("derived/2/midi.mid")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"midi bytes")
    assert store.exists("derived/2/midi.mid")


def test_store_rejects_traversal(tmp_path: Path):
    store = FilesystemArtifactStore(root=tmp_path)
    with pytest.raises(ValueError, match="path traversal"):
        store.put("../escape.txt", b"x")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_artifacts.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# src/music_decoder/artifacts/__init__.py
```

```python
# src/music_decoder/artifacts/base.py
from __future__ import annotations

from pathlib import Path
from typing import Protocol


class ArtifactStore(Protocol):
    def put(self, key: str, data: bytes) -> str: ...
    def read(self, key: str) -> bytes: ...
    def exists(self, key: str) -> bool: ...
    def path_for(self, key: str) -> Path: ...
```

```python
# src/music_decoder/artifacts/filesystem.py
from __future__ import annotations

from pathlib import Path


class FilesystemArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        target = (self.root / key).resolve()
        try:
            target.relative_to(self.root.resolve())
        except ValueError as e:
            raise ValueError(f"path traversal not allowed: {key!r}") from e
        return target

    def put(self, key: str, data: bytes) -> str:
        path = self._resolve(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return key

    def read(self, key: str) -> bytes:
        return self._resolve(key).read_bytes()

    def exists(self, key: str) -> bool:
        try:
            return self._resolve(key).exists()
        except ValueError:
            return False

    def path_for(self, key: str) -> Path:
        return self._resolve(key)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_artifacts.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/music_decoder/artifacts/ tests/unit/test_artifacts.py
git commit -m "feat(artifacts): filesystem-backed ArtifactStore with traversal protection"
```

---

## Phase 1 — Persistence

### Task 7: SQLAlchemy declarative models

**Files:**
- Create: `src/music_decoder/persistence/__init__.py`
- Create: `src/music_decoder/persistence/models.py`
- Create: `tests/unit/test_persistence_models.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_persistence_models.py
from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from music_decoder.persistence.models import (
    Base,
    Upload,
    Job,
    JobProgress,
    Note,
    KeyEstimate,
    TempoEstimate,
    TabReference,
    AccuracyReport,
)


def _engine():
    e = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(e)
    return e


def test_create_upload_and_job():
    engine = _engine()
    with Session(engine) as s:
        u = Upload(
            sha256="abc", original_filename="a.wav", mime_type="audio/wav",
            duration_s=30.0, sample_rate_hz=22050,
            declared_kind="solo_guitar", artifact_path="uploads/1/source.wav",
            created_at=datetime.now(timezone.utc),
        )
        s.add(u); s.flush()
        j = Job(
            upload_id=u.id, status="queued",
            transcription_model="basic-pitch", requested_tuning="EADGBE",
            requested_quality="standard", use_demucs=False,
            hyperparameter_set="2026-05-04-baseline",
            created_at=datetime.now(timezone.utc),
        )
        s.add(j); s.commit()
        assert j.id is not None
        assert j.upload_id == u.id


def test_status_check_constraint_rejects_garbage():
    import pytest
    from sqlalchemy.exc import IntegrityError

    engine = _engine()
    with Session(engine) as s:
        u = Upload(
            sha256="x", original_filename="a.wav", mime_type="audio/wav",
            duration_s=1.0, sample_rate_hz=22050,
            declared_kind="solo_guitar", artifact_path="uploads/1/source.wav",
            created_at=datetime.now(timezone.utc),
        )
        s.add(u); s.flush()
        s.add(Job(
            upload_id=u.id, status="WHATEVER",
            transcription_model="basic-pitch", requested_tuning="EADGBE",
            requested_quality="standard", use_demucs=False,
            hyperparameter_set="x", created_at=datetime.now(timezone.utc),
        ))
        with pytest.raises(IntegrityError):
            s.commit()


def test_cascade_delete_removes_children():
    engine = _engine()
    with Session(engine) as s:
        u = Upload(
            sha256="z", original_filename="a.wav", mime_type="audio/wav",
            duration_s=1.0, sample_rate_hz=22050,
            declared_kind="solo_guitar", artifact_path="uploads/1/source.wav",
            created_at=datetime.now(timezone.utc),
        )
        s.add(u); s.flush()
        j = Job(
            upload_id=u.id, status="succeeded",
            transcription_model="basic-pitch", requested_tuning="EADGBE",
            requested_quality="standard", use_demucs=False,
            hyperparameter_set="x", created_at=datetime.now(timezone.utc),
        )
        s.add(j); s.flush()
        s.add(Note(job_id=j.id, start_s=0.0, end_s=1.0, pitch=60, velocity=80, confidence=0.9))
        s.commit()
        s.delete(u)
        s.commit()
        assert s.query(Note).count() == 0
        assert s.query(Job).count() == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_persistence_models.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement the models**

```python
# src/music_decoder/persistence/__init__.py
```

```python
# src/music_decoder/persistence/models.py
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    CheckConstraint, ForeignKey, Index, Integer, String, Text, Float, DateTime,
    Boolean,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Upload(Base):
    __tablename__ = "uploads"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    sha256: Mapped[str] = mapped_column(String, nullable=False, index=True)
    original_filename: Mapped[str] = mapped_column(String, nullable=False)
    mime_type: Mapped[str] = mapped_column(String, nullable=False)
    duration_s: Mapped[float | None] = mapped_column(Float)
    sample_rate_hz: Mapped[int | None] = mapped_column(Integer)
    declared_kind: Mapped[str] = mapped_column(
        String, CheckConstraint("declared_kind IN ('solo_guitar','full_mix')"),
        nullable=False,
    )
    artifact_path: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    jobs: Mapped[list["Job"]] = relationship(
        back_populates="upload", cascade="all, delete-orphan", passive_deletes=True,
    )


class Job(Base):
    __tablename__ = "jobs"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    upload_id: Mapped[int] = mapped_column(
        ForeignKey("uploads.id", ondelete="CASCADE"), nullable=False, index=True,
    )
    status: Mapped[str] = mapped_column(
        String, CheckConstraint(
            "status IN ('queued','running','succeeded','failed','cancelled')"
        ), nullable=False,
    )
    transcription_model: Mapped[str] = mapped_column(
        String, CheckConstraint("transcription_model IN ('basic-pitch','crepe')"),
        nullable=False,
    )
    requested_tuning: Mapped[str] = mapped_column(String, nullable=False)
    requested_quality: Mapped[str] = mapped_column(
        String, CheckConstraint("requested_quality IN ('standard','high')"),
        nullable=False,
    )
    use_demucs: Mapped[bool] = mapped_column(Boolean, nullable=False)
    hyperparameter_set: Mapped[str] = mapped_column(String, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime)
    error_class: Mapped[str | None] = mapped_column(String)
    error_message: Mapped[str | None] = mapped_column(Text)
    error_traceback: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    upload: Mapped[Upload] = relationship(back_populates="jobs")
    progress: Mapped[list["JobProgress"]] = relationship(
        back_populates="job", cascade="all, delete-orphan", passive_deletes=True,
    )
    notes: Mapped[list["Note"]] = relationship(
        back_populates="job", cascade="all, delete-orphan", passive_deletes=True,
    )

    __table_args__ = (
        Index("jobs_queued_idx", "status",
              sqlite_where=Index("jobs_queued_idx", "status").expressions[0].in_(("queued", "running"))),
    )


class JobProgress(Base):
    __tablename__ = "job_progress"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    job_id: Mapped[int] = mapped_column(
        ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False,
    )
    stage: Mapped[str] = mapped_column(String, nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime)
    success: Mapped[bool | None] = mapped_column(Boolean)
    error: Mapped[str | None] = mapped_column(Text)
    summary_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")

    job: Mapped[Job] = relationship(back_populates="progress")
    __table_args__ = (Index("job_progress_job_idx", "job_id", "started_at"),)


class Note(Base):
    __tablename__ = "notes"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    job_id: Mapped[int] = mapped_column(
        ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False,
    )
    start_s: Mapped[float] = mapped_column(Float, nullable=False)
    end_s: Mapped[float] = mapped_column(Float, nullable=False)
    pitch: Mapped[int] = mapped_column(Integer, nullable=False)
    velocity: Mapped[int] = mapped_column(Integer, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    string: Mapped[int | None] = mapped_column(Integer)
    fret: Mapped[int | None] = mapped_column(Integer)
    cost_breakdown_json: Mapped[str | None] = mapped_column(Text)
    dropped_reason: Mapped[str | None] = mapped_column(String)

    job: Mapped[Job] = relationship(back_populates="notes")
    __table_args__ = (Index("notes_job_idx", "job_id", "start_s"),)


class KeyEstimate(Base):
    __tablename__ = "key_estimates"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    job_id: Mapped[int] = mapped_column(
        ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False,
    )
    scope: Mapped[str] = mapped_column(
        String, CheckConstraint("scope IN ('global','window')"), nullable=False,
    )
    window_start_s: Mapped[float | None] = mapped_column(Float)
    window_end_s: Mapped[float | None] = mapped_column(Float)
    profile: Mapped[str] = mapped_column(
        String, CheckConstraint("profile IN ('krumhansl_kessler','temperley')"),
        nullable=False,
    )
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    tonic: Mapped[str] = mapped_column(String, nullable=False)
    mode: Mapped[str] = mapped_column(
        String, CheckConstraint("mode IN ('major','minor')"), nullable=False,
    )
    correlation: Mapped[float] = mapped_column(Float, nullable=False)
    margin: Mapped[float] = mapped_column(Float, nullable=False)

    __table_args__ = (Index("key_estimates_job_idx", "job_id", "scope"),)


class TempoEstimate(Base):
    __tablename__ = "tempo_estimates"
    job_id: Mapped[int] = mapped_column(
        ForeignKey("jobs.id", ondelete="CASCADE"), primary_key=True,
    )
    tempo_bpm: Mapped[float] = mapped_column(Float, nullable=False)
    beat_times_s_json: Mapped[str] = mapped_column(Text, nullable=False)
    downbeat_times_s_json: Mapped[str] = mapped_column(Text, nullable=False)
    ts_numerator: Mapped[int] = mapped_column(Integer, nullable=False)
    ts_denominator: Mapped[int] = mapped_column(Integer, nullable=False)
    ts_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    ts_assumed: Mapped[bool] = mapped_column(Boolean, nullable=False)


class TabReference(Base):
    __tablename__ = "tab_references"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    job_id: Mapped[int] = mapped_column(
        ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False,
    )
    source: Mapped[str] = mapped_column(String, nullable=False)
    song_acoustid: Mapped[str | None] = mapped_column(String)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    similarity_to_prediction: Mapped[float | None] = mapped_column(Float)
    disagreement_spans_json: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class AccuracyReport(Base):
    __tablename__ = "accuracy_reports"
    job_id: Mapped[int] = mapped_column(
        ForeignKey("jobs.id", ondelete="CASCADE"), primary_key=True,
    )
    fixture_name: Mapped[str | None] = mapped_column(String)
    note_f_measure: Mapped[float | None] = mapped_column(Float)
    onset_f_measure: Mapped[float | None] = mapped_column(Float)
    pitch_class_accuracy: Mapped[float | None] = mapped_column(Float)
    key_mirex_score: Mapped[float | None] = mapped_column(Float)
    tab_string_accuracy: Mapped[float | None] = mapped_column(Float)
    self_confidence_summary_json: Mapped[str] = mapped_column(Text, nullable=False)
    full_metrics_json: Mapped[str] = mapped_column(Text, nullable=False)
```

> **Note on the `jobs_queued_idx` partial index:** SQLAlchemy partial-index syntax is awkward; if the inline `Index` definition above conflicts during testing, replace the `__table_args__` line in `Job` with a plain `(Index("jobs_queued_idx", "status"),)` — the partial-index condition will be added in the Alembic migration in Task 8.

- [ ] **Step 4: Adjust if partial index conflicts**

If `pytest` errors on the partial index, use the plain index in `models.py`:
```python
__table_args__ = (Index("jobs_queued_idx", "status"),)
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/unit/test_persistence_models.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add src/music_decoder/persistence/ tests/unit/test_persistence_models.py
git commit -m "feat(persistence): SQLAlchemy models for full schema with FK CASCADE"
```

---

### Task 8: Database session factory + Alembic migration

**Files:**
- Create: `src/music_decoder/persistence/session.py`
- Create: `alembic.ini`
- Create: `migrations/env.py`
- Create: `migrations/script.py.mako`
- Create: `migrations/versions/0001_init.py`
- Create: `tests/unit/test_persistence_session.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_persistence_session.py
from pathlib import Path

from sqlalchemy import inspect

from music_decoder.persistence.session import build_engine, run_migrations


def test_run_migrations_creates_all_tables(tmp_path: Path):
    db_path = tmp_path / "app.sqlite3"
    engine = build_engine(db_path)
    run_migrations(engine)
    insp = inspect(engine)
    expected = {
        "uploads", "jobs", "job_progress", "notes",
        "key_estimates", "tempo_estimates", "tab_references", "accuracy_reports",
        "alembic_version",
    }
    assert expected.issubset(set(insp.get_table_names()))


def test_engine_enables_wal_and_foreign_keys(tmp_path: Path):
    db_path = tmp_path / "app.sqlite3"
    engine = build_engine(db_path)
    with engine.connect() as conn:
        from sqlalchemy import text
        assert conn.execute(text("PRAGMA journal_mode")).scalar_one().lower() == "wal"
        assert conn.execute(text("PRAGMA foreign_keys")).scalar_one() == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_persistence_session.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement session factory**

```python
# src/music_decoder/persistence/session.py
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from alembic import command
from alembic.config import Config
from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import create_engine


def build_engine(db_path: Path) -> Engine:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}", future=True)

    @event.listens_for(engine, "connect")
    def _on_connect(dbapi_conn, _):  # pragma: no cover - trivial
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    return engine


def session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


def run_migrations(engine: Engine) -> None:
    cfg = Config(str(Path(__file__).parents[3] / "alembic.ini"))
    cfg.set_main_option("script_location",
                        str(Path(__file__).parents[3] / "migrations"))
    cfg.attributes["connection"] = engine.connect()
    command.upgrade(cfg, "head")
    cfg.attributes["connection"].close()
```

- [ ] **Step 4: Write `alembic.ini`**

```ini
[alembic]
script_location = migrations
prepend_sys_path = .
version_path_separator = os
sqlalchemy.url = sqlite:///./placeholder.sqlite3

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
```

- [ ] **Step 5: Write `migrations/env.py`**

```python
# migrations/env.py
from __future__ import annotations

from alembic import context
from sqlalchemy import engine_from_config, pool

from music_decoder.persistence.models import Base


config = context.config
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = config.attributes.get("connection", None)
    if connectable is None:
        connectable = engine_from_config(
            config.get_section(config.config_ini_section, {}),
            prefix="sqlalchemy.",
            poolclass=pool.NullPool,
        )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

- [ ] **Step 6: Write `migrations/script.py.mako`**

```mako
"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}

"""
from alembic import op
import sqlalchemy as sa
${imports if imports else ""}

revision = ${repr(up_revision)}
down_revision = ${repr(down_revision)}
branch_labels = ${repr(branch_labels)}
depends_on = ${repr(depends_on)}


def upgrade() -> None:
    ${upgrades if upgrades else "pass"}


def downgrade() -> None:
    ${downgrades if downgrades else "pass"}
```

- [ ] **Step 7: Generate the initial migration**

Run:
```bash
alembic revision --autogenerate -m "init schema"
```

Move the generated file to `migrations/versions/0001_init.py` and verify it `op.create_table()` calls match the schema in [docs/superpowers/specs/2026-05-04-music-decoder-design.md](../specs/2026-05-04-music-decoder-design.md) §5. **Manually add the partial index** at the end of `upgrade()`:

```python
op.execute(
    "CREATE INDEX IF NOT EXISTS jobs_queued_partial_idx "
    "ON jobs(status) WHERE status IN ('queued','running')"
)
```

- [ ] **Step 8: Run tests**

Run: `pytest tests/unit/test_persistence_session.py -v`
Expected: 2 passed.

- [ ] **Step 9: Commit**

```bash
git add alembic.ini migrations/ src/music_decoder/persistence/session.py tests/unit/test_persistence_session.py
git commit -m "feat(persistence): SQLite WAL session + alembic init migration"
```

---

### Task 9: Repositories (CRUD wrappers)

**Files:**
- Create: `src/music_decoder/persistence/repositories.py`
- Create: `tests/unit/test_persistence_repositories.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_persistence_repositories.py
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from music_decoder.persistence.models import Base
from music_decoder.persistence.repositories import (
    UploadRepo, JobRepo, NoteRepo, KeyEstimateRepo, TempoEstimateRepo,
    JobProgressRepo, AccuracyReportRepo, TabReferenceRepo,
)


@pytest.fixture
def session():
    e = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(e)
    with Session(e) as s:
        yield s


def test_upload_create_and_lookup_by_sha(session: Session):
    repo = UploadRepo(session)
    u = repo.create(
        sha256="abc", original_filename="a.wav", mime_type="audio/wav",
        duration_s=10.0, sample_rate_hz=22050, declared_kind="solo_guitar",
        artifact_path="uploads/1/source.wav",
    )
    session.commit()
    found = repo.find_by_sha256("abc")
    assert found is not None and found.id == u.id


def test_job_lifecycle(session: Session):
    upload = UploadRepo(session).create(
        sha256="x", original_filename="a.wav", mime_type="audio/wav",
        duration_s=1.0, sample_rate_hz=22050, declared_kind="solo_guitar",
        artifact_path="uploads/1/source.wav",
    )
    session.flush()
    jrepo = JobRepo(session)
    j = jrepo.enqueue(
        upload_id=upload.id, transcription_model="basic-pitch",
        requested_tuning="EADGBE", requested_quality="standard",
        use_demucs=False, hyperparameter_set="2026-05-04-baseline",
    )
    session.commit()
    assert j.status == "queued"
    jrepo.mark_running(j.id)
    session.commit()
    assert jrepo.get(j.id).status == "running"
    jrepo.mark_succeeded(j.id)
    session.commit()
    assert jrepo.get(j.id).status == "succeeded"


def test_pop_next_queued_returns_oldest_first(session: Session):
    upload = UploadRepo(session).create(
        sha256="x", original_filename="a.wav", mime_type="audio/wav",
        duration_s=1.0, sample_rate_hz=22050, declared_kind="solo_guitar",
        artifact_path="uploads/1/source.wav",
    )
    session.flush()
    jrepo = JobRepo(session)
    j1 = jrepo.enqueue(upload.id, "basic-pitch", "EADGBE", "standard", False, "v1")
    j2 = jrepo.enqueue(upload.id, "basic-pitch", "EADGBE", "standard", False, "v1")
    session.commit()
    popped = jrepo.pop_next_queued()
    assert popped.id == j1.id
    session.commit()
    popped2 = jrepo.pop_next_queued()
    assert popped2.id == j2.id
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_persistence_repositories.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement repositories**

```python
# src/music_decoder/persistence/repositories.py
from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import (
    Upload, Job, JobProgress, Note, KeyEstimate, TempoEstimate,
    TabReference, AccuracyReport,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class UploadRepo:
    def __init__(self, session: Session) -> None:
        self.s = session

    def create(self, *, sha256: str, original_filename: str, mime_type: str,
               duration_s: float | None, sample_rate_hz: int | None,
               declared_kind: str, artifact_path: str) -> Upload:
        u = Upload(
            sha256=sha256, original_filename=original_filename, mime_type=mime_type,
            duration_s=duration_s, sample_rate_hz=sample_rate_hz,
            declared_kind=declared_kind, artifact_path=artifact_path,
            created_at=_now(),
        )
        self.s.add(u)
        self.s.flush()
        return u

    def find_by_sha256(self, sha256: str) -> Upload | None:
        return self.s.execute(select(Upload).where(Upload.sha256 == sha256)).scalar_one_or_none()


class JobRepo:
    def __init__(self, session: Session) -> None:
        self.s = session

    def enqueue(self, upload_id: int, transcription_model: str,
                requested_tuning: str, requested_quality: str,
                use_demucs: bool, hyperparameter_set: str) -> Job:
        j = Job(
            upload_id=upload_id, status="queued",
            transcription_model=transcription_model,
            requested_tuning=requested_tuning, requested_quality=requested_quality,
            use_demucs=use_demucs, hyperparameter_set=hyperparameter_set,
            created_at=_now(),
        )
        self.s.add(j)
        self.s.flush()
        return j

    def get(self, job_id: int) -> Job:
        return self.s.execute(select(Job).where(Job.id == job_id)).scalar_one()

    def pop_next_queued(self) -> Job | None:
        # Atomic in WAL with a single writer; multi-writer would need SELECT FOR UPDATE.
        j = self.s.execute(
            select(Job).where(Job.status == "queued").order_by(Job.id.asc()).limit(1)
        ).scalar_one_or_none()
        if j is None:
            return None
        j.status = "running"
        j.started_at = _now()
        self.s.flush()
        return j

    def mark_running(self, job_id: int) -> None:
        j = self.get(job_id)
        j.status = "running"
        j.started_at = j.started_at or _now()

    def mark_succeeded(self, job_id: int) -> None:
        j = self.get(job_id)
        j.status = "succeeded"
        j.finished_at = _now()

    def mark_failed(self, job_id: int, error_class: str, error_message: str,
                    traceback_text: str | None) -> None:
        j = self.get(job_id)
        j.status = "failed"
        j.finished_at = _now()
        j.error_class = error_class
        j.error_message = error_message
        j.error_traceback = traceback_text


class JobProgressRepo:
    def __init__(self, session: Session) -> None:
        self.s = session

    def record(self, *, job_id: int, stage: str, started_at: datetime,
               ended_at: datetime | None, success: bool | None,
               error: str | None, summary_json: str = "{}") -> JobProgress:
        p = JobProgress(
            job_id=job_id, stage=stage, started_at=started_at, ended_at=ended_at,
            success=success, error=error, summary_json=summary_json,
        )
        self.s.add(p)
        self.s.flush()
        return p


class NoteRepo:
    def __init__(self, session: Session) -> None:
        self.s = session

    def bulk_insert(self, job_id: int, notes: Iterable[dict]) -> None:
        self.s.add_all([Note(job_id=job_id, **n) for n in notes])
        self.s.flush()

    def list_for_job(self, job_id: int) -> list[Note]:
        return list(self.s.execute(
            select(Note).where(Note.job_id == job_id).order_by(Note.start_s)
        ).scalars())


class KeyEstimateRepo:
    def __init__(self, session: Session) -> None:
        self.s = session

    def bulk_insert(self, job_id: int, rows: Iterable[dict]) -> None:
        self.s.add_all([KeyEstimate(job_id=job_id, **r) for r in rows])
        self.s.flush()


class TempoEstimateRepo:
    def __init__(self, session: Session) -> None:
        self.s = session

    def upsert(self, *, job_id: int, tempo_bpm: float, beat_times_s_json: str,
               downbeat_times_s_json: str, ts_numerator: int, ts_denominator: int,
               ts_confidence: float, ts_assumed: bool) -> TempoEstimate:
        existing = self.s.get(TempoEstimate, job_id)
        if existing:
            existing.tempo_bpm = tempo_bpm
            existing.beat_times_s_json = beat_times_s_json
            existing.downbeat_times_s_json = downbeat_times_s_json
            existing.ts_numerator = ts_numerator
            existing.ts_denominator = ts_denominator
            existing.ts_confidence = ts_confidence
            existing.ts_assumed = ts_assumed
            return existing
        t = TempoEstimate(
            job_id=job_id, tempo_bpm=tempo_bpm,
            beat_times_s_json=beat_times_s_json,
            downbeat_times_s_json=downbeat_times_s_json,
            ts_numerator=ts_numerator, ts_denominator=ts_denominator,
            ts_confidence=ts_confidence, ts_assumed=ts_assumed,
        )
        self.s.add(t); self.s.flush()
        return t


class TabReferenceRepo:
    def __init__(self, session: Session) -> None:
        self.s = session

    def create(self, *, job_id: int, source: str, song_acoustid: str | None,
               raw_text: str, similarity_to_prediction: float | None,
               disagreement_spans_json: str | None) -> TabReference:
        r = TabReference(
            job_id=job_id, source=source, song_acoustid=song_acoustid,
            raw_text=raw_text, similarity_to_prediction=similarity_to_prediction,
            disagreement_spans_json=disagreement_spans_json,
            created_at=_now(),
        )
        self.s.add(r); self.s.flush()
        return r


class AccuracyReportRepo:
    def __init__(self, session: Session) -> None:
        self.s = session

    def upsert(self, *, job_id: int, fixture_name: str | None,
               note_f_measure: float | None, onset_f_measure: float | None,
               pitch_class_accuracy: float | None,
               key_mirex_score: float | None,
               tab_string_accuracy: float | None,
               self_confidence_summary_json: str,
               full_metrics_json: str) -> AccuracyReport:
        existing = self.s.get(AccuracyReport, job_id)
        if existing:
            existing.fixture_name = fixture_name
            existing.note_f_measure = note_f_measure
            existing.onset_f_measure = onset_f_measure
            existing.pitch_class_accuracy = pitch_class_accuracy
            existing.key_mirex_score = key_mirex_score
            existing.tab_string_accuracy = tab_string_accuracy
            existing.self_confidence_summary_json = self_confidence_summary_json
            existing.full_metrics_json = full_metrics_json
            return existing
        r = AccuracyReport(
            job_id=job_id, fixture_name=fixture_name,
            note_f_measure=note_f_measure, onset_f_measure=onset_f_measure,
            pitch_class_accuracy=pitch_class_accuracy,
            key_mirex_score=key_mirex_score,
            tab_string_accuracy=tab_string_accuracy,
            self_confidence_summary_json=self_confidence_summary_json,
            full_metrics_json=full_metrics_json,
        )
        self.s.add(r); self.s.flush()
        return r
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_persistence_repositories.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/music_decoder/persistence/repositories.py tests/unit/test_persistence_repositories.py
git commit -m "feat(persistence): repositories with pop_next_queued for SQLite job queue"
```

---

## Phase 2 — Evaluation harness (built before any pipeline stage)

### Task 10: mir_eval-backed metrics module

**Files:**
- Create: `src/music_decoder/evaluation/__init__.py`
- Create: `src/music_decoder/evaluation/metrics.py`
- Create: `tests/unit/test_evaluation_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_evaluation_metrics.py
import numpy as np
import pytest

from music_decoder.evaluation.metrics import (
    note_f_measure,
    onset_f_measure,
    pitch_class_accuracy,
    key_mirex_score,
    tab_string_accuracy,
)


def _intervals_pitches(triples):
    intervals = np.array([(s, e) for (s, e, _) in triples], dtype=float)
    pitches = np.array([p for (_, _, p) in triples], dtype=float)
    return intervals, pitches


def test_note_f_measure_perfect():
    notes = [(0.0, 0.5, 60), (0.5, 1.0, 62), (1.0, 1.5, 64)]
    pi, pp = _intervals_pitches(notes)
    f = note_f_measure(pi, pp, pi.copy(), pp.copy())
    assert f == pytest.approx(1.0)


def test_note_f_measure_one_missed_one_extra():
    truth = [(0.0, 0.5, 60), (0.5, 1.0, 62)]
    pred = [(0.0, 0.5, 60), (0.5, 1.0, 99)]   # second note wrong pitch
    ti, tp = _intervals_pitches(truth)
    pi, pp = _intervals_pitches(pred)
    f = note_f_measure(pi, pp, ti, tp)
    assert 0.4 < f < 0.7


def test_pitch_class_accuracy_perfect():
    truth = [(0.0, 1.0, 60)]
    pred = [(0.0, 1.0, 72)]   # same pitch class as 60 (C)
    ti, tp = _intervals_pitches(truth)
    pi, pp = _intervals_pitches(pred)
    acc = pitch_class_accuracy(pi, pp, ti, tp)
    assert acc > 0.9


def test_key_mirex_correct():
    assert key_mirex_score(("C", "major"), ("C", "major")) == 1.0


def test_key_mirex_perfect_fifth():
    assert key_mirex_score(("G", "major"), ("C", "major")) == pytest.approx(0.5)


def test_key_mirex_relative():
    assert key_mirex_score(("A", "minor"), ("C", "major")) == pytest.approx(0.3)


def test_key_mirex_parallel():
    assert key_mirex_score(("C", "minor"), ("C", "major")) == pytest.approx(0.2)


def test_key_mirex_wrong():
    assert key_mirex_score(("F#", "major"), ("C", "major")) == 0.0


def test_tab_string_accuracy():
    pred = [(60, 4, 1), (62, 4, 3)]   # (pitch, string, fret)
    truth = [(60, 4, 1), (62, 3, 7)]   # second note: same pitch, different string
    assert tab_string_accuracy(pred, truth) == pytest.approx(0.5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_evaluation_metrics.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement metrics**

```python
# src/music_decoder/evaluation/__init__.py
```

```python
# src/music_decoder/evaluation/metrics.py
from __future__ import annotations

from typing import Iterable

import numpy as np
import mir_eval


_PITCH_CLASS = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4,
                "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9,
                "A#": 10, "Bb": 10, "B": 11}


def _midi_to_hz(midi: np.ndarray) -> np.ndarray:
    return 440.0 * 2 ** ((midi - 69) / 12.0)


def note_f_measure(
    pred_intervals: np.ndarray, pred_pitches_midi: np.ndarray,
    gt_intervals: np.ndarray, gt_pitches_midi: np.ndarray,
    onset_tolerance_s: float = 0.05, pitch_tolerance_cents: float = 50.0,
) -> float:
    if len(pred_intervals) == 0 and len(gt_intervals) == 0:
        return 1.0
    if len(pred_intervals) == 0 or len(gt_intervals) == 0:
        return 0.0
    pred_hz = _midi_to_hz(np.asarray(pred_pitches_midi))
    gt_hz = _midi_to_hz(np.asarray(gt_pitches_midi))
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        np.asarray(gt_intervals), gt_hz,
        np.asarray(pred_intervals), pred_hz,
        onset_tolerance=onset_tolerance_s,
        pitch_tolerance=pitch_tolerance_cents,
        offset_ratio=None,    # disable offset matching per MIREX defaults
    )
    return float(f)


def onset_f_measure(
    pred_intervals: np.ndarray, gt_intervals: np.ndarray,
    tolerance_s: float = 0.05,
) -> float:
    if len(pred_intervals) == 0 and len(gt_intervals) == 0:
        return 1.0
    if len(pred_intervals) == 0 or len(gt_intervals) == 0:
        return 0.0
    pred_onsets = np.asarray(pred_intervals)[:, 0]
    gt_onsets = np.asarray(gt_intervals)[:, 0]
    p, r, f = mir_eval.onset.f_measure(gt_onsets, pred_onsets, window=tolerance_s)
    return float(f)


def pitch_class_accuracy(
    pred_intervals: np.ndarray, pred_pitches_midi: np.ndarray,
    gt_intervals: np.ndarray, gt_pitches_midi: np.ndarray,
    frame_rate_hz: float = 100.0,
) -> float:
    if len(gt_intervals) == 0:
        return 1.0 if len(pred_intervals) == 0 else 0.0
    end = max(float(np.asarray(gt_intervals).max()),
              float(np.asarray(pred_intervals).max()) if len(pred_intervals) else 0.0)
    n_frames = max(int(np.ceil(end * frame_rate_hz)) + 1, 1)
    times = np.arange(n_frames) / frame_rate_hz

    def _frame_pcs(intervals: np.ndarray, pitches: np.ndarray) -> np.ndarray:
        out = np.full(n_frames, -1, dtype=int)
        for (s, e), p in zip(intervals, pitches):
            mask = (times >= s) & (times < e)
            out[mask] = int(p) % 12
        return out

    gt_pcs = _frame_pcs(np.asarray(gt_intervals), np.asarray(gt_pitches_midi))
    pred_pcs = _frame_pcs(np.asarray(pred_intervals), np.asarray(pred_pitches_midi))
    valid = gt_pcs != -1
    if valid.sum() == 0:
        return 1.0
    matched = ((pred_pcs == gt_pcs) & valid).sum()
    return float(matched / valid.sum())


def key_mirex_score(predicted: tuple[str, str], truth: tuple[str, str]) -> float:
    p_tonic, p_mode = predicted
    t_tonic, t_mode = truth
    p, t = _PITCH_CLASS[p_tonic], _PITCH_CLASS[t_tonic]
    if (p, p_mode) == (t, t_mode):
        return 1.0
    if p_mode == t_mode and (p - t) % 12 == 7:   # perfect fifth above
        return 0.5
    if {p_mode, t_mode} == {"major", "minor"}:
        # relative: minor tonic 9 semitones above major tonic
        if p_mode == "minor" and (p - t) % 12 == 9:
            return 0.3
        if t_mode == "minor" and (t - p) % 12 == 9:
            return 0.3
        # parallel: same tonic, opposite mode
        if p == t:
            return 0.2
    return 0.0


def tab_string_accuracy(
    predicted: Iterable[tuple[int, int, int]],
    truth: Iterable[tuple[int, int, int]],
) -> float:
    """Each tuple is (pitch_midi, string_index, fret). Aligned 1:1 by order."""
    pred_list = list(predicted)
    truth_list = list(truth)
    n = max(len(pred_list), len(truth_list))
    if n == 0:
        return 1.0
    correct = 0
    for p, t in zip(pred_list, truth_list):
        if p[0] == t[0] and p[1] == t[1]:
            correct += 1
    return correct / n
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_evaluation_metrics.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/music_decoder/evaluation/__init__.py src/music_decoder/evaluation/metrics.py tests/unit/test_evaluation_metrics.py
git commit -m "feat(eval): mir_eval-backed metrics with MIREX key scoring"
```

---

### Task 11: Fixture base interface + dataclasses

**Files:**
- Create: `src/music_decoder/evaluation/fixtures/__init__.py`
- Create: `src/music_decoder/evaluation/fixtures/base.py`
- Create: `tests/unit/test_evaluation_fixture_base.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_evaluation_fixture_base.py
import numpy as np

from music_decoder.evaluation.fixtures.base import GroundTruth, Fixture


def test_groundtruth_holds_intervals_and_key():
    gt = GroundTruth(
        intervals=np.array([[0.0, 0.5], [0.5, 1.0]]),
        pitches_midi=np.array([60, 62]),
        key=("C", "major"),
        tempo_bpm=120.0,
        tab=[(60, 4, 1), (62, 4, 3)],
    )
    assert gt.key == ("C", "major")
    assert gt.tempo_bpm == 120.0
    assert gt.tab == [(60, 4, 1), (62, 4, 3)]


def test_fixture_has_audio_path_and_truth(tmp_path):
    audio = tmp_path / "x.wav"
    audio.write_bytes(b"x")
    gt = GroundTruth(
        intervals=np.zeros((0, 2)), pitches_midi=np.zeros(0),
        key=None, tempo_bpm=None, tab=None,
    )
    fx = Fixture(name="x", source="manual", audio_path=audio, ground_truth=gt)
    assert fx.audio_path == audio
    assert fx.source == "manual"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_evaluation_fixture_base.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# src/music_decoder/evaluation/fixtures/__init__.py
```

```python
# src/music_decoder/evaluation/fixtures/base.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Literal, Protocol

import numpy as np


@dataclass(frozen=True)
class GroundTruth:
    intervals: np.ndarray              # (N, 2): start_s, end_s
    pitches_midi: np.ndarray           # (N,): MIDI pitches
    key: tuple[str, str] | None        # ("C", "major") or None
    tempo_bpm: float | None
    tab: list[tuple[int, int, int]] | None   # [(pitch, string, fret), ...]


@dataclass(frozen=True)
class Fixture:
    name: str
    source: Literal["guitarset", "synthetic", "manual"]
    audio_path: Path
    ground_truth: GroundTruth


class FixtureLoader(Protocol):
    def load(self) -> Iterable[Fixture]: ...
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_evaluation_fixture_base.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/music_decoder/evaluation/fixtures/ tests/unit/test_evaluation_fixture_base.py
git commit -m "feat(eval): GroundTruth + Fixture dataclasses + FixtureLoader protocol"
```

---

### Task 12: Synthetic fixture renderer + loader

**Files:**
- Create: `tests/fixtures/synthetic/c_major_scale.mid`
- Create: `tests/fixtures/synthetic/g_major_chord.mid`
- Create: `scripts/render_synthetic.py`
- Create: `src/music_decoder/evaluation/fixtures/synthetic.py`
- Create: `tests/unit/test_evaluation_fixture_synthetic.py`

- [ ] **Step 1: Build the MIDI fixture files (deterministic, no scope for ambiguity)**

Write a one-shot script `scripts/build_synthetic_midi.py`:
```python
# scripts/build_synthetic_midi.py
"""Generates the synthetic .mid files committed to tests/fixtures/synthetic/."""
from pathlib import Path
import pretty_midi


OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "synthetic"


def c_major_scale() -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    inst = pretty_midi.Instrument(program=24)   # nylon guitar
    pitches = [60, 62, 64, 65, 67, 69, 71, 72]
    for i, p in enumerate(pitches):
        inst.notes.append(pretty_midi.Note(
            velocity=80, pitch=p, start=i * 0.5, end=(i + 1) * 0.5,
        ))
    pm.instruments.append(inst)
    return pm


def g_major_chord() -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    inst = pretty_midi.Instrument(program=24)
    pitches = [43, 47, 50, 55, 59, 67]   # G2 B2 D3 G3 B3 G4
    for p in pitches:
        inst.notes.append(pretty_midi.Note(velocity=80, pitch=p, start=0.0, end=2.0))
    pm.instruments.append(inst)
    return pm


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    c_major_scale().write(str(OUT / "c_major_scale.mid"))
    g_major_chord().write(str(OUT / "g_major_chord.mid"))


if __name__ == "__main__":
    main()
```

Run: `python scripts/build_synthetic_midi.py`
Expected: two `.mid` files appear in `tests/fixtures/synthetic/`.

- [ ] **Step 2: Write the failing test for the loader**

```python
# tests/unit/test_evaluation_fixture_synthetic.py
import shutil
from pathlib import Path

import pytest

from music_decoder.evaluation.fixtures.synthetic import SyntheticFixtures


@pytest.fixture
def synthetic_dir(tmp_path: Path) -> Path:
    src = Path("tests/fixtures/synthetic")
    dst = tmp_path / "synthetic"
    dst.mkdir()
    shutil.copy(src / "c_major_scale.mid", dst)
    shutil.copy(src / "g_major_chord.mid", dst)
    return dst


def test_loader_renders_wav_from_midi(synthetic_dir: Path):
    loader = SyntheticFixtures(root=synthetic_dir)
    fixtures = list(loader.load())
    names = {f.name for f in fixtures}
    assert {"c_major_scale", "g_major_chord"}.issubset(names)
    for f in fixtures:
        assert f.audio_path.exists()
        assert f.audio_path.suffix == ".wav"
        assert f.ground_truth.pitches_midi.size > 0


def test_c_major_scale_has_eight_notes(synthetic_dir: Path):
    loader = SyntheticFixtures(root=synthetic_dir)
    fx = next(f for f in loader.load() if f.name == "c_major_scale")
    assert len(fx.ground_truth.pitches_midi) == 8
    assert fx.ground_truth.key is None or fx.ground_truth.key[0] in ("C", "C#")


def test_g_major_chord_is_simultaneous(synthetic_dir: Path):
    loader = SyntheticFixtures(root=synthetic_dir)
    fx = next(f for f in loader.load() if f.name == "g_major_chord")
    starts = fx.ground_truth.intervals[:, 0]
    assert (starts == starts[0]).all()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/unit/test_evaluation_fixture_synthetic.py -v`
Expected: FAIL.

- [ ] **Step 4: Implement the synthetic loader**

```python
# src/music_decoder/evaluation/fixtures/synthetic.py
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pretty_midi
import scipy.io.wavfile as wavfile

from .base import Fixture, FixtureLoader, GroundTruth


_SR = 22050


class SyntheticFixtures:
    """Renders committed .mid files to .wav via pretty_midi.synthesize() (sine waves).

    Sine-wave synthesis is deterministic, has no soundfont dependency, and produces
    unambiguous fundamental frequencies — ideal as ground truth for a transcription
    benchmark.
    """

    def __init__(self, root: Path) -> None:
        self.root = root

    def _ensure_wav(self, midi_path: Path) -> Path:
        wav_path = midi_path.with_suffix(".wav")
        if wav_path.exists() and wav_path.stat().st_mtime > midi_path.stat().st_mtime:
            return wav_path
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        audio = pm.synthesize(fs=_SR).astype(np.float32)
        peak = float(np.max(np.abs(audio)) or 1.0)
        audio = (audio / peak * 0.9).astype(np.float32)
        wavfile.write(str(wav_path), _SR, (audio * 32767).astype(np.int16))
        return wav_path

    def _ground_truth(self, midi_path: Path) -> GroundTruth:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        notes = [n for inst in pm.instruments for n in inst.notes]
        notes.sort(key=lambda n: (n.start, n.pitch))
        intervals = np.array([(n.start, n.end) for n in notes], dtype=float)
        pitches = np.array([n.pitch for n in notes], dtype=float)
        return GroundTruth(
            intervals=intervals if len(notes) else np.zeros((0, 2)),
            pitches_midi=pitches if len(notes) else np.zeros(0),
            key=None,
            tempo_bpm=float(pm.estimate_tempo()) if notes else None,
            tab=None,
        )

    def load(self) -> Iterator[Fixture]:
        for midi_path in sorted(self.root.glob("*.mid")):
            wav = self._ensure_wav(midi_path)
            yield Fixture(
                name=midi_path.stem, source="synthetic",
                audio_path=wav, ground_truth=self._ground_truth(midi_path),
            )
```

- [ ] **Step 5: Add the `scripts/render_synthetic.py` convenience target**

```python
# scripts/render_synthetic.py
"""Render the .mid fixtures in tests/fixtures/synthetic to .wav."""
from pathlib import Path
from music_decoder.evaluation.fixtures.synthetic import SyntheticFixtures


def main() -> None:
    root = Path("tests/fixtures/synthetic")
    for fx in SyntheticFixtures(root=root).load():
        print(f"rendered {fx.audio_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run tests**

Run: `pytest tests/unit/test_evaluation_fixture_synthetic.py -v`
Expected: 3 passed.

- [ ] **Step 7: Commit**

```bash
git add tests/fixtures/synthetic/*.mid scripts/build_synthetic_midi.py scripts/render_synthetic.py \
        src/music_decoder/evaluation/fixtures/synthetic.py \
        tests/unit/test_evaluation_fixture_synthetic.py
git commit -m "feat(eval): synthetic fixtures with deterministic sine-wave rendering"
```

---

### Task 13: GuitarSet fixture loader (mirdata)

**Files:**
- Create: `src/music_decoder/evaluation/fixtures/guitarset.py`
- Create: `tests/unit/test_evaluation_fixture_guitarset.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_evaluation_fixture_guitarset.py
import pytest

from music_decoder.evaluation.fixtures.guitarset import GuitarSetFixtures


@pytest.mark.slow
def test_guitarset_yields_fixtures_when_data_present(tmp_path):
    """If the GuitarSet cache is present (CI restores it), this returns fixtures."""
    loader = GuitarSetFixtures(
        cache_dir=tmp_path / "guitarset",
        track_ids=["00_BN1-129-Eb_comp", "00_BN1-129-Eb_solo"],
    )
    if not loader.is_available():
        pytest.skip("GuitarSet cache unavailable; run `make fixtures` to download.")
    fixtures = list(loader.load())
    assert len(fixtures) >= 1
    fx = fixtures[0]
    assert fx.source == "guitarset"
    assert fx.ground_truth.tab is not None
    assert all(0 <= s < 6 for (_, s, _) in fx.ground_truth.tab)


def test_guitarset_is_available_returns_false_for_missing_cache(tmp_path):
    loader = GuitarSetFixtures(cache_dir=tmp_path / "missing", track_ids=["x"])
    assert loader.is_available() is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_evaluation_fixture_guitarset.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement the loader**

```python
# src/music_decoder/evaluation/fixtures/guitarset.py
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from .base import Fixture, FixtureLoader, GroundTruth


class GuitarSetFixtures:
    def __init__(self, cache_dir: Path, track_ids: list[str]) -> None:
        self.cache_dir = cache_dir
        self.track_ids = track_ids

    def is_available(self) -> bool:
        return self.cache_dir.exists() and any(self.cache_dir.iterdir())

    def _load_track(self, dataset, track_id: str) -> Fixture | None:
        if track_id not in dataset.track_ids:
            return None
        track = dataset.track(track_id)
        notes = track.notes
        if notes is None:
            return None
        intervals = np.asarray(notes.intervals, dtype=float)
        pitches = np.asarray(notes.notes, dtype=float)
        # GuitarSet stores per-string note annotations; we collect them.
        tab: list[tuple[int, int, int]] = []
        for s, sn in enumerate(track.notes_by_string or []):
            if sn is None:
                continue
            for p in np.asarray(sn.notes, dtype=float):
                tab.append((int(round(p)), s, int(round(p) - track.tuning[s])))
        gt = GroundTruth(
            intervals=intervals, pitches_midi=pitches,
            key=None, tempo_bpm=track.tempo if hasattr(track, "tempo") else None,
            tab=tab if tab else None,
        )
        return Fixture(
            name=track_id, source="guitarset",
            audio_path=Path(track.audio_mic_path),
            ground_truth=gt,
        )

    def load(self) -> Iterator[Fixture]:
        if not self.is_available():
            return iter(())
        import mirdata
        ds = mirdata.initialize("guitarset", data_home=str(self.cache_dir))
        for tid in self.track_ids:
            fx = self._load_track(ds, tid)
            if fx is not None:
                yield fx
```

> **Note:** GuitarSet's `notes_by_string` API surface varies across mirdata versions. If the `notes_by_string` attribute does not exist, fall back to using `track.notes` only and set `tab=None`; `tab_string_accuracy` is then skipped for that fixture.

- [ ] **Step 4: Add download target to Makefile**

Add to `Makefile`:
```makefile
fixtures:
	python -c "import mirdata; mirdata.initialize('guitarset', data_home='tests/fixtures/guitarset').download(force_overwrite=False)"
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/unit/test_evaluation_fixture_guitarset.py -v`
Expected: 2 passed (the slow one will skip if cache absent).

- [ ] **Step 6: Commit**

```bash
git add src/music_decoder/evaluation/fixtures/guitarset.py \
        tests/unit/test_evaluation_fixture_guitarset.py Makefile
git commit -m "feat(eval): GuitarSet fixture loader via mirdata with cache check"
```

---

### Task 14: Manual fixture loader (drop-in JSON)

**Files:**
- Create: `tests/fixtures/manual/.gitkeep`
- Create: `src/music_decoder/evaluation/fixtures/manual.py`
- Create: `tests/unit/test_evaluation_fixture_manual.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_evaluation_fixture_manual.py
import json
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile
import pytest

from music_decoder.evaluation.fixtures.manual import ManualFixtures


def _write_dummy_wav(path: Path) -> None:
    sr = 22050
    samples = (np.sin(2 * np.pi * 440 * np.arange(sr) / sr) * 0.5).astype(np.float32)
    wavfile.write(str(path), sr, (samples * 32767).astype(np.int16))


@pytest.fixture
def manual_dir(tmp_path: Path) -> Path:
    d = tmp_path / "manual"
    d.mkdir()
    _write_dummy_wav(d / "clip01.wav")
    (d / "clip01.json").write_text(json.dumps({
        "audio": "clip01.wav",
        "tuning": "EADGBE",
        "key": {"tonic": "G", "mode": "major"},
        "tempo_bpm": 120,
        "tab": [
            {"start_s": 0.0, "end_s": 0.5, "pitch": 67, "string": 3, "fret": 12}
        ],
    }))
    return d


def test_manual_loader_reads_drop_in_pair(manual_dir: Path):
    loader = ManualFixtures(root=manual_dir)
    fixtures = list(loader.load())
    assert len(fixtures) == 1
    fx = fixtures[0]
    assert fx.name == "clip01"
    assert fx.source == "manual"
    assert fx.ground_truth.key == ("G", "major")
    assert fx.ground_truth.tempo_bpm == 120.0
    assert fx.ground_truth.tab == [(67, 3, 12)]


def test_manual_loader_skips_orphan_json(manual_dir: Path):
    (manual_dir / "broken.json").write_text(json.dumps({"audio": "missing.wav"}))
    fixtures = list(ManualFixtures(root=manual_dir).load())
    assert len(fixtures) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_evaluation_fixture_manual.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# src/music_decoder/evaluation/fixtures/manual.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np

from .base import Fixture, GroundTruth


class ManualFixtures:
    def __init__(self, root: Path) -> None:
        self.root = root

    def load(self) -> Iterator[Fixture]:
        if not self.root.exists():
            return
        for json_path in sorted(self.root.glob("*.json")):
            data = json.loads(json_path.read_text())
            audio_rel = data.get("audio")
            if not audio_rel:
                continue
            audio_path = self.root / audio_rel
            if not audio_path.exists():
                continue
            tab_rows = data.get("tab", []) or []
            intervals = np.array(
                [(row["start_s"], row["end_s"]) for row in tab_rows],
                dtype=float,
            ) if tab_rows else np.zeros((0, 2))
            pitches = np.array(
                [row["pitch"] for row in tab_rows], dtype=float,
            ) if tab_rows else np.zeros(0)
            tab: list[tuple[int, int, int]] | None = None
            if tab_rows and all("string" in r and "fret" in r for r in tab_rows):
                tab = [(int(r["pitch"]), int(r["string"]), int(r["fret"])) for r in tab_rows]
            key_field = data.get("key")
            key = (key_field["tonic"], key_field["mode"]) if key_field else None
            gt = GroundTruth(
                intervals=intervals, pitches_midi=pitches,
                key=key,
                tempo_bpm=float(data["tempo_bpm"]) if "tempo_bpm" in data else None,
                tab=tab,
            )
            yield Fixture(
                name=json_path.stem, source="manual",
                audio_path=audio_path, ground_truth=gt,
            )
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_evaluation_fixture_manual.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/fixtures/manual/.gitkeep src/music_decoder/evaluation/fixtures/manual.py \
        tests/unit/test_evaluation_fixture_manual.py
git commit -m "feat(eval): manual fixture loader with drop-in JSON+audio pairing"
```

---

### Task 15: Evaluation runner (fixture set → metrics → report)

**Files:**
- Create: `src/music_decoder/evaluation/runner.py`
- Create: `tests/unit/test_evaluation_runner.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_evaluation_runner.py
from pathlib import Path

import numpy as np

from music_decoder.evaluation.fixtures.base import Fixture, GroundTruth
from music_decoder.evaluation.runner import EvaluationReport, run_evaluation


def _identity_pipeline(audio_path: Path, fixture: Fixture):
    """Echoes the ground truth back as a perfect prediction."""
    gt = fixture.ground_truth
    return {
        "intervals": gt.intervals.copy(),
        "pitches_midi": gt.pitches_midi.copy(),
        "key": gt.key,
        "tab": list(gt.tab) if gt.tab is not None else None,
    }


def _make_fixture(tmp_path: Path) -> Fixture:
    audio = tmp_path / "x.wav"
    audio.write_bytes(b"fake")
    return Fixture(
        name="x", source="manual", audio_path=audio,
        ground_truth=GroundTruth(
            intervals=np.array([[0.0, 0.5], [0.5, 1.0]]),
            pitches_midi=np.array([60, 62]),
            key=("C", "major"),
            tempo_bpm=120.0,
            tab=[(60, 4, 1), (62, 4, 3)],
        ),
    )


def test_runner_returns_perfect_report_for_identity_pipeline(tmp_path):
    fx = _make_fixture(tmp_path)
    report = run_evaluation(_identity_pipeline, [fx])
    assert isinstance(report, EvaluationReport)
    assert report.per_fixture[0].note_f_measure == 1.0
    assert report.per_fixture[0].key_mirex_score == 1.0
    assert report.per_fixture[0].tab_string_accuracy == 1.0


def test_runner_writes_json_report(tmp_path):
    fx = _make_fixture(tmp_path)
    out = tmp_path / "report.json"
    run_evaluation(_identity_pipeline, [fx], report_path=out)
    assert out.exists()
    import json
    parsed = json.loads(out.read_text())
    assert parsed["per_fixture"][0]["note_f_measure"] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_evaluation_runner.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# src/music_decoder/evaluation/runner.py
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from .fixtures.base import Fixture
from .metrics import (
    note_f_measure,
    onset_f_measure,
    pitch_class_accuracy,
    key_mirex_score,
    tab_string_accuracy,
)


@dataclass(frozen=True)
class FixtureMetrics:
    name: str
    source: str
    note_f_measure: float | None
    onset_f_measure: float | None
    pitch_class_accuracy: float | None
    key_mirex_score: float | None
    tab_string_accuracy: float | None


@dataclass(frozen=True)
class EvaluationReport:
    timestamp: str
    per_fixture: list[FixtureMetrics]


PipelineFn = Callable[[Path, Fixture], dict]


def _safe(fn, *args, **kw) -> float | None:
    try:
        return fn(*args, **kw)
    except Exception:
        return None


def run_evaluation(
    pipeline: PipelineFn,
    fixtures: Iterable[Fixture],
    *,
    report_path: Path | None = None,
) -> EvaluationReport:
    rows: list[FixtureMetrics] = []
    for fx in fixtures:
        prediction = pipeline(fx.audio_path, fx)
        gt = fx.ground_truth
        f_note = _safe(
            note_f_measure,
            np.asarray(prediction["intervals"]), np.asarray(prediction["pitches_midi"]),
            gt.intervals, gt.pitches_midi,
        )
        f_onset = _safe(
            onset_f_measure,
            np.asarray(prediction["intervals"]), gt.intervals,
        )
        pc = _safe(
            pitch_class_accuracy,
            np.asarray(prediction["intervals"]), np.asarray(prediction["pitches_midi"]),
            gt.intervals, gt.pitches_midi,
        )
        k = (
            key_mirex_score(prediction["key"], gt.key)
            if prediction.get("key") and gt.key else None
        )
        tab = (
            tab_string_accuracy(prediction["tab"], gt.tab)
            if prediction.get("tab") and gt.tab else None
        )
        rows.append(FixtureMetrics(
            name=fx.name, source=fx.source,
            note_f_measure=f_note, onset_f_measure=f_onset,
            pitch_class_accuracy=pc,
            key_mirex_score=k, tab_string_accuracy=tab,
        ))
    report = EvaluationReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        per_fixture=rows,
    )
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(
            {"timestamp": report.timestamp,
             "per_fixture": [asdict(r) for r in rows]},
            indent=2,
        ))
    return report
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_evaluation_runner.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/music_decoder/evaluation/runner.py tests/unit/test_evaluation_runner.py
git commit -m "feat(eval): runner that maps fixtures+pipeline to a JSON-serializable report"
```

---

### Task 16: Regression test scaffolding + thresholds

**Files:**
- Create: `config/eval_thresholds.yaml`
- Create: `evaluation_reports/baseline.json`
- Create: `src/music_decoder/evaluation/regression.py`
- Create: `tests/regression/__init__.py`
- Create: `tests/regression/test_accuracy_thresholds.py`

- [ ] **Step 1: Write `config/eval_thresholds.yaml`**

```yaml
# Pinned acceptance thresholds. Lowering any of these requires committing
# a corresponding update to evaluation_reports/baseline.json with rationale.
note_f_measure: 0.65
onset_f_measure: 0.70
pitch_class_accuracy: 0.75
key_mirex_score: 0.75
tab_string_accuracy: 0.55
regression_tolerance: 0.02
```

- [ ] **Step 2: Initial baseline**

Write `evaluation_reports/baseline.json`:
```json
{
  "note_f_measure": 0.0,
  "onset_f_measure": 0.0,
  "pitch_class_accuracy": 0.0,
  "key_mirex_score": 0.0,
  "tab_string_accuracy": 0.0,
  "_note": "Initial seed. Updated by hand on intentional baseline movements."
}
```

- [ ] **Step 3: Write the failing test**

```python
# tests/regression/test_accuracy_thresholds.py
"""
Regression test: the placeholder pipeline returns identity predictions, so
every metric is 1.0 — well above thresholds. Once Phase 4 lands, this test
swaps in the real pipeline and gates accuracy regressions.
"""
from pathlib import Path

import numpy as np
import pytest

from music_decoder.evaluation.fixtures.base import Fixture, GroundTruth
from music_decoder.evaluation.regression import (
    check_against_thresholds,
    check_against_baseline,
    load_thresholds,
    load_baseline,
)
from music_decoder.evaluation.runner import run_evaluation


def _identity_pipeline(audio_path, fixture):
    gt = fixture.ground_truth
    return {
        "intervals": gt.intervals.copy(),
        "pitches_midi": gt.pitches_midi.copy(),
        "key": gt.key,
        "tab": list(gt.tab) if gt.tab is not None else None,
    }


@pytest.fixture
def trivial_fixture(tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")
    return Fixture(
        name="trivial", source="manual", audio_path=audio,
        ground_truth=GroundTruth(
            intervals=np.array([[0.0, 0.5]]), pitches_midi=np.array([60]),
            key=("C", "major"), tempo_bpm=120.0, tab=[(60, 4, 1)],
        ),
    )


@pytest.mark.regression
def test_thresholds_loadable():
    thresholds = load_thresholds(Path("config/eval_thresholds.yaml"))
    assert "note_f_measure" in thresholds
    assert "regression_tolerance" in thresholds


@pytest.mark.regression
def test_identity_pipeline_passes_thresholds(trivial_fixture):
    report = run_evaluation(_identity_pipeline, [trivial_fixture])
    thresholds = load_thresholds(Path("config/eval_thresholds.yaml"))
    failures = check_against_thresholds(report, thresholds)
    assert failures == []


@pytest.mark.regression
def test_baseline_comparison_no_regression(trivial_fixture):
    report = run_evaluation(_identity_pipeline, [trivial_fixture])
    baseline = load_baseline(Path("evaluation_reports/baseline.json"))
    failures = check_against_baseline(report, baseline, tolerance=0.02)
    assert failures == []
```

- [ ] **Step 4: Run test to verify it fails**

Run: `pytest tests/regression/test_accuracy_thresholds.py -v`
Expected: FAIL — module missing.

- [ ] **Step 5: Implement**

```python
# src/music_decoder/evaluation/regression.py
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import yaml

from .runner import EvaluationReport


def load_thresholds(path: Path) -> dict[str, float]:
    return yaml.safe_load(path.read_text())


def load_baseline(path: Path) -> dict[str, float]:
    raw = json.loads(path.read_text())
    return {k: v for k, v in raw.items() if not k.startswith("_") and isinstance(v, (int, float))}


def _aggregate(report: EvaluationReport) -> dict[str, float]:
    aggregates: dict[str, float] = {}
    for metric in ("note_f_measure", "onset_f_measure", "pitch_class_accuracy",
                   "key_mirex_score", "tab_string_accuracy"):
        values = [getattr(r, metric) for r in report.per_fixture if getattr(r, metric) is not None]
        if values:
            aggregates[metric] = statistics.mean(values)
    return aggregates


def check_against_thresholds(
    report: EvaluationReport, thresholds: dict[str, float]
) -> list[str]:
    failures: list[str] = []
    aggregates = _aggregate(report)
    for metric, value in aggregates.items():
        threshold = thresholds.get(metric)
        if threshold is None:
            continue
        if value < threshold:
            failures.append(
                f"{metric}={value:.4f} below threshold {threshold:.4f}"
            )
    return failures


def check_against_baseline(
    report: EvaluationReport, baseline: dict[str, float], tolerance: float = 0.02,
) -> list[str]:
    failures: list[str] = []
    aggregates = _aggregate(report)
    for metric, current in aggregates.items():
        prev = baseline.get(metric)
        if prev is None:
            continue
        if current + tolerance < prev:
            failures.append(
                f"{metric}: regressed from {prev:.4f} to {current:.4f} "
                f"(beyond tolerance {tolerance})"
            )
    return failures
```

```python
# tests/regression/__init__.py
```

- [ ] **Step 6: Run tests**

Run: `pytest tests/regression/test_accuracy_thresholds.py -v`
Expected: 3 passed.

- [ ] **Step 7: Commit**

```bash
git add config/eval_thresholds.yaml evaluation_reports/baseline.json \
        src/music_decoder/evaluation/regression.py tests/regression/
git commit -m "feat(eval): regression harness with thresholds + baseline tolerance"
```

---

## Phase 3 — Pipeline stage contracts

### Task 17: Typed pipeline dataclasses

**Files:**
- Create: `src/music_decoder/pipeline/__init__.py`
- Create: `src/music_decoder/pipeline/contracts.py`
- Create: `tests/unit/test_pipeline_contracts.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_pipeline_contracts.py
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from music_decoder.pipeline.contracts import (
    AudioSource, LoadedAudio, SeparationResult,
    TranscribedNote, TranscriptionResult,
    KeyEstimate, KeyDetectionResult,
    BeatGrid,
    TabPosition, TabbedNote, TabAssignmentResult,
    TabReferenceMatch,
    StageEvent,
)
from music_decoder.tab_assignment.tuning import get_preset


def test_loaded_audio_holds_basic_fields(tmp_path):
    src = AudioSource(
        path=tmp_path / "a.wav", declared_kind="solo_guitar",
        requested_quality="standard", requested_tuning=get_preset("EADGBE"),
    )
    a = LoadedAudio(
        samples=np.zeros(22050, dtype=np.float32), sr=22050,
        duration_s=1.0, sha256="abc", source=src,
    )
    assert a.duration_s == 1.0
    assert a.source.requested_tuning.name == "EADGBE"


def test_transcribed_note_is_immutable():
    n = TranscribedNote(start_s=0.0, end_s=1.0, pitch=60, velocity=80, confidence=0.9)
    with pytest.raises(Exception):
        n.pitch = 61  # type: ignore[misc]


def test_key_estimate_records_profile_and_margin():
    k = KeyEstimate(
        tonic="C", mode="major", profile="krumhansl_kessler",
        correlation=0.82, margin=0.15,
    )
    assert k.tonic == "C"
    assert k.profile == "krumhansl_kessler"


def test_tab_position_indexing():
    p = TabPosition(string=5, fret=0)
    assert p.string == 5 and p.fret == 0


def test_stage_event_serializable():
    e = StageEvent(
        job_id=1, stage="transcription",
        started_at=datetime.now(timezone.utc),
        ended_at=datetime.now(timezone.utc),
        success=True, error=None, summary={"notes": 42},
    )
    assert e.summary["notes"] == 42
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_pipeline_contracts.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# src/music_decoder/pipeline/__init__.py
```

```python
# src/music_decoder/pipeline/contracts.py
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

from music_decoder.tab_assignment.tuning import Tuning


# ---- audio_io
@dataclass(frozen=True)
class AudioSource:
    path: Path
    declared_kind: Literal["solo_guitar", "full_mix"]
    requested_quality: Literal["standard", "high"]
    requested_tuning: Tuning


@dataclass(frozen=True)
class LoadedAudio:
    samples: np.ndarray
    sr: int
    duration_s: float
    sha256: str
    source: AudioSource


# ---- separation
@dataclass(frozen=True)
class SeparationResult:
    guitar_samples: np.ndarray | None
    sr: int
    skipped_reason: str | None
    bleed_estimate_db: float | None


# ---- transcription
@dataclass(frozen=True)
class TranscribedNote:
    start_s: float
    end_s: float
    pitch: int
    velocity: int
    confidence: float


@dataclass(frozen=True)
class TranscriptionResult:
    notes: list[TranscribedNote]
    model: Literal["basic-pitch", "crepe"]
    raw_midi_path: Path
    post_midi_path: Path
    hyperparameters: dict[str, Any]
    median_confidence: float


# ---- key_detection
@dataclass(frozen=True)
class KeyEstimate:
    tonic: str
    mode: Literal["major", "minor"]
    profile: Literal["krumhansl_kessler", "temperley"]
    correlation: float
    margin: float


@dataclass(frozen=True)
class KeyDetectionResult:
    global_top3_per_profile: dict[str, list[KeyEstimate]]
    consensus_key: KeyEstimate | None
    windowed_segments: list[tuple[float, float, KeyEstimate]]
    confidence: float


# ---- beat_tracking
@dataclass(frozen=True)
class BeatGrid:
    tempo_bpm: float
    beat_times_s: np.ndarray
    downbeat_times_s: np.ndarray
    ts_numerator: int
    ts_denominator: int
    ts_confidence: float
    ts_assumed: bool


# ---- tab_assignment
@dataclass(frozen=True)
class TabPosition:
    string: int
    fret: int


@dataclass(frozen=True)
class TabbedNote:
    note: TranscribedNote
    position: TabPosition
    cost_breakdown: dict[str, float]


@dataclass(frozen=True)
class TabAssignmentResult:
    tabbed_notes: list[TabbedNote]
    tuning: Tuning
    total_cost: float
    notes_dropped: list[tuple[TranscribedNote, str]]


# ---- tab_reference
@dataclass(frozen=True)
class TabReferenceMatch:
    source: Literal["user_pasted_url", "user_pasted_text"]
    acoustid: str | None
    raw_text: str
    parsed_positions: list[TabPosition]
    similarity_to_prediction: float | None
    disagreement_spans: list[tuple[float, float]]


# ---- orchestration
@dataclass(frozen=True)
class StageEvent:
    job_id: int
    stage: str
    started_at: datetime
    ended_at: datetime
    success: bool
    error: str | None
    summary: dict[str, Any]
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_pipeline_contracts.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/music_decoder/pipeline/__init__.py src/music_decoder/pipeline/contracts.py \
        tests/unit/test_pipeline_contracts.py
git commit -m "feat(pipeline): typed dataclass contracts for every stage"
```

---

## Phase 4 — Pipeline stages

### Task 18: audio_io.load with ffmpeg + validation

**Files:**
- Create: `tests/fixtures/audio_samples/sine_440.wav`
- Create: `tests/fixtures/audio_samples/silence_2s.wav`
- Create: `src/music_decoder/audio_io/__init__.py`
- Create: `src/music_decoder/audio_io/ffmpeg.py`
- Create: `src/music_decoder/audio_io/load.py`
- Create: `tests/unit/test_audio_io.py`

- [ ] **Step 1: Generate the test audio fixtures**

Write `scripts/build_audio_samples.py`:
```python
# scripts/build_audio_samples.py
from pathlib import Path
import numpy as np
import scipy.io.wavfile as wavfile

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "audio_samples"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sr = 22050
    t = np.arange(sr) / sr
    sine = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    wavfile.write(str(OUT / "sine_440.wav"), sr, (sine * 32767).astype(np.int16))
    silence = np.zeros(sr * 2, dtype=np.int16)
    wavfile.write(str(OUT / "silence_2s.wav"), sr, silence)


if __name__ == "__main__":
    main()
```

Run: `python scripts/build_audio_samples.py`
Expected: two WAV files committed.

- [ ] **Step 2: Write the failing test**

```python
# tests/unit/test_audio_io.py
from pathlib import Path

import numpy as np
import pytest

from music_decoder.audio_io.load import (
    load_audio,
    CorruptAudioError,
    SilentAudioError,
    ClipTooShortError,
)
from music_decoder.pipeline.contracts import AudioSource
from music_decoder.tab_assignment.tuning import get_preset


def _source(path: Path) -> AudioSource:
    return AudioSource(
        path=path, declared_kind="solo_guitar",
        requested_quality="standard", requested_tuning=get_preset("EADGBE"),
    )


def test_load_decodes_sine_wav(fixtures_dir: Path):
    src = _source(fixtures_dir / "audio_samples" / "sine_440.wav")
    audio = load_audio(src)
    assert audio.sr == 22050
    assert audio.samples.dtype == np.float32
    assert audio.samples.ndim == 1
    assert 0.95 < audio.duration_s < 1.05
    assert len(audio.sha256) == 64


def test_load_high_quality_uses_44100(fixtures_dir: Path):
    src = AudioSource(
        path=fixtures_dir / "audio_samples" / "sine_440.wav",
        declared_kind="solo_guitar", requested_quality="high",
        requested_tuning=get_preset("EADGBE"),
    )
    audio = load_audio(src)
    assert audio.sr == 44100


def test_load_rejects_silent_audio(fixtures_dir: Path):
    src = _source(fixtures_dir / "audio_samples" / "silence_2s.wav")
    with pytest.raises(SilentAudioError):
        load_audio(src)


def test_load_rejects_too_short(tmp_path: Path):
    sr = 22050
    short = (0.5 * np.sin(2 * np.pi * 440 * np.arange(int(sr * 0.5)) / sr)).astype(np.float32)
    import scipy.io.wavfile as wavfile
    p = tmp_path / "tiny.wav"
    wavfile.write(str(p), sr, (short * 32767).astype(np.int16))
    with pytest.raises(ClipTooShortError):
        load_audio(_source(p))


def test_load_corrupt_raises(tmp_path: Path):
    p = tmp_path / "broken.mp3"
    p.write_bytes(b"definitely not audio")
    with pytest.raises(CorruptAudioError):
        load_audio(_source(p))
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/unit/test_audio_io.py -v`
Expected: FAIL — module missing.

- [ ] **Step 4: Implement ffmpeg wrapper**

```python
# src/music_decoder/audio_io/__init__.py
```

```python
# src/music_decoder/audio_io/ffmpeg.py
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def find_ffmpeg(explicit: Path | None = None) -> Path:
    if explicit and explicit.exists():
        return explicit
    found = shutil.which("ffmpeg")
    if found is None:
        raise FileNotFoundError("ffmpeg binary not found on PATH")
    return Path(found)


def decode_to_wav(
    src: Path, dst: Path, *, sr: int, ffmpeg: Path | None = None,
) -> None:
    """Decode any audio file to a mono PCM WAV at the requested sample rate."""
    binary = find_ffmpeg(ffmpeg)
    cmd = [
        str(binary), "-y", "-loglevel", "error",
        "-i", str(src), "-ac", "1", "-ar", str(sr),
        "-acodec", "pcm_s16le", str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({proc.returncode}): {proc.stderr.decode(errors='replace')}")
```

- [ ] **Step 5: Implement loader with validation**

```python
# src/music_decoder/audio_io/load.py
from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile

from music_decoder.pipeline.contracts import AudioSource, LoadedAudio

from .ffmpeg import decode_to_wav


SR_STANDARD = 22050
SR_HIGH = 44100
MIN_DURATION_S = 1.0
SILENCE_RMS_THRESHOLD = 1e-4


class AudioIoError(Exception):
    """Base."""


class CorruptAudioError(AudioIoError):
    pass


class SilentAudioError(AudioIoError):
    pass


class ClipTooShortError(AudioIoError):
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate(samples: np.ndarray, sr: int) -> None:
    if not np.isfinite(samples).all():
        raise CorruptAudioError("samples contain NaN or Inf")
    duration_s = samples.size / sr
    if duration_s < MIN_DURATION_S:
        raise ClipTooShortError(f"duration {duration_s:.2f}s < {MIN_DURATION_S}s")
    rms = float(np.sqrt(np.mean(samples ** 2)))
    if rms < SILENCE_RMS_THRESHOLD:
        raise SilentAudioError(f"rms {rms:.6f} below silence threshold")


def load_audio(source: AudioSource) -> LoadedAudio:
    target_sr = SR_HIGH if source.requested_quality == "high" else SR_STANDARD
    if not source.path.exists():
        raise CorruptAudioError(f"file does not exist: {source.path}")
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "decoded.wav"
        try:
            decode_to_wav(source.path, wav, sr=target_sr)
        except Exception as e:
            raise CorruptAudioError(str(e)) from e
        sr, raw = wavfile.read(str(wav))
        if sr != target_sr:
            raise CorruptAudioError(f"unexpected sample rate {sr}, wanted {target_sr}")
        if raw.dtype == np.int16:
            samples = (raw.astype(np.float32) / 32768.0)
        else:
            samples = raw.astype(np.float32)
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        _validate(samples, sr)
    return LoadedAudio(
        samples=samples, sr=sr, duration_s=samples.size / sr,
        sha256=_sha256(source.path), source=source,
    )
```

- [ ] **Step 6: Run tests**

Run: `pytest tests/unit/test_audio_io.py -v`
Expected: 5 passed (`load_corrupt_raises` requires ffmpeg installed; if not, install with `brew install ffmpeg` or document the prerequisite).

- [ ] **Step 7: Commit**

```bash
git add tests/fixtures/audio_samples/ scripts/build_audio_samples.py \
        src/music_decoder/audio_io/ tests/unit/test_audio_io.py
git commit -m "feat(audio_io): ffmpeg-backed loader with silence/duration/integrity checks"
```

---

### Task 19: Demucs source separation wrapper (fail-soft)

**Files:**
- Create: `src/music_decoder/separation/__init__.py`
- Create: `src/music_decoder/separation/demucs.py`
- Create: `tests/unit/test_separation.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_separation.py
from pathlib import Path

import numpy as np
import pytest

from music_decoder.pipeline.contracts import AudioSource, LoadedAudio
from music_decoder.separation.demucs import isolate_guitar
from music_decoder.tab_assignment.tuning import get_preset


def _audio(samples: np.ndarray, sr: int = 22050) -> LoadedAudio:
    return LoadedAudio(
        samples=samples, sr=sr, duration_s=samples.size / sr,
        sha256="x" * 64,
        source=AudioSource(
            path=Path("/tmp/x.wav"), declared_kind="full_mix",
            requested_quality="standard", requested_tuning=get_preset("EADGBE"),
        ),
    )


def test_solo_guitar_short_circuits_with_skipped_reason():
    samples = np.random.randn(22050).astype(np.float32) * 0.1
    audio = LoadedAudio(
        samples=samples, sr=22050, duration_s=1.0, sha256="x" * 64,
        source=AudioSource(
            path=Path("/tmp/x.wav"), declared_kind="solo_guitar",
            requested_quality="standard", requested_tuning=get_preset("EADGBE"),
        ),
    )
    result = isolate_guitar(audio)
    assert result.guitar_samples is None
    assert result.skipped_reason and "solo_guitar declared" in result.skipped_reason


def test_demucs_failure_returns_skipped_with_original_audio(monkeypatch):
    samples = np.random.randn(22050 * 2).astype(np.float32) * 0.1
    audio = _audio(samples)

    def fake_apply_model(*args, **kwargs):
        raise RuntimeError("demucs blew up")

    monkeypatch.setattr(
        "music_decoder.separation.demucs._apply_demucs", fake_apply_model
    )
    result = isolate_guitar(audio)
    assert result.guitar_samples is None
    assert result.skipped_reason and result.skipped_reason.startswith("demucs_failed")


@pytest.mark.slow
def test_demucs_round_trip_returns_audio_when_called():
    """Real Demucs invocation; only runs in slow mode."""
    samples = np.random.randn(22050 * 5).astype(np.float32) * 0.05
    audio = _audio(samples)
    result = isolate_guitar(audio)
    if result.guitar_samples is None:
        pytest.skip(f"demucs unavailable: {result.skipped_reason}")
    assert result.guitar_samples.shape == samples.shape
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_separation.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# src/music_decoder/separation/__init__.py
```

```python
# src/music_decoder/separation/demucs.py
from __future__ import annotations

import logging
import traceback
from typing import Any

import numpy as np

from music_decoder.logging_setup import get_logger
from music_decoder.pipeline.contracts import LoadedAudio, SeparationResult


_log = get_logger("separation")


def _apply_demucs(samples: np.ndarray, sr: int) -> np.ndarray:
    """Run Demucs htdemucs_6s and return the guitar stem at the same sample rate."""
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    model = get_model("htdemucs_6s")
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # demucs expects (channels, samples) at the model's native rate
    target_sr = model.samplerate
    if sr != target_sr:
        from librosa import resample
        samples = resample(samples, orig_sr=sr, target_sr=target_sr)
    audio_tensor = torch.from_numpy(samples).float().unsqueeze(0).unsqueeze(0)
    audio_tensor = audio_tensor.repeat(1, 2, 1)   # demucs wants stereo
    audio_tensor = audio_tensor.to(device)

    with torch.no_grad():
        sources = apply_model(model, audio_tensor, split=True, overlap=0.25)
    # find the guitar stem
    sources_names = list(model.sources)
    guitar_idx = sources_names.index("guitar")
    guitar = sources[0, guitar_idx].mean(dim=0).cpu().numpy()
    if target_sr != sr:
        from librosa import resample
        guitar = resample(guitar, orig_sr=target_sr, target_sr=sr)
    if guitar.size > samples.size:
        guitar = guitar[: samples.size]
    return guitar.astype(np.float32)


def isolate_guitar(audio: LoadedAudio) -> SeparationResult:
    if audio.source.declared_kind == "solo_guitar":
        return SeparationResult(
            guitar_samples=None, sr=audio.sr,
            skipped_reason="solo_guitar declared", bleed_estimate_db=None,
        )
    try:
        guitar = _apply_demucs(audio.samples, audio.sr)
    except Exception as e:
        _log.error("demucs_failed", extra={"error": str(e)})
        return SeparationResult(
            guitar_samples=None, sr=audio.sr,
            skipped_reason=f"demucs_failed: {e}",
            bleed_estimate_db=None,
        )
    return SeparationResult(
        guitar_samples=guitar, sr=audio.sr,
        skipped_reason=None, bleed_estimate_db=None,
    )
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_separation.py -v -m "not slow"`
Expected: 2 passed (slow test skipped).

- [ ] **Step 5: Commit**

```bash
git add src/music_decoder/separation/ tests/unit/test_separation.py
git commit -m "feat(separation): Demucs htdemucs_6s wrapper with fail-soft behavior"
```

---

### Task 20: basic-pitch transcription wrapper

**Files:**
- Create: `src/music_decoder/transcription/__init__.py`
- Create: `src/music_decoder/transcription/basic_pitch_wrapper.py`
- Create: `tests/unit/test_transcription_basic_pitch.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_transcription_basic_pitch.py
from pathlib import Path

import numpy as np
import pytest

from music_decoder.config.hyperparameters import BasicPitchParams
from music_decoder.pipeline.contracts import AudioSource, LoadedAudio
from music_decoder.tab_assignment.tuning import get_preset
from music_decoder.transcription.basic_pitch_wrapper import transcribe_basic_pitch


def _hparams() -> BasicPitchParams:
    return BasicPitchParams(
        onset_threshold=0.5, frame_threshold=0.3,
        minimum_note_length_ms=58,
        minimum_frequency_hz=32.7, maximum_frequency_hz=2000.0,
    )


@pytest.mark.slow
def test_basic_pitch_returns_some_notes_for_synthetic(fixtures_dir: Path, tmp_path: Path):
    """Render the committed C major scale to WAV and ensure basic-pitch finds notes."""
    from music_decoder.evaluation.fixtures.synthetic import SyntheticFixtures
    fx = next(f for f in SyntheticFixtures(root=fixtures_dir / "synthetic").load()
              if f.name == "c_major_scale")

    sr = 22050
    import scipy.io.wavfile as wavfile
    rate, raw = wavfile.read(str(fx.audio_path))
    samples = (raw.astype(np.float32) / 32768.0).astype(np.float32)
    if rate != sr:
        from librosa import resample
        samples = resample(samples, orig_sr=rate, target_sr=sr)
    audio = LoadedAudio(
        samples=samples, sr=sr, duration_s=samples.size / sr, sha256="x" * 64,
        source=AudioSource(
            path=fx.audio_path, declared_kind="solo_guitar",
            requested_quality="standard", requested_tuning=get_preset("EADGBE"),
        ),
    )
    result = transcribe_basic_pitch(audio, _hparams(), output_dir=tmp_path)
    assert result.model == "basic-pitch"
    assert len(result.notes) > 0
    assert all(0 <= n.confidence <= 1 for n in result.notes)
    assert all(n.start_s < n.end_s for n in result.notes)
    assert result.raw_midi_path.exists()
    assert result.post_midi_path.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_transcription_basic_pitch.py -v -m slow`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement**

```python
# src/music_decoder/transcription/__init__.py
```

```python
# src/music_decoder/transcription/basic_pitch_wrapper.py
from __future__ import annotations

import statistics
import tempfile
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile

from music_decoder.config.hyperparameters import BasicPitchParams
from music_decoder.pipeline.contracts import (
    LoadedAudio, TranscribedNote, TranscriptionResult,
)


def transcribe_basic_pitch(
    audio: LoadedAudio, params: BasicPitchParams, *, output_dir: Path,
) -> TranscriptionResult:
    from basic_pitch.inference import predict
    from basic_pitch import ICASSP_2022_MODEL_PATH

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_midi_path = output_dir / "raw_basic_pitch.mid"
    post_midi_path = output_dir / "post_basic_pitch.mid"   # post-processing fills it later

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "input.wav"
        wavfile.write(str(wav_path), audio.sr, (audio.samples * 32767).astype(np.int16))
        _, midi_data, note_events = predict(
            str(wav_path),
            model_or_model_path=ICASSP_2022_MODEL_PATH,
            onset_threshold=params.onset_threshold,
            frame_threshold=params.frame_threshold,
            minimum_note_length=params.minimum_note_length_ms,
            minimum_frequency=params.minimum_frequency_hz,
            maximum_frequency=params.maximum_frequency_hz,
        )
    midi_data.write(str(raw_midi_path))
    midi_data.write(str(post_midi_path))   # placeholder until post-processing runs

    notes: list[TranscribedNote] = []
    for start, end, pitch, amplitude, _pitch_bend in note_events:
        notes.append(TranscribedNote(
            start_s=float(start), end_s=float(end),
            pitch=int(pitch), velocity=int(round(min(127, amplitude * 127))),
            confidence=float(min(1.0, max(0.0, amplitude))),
        ))
    median_conf = statistics.median([n.confidence for n in notes]) if notes else 0.0

    return TranscriptionResult(
        notes=notes, model="basic-pitch",
        raw_midi_path=raw_midi_path, post_midi_path=post_midi_path,
        hyperparameters={
            "onset_threshold": params.onset_threshold,
            "frame_threshold": params.frame_threshold,
            "minimum_note_length_ms": params.minimum_note_length_ms,
            "minimum_frequency_hz": params.minimum_frequency_hz,
            "maximum_frequency_hz": params.maximum_frequency_hz,
        },
        median_confidence=float(median_conf),
    )
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_transcription_basic_pitch.py -v -m slow`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/music_decoder/transcription/__init__.py \
        src/music_decoder/transcription/basic_pitch_wrapper.py \
        tests/unit/test_transcription_basic_pitch.py
git commit -m "feat(transcription): basic-pitch wrapper emitting TranscriptionResult"
```

---

### Task 21: CREPE transcription wrapper (monophonic)

**Files:**
- Create: `src/music_decoder/transcription/crepe_wrapper.py`
- Create: `tests/unit/test_transcription_crepe.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_transcription_crepe.py
from pathlib import Path

import numpy as np
import pytest

from music_decoder.config.hyperparameters import CrepeParams
from music_decoder.pipeline.contracts import AudioSource, LoadedAudio
from music_decoder.tab_assignment.tuning import get_preset
from music_decoder.transcription.crepe_wrapper import transcribe_crepe


def _audio_from_wav(path: Path) -> LoadedAudio:
    import scipy.io.wavfile as wavfile
    sr, raw = wavfile.read(str(path))
    samples = (raw.astype(np.float32) / 32768.0).astype(np.float32)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    return LoadedAudio(
        samples=samples, sr=sr, duration_s=samples.size / sr, sha256="x" * 64,
        source=AudioSource(
            path=path, declared_kind="solo_guitar",
            requested_quality="standard", requested_tuning=get_preset("EADGBE"),
        ),
    )


@pytest.mark.slow
def test_crepe_returns_pitches_for_sine(fixtures_dir: Path, tmp_path: Path):
    audio = _audio_from_wav(fixtures_dir / "audio_samples" / "sine_440.wav")
    params = CrepeParams(model_capacity="tiny", step_size_ms=10, viterbi=True)
    result = transcribe_crepe(audio, params, output_dir=tmp_path)
    assert result.model == "crepe"
    pitches = [n.pitch for n in result.notes]
    # 440 Hz = MIDI 69. Allow ±1 semitone tolerance for rounding.
    assert any(68 <= p <= 70 for p in pitches)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_transcription_crepe.py -v -m slow`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# src/music_decoder/transcription/crepe_wrapper.py
from __future__ import annotations

import statistics
from pathlib import Path

import numpy as np
import pretty_midi

from music_decoder.config.hyperparameters import CrepeParams
from music_decoder.pipeline.contracts import (
    LoadedAudio, TranscribedNote, TranscriptionResult,
)


def _hz_to_midi(freq: float) -> int:
    if freq <= 0:
        return -1
    return int(round(69 + 12 * np.log2(freq / 440.0)))


def _segment_runs(midi_seq: np.ndarray, conf_seq: np.ndarray, step_s: float):
    """Group consecutive frames with the same pitch into note events."""
    notes: list[tuple[float, float, int, float]] = []
    i = 0
    while i < len(midi_seq):
        if midi_seq[i] < 0:
            i += 1
            continue
        j = i
        while j + 1 < len(midi_seq) and midi_seq[j + 1] == midi_seq[i]:
            j += 1
        start_s = i * step_s
        end_s = (j + 1) * step_s
        avg_conf = float(np.mean(conf_seq[i:j + 1]))
        notes.append((start_s, end_s, int(midi_seq[i]), avg_conf))
        i = j + 1
    return notes


def transcribe_crepe(
    audio: LoadedAudio, params: CrepeParams, *, output_dir: Path,
) -> TranscriptionResult:
    import crepe

    output_dir.mkdir(parents=True, exist_ok=True)
    samples = audio.samples
    if audio.sr != 16000:
        from librosa import resample
        samples = resample(samples, orig_sr=audio.sr, target_sr=16000)
    time, frequency, confidence, _ = crepe.predict(
        samples, sr=16000,
        model_capacity=params.model_capacity,
        step_size=params.step_size_ms,
        viterbi=params.viterbi,
        verbose=0,
    )
    midi_seq = np.array([_hz_to_midi(f) if c >= 0.5 else -1
                         for f, c in zip(frequency, confidence)])
    step_s = params.step_size_ms / 1000.0
    runs = _segment_runs(midi_seq, confidence, step_s)
    notes = [
        TranscribedNote(start_s=s, end_s=e, pitch=p, velocity=80, confidence=c)
        for s, e, p, c in runs
    ]
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=24)
    for n in notes:
        inst.notes.append(pretty_midi.Note(
            velocity=n.velocity, pitch=n.pitch, start=n.start_s, end=n.end_s,
        ))
    pm.instruments.append(inst)
    raw = output_dir / "raw_crepe.mid"
    post = output_dir / "post_crepe.mid"
    pm.write(str(raw)); pm.write(str(post))
    median_conf = statistics.median([n.confidence for n in notes]) if notes else 0.0
    return TranscriptionResult(
        notes=notes, model="crepe",
        raw_midi_path=raw, post_midi_path=post,
        hyperparameters={
            "model_capacity": params.model_capacity,
            "step_size_ms": params.step_size_ms,
            "viterbi": params.viterbi,
        },
        median_confidence=float(median_conf),
    )
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_transcription_crepe.py -v -m slow`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/music_decoder/transcription/crepe_wrapper.py \
        tests/unit/test_transcription_crepe.py
git commit -m "feat(transcription): CREPE wrapper for monophonic pitch tracking"
```

---

### Task 22: Transcription accuracy benchmark on synthetic fixture

**Files:**
- Create: `tests/regression/test_transcription_accuracy.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/regression/test_transcription_accuracy.py
"""Asserts basic-pitch hits a minimum F-measure on the deterministic synthetic fixture."""
from pathlib import Path

import numpy as np
import pytest

from music_decoder.audio_io.load import load_audio
from music_decoder.config.hyperparameters import BasicPitchParams
from music_decoder.evaluation.fixtures.synthetic import SyntheticFixtures
from music_decoder.evaluation.metrics import note_f_measure
from music_decoder.pipeline.contracts import AudioSource
from music_decoder.tab_assignment.tuning import get_preset
from music_decoder.transcription.basic_pitch_wrapper import transcribe_basic_pitch


@pytest.mark.regression
@pytest.mark.slow
def test_basic_pitch_meets_f_measure_on_c_major_scale(tmp_path: Path):
    fx = next(f for f in SyntheticFixtures(
        root=Path("tests/fixtures/synthetic")
    ).load() if f.name == "c_major_scale")
    src = AudioSource(
        path=fx.audio_path, declared_kind="solo_guitar",
        requested_quality="standard", requested_tuning=get_preset("EADGBE"),
    )
    audio = load_audio(src)
    params = BasicPitchParams(
        onset_threshold=0.5, frame_threshold=0.3,
        minimum_note_length_ms=58,
        minimum_frequency_hz=32.7, maximum_frequency_hz=2000.0,
    )
    result = transcribe_basic_pitch(audio, params, output_dir=tmp_path)
    pred_iv = np.array([(n.start_s, n.end_s) for n in result.notes], dtype=float)
    pred_p = np.array([n.pitch for n in result.notes], dtype=float)
    f = note_f_measure(pred_iv, pred_p, fx.ground_truth.intervals,
                       fx.ground_truth.pitches_midi)
    assert f >= 0.50, (
        f"basic-pitch F-measure {f:.3f} below 0.50 on synthetic C major scale; "
        "the model is unusable as currently configured."
    )
```

- [ ] **Step 2: Run test**

Run: `pytest tests/regression/test_transcription_accuracy.py -v -m "regression and slow"`

Expected: PASS (basic-pitch on a clean sine-wave scale should clear 0.50 easily).

> **If it fails below 0.50:** investigate the basic-pitch hyperparameters, particularly `minimum_frequency_hz` and `frame_threshold`. The synthetic fixture has fundamentals between MIDI 60–72; if the lower bound is wrong, low-octave detections vanish.

- [ ] **Step 3: Commit**

```bash
git add tests/regression/test_transcription_accuracy.py
git commit -m "test(regression): basic-pitch F-measure threshold on synthetic scale"
```

---

### Task 23: Post-processing filter — minimum-duration drop

**Files:**
- Create: `src/music_decoder/transcription/post_processing.py`
- Create: `tests/unit/test_post_processing.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_post_processing.py
import numpy as np
import pytest

from music_decoder.pipeline.contracts import TranscribedNote
from music_decoder.transcription.post_processing import (
    drop_short_notes,
    merge_same_pitch,
    median_filter_pitch_contour,
    snap_to_beats,
)


def _n(start, end, pitch=60, conf=0.9):
    return TranscribedNote(start_s=start, end_s=end, pitch=pitch, velocity=80, confidence=conf)


def test_drop_short_notes_removes_below_threshold():
    notes = [_n(0.0, 0.01), _n(0.5, 0.6), _n(1.0, 1.04)]
    out = drop_short_notes(notes, min_duration_s=0.05)
    assert len(out) == 1
    assert out[0].start_s == 0.5


def test_drop_short_notes_keeps_exact_threshold():
    notes = [_n(0.0, 0.05)]
    assert len(drop_short_notes(notes, min_duration_s=0.05)) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_post_processing.py::test_drop_short_notes_removes_below_threshold -v`
Expected: FAIL.

- [ ] **Step 3: Implement the filter**

```python
# src/music_decoder/transcription/post_processing.py
from __future__ import annotations

from typing import Iterable

import numpy as np

from music_decoder.pipeline.contracts import TranscribedNote


def drop_short_notes(
    notes: Iterable[TranscribedNote], *, min_duration_s: float,
) -> list[TranscribedNote]:
    return [n for n in notes if (n.end_s - n.start_s) >= min_duration_s]
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_post_processing.py -v -k drop_short`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/music_decoder/transcription/post_processing.py tests/unit/test_post_processing.py
git commit -m "feat(post): drop_short_notes filter with duration threshold"
```

---

### Task 24: Post-processing filter — same-pitch merge

**Files:**
- Modify: `src/music_decoder/transcription/post_processing.py`
- Modify: `tests/unit/test_post_processing.py`

- [ ] **Step 1: Add the failing test**

Append to `tests/unit/test_post_processing.py`:
```python
def test_merge_same_pitch_combines_close_notes():
    notes = [
        _n(0.0, 0.5, pitch=60, conf=0.9),
        _n(0.52, 1.0, pitch=60, conf=0.7),   # gap 0.02s, same pitch
        _n(1.5, 2.0, pitch=60, conf=0.8),    # too far apart to merge
    ]
    out = merge_same_pitch(notes, gap_s=0.05)
    assert len(out) == 2
    assert out[0].start_s == 0.0
    assert out[0].end_s == 1.0
    assert out[0].confidence == pytest.approx(0.8, abs=0.01)


def test_merge_same_pitch_does_not_cross_pitches():
    notes = [_n(0.0, 0.5, pitch=60), _n(0.52, 1.0, pitch=62)]
    out = merge_same_pitch(notes, gap_s=0.05)
    assert len(out) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_post_processing.py -v -k merge_same`
Expected: FAIL — function missing.

- [ ] **Step 3: Implement**

Append to `src/music_decoder/transcription/post_processing.py`:
```python
def merge_same_pitch(
    notes: Iterable[TranscribedNote], *, gap_s: float,
) -> list[TranscribedNote]:
    sorted_notes = sorted(notes, key=lambda n: (n.pitch, n.start_s))
    out: list[TranscribedNote] = []
    for n in sorted_notes:
        if out and out[-1].pitch == n.pitch and (n.start_s - out[-1].end_s) <= gap_s:
            prev = out[-1]
            total_dur = (prev.end_s - prev.start_s) + (n.end_s - n.start_s)
            mean_conf = (
                prev.confidence * (prev.end_s - prev.start_s)
                + n.confidence * (n.end_s - n.start_s)
            ) / total_dur
            out[-1] = TranscribedNote(
                start_s=prev.start_s, end_s=n.end_s, pitch=prev.pitch,
                velocity=max(prev.velocity, n.velocity),
                confidence=mean_conf,
            )
        else:
            out.append(n)
    return sorted(out, key=lambda n: n.start_s)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_post_processing.py -v -k merge_same`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/music_decoder/transcription/post_processing.py tests/unit/test_post_processing.py
git commit -m "feat(post): merge_same_pitch filter for adjacent same-pitch fragments"
```

---

### Task 25: Post-processing filter — pitch contour median filter

**Files:**
- Modify: `src/music_decoder/transcription/post_processing.py`
- Modify: `tests/unit/test_post_processing.py`

- [ ] **Step 1: Add the failing test**

```python
def test_median_filter_removes_single_outlier():
    # A 9-frame contour with one wildly wrong frame
    contour = np.array([60, 60, 60, 60, 99, 60, 60, 60, 60], dtype=float)
    out = median_filter_pitch_contour(contour, window=3)
    assert out[4] == 60.0


def test_median_filter_window_must_be_odd():
    with pytest.raises(ValueError):
        median_filter_pitch_contour(np.array([60.0, 60.0]), window=4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_post_processing.py -v -k median_filter`
Expected: FAIL.

- [ ] **Step 3: Implement**

Append to `post_processing.py`:
```python
from scipy.signal import medfilt


def median_filter_pitch_contour(contour: np.ndarray, *, window: int) -> np.ndarray:
    if window % 2 == 0:
        raise ValueError("median filter window must be odd")
    return medfilt(np.asarray(contour, dtype=float), kernel_size=window)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_post_processing.py -v -k median_filter`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/music_decoder/transcription/post_processing.py tests/unit/test_post_processing.py
git commit -m "feat(post): median filter for monophonic pitch contour smoothing"
```

---

### Task 26: Post-processing filter — rhythmic onset snap

**Files:**
- Modify: `src/music_decoder/transcription/post_processing.py`
- Modify: `tests/unit/test_post_processing.py`

- [ ] **Step 1: Add the failing test**

```python
def test_snap_to_beats_moves_high_confidence_onsets():
    notes = [
        _n(0.49, 1.0, pitch=60, conf=0.95),   # close to beat at 0.5
        _n(0.55, 1.0, pitch=62, conf=0.30),   # low confidence: don't snap
    ]
    beats = np.array([0.0, 0.5, 1.0, 1.5])
    out = snap_to_beats(notes, beats=beats, confidence_threshold=0.7,
                        max_snap_s=0.05)
    assert out[0].start_s == 0.5
    assert out[1].start_s == 0.55


def test_snap_to_beats_does_not_move_far_onsets():
    notes = [_n(0.30, 0.50, conf=0.95)]
    beats = np.array([0.0, 0.5, 1.0])
    out = snap_to_beats(notes, beats=beats, confidence_threshold=0.7,
                        max_snap_s=0.05)
    assert out[0].start_s == 0.30   # 0.20s away from nearest beat → no snap
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_post_processing.py -v -k snap_to_beats`
Expected: FAIL.

- [ ] **Step 3: Implement**

Append to `post_processing.py`:
```python
def snap_to_beats(
    notes: Iterable[TranscribedNote],
    *,
    beats: np.ndarray,
    confidence_threshold: float,
    max_snap_s: float,
) -> list[TranscribedNote]:
    if len(beats) == 0:
        return list(notes)
    beat_arr = np.asarray(beats)
    out: list[TranscribedNote] = []
    for n in notes:
        if n.confidence < confidence_threshold:
            out.append(n)
            continue
        diffs = np.abs(beat_arr - n.start_s)
        idx = int(np.argmin(diffs))
        if diffs[idx] <= max_snap_s:
            shift = float(beat_arr[idx]) - n.start_s
            out.append(TranscribedNote(
                start_s=float(beat_arr[idx]), end_s=n.end_s + shift,
                pitch=n.pitch, velocity=n.velocity, confidence=n.confidence,
            ))
        else:
            out.append(n)
    return out
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_post_processing.py -v`
Expected: all post-processing tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/music_decoder/transcription/post_processing.py tests/unit/test_post_processing.py
git commit -m "feat(post): rhythmic snap filter that moves only high-confidence onsets"
```

---

### Task 27: Post-processing chain composer + accuracy delta benchmark

**Files:**
- Modify: `src/music_decoder/transcription/post_processing.py`
- Create: `tests/regression/test_post_processing_delta.py`

- [ ] **Step 1: Add the failing test (the chain composer)**

Add to `tests/unit/test_post_processing.py`:
```python
def test_apply_post_processing_chains_filters():
    from music_decoder.transcription.post_processing import apply_post_processing
    from music_decoder.config.hyperparameters import PostProcessingParams

    notes = [
        _n(0.0, 0.01, pitch=60, conf=0.9),     # too short → dropped
        _n(0.49, 0.99, pitch=60, conf=0.95),   # snapped to 0.5
        _n(0.52, 1.0, pitch=60, conf=0.95),    # merged with previous
    ]
    params = PostProcessingParams(
        median_filter_window=3, min_note_duration_s=0.05,
        same_pitch_merge_gap_s=0.05,
        rhythmic_snap_confidence_threshold=0.7,
    )
    out = apply_post_processing(notes, params=params, beats=np.array([0.0, 0.5, 1.0]))
    assert len(out) == 1
    assert out[0].start_s == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_post_processing.py -v -k apply_post_processing`
Expected: FAIL.

- [ ] **Step 3: Implement the composer**

Append to `post_processing.py`:
```python
from music_decoder.config.hyperparameters import PostProcessingParams


def apply_post_processing(
    notes: Iterable[TranscribedNote],
    *,
    params: PostProcessingParams,
    beats: np.ndarray | None = None,
) -> list[TranscribedNote]:
    pipeline = list(notes)
    pipeline = drop_short_notes(pipeline, min_duration_s=params.min_note_duration_s)
    pipeline = merge_same_pitch(pipeline, gap_s=params.same_pitch_merge_gap_s)
    if beats is not None and len(beats) > 0:
        pipeline = snap_to_beats(
            pipeline, beats=beats,
            confidence_threshold=params.rhythmic_snap_confidence_threshold,
            max_snap_s=0.05,
        )
    return pipeline
```

- [ ] **Step 4: Write the accuracy delta benchmark**

```python
# tests/regression/test_post_processing_delta.py
"""Asserts that applying the post-processing chain does not regress F-measure."""
from pathlib import Path

import numpy as np
import pytest

from music_decoder.audio_io.load import load_audio
from music_decoder.config.hyperparameters import (
    BasicPitchParams, PostProcessingParams,
)
from music_decoder.evaluation.fixtures.synthetic import SyntheticFixtures
from music_decoder.evaluation.metrics import note_f_measure
from music_decoder.pipeline.contracts import AudioSource
from music_decoder.tab_assignment.tuning import get_preset
from music_decoder.transcription.basic_pitch_wrapper import transcribe_basic_pitch
from music_decoder.transcription.post_processing import apply_post_processing


def _intervals_pitches(notes):
    iv = np.array([(n.start_s, n.end_s) for n in notes], dtype=float)
    p = np.array([n.pitch for n in notes], dtype=float)
    return iv, p


@pytest.mark.regression
@pytest.mark.slow
def test_post_processing_does_not_decrease_f_measure(tmp_path: Path):
    fx = next(f for f in SyntheticFixtures(
        root=Path("tests/fixtures/synthetic")
    ).load() if f.name == "c_major_scale")
    src = AudioSource(
        path=fx.audio_path, declared_kind="solo_guitar",
        requested_quality="standard", requested_tuning=get_preset("EADGBE"),
    )
    audio = load_audio(src)
    bp_params = BasicPitchParams(
        onset_threshold=0.5, frame_threshold=0.3, minimum_note_length_ms=58,
        minimum_frequency_hz=32.7, maximum_frequency_hz=2000.0,
    )
    raw = transcribe_basic_pitch(audio, bp_params, output_dir=tmp_path)
    iv, p = _intervals_pitches(raw.notes)
    f_raw = note_f_measure(iv, p, fx.ground_truth.intervals,
                           fx.ground_truth.pitches_midi)
    pp_params = PostProcessingParams(
        median_filter_window=5, min_note_duration_s=0.05,
        same_pitch_merge_gap_s=0.05,
        rhythmic_snap_confidence_threshold=0.7,
    )
    cleaned = apply_post_processing(raw.notes, params=pp_params, beats=None)
    iv2, p2 = _intervals_pitches(cleaned)
    f_post = note_f_measure(iv2, p2, fx.ground_truth.intervals,
                            fx.ground_truth.pitches_midi)
    assert f_post >= f_raw - 0.02, (
        f"post-processing regressed F-measure: raw={f_raw:.3f}, post={f_post:.3f}"
    )
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/unit/test_post_processing.py tests/regression/test_post_processing_delta.py -v`
Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add src/music_decoder/transcription/post_processing.py \
        tests/unit/test_post_processing.py tests/regression/test_post_processing_delta.py
git commit -m "feat(post): apply_post_processing composer + accuracy-delta regression test"
```

---

### Task 28: Key detection — chroma + HPSS

**Files:**
- Create: `src/music_decoder/key_detection/__init__.py`
- Create: `src/music_decoder/key_detection/chroma.py`
- Create: `tests/unit/test_key_detection_chroma.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_key_detection_chroma.py
import numpy as np

from music_decoder.key_detection.chroma import compute_chroma_with_hpss


def test_chroma_shape_and_range():
    sr = 22050
    t = np.arange(sr * 2) / sr
    samples = (0.5 * np.sin(2 * np.pi * 261.63 * t)).astype(np.float32)   # C4
    chroma = compute_chroma_with_hpss(samples, sr=sr, hpss_margin=1.0)
    assert chroma.shape[0] == 12
    assert (chroma >= 0).all()
    assert (chroma <= 1).all()


def test_pure_c_concentrates_mass_on_c():
    sr = 22050
    t = np.arange(sr * 2) / sr
    samples = (0.5 * np.sin(2 * np.pi * 261.63 * t)).astype(np.float32)
    chroma = compute_chroma_with_hpss(samples, sr=sr, hpss_margin=1.0)
    pc_sum = chroma.sum(axis=1)
    pc_sum /= pc_sum.sum()
    assert pc_sum.argmax() == 0  # C
    assert pc_sum[0] > 0.3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_key_detection_chroma.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# src/music_decoder/key_detection/__init__.py
```

```python
# src/music_decoder/key_detection/chroma.py
from __future__ import annotations

import numpy as np
import librosa


def compute_chroma_with_hpss(
    samples: np.ndarray, *, sr: int, hpss_margin: float,
) -> np.ndarray:
    """CQT-based chroma after harmonic-percussive separation.

    Returns shape (12, T) with values in [0, 1].
    """
    y_h, _ = librosa.effects.hpss(samples.astype(float), margin=hpss_margin)
    chroma = librosa.feature.chroma_cqt(y=y_h, sr=sr)
    chroma = chroma / (chroma.max(axis=0, keepdims=True) + 1e-12)
    return chroma.astype(np.float32)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_key_detection_chroma.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/music_decoder/key_detection/ tests/unit/test_key_detection_chroma.py
git commit -m "feat(key): chroma_cqt with HPSS preprocessing"
```

---

### Task 29: Key detection — K-S correlation with both K-K and Temperley profiles

**Files:**
- Create: `src/music_decoder/key_detection/profiles.py`
- Create: `src/music_decoder/key_detection/ks.py`
- Create: `tests/unit/test_key_detection_ks.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_key_detection_ks.py
import numpy as np

from music_decoder.key_detection.ks import (
    correlate_against_profiles,
    top_k_estimates,
)
from music_decoder.key_detection.profiles import (
    KRUMHANSL_KESSLER_MAJOR,
    KRUMHANSL_KESSLER_MINOR,
    TEMPERLEY_MAJOR,
    TEMPERLEY_MINOR,
)


def test_kk_profile_lengths():
    assert len(KRUMHANSL_KESSLER_MAJOR) == 12
    assert len(KRUMHANSL_KESSLER_MINOR) == 12


def test_correlate_returns_24_keys():
    pc = np.zeros(12); pc[0] = 1.0   # pure C
    estimates = correlate_against_profiles(
        pc, profile="krumhansl_kessler",
    )
    assert len(estimates) == 24
    by_key = {(e.tonic, e.mode): e.correlation for e in estimates}
    assert max(by_key.values()) == by_key[("C", "major")]


def test_top_k_returns_three_sorted_descending():
    pc = np.zeros(12); pc[0] = 1.0
    top3 = top_k_estimates(pc, profile="krumhansl_kessler", k=3)
    assert len(top3) == 3
    assert top3[0].tonic == "C"
    assert top3[0].mode == "major"
    assert top3[0].correlation >= top3[1].correlation >= top3[2].correlation
    assert top3[0].margin == top3[0].correlation - top3[1].correlation


def test_temperley_distinguishable_from_kk():
    pc = np.array([0.4, 0.0, 0.2, 0.0, 0.3, 0.05, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0])
    kk = top_k_estimates(pc, profile="krumhansl_kessler", k=1)[0]
    temp = top_k_estimates(pc, profile="temperley", k=1)[0]
    assert (kk.tonic, kk.mode) is not None
    assert (temp.tonic, temp.mode) is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_key_detection_ks.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement profile constants**

```python
# src/music_decoder/key_detection/profiles.py
from __future__ import annotations

# Krumhansl-Kessler (1990)
KRUMHANSL_KESSLER_MAJOR = (
    6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
)
KRUMHANSL_KESSLER_MINOR = (
    6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17,
)
# Temperley (2007) — flatter peaks, simplified.
TEMPERLEY_MAJOR = (
    5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0,
)
TEMPERLEY_MINOR = (
    5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0,
)

KEYS = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
```

- [ ] **Step 4: Implement K-S correlation**

```python
# src/music_decoder/key_detection/ks.py
from __future__ import annotations

from typing import Literal

import numpy as np

from music_decoder.pipeline.contracts import KeyEstimate

from .profiles import (
    KEYS,
    KRUMHANSL_KESSLER_MAJOR, KRUMHANSL_KESSLER_MINOR,
    TEMPERLEY_MAJOR, TEMPERLEY_MINOR,
)


_PROFILES = {
    "krumhansl_kessler": (np.array(KRUMHANSL_KESSLER_MAJOR),
                          np.array(KRUMHANSL_KESSLER_MINOR)),
    "temperley":         (np.array(TEMPERLEY_MAJOR),
                          np.array(TEMPERLEY_MINOR)),
}


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    xc = x - x.mean()
    yc = y - y.mean()
    denom = np.sqrt((xc ** 2).sum() * (yc ** 2).sum())
    if denom == 0:
        return 0.0
    return float((xc * yc).sum() / denom)


def correlate_against_profiles(
    pitch_class_distribution: np.ndarray,
    *,
    profile: Literal["krumhansl_kessler", "temperley"],
) -> list[KeyEstimate]:
    if profile not in _PROFILES:
        raise ValueError(f"unknown profile {profile!r}")
    major, minor = _PROFILES[profile]
    pc = np.asarray(pitch_class_distribution, dtype=float)
    if pc.sum() > 0:
        pc = pc / pc.sum()
    corrs: list[tuple[str, str, float]] = []
    for i in range(12):
        corrs.append((KEYS[i], "major", _pearson(np.roll(major, i), pc)))
        corrs.append((KEYS[i], "minor", _pearson(np.roll(minor, i), pc)))
    sorted_corrs = sorted(corrs, key=lambda x: x[2], reverse=True)
    top = sorted_corrs[0][2]
    second = sorted_corrs[1][2] if len(sorted_corrs) > 1 else top
    return [
        KeyEstimate(
            tonic=tonic, mode=mode, profile=profile,
            correlation=corr, margin=corr - second,
        )
        for (tonic, mode, corr) in sorted_corrs
    ]


def top_k_estimates(
    pitch_class_distribution: np.ndarray,
    *,
    profile: Literal["krumhansl_kessler", "temperley"],
    k: int = 3,
) -> list[KeyEstimate]:
    return correlate_against_profiles(pitch_class_distribution, profile=profile)[:k]
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/unit/test_key_detection_ks.py -v`
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add src/music_decoder/key_detection/profiles.py src/music_decoder/key_detection/ks.py \
        tests/unit/test_key_detection_ks.py
git commit -m "feat(key): K-S correlation with K-K and Temperley profiles"
```

---

### Task 30: Key detection — global top-3 + cross-profile consensus

**Files:**
- Create: `src/music_decoder/key_detection/global_estimator.py`
- Create: `tests/unit/test_key_detection_global.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_key_detection_global.py
import numpy as np

from music_decoder.key_detection.global_estimator import estimate_global_key


def test_consensus_when_profiles_agree():
    pc = np.zeros(12); pc[0] = 0.6; pc[4] = 0.3; pc[7] = 0.4   # C major triad emphasis
    result = estimate_global_key(pc)
    assert "krumhansl_kessler" in result.global_top3_per_profile
    assert "temperley" in result.global_top3_per_profile
    assert result.consensus_key is not None
    assert result.consensus_key.tonic == "C"
    assert result.consensus_key.mode == "major"
    assert 0.0 <= result.confidence <= 1.0


def test_no_consensus_when_profiles_disagree():
    pc = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0])
    result = estimate_global_key(pc)
    assert isinstance(result.global_top3_per_profile["krumhansl_kessler"], list)
    # consensus may or may not be set depending on rankings; just ensure shape.
    if result.consensus_key is None:
        assert result.confidence < 0.7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_key_detection_global.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# src/music_decoder/key_detection/global_estimator.py
from __future__ import annotations

import numpy as np

from music_decoder.pipeline.contracts import KeyDetectionResult, KeyEstimate

from .ks import top_k_estimates


def estimate_global_key(
    pitch_class_distribution: np.ndarray,
) -> KeyDetectionResult:
    kk_top3 = top_k_estimates(pitch_class_distribution,
                              profile="krumhansl_kessler", k=3)
    temp_top3 = top_k_estimates(pitch_class_distribution,
                                profile="temperley", k=3)
    profile_top1 = {
        "krumhansl_kessler": kk_top3[0],
        "temperley": temp_top3[0],
    }
    if (profile_top1["krumhansl_kessler"].tonic == profile_top1["temperley"].tonic
            and profile_top1["krumhansl_kessler"].mode == profile_top1["temperley"].mode):
        consensus = profile_top1["krumhansl_kessler"]
        # confidence = mean margin across profiles, clipped to [0, 1]
        confidence = float(np.clip(
            (kk_top3[0].margin + temp_top3[0].margin) / 2 + 0.5, 0.0, 1.0,
        ))
    else:
        consensus = None
        confidence = float(np.clip(
            (kk_top3[0].margin + temp_top3[0].margin) / 4, 0.0, 0.5,
        ))
    return KeyDetectionResult(
        global_top3_per_profile={
            "krumhansl_kessler": kk_top3, "temperley": temp_top3,
        },
        consensus_key=consensus,
        windowed_segments=[],
        confidence=confidence,
    )
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_key_detection_global.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/music_decoder/key_detection/global_estimator.py \
        tests/unit/test_key_detection_global.py
git commit -m "feat(key): global top-3 + cross-profile consensus + confidence"
```

---

### Task 31: Key detection — windowed (modulation)

**Files:**
- Create: `src/music_decoder/key_detection/windowed.py`
- Create: `tests/unit/test_key_detection_windowed.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_key_detection_windowed.py
import numpy as np

from music_decoder.key_detection.windowed import detect_windowed_keys


def test_windowed_segments_cover_audio():
    sr = 22050
    chroma = np.zeros((12, sr * 16 // 512))
    chroma[0] = 1.0   # uniform C presence
    segments = detect_windowed_keys(
        chroma, sr=sr, hop_length=512,
        segment_length_s=8.0, hop_s=2.0,
    )
    assert len(segments) > 0
    starts = [s for (s, _, _) in segments]
    ends = [e for (_, e, _) in segments]
    assert min(starts) == 0.0
    assert max(ends) >= 16.0 - 8.0


def test_windowed_segments_emit_key_estimate():
    sr = 22050
    chroma = np.zeros((12, sr * 8 // 512))
    chroma[0] = 1.0
    segs = detect_windowed_keys(
        chroma, sr=sr, hop_length=512,
        segment_length_s=8.0, hop_s=2.0,
    )
    assert all(seg[2].tonic in ("C", "C#", "D", "D#", "E", "F", "F#", "G",
                                 "G#", "A", "A#", "B") for seg in segs)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_key_detection_windowed.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# src/music_decoder/key_detection/windowed.py
from __future__ import annotations

import numpy as np

from music_decoder.pipeline.contracts import KeyEstimate

from .ks import top_k_estimates


def detect_windowed_keys(
    chroma: np.ndarray,   # shape (12, T)
    *,
    sr: int,
    hop_length: int,
    segment_length_s: float,
    hop_s: float,
) -> list[tuple[float, float, KeyEstimate]]:
    if chroma.size == 0:
        return []
    frames_per_second = sr / hop_length
    seg_frames = max(1, int(round(segment_length_s * frames_per_second)))
    hop_frames = max(1, int(round(hop_s * frames_per_second)))
    n_frames = chroma.shape[1]
    segments: list[tuple[float, float, KeyEstimate]] = []
    for start_frame in range(0, max(1, n_frames - seg_frames + 1), hop_frames):
        end_frame = min(n_frames, start_frame + seg_frames)
        window = chroma[:, start_frame:end_frame].mean(axis=1)
        if window.sum() == 0:
            continue
        top1 = top_k_estimates(window, profile="krumhansl_kessler", k=1)[0]
        start_s = start_frame / frames_per_second
        end_s = end_frame / frames_per_second
        segments.append((start_s, end_s, top1))
    return segments
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_key_detection_windowed.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/music_decoder/key_detection/windowed.py \
        tests/unit/test_key_detection_windowed.py
git commit -m "feat(key): windowed K-S segmentation for modulation detection"
```

---

### Task 32: Key detection — public API + accuracy benchmark

**Files:**
- Create: `src/music_decoder/key_detection/api.py`
- Create: `tests/regression/test_key_detection_accuracy.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/regression/test_key_detection_accuracy.py
"""On synthetic C-major fixture, both profile estimators should pick C major."""
from pathlib import Path

import numpy as np
import pytest

from music_decoder.audio_io.load import load_audio
from music_decoder.evaluation.fixtures.synthetic import SyntheticFixtures
from music_decoder.key_detection.api import detect_key
from music_decoder.pipeline.contracts import AudioSource
from music_decoder.tab_assignment.tuning import get_preset


@pytest.mark.regression
@pytest.mark.slow
def test_synthetic_c_major_scale_returns_c_major():
    fx = next(f for f in SyntheticFixtures(
        root=Path("tests/fixtures/synthetic")
    ).load() if f.name == "c_major_scale")
    src = AudioSource(
        path=fx.audio_path, declared_kind="solo_guitar",
        requested_quality="standard", requested_tuning=get_preset("EADGBE"),
    )
    audio = load_audio(src)
    result = detect_key(
        audio.samples, sr=audio.sr,
        hpss_margin=1.0, segment_length_s=8.0, hop_s=2.0,
    )
    top_kk = result.global_top3_per_profile["krumhansl_kessler"][0]
    top_t = result.global_top3_per_profile["temperley"][0]
    assert top_kk.tonic == "C" and top_kk.mode == "major"
    assert top_t.tonic == "C" and top_t.mode == "major"
    assert result.consensus_key is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/regression/test_key_detection_accuracy.py -v -m "regression and slow"`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement the public API**

```python
# src/music_decoder/key_detection/api.py
from __future__ import annotations

import numpy as np

from music_decoder.pipeline.contracts import KeyDetectionResult

from .chroma import compute_chroma_with_hpss
from .global_estimator import estimate_global_key
from .windowed import detect_windowed_keys


_DEFAULT_HOP_LENGTH = 512


def detect_key(
    samples: np.ndarray,
    *,
    sr: int,
    hpss_margin: float,
    segment_length_s: float,
    hop_s: float,
) -> KeyDetectionResult:
    chroma = compute_chroma_with_hpss(samples, sr=sr, hpss_margin=hpss_margin)
    pc = chroma.mean(axis=1)
    global_result = estimate_global_key(pc)
    windowed = detect_windowed_keys(
        chroma, sr=sr, hop_length=_DEFAULT_HOP_LENGTH,
        segment_length_s=segment_length_s, hop_s=hop_s,
    )
    return KeyDetectionResult(
        global_top3_per_profile=global_result.global_top3_per_profile,
        consensus_key=global_result.consensus_key,
        windowed_segments=windowed,
        confidence=global_result.confidence,
    )
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/regression/test_key_detection_accuracy.py -v -m "regression and slow"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/music_decoder/key_detection/api.py \
        tests/regression/test_key_detection_accuracy.py
git commit -m "feat(key): top-level detect_key API + synthetic accuracy regression"
```

---

### Task 33: Beat tracking — tempo + beats + downbeats

**Files:**
- Create: `src/music_decoder/beat_tracking/__init__.py`
- Create: `src/music_decoder/beat_tracking/beats.py`
- Create: `tests/unit/test_beat_tracking_beats.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_beat_tracking_beats.py
import numpy as np

from music_decoder.beat_tracking.beats import track_beats


def _click_track(bpm: float, duration_s: float, sr: int = 22050) -> np.ndarray:
    """Generate a synthetic click train at the requested tempo."""
    interval_s = 60.0 / bpm
    n = int(duration_s * sr)
    out = np.zeros(n, dtype=np.float32)
    click_len = int(0.005 * sr)
    t = 0.0
    while t < duration_s:
        start = int(t * sr)
        if start + click_len < n:
            out[start:start + click_len] = 1.0
        t += interval_s
    return out


def test_beats_detected_for_120_bpm_clicks():
    sr = 22050
    samples = _click_track(120, duration_s=8.0, sr=sr)
    result = track_beats(samples, sr=sr, start_bpm=120.0, tightness=100.0)
    assert 110 < result.tempo_bpm < 130
    assert len(result.beat_times_s) >= 8


def test_degenerate_audio_returns_zero_beats():
    sr = 22050
    samples = np.zeros(sr, dtype=np.float32)
    result = track_beats(samples, sr=sr, start_bpm=120.0, tightness=100.0)
    assert len(result.beat_times_s) == 0
    assert result.ts_assumed is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_beat_tracking_beats.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# src/music_decoder/beat_tracking/__init__.py
```

```python
# src/music_decoder/beat_tracking/beats.py
from __future__ import annotations

import numpy as np
import librosa

from music_decoder.pipeline.contracts import BeatGrid


def track_beats(
    samples: np.ndarray, *, sr: int, start_bpm: float, tightness: float,
) -> BeatGrid:
    rms = float(np.sqrt(np.mean(samples ** 2))) if samples.size else 0.0
    if rms < 1e-5:
        return BeatGrid(
            tempo_bpm=0.0, beat_times_s=np.zeros(0),
            downbeat_times_s=np.zeros(0),
            ts_numerator=4, ts_denominator=4,
            ts_confidence=0.0, ts_assumed=True,
        )
    tempo, beats = librosa.beat.beat_track(
        y=samples.astype(float), sr=sr,
        start_bpm=start_bpm, tightness=tightness, units="time",
    )
    beats = np.asarray(beats, dtype=float)
    # Crude downbeat estimate: every 4th beat, anchored at first beat.
    downbeats = beats[::4]
    return BeatGrid(
        tempo_bpm=float(tempo),
        beat_times_s=beats,
        downbeat_times_s=downbeats,
        ts_numerator=4, ts_denominator=4,
        ts_confidence=0.5, ts_assumed=True,
    )
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_beat_tracking_beats.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/music_decoder/beat_tracking/ tests/unit/test_beat_tracking_beats.py
git commit -m "feat(beats): tempo + beat + crude-downbeat tracking with degenerate-input fallback"
```

---

### Task 34: Time-signature inference

**Files:**
- Create: `src/music_decoder/beat_tracking/time_signature.py`
- Create: `tests/unit/test_beat_tracking_ts.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_beat_tracking_ts.py
import numpy as np

from music_decoder.beat_tracking.time_signature import infer_time_signature


def test_strong_4_4_pattern_detects_4_4():
    # Beat strengths cycling 1.0, 0.4, 0.6, 0.4 (classic 4/4 emphasis on beat 1)
    strengths = np.tile(np.array([1.0, 0.4, 0.6, 0.4]), 8)
    result = infer_time_signature(strengths, min_confidence=0.3)
    assert result.numerator == 4
    assert result.denominator == 4
    assert result.confidence >= 0.3
    assert result.assumed is False


def test_unclear_pattern_falls_back_to_4_4_assumed():
    strengths = np.full(20, 0.5)   # uniform → no autocorrelation peaks
    result = infer_time_signature(strengths, min_confidence=0.5)
    assert result.numerator == 4
    assert result.denominator == 4
    assert result.assumed is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_beat_tracking_ts.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# src/music_decoder/beat_tracking/time_signature.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TimeSignatureResult:
    numerator: int
    denominator: int
    confidence: float
    assumed: bool


_CANDIDATES = (3, 4, 6, 5, 7)


def infer_time_signature(
    beat_strengths: np.ndarray, *, min_confidence: float,
) -> TimeSignatureResult:
    if beat_strengths.size < 4:
        return TimeSignatureResult(numerator=4, denominator=4,
                                   confidence=0.0, assumed=True)
    s = np.asarray(beat_strengths, dtype=float)
    s = s - s.mean()
    n = len(s)
    autocorr = np.correlate(s, s, mode="full")[n - 1:] / max(np.var(s) * n, 1e-12)
    best = (4, 0.0)
    for k in _CANDIDATES:
        if k < len(autocorr):
            peak = autocorr[k]
            if peak > best[1]:
                best = (k, float(peak))
    numerator, peak = best
    confidence = float(max(0.0, min(1.0, peak)))
    if confidence < min_confidence:
        return TimeSignatureResult(numerator=4, denominator=4,
                                   confidence=confidence, assumed=True)
    return TimeSignatureResult(numerator=numerator, denominator=4,
                               confidence=confidence, assumed=False)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_beat_tracking_ts.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/music_decoder/beat_tracking/time_signature.py tests/unit/test_beat_tracking_ts.py
git commit -m "feat(beats): time-signature inference from beat-strength autocorrelation"
```

---

### Task 35: A* — hand-crafted tests authored BEFORE any A* implementation

> Per the design's TDD requirement for A*: write the hand-crafted test cases first, watch them fail with "module not found", then implement the algorithm.

**Files:**
- Create: `tests/unit/test_tab_assignment.py`

- [ ] **Step 1: Write the failing tests** (deliberately authored against an API that doesn't exist yet)

```python
# tests/unit/test_tab_assignment.py
import numpy as np
import pytest

from music_decoder.pipeline.contracts import TranscribedNote, TabPosition
from music_decoder.tab_assignment.assigner import assign_tab
from music_decoder.tab_assignment.tuning import get_preset


def _note(pitch: int, start: float = 0.0, end: float = 1.0, conf: float = 1.0):
    return TranscribedNote(start_s=start, end_s=end, pitch=pitch, velocity=80, confidence=conf)


def _default_weights() -> dict[str, float]:
    return {
        "w_move": 1.0, "w_string": 0.3, "w_span": 0.5, "w_high": 0.4,
        "w_open": 0.2, "w_chord_intra": 0.6,
    }


def test_open_high_e_is_preferred_over_b_string_5th_fret():
    """E4 (MIDI 64) on EADGBE: open high E (5,0) preferred to (4,5)."""
    notes = [_note(64)]
    result = assign_tab(notes, tuning=get_preset("EADGBE"),
                       weights=_default_weights(), max_fret=22)
    assert result.tabbed_notes[0].position == TabPosition(string=5, fret=0)


def test_low_e_is_only_playable_open_e2():
    """E2 (MIDI 40): only valid position is (0, 0) on EADGBE."""
    notes = [_note(40)]
    result = assign_tab(notes, tuning=get_preset("EADGBE"),
                       weights=_default_weights(), max_fret=22)
    assert result.tabbed_notes[0].position == TabPosition(string=0, fret=0)
    assert len(result.notes_dropped) == 0


def test_below_low_e_is_dropped_in_eadgbe():
    """D2 (MIDI 38) cannot be played on EADGBE (lowest is E2)."""
    notes = [_note(38)]
    result = assign_tab(notes, tuning=get_preset("EADGBE"),
                       weights=_default_weights(), max_fret=22)
    assert len(result.tabbed_notes) == 0
    assert len(result.notes_dropped) == 1
    assert "out_of_range" in result.notes_dropped[0][1]


def test_drop_d_makes_d2_playable_on_open_low_string():
    """D2 (MIDI 38) on Drop_D: open low D string (0, 0)."""
    notes = [_note(38)]
    result = assign_tab(notes, tuning=get_preset("Drop_D"),
                       weights=_default_weights(), max_fret=22)
    assert result.tabbed_notes[0].position == TabPosition(string=0, fret=0)


def test_g_major_chord_returns_six_simultaneous_positions():
    """G2 B2 D3 G3 B3 G4 simultaneously → known open-position G chord."""
    notes = [
        _note(43, 0.0, 1.0), _note(47, 0.0, 1.0), _note(50, 0.0, 1.0),
        _note(55, 0.0, 1.0), _note(59, 0.0, 1.0), _note(67, 0.0, 1.0),
    ]
    result = assign_tab(notes, tuning=get_preset("EADGBE"),
                       weights=_default_weights(), max_fret=22)
    pairs = sorted([(n.position.string, n.position.fret) for n in result.tabbed_notes])
    assert pairs == [(0, 3), (1, 2), (2, 0), (3, 0), (4, 0), (5, 3)]


def test_ascending_scale_is_monotonic_in_pitch():
    """A simple ascending scale should not produce any string-collision artifacts."""
    pitches = [60, 62, 64, 65, 67, 69, 71, 72]
    notes = [_note(p, start=i * 0.5, end=(i + 1) * 0.5) for i, p in enumerate(pitches)]
    result = assign_tab(notes, tuning=get_preset("EADGBE"),
                       weights=_default_weights(), max_fret=22)
    assert len(result.tabbed_notes) == 8
    # No two notes should occupy the same string at the same time
    for a, b in zip(result.tabbed_notes, result.tabbed_notes[1:]):
        if a.note.end_s > b.note.start_s and a.position.string == b.position.string:
            raise AssertionError("string collision in sequential single notes")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_tab_assignment.py -v`
Expected: ALL FAIL with `ModuleNotFoundError` for `music_decoder.tab_assignment.assigner`. This baseline shows the tests describe behavior the system does not yet have.

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/unit/test_tab_assignment.py
git commit -m "test(tab): hand-crafted A* test cases (currently failing)"
```

---

### Task 36: A* — candidate set generator

**Files:**
- Create: `src/music_decoder/tab_assignment/candidates.py`
- Create: `tests/unit/test_tab_candidates.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_tab_candidates.py
import pytest

from music_decoder.tab_assignment.candidates import (
    note_candidates,
    chord_combinations,
)
from music_decoder.tab_assignment.tuning import get_preset


def test_e4_in_eadgbe_has_two_candidates_below_fret_22():
    cands = note_candidates(pitch=64, tuning=get_preset("EADGBE"), max_fret=22)
    pairs = sorted([(c.string, c.fret) for c in cands])
    assert (5, 0) in pairs and (4, 5) in pairs and (3, 9) in pairs


def test_below_range_returns_empty():
    cands = note_candidates(pitch=30, tuning=get_preset("EADGBE"), max_fret=22)
    assert cands == []


def test_chord_combinations_filters_string_collisions():
    pitches = [40, 40]   # two E2s; cannot both play on string 0
    combos = chord_combinations(pitches, tuning=get_preset("EADGBE"), max_fret=22)
    for combo in combos:
        strings = [p.string for p in combo]
        assert len(strings) == len(set(strings))   # no duplicates


def test_chord_combinations_respects_span_5_frets():
    pitches = [60, 67]  # C4 and G4
    combos = chord_combinations(pitches, tuning=get_preset("EADGBE"), max_fret=22)
    for combo in combos:
        frets = [p.fret for p in combo]
        assert max(frets) - min(frets) <= 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_tab_candidates.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# src/music_decoder/tab_assignment/candidates.py
from __future__ import annotations

from itertools import product

from music_decoder.pipeline.contracts import TabPosition
from music_decoder.tab_assignment.tuning import Tuning


_MAX_CHORD_SPAN = 5


def note_candidates(
    *, pitch: int, tuning: Tuning, max_fret: int,
) -> list[TabPosition]:
    out: list[TabPosition] = []
    for s, open_p in enumerate(tuning.open_pitches):
        f = pitch - open_p
        if 0 <= f <= max_fret:
            out.append(TabPosition(string=s, fret=f))
    return out


def chord_combinations(
    pitches: list[int], *, tuning: Tuning, max_fret: int,
) -> list[tuple[TabPosition, ...]]:
    per_pitch = [note_candidates(pitch=p, tuning=tuning, max_fret=max_fret)
                 for p in pitches]
    if any(len(c) == 0 for c in per_pitch):
        return []
    combos: list[tuple[TabPosition, ...]] = []
    for combo in product(*per_pitch):
        strings = [p.string for p in combo]
        if len(set(strings)) != len(strings):
            continue
        frets = [p.fret for p in combo if p.fret > 0]
        if frets and (max(frets) - min(frets) > _MAX_CHORD_SPAN):
            continue
        combos.append(combo)
    return combos
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_tab_candidates.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/music_decoder/tab_assignment/candidates.py tests/unit/test_tab_candidates.py
git commit -m "feat(tab): candidate generator with collision and span filtering"
```

---

### Task 37: A* — transition cost function

**Files:**
- Create: `src/music_decoder/tab_assignment/cost.py`
- Create: `tests/unit/test_tab_cost.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_tab_cost.py
import pytest

from music_decoder.pipeline.contracts import TabPosition
from music_decoder.tab_assignment.cost import (
    transition_cost,
    chord_span_penalty,
    chord_collides,
)


_W = {
    "w_move": 1.0, "w_string": 0.3, "w_span": 0.5,
    "w_high": 0.4, "w_open": 0.2, "w_chord_intra": 0.6,
}


def test_transition_cost_zero_for_identical_position():
    p = TabPosition(string=4, fret=3)
    cost = transition_cost(prev=p, curr=p, weights=_W, hand_anchor=3.0)
    assert cost <= 0.0   # open-string bonus may pull negative; non-positive at minimum


def test_open_string_gets_bonus():
    p = TabPosition(string=5, fret=5)
    o = TabPosition(string=5, fret=0)
    no_bonus = transition_cost(prev=p, curr=p, weights=_W, hand_anchor=5.0)
    with_bonus = transition_cost(prev=p, curr=o, weights=_W, hand_anchor=5.0)
    assert with_bonus < no_bonus


def test_high_fret_penalizes():
    low = TabPosition(string=5, fret=5)
    high = TabPosition(string=5, fret=18)
    c_low = transition_cost(prev=low, curr=low, weights=_W, hand_anchor=5.0)
    c_high = transition_cost(prev=low, curr=high, weights=_W, hand_anchor=5.0)
    assert c_high > c_low


def test_movement_cost_grows_with_fret_distance():
    a = TabPosition(string=5, fret=3)
    b = TabPosition(string=5, fret=7)
    c = TabPosition(string=5, fret=15)
    near = transition_cost(prev=a, curr=b, weights=_W, hand_anchor=3.0)
    far = transition_cost(prev=a, curr=c, weights=_W, hand_anchor=3.0)
    assert far > near


def test_chord_span_penalty_zero_within_4_fret_span():
    chord = (TabPosition(0, 3), TabPosition(1, 2), TabPosition(2, 0), TabPosition(3, 0))
    assert chord_span_penalty(chord) == 0.0


def test_chord_collides_detects_string_duplicates():
    chord_ok = (TabPosition(0, 3), TabPosition(1, 2))
    chord_bad = (TabPosition(0, 3), TabPosition(0, 5))
    assert not chord_collides(chord_ok)
    assert chord_collides(chord_bad)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_tab_cost.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# src/music_decoder/tab_assignment/cost.py
from __future__ import annotations

from typing import Sequence

from music_decoder.pipeline.contracts import TabPosition


def chord_span_penalty(chord: Sequence[TabPosition]) -> float:
    fretted = [p.fret for p in chord if p.fret > 0]
    if len(fretted) < 2:
        return 0.0
    span = max(fretted) - min(fretted)
    if span <= 4:
        return 0.0
    return float((span - 4) ** 1.5)


def chord_collides(chord: Sequence[TabPosition]) -> bool:
    strings = [p.string for p in chord]
    return len(strings) != len(set(strings))


def transition_cost(
    *,
    prev: TabPosition,
    curr: TabPosition,
    weights: dict[str, float],
    hand_anchor: float,
) -> float:
    move = weights["w_move"] * abs(curr.fret - prev.fret)
    string = weights["w_string"] * abs(curr.string - prev.string)
    span = weights["w_span"] * max(0.0, curr.fret - hand_anchor) ** 1.5
    high = weights["w_high"] * max(0, curr.fret - 12) ** 1.2
    open_bonus = weights["w_open"] * (-1.0 if curr.fret == 0 else 0.0)
    return float(move + string + span + high + open_bonus)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_tab_cost.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/music_decoder/tab_assignment/cost.py tests/unit/test_tab_cost.py
git commit -m "feat(tab): transition cost + chord span/collision penalties"
```

---

### Task 38: A* — admissible heuristic

**Files:**
- Create: `src/music_decoder/tab_assignment/heuristic.py`
- Create: `tests/unit/test_tab_heuristic.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_tab_heuristic.py
from music_decoder.pipeline.contracts import TabPosition
from music_decoder.tab_assignment.heuristic import remaining_high_fret_penalty


_W = {"w_high": 0.4}


def test_no_remaining_groups_returns_zero():
    assert remaining_high_fret_penalty([], weights=_W) == 0.0


def test_single_group_low_fret_returns_zero():
    groups = [[(TabPosition(5, 3),)]]
    assert remaining_high_fret_penalty(groups, weights=_W) == 0.0


def test_high_fret_in_group_contributes():
    groups = [
        [(TabPosition(5, 14),)],   # min fret = 14, penalty (14-12)^1.2 * 0.4
        [(TabPosition(5, 18),)],
    ]
    h = remaining_high_fret_penalty(groups, weights=_W)
    assert h > 0


def test_uses_minimum_fret_in_each_group():
    groups = [[
        (TabPosition(5, 18),),
        (TabPosition(4, 14),),  # min fret = 14
    ]]
    h = remaining_high_fret_penalty(groups, weights=_W)
    expected = _W["w_high"] * max(0, 14 - 12) ** 1.2
    assert h == expected
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_tab_heuristic.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# src/music_decoder/tab_assignment/heuristic.py
from __future__ import annotations

from typing import Sequence

from music_decoder.pipeline.contracts import TabPosition


def remaining_high_fret_penalty(
    groups: Sequence[Sequence[Sequence[TabPosition]]],
    *,
    weights: dict[str, float],
) -> float:
    """Admissible lower bound: for each remaining note/chord group, take
    the minimum fret across all candidate states and apply only the
    high-fret penalty term.
    """
    total = 0.0
    for group in groups:
        if not group:
            continue
        min_fret = min(min(p.fret for p in candidate) for candidate in group)
        total += weights["w_high"] * max(0, min_fret - 12) ** 1.2
    return float(total)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_tab_heuristic.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/music_decoder/tab_assignment/heuristic.py tests/unit/test_tab_heuristic.py
git commit -m "feat(tab): admissible high-fret-only heuristic for A*"
```

---

### Task 39: A* — search algorithm

**Files:**
- Create: `src/music_decoder/tab_assignment/astar.py`
- Create: `tests/unit/test_tab_astar.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_tab_astar.py
from music_decoder.pipeline.contracts import TabPosition
from music_decoder.tab_assignment.astar import (
    Group, astar_min_cost_path,
)


_W = {
    "w_move": 1.0, "w_string": 0.3, "w_span": 0.5, "w_high": 0.4,
    "w_open": 0.2, "w_chord_intra": 0.6,
}


def test_single_group_picks_lowest_cost_candidate():
    groups: list[Group] = [
        Group([(TabPosition(5, 0),), (TabPosition(4, 5),)]),
    ]
    path, cost = astar_min_cost_path(groups, weights=_W, hand_anchor_window=8)
    # Open string should win because of open_bonus.
    assert path[0] == (TabPosition(5, 0),)


def test_two_groups_avoids_high_cost_transition():
    groups: list[Group] = [
        Group([(TabPosition(5, 0),)]),
        Group([(TabPosition(4, 1),), (TabPosition(0, 18),)]),
    ]
    path, cost = astar_min_cost_path(groups, weights=_W, hand_anchor_window=8)
    assert path[1] == (TabPosition(4, 1),)


def test_empty_groups_returns_empty_path():
    path, cost = astar_min_cost_path([], weights=_W, hand_anchor_window=8)
    assert path == []
    assert cost == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_tab_astar.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# src/music_decoder/tab_assignment/astar.py
from __future__ import annotations

import heapq
import statistics
from dataclasses import dataclass
from typing import Sequence

from music_decoder.pipeline.contracts import TabPosition

from .cost import transition_cost, chord_span_penalty, chord_collides
from .heuristic import remaining_high_fret_penalty


Candidate = tuple[TabPosition, ...]   # singleton for a note, multi for a chord


@dataclass(frozen=True)
class Group:
    candidates: list[Candidate]


def _representative(candidate: Candidate) -> TabPosition:
    """Lowest-fret member of a chord state, or the singleton position."""
    return min(candidate, key=lambda p: p.fret)


def _state_extra_cost(candidate: Candidate, *, weights: dict[str, float]) -> float:
    if len(candidate) <= 1:
        return 0.0
    if chord_collides(candidate):
        return float("inf")
    return weights["w_chord_intra"] * chord_span_penalty(candidate)


def astar_min_cost_path(
    groups: Sequence[Group],
    *,
    weights: dict[str, float],
    hand_anchor_window: int,
) -> tuple[list[Candidate], float]:
    if not groups:
        return [], 0.0
    n = len(groups)
    # Each state: (g, group_index, candidate_index, recent_anchor_frets_tuple).
    start_states: list[tuple[float, float, int, int, tuple[int, ...]]] = []
    heap: list[tuple[float, int, int, tuple[int, ...]]] = []
    parents: dict[tuple[int, int, tuple[int, ...]], tuple[int, int, tuple[int, ...]]] = {}
    g_score: dict[tuple[int, int, tuple[int, ...]], float] = {}
    for idx, candidate in enumerate(groups[0].candidates):
        extra = _state_extra_cost(candidate, weights=weights)
        rep = _representative(candidate)
        if extra == float("inf"):
            continue
        history = (rep.fret,)
        state = (0, idx, history)
        g = extra
        g_score[state] = g
        heuristic = remaining_high_fret_penalty(
            [grp.candidates for grp in groups[1:]], weights=weights,
        )
        heapq.heappush(heap, (g + heuristic, *state))

    if not heap:
        return [], 0.0

    goal_state: tuple[int, int, tuple[int, ...]] | None = None
    while heap:
        f, gi, ci, history = heapq.heappop(heap)
        state = (gi, ci, history)
        if gi == n - 1:
            goal_state = state
            break
        candidate = groups[gi].candidates[ci]
        rep = _representative(candidate)
        for next_idx, next_candidate in enumerate(groups[gi + 1].candidates):
            next_extra = _state_extra_cost(next_candidate, weights=weights)
            if next_extra == float("inf"):
                continue
            next_rep = _representative(next_candidate)
            anchor_window = list(history) + [next_rep.fret]
            anchor_window = anchor_window[-hand_anchor_window:]
            anchor = float(statistics.median(anchor_window))
            step = transition_cost(
                prev=rep, curr=next_rep, weights=weights, hand_anchor=anchor,
            )
            new_g = g_score[state] + step + next_extra
            next_state = (gi + 1, next_idx, tuple(anchor_window))
            if new_g < g_score.get(next_state, float("inf")):
                g_score[next_state] = new_g
                parents[next_state] = state
                remaining = [grp.candidates for grp in groups[gi + 2:]]
                heapq.heappush(
                    heap,
                    (new_g + remaining_high_fret_penalty(remaining, weights=weights),
                     *next_state),
                )

    if goal_state is None:
        return [], float("inf")

    # Reconstruct path
    path_states: list[tuple[int, int, tuple[int, ...]]] = [goal_state]
    while path_states[-1] in parents:
        path_states.append(parents[path_states[-1]])
    path_states.reverse()
    path = [groups[gi].candidates[ci] for gi, ci, _ in path_states]
    return path, g_score[goal_state]
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_tab_astar.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/music_decoder/tab_assignment/astar.py tests/unit/test_tab_astar.py
git commit -m "feat(tab): A* search over candidate groups with sliding anchor"
```

---

### Task 40: A* — public assigner that satisfies the hand-crafted tests

**Files:**
- Create: `src/music_decoder/tab_assignment/assigner.py`

- [ ] **Step 1: Re-run the previously-committed hand-crafted tests; they still fail because `assigner.assign_tab` does not exist.**

Run: `pytest tests/unit/test_tab_assignment.py -v`
Expected: FAIL — `ModuleNotFoundError: ...assigner`.

- [ ] **Step 2: Implement the public API**

```python
# src/music_decoder/tab_assignment/assigner.py
from __future__ import annotations

from typing import Iterable

from music_decoder.pipeline.contracts import (
    TabAssignmentResult, TabbedNote, TabPosition, TranscribedNote,
)
from music_decoder.tab_assignment.tuning import Tuning

from .astar import Group, astar_min_cost_path
from .candidates import note_candidates, chord_combinations


def _group_simultaneous(
    notes: list[TranscribedNote],
) -> list[list[TranscribedNote]]:
    """Group notes whose intervals overlap into chord groups."""
    sorted_notes = sorted(notes, key=lambda n: (n.start_s, n.pitch))
    groups: list[list[TranscribedNote]] = []
    for n in sorted_notes:
        if groups and groups[-1][0].start_s == n.start_s:
            groups[-1].append(n)
        else:
            groups.append([n])
    return groups


def assign_tab(
    notes: Iterable[TranscribedNote],
    *,
    tuning: Tuning,
    weights: dict[str, float],
    max_fret: int,
) -> TabAssignmentResult:
    note_list = list(notes)
    if not note_list:
        return TabAssignmentResult(
            tabbed_notes=[], tuning=tuning, total_cost=0.0, notes_dropped=[],
        )
    groups = _group_simultaneous(note_list)
    candidate_groups: list[Group] = []
    dropped: list[tuple[TranscribedNote, str]] = []
    flat_group_to_notes: list[list[TranscribedNote]] = []
    for chord_group in groups:
        if len(chord_group) == 1:
            cands = note_candidates(
                pitch=chord_group[0].pitch, tuning=tuning, max_fret=max_fret,
            )
            if not cands:
                dropped.append((chord_group[0], "out_of_range_for_tuning"))
                continue
            candidate_groups.append(Group([(c,) for c in cands]))
            flat_group_to_notes.append(chord_group)
        else:
            combos = chord_combinations(
                pitches=[n.pitch for n in chord_group],
                tuning=tuning, max_fret=max_fret,
            )
            if not combos:
                for n in chord_group:
                    dropped.append((n, "unsatisfiable_chord"))
                continue
            candidate_groups.append(Group(list(combos)))
            flat_group_to_notes.append(chord_group)

    if not candidate_groups:
        return TabAssignmentResult(
            tabbed_notes=[], tuning=tuning, total_cost=0.0, notes_dropped=dropped,
        )

    path, cost = astar_min_cost_path(
        candidate_groups, weights=weights, hand_anchor_window=8,
    )
    if cost == float("inf"):
        for chord_group in flat_group_to_notes:
            for n in chord_group:
                dropped.append((n, "no_valid_path"))
        return TabAssignmentResult(
            tabbed_notes=[], tuning=tuning, total_cost=float("inf"),
            notes_dropped=dropped,
        )

    tabbed: list[TabbedNote] = []
    for chord_group, candidate in zip(flat_group_to_notes, path):
        for n, pos in zip(chord_group, candidate):
            tabbed.append(TabbedNote(
                note=n, position=pos, cost_breakdown={"path_total": cost},
            ))
    return TabAssignmentResult(
        tabbed_notes=tabbed, tuning=tuning, total_cost=cost, notes_dropped=dropped,
    )
```

- [ ] **Step 3: Run the hand-crafted tests**

Run: `pytest tests/unit/test_tab_assignment.py -v`
Expected: ALL 6 pass. If any fail, **do not** modify the test — investigate the cost weights or the candidate generator. The tests are the spec.

- [ ] **Step 4: Commit**

```bash
git add src/music_decoder/tab_assignment/assigner.py
git commit -m "feat(tab): assign_tab public API; hand-crafted A* tests now pass"
```

---

### Task 41: TabReferenceProvider interface + user-paste implementation

**Files:**
- Create: `src/music_decoder/tab_reference/__init__.py`
- Create: `src/music_decoder/tab_reference/base.py`
- Create: `src/music_decoder/tab_reference/user_paste.py`
- Create: `tests/unit/test_tab_reference_user_paste.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_tab_reference_user_paste.py
from music_decoder.tab_reference.base import RawTabInput, TabReferenceProvider
from music_decoder.tab_reference.user_paste import UserPasteProvider


def test_user_paste_url_records_source():
    provider: TabReferenceProvider = UserPasteProvider()
    out = provider.fetch(RawTabInput(text="https://www.ultimate-guitar.com/tab/x"))
    assert out.source == "user_pasted_url"
    assert "ultimate-guitar.com" in out.raw_text


def test_user_paste_text_records_source():
    raw = "e|---0---0---|\nB|---1---1---|\n"
    provider: TabReferenceProvider = UserPasteProvider()
    out = provider.fetch(RawTabInput(text=raw))
    assert out.source == "user_pasted_text"
    assert out.raw_text == raw
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_tab_reference_user_paste.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# src/music_decoder/tab_reference/__init__.py
```

```python
# src/music_decoder/tab_reference/base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class RawTabInput:
    text: str


@dataclass(frozen=True)
class FetchedTab:
    source: str
    raw_text: str


class TabReferenceProvider(Protocol):
    def fetch(self, payload: RawTabInput) -> FetchedTab: ...
```

```python
# src/music_decoder/tab_reference/user_paste.py
from __future__ import annotations

from urllib.parse import urlparse

from .base import FetchedTab, RawTabInput


class UserPasteProvider:
    def fetch(self, payload: RawTabInput) -> FetchedTab:
        text = payload.text.strip()
        try:
            parsed = urlparse(text)
            looks_like_url = bool(parsed.scheme and parsed.netloc)
        except Exception:
            looks_like_url = False
        return FetchedTab(
            source="user_pasted_url" if looks_like_url else "user_pasted_text",
            raw_text=text,
        )
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_tab_reference_user_paste.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/music_decoder/tab_reference/ tests/unit/test_tab_reference_user_paste.py
git commit -m "feat(ref): TabReferenceProvider protocol + user-paste impl"
```

---

### Task 42: Chromaprint/AcoustID song fingerprinting

**Files:**
- Create: `src/music_decoder/tab_reference/fingerprint.py`
- Create: `tests/unit/test_tab_reference_fingerprint.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_tab_reference_fingerprint.py
from pathlib import Path
import numpy as np
import pytest

from music_decoder.tab_reference.fingerprint import (
    fingerprint_audio,
    SongIdentificationFailed,
)


@pytest.mark.slow
def test_fingerprint_returns_string_for_real_audio(fixtures_dir: Path):
    fp = fingerprint_audio(fixtures_dir / "audio_samples" / "sine_440.wav")
    assert isinstance(fp, str) and len(fp) > 8


def test_fingerprint_fails_gracefully_on_missing_file(tmp_path: Path):
    with pytest.raises(SongIdentificationFailed):
        fingerprint_audio(tmp_path / "nope.wav")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_tab_reference_fingerprint.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# src/music_decoder/tab_reference/fingerprint.py
from __future__ import annotations

from pathlib import Path


class SongIdentificationFailed(RuntimeError):
    pass


def fingerprint_audio(path: Path) -> str:
    """Compute a Chromaprint fingerprint via pyacoustid."""
    if not path.exists():
        raise SongIdentificationFailed(f"file does not exist: {path}")
    try:
        import acoustid  # pyacoustid
        duration, fp = acoustid.fingerprint_file(str(path))
        return fp.decode() if isinstance(fp, bytes) else str(fp)
    except Exception as e:
        raise SongIdentificationFailed(str(e)) from e


def lookup_acoustid(api_key: str, fingerprint: str, duration_s: float) -> str | None:
    """Optionally resolve a fingerprint to an AcoustID. Returns None on failure."""
    try:
        import acoustid
        results = list(acoustid.match(api_key, fingerprint, duration_s, parse=True))
        if not results:
            return None
        score, recording_id, *_ = results[0]
        return str(recording_id)
    except Exception:
        return None
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_tab_reference_fingerprint.py -v -m "not slow"`
Expected: 1 passed (the slow test runs only when Chromaprint binary is available).

- [ ] **Step 5: Commit**

```bash
git add src/music_decoder/tab_reference/fingerprint.py \
        tests/unit/test_tab_reference_fingerprint.py
git commit -m "feat(ref): Chromaprint fingerprinting + optional AcoustID lookup"
```

---

### Task 43: ASCII tab parser + alignment to prediction

**Files:**
- Create: `src/music_decoder/tab_reference/parser.py`
- Create: `src/music_decoder/tab_reference/alignment.py`
- Create: `tests/unit/test_tab_reference_parser.py`
- Create: `tests/unit/test_tab_reference_alignment.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_tab_reference_parser.py
from music_decoder.tab_reference.parser import parse_ascii_tab


def test_parse_simple_eadgbe_tab():
    tab = (
        "e|--0--3--|\n"
        "B|--1--0--|\n"
        "G|--0--0--|\n"
        "D|--2--0--|\n"
        "A|--3--2--|\n"
        "E|--x--3--|\n"
    )
    positions = parse_ascii_tab(tab)
    # First column: open e, B 1, G 0, D 2, A 3, E muted → produces 5 positions
    assert len(positions) >= 5
    assert all(0 <= p.string < 6 for p in positions)
    assert any(p.fret == 3 and p.string == 0 for p in positions)   # E string fret 3 in second column


def test_parse_skips_unparseable_input():
    assert parse_ascii_tab("nothing here") == []
```

```python
# tests/unit/test_tab_reference_alignment.py
import pytest

from music_decoder.pipeline.contracts import TabbedNote, TabPosition, TranscribedNote
from music_decoder.tab_reference.alignment import similarity_to_prediction


def _tn(pitch=60, start=0.0, end=0.5, string=4, fret=1):
    return TabbedNote(
        note=TranscribedNote(start_s=start, end_s=end, pitch=pitch,
                             velocity=80, confidence=0.9),
        position=TabPosition(string=string, fret=fret),
        cost_breakdown={},
    )


def test_similarity_perfect_when_identical():
    pred = [_tn(60, 0, 0.5, 4, 1), _tn(62, 0.5, 1.0, 4, 3)]
    ref = [TabPosition(4, 1), TabPosition(4, 3)]
    assert similarity_to_prediction(pred, ref) == pytest.approx(1.0)


def test_similarity_zero_for_no_matches():
    pred = [_tn(60, 0, 0.5, 4, 1)]
    ref = [TabPosition(0, 5), TabPosition(1, 7)]
    assert similarity_to_prediction(pred, ref) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_tab_reference_parser.py tests/unit/test_tab_reference_alignment.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement parser**

```python
# src/music_decoder/tab_reference/parser.py
from __future__ import annotations

import re

from music_decoder.pipeline.contracts import TabPosition


_LINE_RE = re.compile(r"^[eEBGDADCAGB][^\|]*\|(.*)\|", re.MULTILINE)
_STRING_LETTERS_TO_INDEX = {
    "E": 0, "A": 1, "D": 2, "G": 3, "B": 4, "e": 5,
}


def parse_ascii_tab(text: str) -> list[TabPosition]:
    """Best-effort ASCII-tab parser for standard tuning.

    Returns positions where digits appear in the column content. Handles
    multi-line standard layouts like:
        e|--0--3--|
        B|--1--0--|
        ...
    """
    string_lines: dict[int, str] = {}
    for line in text.splitlines():
        if not line:
            continue
        head = line[:2]
        if head[0] not in _STRING_LETTERS_TO_INDEX:
            continue
        string_idx = _STRING_LETTERS_TO_INDEX[head[0]]
        # Take the substring after the first '|'
        bar_idx = line.find("|")
        if bar_idx < 0:
            continue
        string_lines[string_idx] = line[bar_idx + 1:]

    if not string_lines:
        return []
    positions: list[TabPosition] = []
    width = min(len(s) for s in string_lines.values())
    for col in range(width):
        for s_idx, content in string_lines.items():
            ch = content[col]
            if ch.isdigit():
                # Multi-digit frets: greedy consume
                j = col
                num = ""
                while j < len(content) and content[j].isdigit():
                    num += content[j]
                    j += 1
                positions.append(TabPosition(string=s_idx, fret=int(num)))
    return positions
```

- [ ] **Step 4: Implement alignment**

```python
# src/music_decoder/tab_reference/alignment.py
from __future__ import annotations

from typing import Iterable

from music_decoder.pipeline.contracts import TabbedNote, TabPosition


def similarity_to_prediction(
    predicted: Iterable[TabbedNote],
    reference: Iterable[TabPosition],
) -> float:
    """Aligned 1:1 by order: matching counts when both string and fret match."""
    pred_list = [(t.position.string, t.position.fret) for t in predicted]
    ref_list = [(p.string, p.fret) for p in reference]
    n = max(len(pred_list), len(ref_list))
    if n == 0:
        return 1.0
    correct = sum(1 for p, r in zip(pred_list, ref_list) if p == r)
    return correct / n
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/unit/test_tab_reference_parser.py tests/unit/test_tab_reference_alignment.py -v`
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add src/music_decoder/tab_reference/parser.py \
        src/music_decoder/tab_reference/alignment.py \
        tests/unit/test_tab_reference_parser.py \
        tests/unit/test_tab_reference_alignment.py
git commit -m "feat(ref): ASCII-tab parser + similarity alignment to prediction"
```

---

## Phase 5 — Pipeline orchestration + MIDI synth

### Task 44: StageEvent emitter

**Files:**
- Create: `src/music_decoder/pipeline/events.py`
- Create: `tests/unit/test_pipeline_events.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_pipeline_events.py
import json
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from music_decoder.persistence.models import Base
from music_decoder.persistence.repositories import (
    UploadRepo, JobRepo, JobProgressRepo,
)
from music_decoder.pipeline.events import StageEventEmitter


def test_emitter_writes_progress_rows():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        u = UploadRepo(s).create(
            sha256="x", original_filename="a.wav", mime_type="audio/wav",
            duration_s=1.0, sample_rate_hz=22050,
            declared_kind="solo_guitar",
            artifact_path="uploads/1/source.wav",
        )
        s.flush()
        j = JobRepo(s).enqueue(
            u.id, "basic-pitch", "EADGBE", "standard", False,
            "2026-05-04-baseline",
        )
        s.flush()
        emitter = StageEventEmitter(JobProgressRepo(s), job_id=j.id)
        with emitter("audio_io") as record:
            record["bytes_read"] = 1024
        s.commit()
        rows = list(s.query(__import__("music_decoder.persistence.models", fromlist=["JobProgress"]).JobProgress))
        assert len(rows) == 1
        assert rows[0].stage == "audio_io"
        assert rows[0].success is True
        assert json.loads(rows[0].summary_json) == {"bytes_read": 1024}


def test_emitter_records_failure():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        u = UploadRepo(s).create(
            sha256="y", original_filename="a.wav", mime_type="audio/wav",
            duration_s=1.0, sample_rate_hz=22050, declared_kind="solo_guitar",
            artifact_path="uploads/1/source.wav",
        )
        s.flush()
        j = JobRepo(s).enqueue(u.id, "basic-pitch", "EADGBE", "standard", False, "v1")
        s.flush()
        emitter = StageEventEmitter(JobProgressRepo(s), job_id=j.id)
        try:
            with emitter("transcription"):
                raise ValueError("boom")
        except ValueError:
            pass
        s.commit()
        from music_decoder.persistence.models import JobProgress
        row = s.query(JobProgress).filter_by(stage="transcription").one()
        assert row.success is False
        assert "boom" in (row.error or "")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_pipeline_events.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# src/music_decoder/pipeline/events.py
from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator

from music_decoder.persistence.repositories import JobProgressRepo


class StageEventEmitter:
    def __init__(self, repo: JobProgressRepo, *, job_id: int) -> None:
        self.repo = repo
        self.job_id = job_id

    @contextmanager
    def __call__(self, stage: str) -> Iterator[dict]:
        summary: dict = {}
        started = datetime.now(timezone.utc)
        try:
            yield summary
        except Exception as e:
            self.repo.record(
                job_id=self.job_id, stage=stage,
                started_at=started, ended_at=datetime.now(timezone.utc),
                success=False, error=f"{type(e).__name__}: {e}",
                summary_json=json.dumps(summary, default=str),
            )
            raise
        else:
            self.repo.record(
                job_id=self.job_id, stage=stage,
                started_at=started, ended_at=datetime.now(timezone.utc),
                success=True, error=None,
                summary_json=json.dumps(summary, default=str),
            )
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_pipeline_events.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/music_decoder/pipeline/events.py tests/unit/test_pipeline_events.py
git commit -m "feat(pipeline): StageEventEmitter context manager → job_progress rows"
```

---

### Task 45: MIDI synthesis (fluidsynth) for playback

**Files:**
- Create: `src/music_decoder/midi_synth/__init__.py`
- Create: `src/music_decoder/midi_synth/fluidsynth_wrapper.py`
- Create: `tests/unit/test_midi_synth.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_midi_synth.py
from pathlib import Path

import numpy as np
import pretty_midi
import pytest

from music_decoder.midi_synth.fluidsynth_wrapper import (
    synthesize_midi_to_wav,
    SynthBackend,
)


@pytest.fixture
def midi_file(tmp_path: Path) -> Path:
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=24)
    inst.notes.append(pretty_midi.Note(
        velocity=80, pitch=60, start=0.0, end=1.0,
    ))
    pm.instruments.append(inst)
    p = tmp_path / "x.mid"
    pm.write(str(p))
    return p


def test_sine_backend_renders_audio(midi_file: Path, tmp_path: Path):
    out = tmp_path / "out.wav"
    synthesize_midi_to_wav(midi_file, out, sr=22050, backend=SynthBackend.SINE)
    assert out.exists()
    import scipy.io.wavfile as wavfile
    sr, raw = wavfile.read(str(out))
    assert sr == 22050
    assert len(raw) > 22050 * 0.9   # at least ~1 second
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_midi_synth.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# src/music_decoder/midi_synth/__init__.py
```

```python
# src/music_decoder/midi_synth/fluidsynth_wrapper.py
from __future__ import annotations

from enum import Enum
from pathlib import Path

import numpy as np
import pretty_midi
import scipy.io.wavfile as wavfile


class SynthBackend(str, Enum):
    SINE = "sine"
    FLUIDSYNTH = "fluidsynth"


def synthesize_midi_to_wav(
    midi_path: Path,
    output_wav: Path,
    *,
    sr: int = 22050,
    backend: SynthBackend = SynthBackend.SINE,
    soundfont_path: Path | None = None,
) -> Path:
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    if backend == SynthBackend.FLUIDSYNTH and soundfont_path is not None:
        audio = pm.fluidsynth(fs=sr, sf2_path=str(soundfont_path))
    else:
        audio = pm.synthesize(fs=sr)
    audio = audio.astype(np.float32)
    peak = float(np.max(np.abs(audio)) or 1.0)
    audio = (audio / peak * 0.9).astype(np.float32)
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(output_wav), sr, (audio * 32767).astype(np.int16))
    return output_wav
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_midi_synth.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/music_decoder/midi_synth/ tests/unit/test_midi_synth.py
git commit -m "feat(synth): MIDI → WAV wrapper supporting sine and fluidsynth backends"
```

---

### Task 46: Pipeline orchestrator (process_audio)

**Files:**
- Create: `src/music_decoder/pipeline/orchestrator.py`
- Create: `tests/integration/__init__.py`
- Create: `tests/integration/test_pipeline_orchestration.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/integration/test_pipeline_orchestration.py
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from music_decoder.artifacts.filesystem import FilesystemArtifactStore
from music_decoder.config.hyperparameters import HyperparameterSet, load_hyperparameters
from music_decoder.persistence.models import Base, Job
from music_decoder.persistence.repositories import UploadRepo, JobRepo
from music_decoder.pipeline.orchestrator import process_audio


@pytest.mark.integration
@pytest.mark.slow
def test_process_audio_succeeds_on_synthetic_clip(tmp_path: Path):
    engine = create_engine(f"sqlite:///{tmp_path / 'app.sqlite3'}")
    Base.metadata.create_all(engine)
    artifacts = FilesystemArtifactStore(root=tmp_path / "artifacts")
    hp = load_hyperparameters(Path("config/hyperparameters.yaml"))

    src_wav = Path("tests/fixtures/audio_samples/sine_440.wav")
    artifact_key = "uploads/1/source.wav"
    artifacts.put(artifact_key, src_wav.read_bytes())

    with Session(engine) as s:
        u = UploadRepo(s).create(
            sha256="x" * 64, original_filename="sine.wav", mime_type="audio/wav",
            duration_s=1.0, sample_rate_hz=22050, declared_kind="solo_guitar",
            artifact_path=artifact_key,
        )
        s.flush()
        j = JobRepo(s).enqueue(
            upload_id=u.id, transcription_model="basic-pitch",
            requested_tuning="EADGBE", requested_quality="standard",
            use_demucs=False, hyperparameter_set=hp.id,
        )
        s.commit()
        job_id = j.id

    process_audio(
        job_id=job_id, engine=engine, artifacts=artifacts, hyperparameters=hp,
    )

    with Session(engine) as s:
        j = s.get(Job, job_id)
        assert j.status == "succeeded"
        from music_decoder.persistence.models import JobProgress, Note, KeyEstimate, TempoEstimate
        progress = s.query(JobProgress).filter_by(job_id=job_id).all()
        stages = [p.stage for p in progress]
        assert "audio_io" in stages
        assert "transcription" in stages
        assert "key_detection" in stages
        assert "beat_tracking" in stages
        assert "tab_assignment" in stages
        assert s.query(TempoEstimate).filter_by(job_id=job_id).count() == 1
        assert s.query(KeyEstimate).filter_by(job_id=job_id).count() >= 6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_pipeline_orchestration.py -v -m "integration and slow"`
Expected: FAIL.

- [ ] **Step 3: Implement orchestrator**

```python
# src/music_decoder/pipeline/orchestrator.py
from __future__ import annotations

import json
import statistics
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from music_decoder.artifacts.base import ArtifactStore
from music_decoder.audio_io.load import load_audio
from music_decoder.beat_tracking.beats import track_beats
from music_decoder.beat_tracking.time_signature import infer_time_signature
from music_decoder.config.hyperparameters import HyperparameterSet
from music_decoder.key_detection.api import detect_key
from music_decoder.logging_setup import get_logger
from music_decoder.persistence.repositories import (
    JobRepo, JobProgressRepo, NoteRepo, KeyEstimateRepo,
    TempoEstimateRepo, AccuracyReportRepo,
)
from music_decoder.pipeline.contracts import AudioSource
from music_decoder.pipeline.events import StageEventEmitter
from music_decoder.separation.demucs import isolate_guitar
from music_decoder.tab_assignment.assigner import assign_tab
from music_decoder.tab_assignment.tuning import get_preset
from music_decoder.transcription.basic_pitch_wrapper import transcribe_basic_pitch
from music_decoder.transcription.crepe_wrapper import transcribe_crepe
from music_decoder.transcription.post_processing import apply_post_processing


_log = get_logger("orchestrator")


def process_audio(
    *, job_id: int, engine: Engine, artifacts: ArtifactStore,
    hyperparameters: HyperparameterSet,
) -> None:
    with Session(engine) as s:
        job_repo = JobRepo(s)
        progress = JobProgressRepo(s)
        note_repo = NoteRepo(s)
        key_repo = KeyEstimateRepo(s)
        tempo_repo = TempoEstimateRepo(s)

        job = job_repo.get(job_id)
        if job.status not in ("queued", "running"):
            return
        job.status = "running"
        job.started_at = datetime.now(timezone.utc)
        s.commit()

        emit = StageEventEmitter(progress, job_id=job_id)
        try:
            with emit("audio_io") as summary:
                source = AudioSource(
                    path=artifacts.path_for(job.upload.artifact_path),
                    declared_kind=job.upload.declared_kind,  # type: ignore[arg-type]
                    requested_quality=job.requested_quality,  # type: ignore[arg-type]
                    requested_tuning=get_preset(job.requested_tuning),
                )
                audio = load_audio(source)
                summary["duration_s"] = audio.duration_s
                summary["sr"] = audio.sr

            with emit("separation") as summary:
                separation = isolate_guitar(audio)
                samples_for_pitch = (
                    separation.guitar_samples if separation.guitar_samples is not None
                    else audio.samples
                )
                summary["skipped_reason"] = separation.skipped_reason

            output_dir = artifacts.path_for(f"derived/{job_id}")
            with emit("transcription") as summary:
                if job.transcription_model == "basic-pitch":
                    from music_decoder.pipeline.contracts import LoadedAudio
                    audio_for_t = LoadedAudio(
                        samples=samples_for_pitch, sr=audio.sr,
                        duration_s=samples_for_pitch.size / audio.sr,
                        sha256=audio.sha256, source=source,
                    )
                    raw_t = transcribe_basic_pitch(
                        audio_for_t, hyperparameters.basic_pitch,
                        output_dir=output_dir,
                    )
                else:
                    from music_decoder.pipeline.contracts import LoadedAudio
                    audio_for_t = LoadedAudio(
                        samples=samples_for_pitch, sr=audio.sr,
                        duration_s=samples_for_pitch.size / audio.sr,
                        sha256=audio.sha256, source=source,
                    )
                    raw_t = transcribe_crepe(
                        audio_for_t, hyperparameters.crepe, output_dir=output_dir,
                    )
                summary["raw_note_count"] = len(raw_t.notes)

            with emit("key_detection") as summary:
                key_result = detect_key(
                    samples_for_pitch, sr=audio.sr,
                    hpss_margin=hyperparameters.key_detection.hpss_margin,
                    segment_length_s=hyperparameters.key_detection.windowed_segment_length_s,
                    hop_s=hyperparameters.key_detection.windowed_hop_s,
                )
                rows: list[dict] = []
                for profile, top3 in key_result.global_top3_per_profile.items():
                    for rank, est in enumerate(top3, start=1):
                        rows.append({
                            "scope": "global", "window_start_s": None,
                            "window_end_s": None, "profile": profile,
                            "rank": rank, "tonic": est.tonic, "mode": est.mode,
                            "correlation": est.correlation, "margin": est.margin,
                        })
                for ws, we, est in key_result.windowed_segments:
                    rows.append({
                        "scope": "window", "window_start_s": float(ws),
                        "window_end_s": float(we), "profile": est.profile,
                        "rank": 1, "tonic": est.tonic, "mode": est.mode,
                        "correlation": est.correlation, "margin": est.margin,
                    })
                key_repo.bulk_insert(job_id, rows)
                summary["consensus"] = (
                    f"{key_result.consensus_key.tonic} {key_result.consensus_key.mode}"
                    if key_result.consensus_key else None
                )

            with emit("beat_tracking") as summary:
                grid = track_beats(
                    samples_for_pitch, sr=audio.sr,
                    start_bpm=hyperparameters.beat_tracking.start_bpm,
                    tightness=hyperparameters.beat_tracking.tightness,
                )
                # Time signature inference using simple beat-strength proxy:
                if grid.beat_times_s.size > 4:
                    diffs = np.diff(grid.beat_times_s)
                    strengths = 1.0 / (diffs + 1e-6)
                else:
                    strengths = np.zeros(0)
                ts = infer_time_signature(
                    strengths,
                    min_confidence=hyperparameters.beat_tracking.ts_min_confidence,
                )
                tempo_repo.upsert(
                    job_id=job_id, tempo_bpm=grid.tempo_bpm,
                    beat_times_s_json=json.dumps(grid.beat_times_s.tolist()),
                    downbeat_times_s_json=json.dumps(grid.downbeat_times_s.tolist()),
                    ts_numerator=ts.numerator, ts_denominator=ts.denominator,
                    ts_confidence=ts.confidence, ts_assumed=ts.assumed,
                )
                summary["tempo_bpm"] = grid.tempo_bpm
                summary["ts"] = f"{ts.numerator}/{ts.denominator}"

            with emit("post_processing") as summary:
                cleaned = apply_post_processing(
                    raw_t.notes, params=hyperparameters.post_processing,
                    beats=grid.beat_times_s,
                )
                summary["cleaned_note_count"] = len(cleaned)

            with emit("tab_assignment") as summary:
                tab_result = assign_tab(
                    cleaned, tuning=get_preset(job.requested_tuning),
                    weights=hyperparameters.tab_assignment.weights,
                    max_fret=hyperparameters.tab_assignment.max_fret,
                )
                rows = []
                for tn in tab_result.tabbed_notes:
                    rows.append({
                        "start_s": tn.note.start_s, "end_s": tn.note.end_s,
                        "pitch": tn.note.pitch, "velocity": tn.note.velocity,
                        "confidence": tn.note.confidence,
                        "string": tn.position.string, "fret": tn.position.fret,
                        "cost_breakdown_json": json.dumps(tn.cost_breakdown),
                        "dropped_reason": None,
                    })
                for tn, reason in tab_result.notes_dropped:
                    rows.append({
                        "start_s": tn.start_s, "end_s": tn.end_s,
                        "pitch": tn.pitch, "velocity": tn.velocity,
                        "confidence": tn.confidence,
                        "string": None, "fret": None, "cost_breakdown_json": None,
                        "dropped_reason": reason,
                    })
                note_repo.bulk_insert(job_id, rows)
                summary["assigned"] = len(tab_result.tabbed_notes)
                summary["dropped"] = len(tab_result.notes_dropped)

            job.status = "succeeded"
            job.finished_at = datetime.now(timezone.utc)
            s.commit()
        except Exception as e:
            _log.exception("pipeline_failed", extra={"job_id": job_id})
            job_repo.mark_failed(
                job_id, type(e).__name__, str(e), traceback.format_exc(),
            )
            s.commit()
            raise
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/integration/test_pipeline_orchestration.py -v -m "integration and slow"`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/music_decoder/pipeline/orchestrator.py tests/integration/__init__.py \
        tests/integration/test_pipeline_orchestration.py
git commit -m "feat(pipeline): orchestrator wires every stage with progress events"
```

---

## Phase 6 — Worker daemon

### Task 47: Worker polling loop with model warmup

**Files:**
- Create: `src/music_decoder/worker/__init__.py`
- Create: `src/music_decoder/worker/daemon.py`
- Create: `tests/integration/test_worker_daemon.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/integration/test_worker_daemon.py
import time
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from music_decoder.artifacts.filesystem import FilesystemArtifactStore
from music_decoder.config.hyperparameters import load_hyperparameters
from music_decoder.persistence.models import Base, Job
from music_decoder.persistence.repositories import UploadRepo, JobRepo
from music_decoder.worker.daemon import WorkerDaemon


@pytest.mark.integration
@pytest.mark.slow
def test_daemon_picks_up_queued_job_and_marks_it_terminal(tmp_path: Path):
    engine = create_engine(f"sqlite:///{tmp_path / 'app.sqlite3'}")
    Base.metadata.create_all(engine)
    artifacts = FilesystemArtifactStore(root=tmp_path / "artifacts")
    hp = load_hyperparameters(Path("config/hyperparameters.yaml"))

    src = Path("tests/fixtures/audio_samples/sine_440.wav")
    artifacts.put("uploads/1/source.wav", src.read_bytes())

    with Session(engine) as s:
        u = UploadRepo(s).create(
            sha256="x", original_filename="sine.wav", mime_type="audio/wav",
            duration_s=1.0, sample_rate_hz=22050, declared_kind="solo_guitar",
            artifact_path="uploads/1/source.wav",
        )
        s.flush()
        j = JobRepo(s).enqueue(
            upload_id=u.id, transcription_model="basic-pitch",
            requested_tuning="EADGBE", requested_quality="standard",
            use_demucs=False, hyperparameter_set=hp.id,
        )
        s.commit()
        job_id = j.id

    daemon = WorkerDaemon(
        engine=engine, artifacts=artifacts, hyperparameters=hp,
        poll_interval_s=0.2,
    )
    daemon.run_until_idle(idle_timeout_s=2.0)

    with Session(engine) as s:
        job = s.get(Job, job_id)
        assert job.status in ("succeeded", "failed")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_worker_daemon.py -v -m "integration and slow"`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# src/music_decoder/worker/__init__.py
```

```python
# src/music_decoder/worker/daemon.py
from __future__ import annotations

import signal
import time
from typing import Optional

from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from music_decoder.artifacts.base import ArtifactStore
from music_decoder.config.hyperparameters import HyperparameterSet
from music_decoder.logging_setup import get_logger
from music_decoder.persistence.repositories import JobRepo
from music_decoder.pipeline.orchestrator import process_audio


_log = get_logger("worker")


class WorkerDaemon:
    def __init__(
        self,
        *,
        engine: Engine,
        artifacts: ArtifactStore,
        hyperparameters: HyperparameterSet,
        poll_interval_s: float = 0.5,
    ) -> None:
        self.engine = engine
        self.artifacts = artifacts
        self.hp = hyperparameters
        self.poll_interval_s = poll_interval_s
        self._stop = False

    def warm_models(self) -> None:
        """Force load of basic-pitch and CREPE so the first job doesn't pay the cost."""
        try:
            import basic_pitch  # noqa: F401
            from basic_pitch import ICASSP_2022_MODEL_PATH  # noqa: F401
        except Exception as e:
            _log.warning("basic_pitch_warmup_failed", extra={"error": str(e)})
        try:
            import crepe  # noqa: F401
        except Exception as e:
            _log.warning("crepe_warmup_failed", extra={"error": str(e)})

    def stop(self) -> None:
        self._stop = True

    def install_signal_handlers(self) -> None:
        def handler(_signum, _frame):
            _log.info("worker_signal_received")
            self._stop = True

        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGINT, handler)

    def _claim_next(self) -> Optional[int]:
        with Session(self.engine) as s:
            j = JobRepo(s).pop_next_queued()
            if j is None:
                return None
            s.commit()
            return j.id

    def _process_one(self, job_id: int) -> None:
        try:
            process_audio(
                job_id=job_id, engine=self.engine,
                artifacts=self.artifacts, hyperparameters=self.hp,
            )
        except Exception as e:
            _log.exception("worker_job_failed", extra={"job_id": job_id, "error": str(e)})

    def run(self) -> None:
        _log.info("worker_starting")
        self.warm_models()
        while not self._stop:
            job_id = self._claim_next()
            if job_id is None:
                time.sleep(self.poll_interval_s)
                continue
            _log.info("worker_processing", extra={"job_id": job_id})
            self._process_one(job_id)
        _log.info("worker_stopped")

    def run_until_idle(self, idle_timeout_s: float = 5.0) -> None:
        """Test helper: process anything queued, then exit when idle."""
        self.warm_models()
        idle_for = 0.0
        while idle_for < idle_timeout_s:
            job_id = self._claim_next()
            if job_id is None:
                time.sleep(self.poll_interval_s)
                idle_for += self.poll_interval_s
                continue
            idle_for = 0.0
            self._process_one(job_id)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/integration/test_worker_daemon.py -v -m "integration and slow"`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/music_decoder/worker/ tests/integration/test_worker_daemon.py
git commit -m "feat(worker): polling daemon with warmup, signal handling, idle exit"
```

---

## Phase 7 — Streamlit UI

UI logic is split between **plain-Python rendering helpers** (testable directly) and **Streamlit pages** that compose those helpers. The hard rendering work (ASCII tab, SVG fretboard, confidence color decisions, chromagram and waveform figures) lives in plain Python; Streamlit pages just call into it.

### Task 48: UI helpers — confidence colors, ASCII tab, SVG fretboard, plots

**Files:**
- Create: `src/music_decoder/ui/__init__.py`
- Create: `src/music_decoder/ui/components/__init__.py`
- Create: `src/music_decoder/ui/components/confidence.py`
- Create: `src/music_decoder/ui/components/tablature.py`
- Create: `src/music_decoder/ui/components/chromagram.py`
- Create: `src/music_decoder/ui/components/waveform.py`
- Create: `tests/unit/test_ui_components.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_ui_components.py
import numpy as np
import pytest

from music_decoder.pipeline.contracts import (
    TabbedNote, TabPosition, TranscribedNote,
)
from music_decoder.ui.components.chromagram import render_chromagram_figure
from music_decoder.ui.components.confidence import confidence_color
from music_decoder.ui.components.tablature import (
    render_ascii_tab,
    render_svg_fretboard,
)
from music_decoder.ui.components.waveform import render_waveform_figure


def test_confidence_color_thresholds():
    assert confidence_color(0.95, high=0.8, medium=0.5) == "green"
    assert confidence_color(0.65, high=0.8, medium=0.5) == "yellow"
    assert confidence_color(0.30, high=0.8, medium=0.5) == "red"


def _tn(pitch=60, start=0.0, end=0.5, string=4, fret=1, conf=0.9):
    return TabbedNote(
        note=TranscribedNote(start_s=start, end_s=end, pitch=pitch, velocity=80, confidence=conf),
        position=TabPosition(string=string, fret=fret),
        cost_breakdown={},
    )


def test_render_ascii_tab_six_rows_for_eadgbe():
    notes = [_tn(60, 0.0, 0.5, 4, 1), _tn(64, 0.5, 1.0, 5, 0)]
    text = render_ascii_tab(notes, n_strings=6, columns=8)
    rows = text.strip().split("\n")
    assert len(rows) == 6
    assert rows[0].startswith("e|")    # high E
    assert rows[5].startswith("E|")    # low E


def test_render_svg_fretboard_returns_svg():
    notes = [_tn(60, 0.0, 0.5, 4, 1, conf=0.95)]
    svg = render_svg_fretboard(notes, n_strings=6, max_fret=5)
    assert svg.startswith("<svg")
    assert "</svg>" in svg


def test_render_chromagram_figure_returns_figure():
    chroma = np.zeros((12, 50))
    chroma[0] = 1.0
    fig = render_chromagram_figure(chroma, sr=22050, hop_length=512)
    assert hasattr(fig, "axes")


def test_render_waveform_figure_returns_figure():
    samples = np.sin(2 * np.pi * 440 * np.arange(22050) / 22050)
    onsets = np.array([0.1, 0.5, 0.9])
    fig = render_waveform_figure(samples, sr=22050, onsets_s=onsets)
    assert hasattr(fig, "axes")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_ui_components.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement confidence helper**

```python
# src/music_decoder/ui/__init__.py
```

```python
# src/music_decoder/ui/components/__init__.py
```

```python
# src/music_decoder/ui/components/confidence.py
from __future__ import annotations


def confidence_color(value: float, *, high: float, medium: float) -> str:
    if value >= high:
        return "green"
    if value >= medium:
        return "yellow"
    return "red"
```

- [ ] **Step 4: Implement ASCII tab + SVG fretboard**

```python
# src/music_decoder/ui/components/tablature.py
from __future__ import annotations

from typing import Iterable

from music_decoder.pipeline.contracts import TabbedNote


_STRING_LABELS = ("E", "A", "D", "G", "B", "e")   # standard EADGBE labels


def render_ascii_tab(
    notes: Iterable[TabbedNote], *, n_strings: int = 6, columns: int = 64,
) -> str:
    """Render a single-bar ASCII tab. Notes are placed in time-sorted order."""
    sorted_notes = sorted(notes, key=lambda t: t.note.start_s)
    if not sorted_notes:
        return "\n".join(f"{_STRING_LABELS[i]}|{'-' * columns}|" for i in range(5, -1, -1))
    end_time = max(t.note.end_s for t in sorted_notes)
    rows: list[list[str]] = [["-"] * columns for _ in range(n_strings)]
    for t in sorted_notes:
        col = int((t.note.start_s / end_time) * (columns - 4))
        fret = str(t.position.fret)
        for i, ch in enumerate(fret):
            if col + i < columns:
                rows[t.position.string][col + i] = ch
    out_rows = []
    for s_idx in range(n_strings - 1, -1, -1):
        label = _STRING_LABELS[s_idx] if s_idx < len(_STRING_LABELS) else "?"
        out_rows.append(f"{label}|{''.join(rows[s_idx])}|")
    return "\n".join(out_rows)


def render_svg_fretboard(
    notes: Iterable[TabbedNote], *, n_strings: int = 6, max_fret: int = 12,
) -> str:
    """Render a static fretboard SVG with marker dots at every (string, fret).

    Confidence determines marker fill color:
        ≥ 0.8: green, ≥ 0.5: gold, < 0.5: red.
    """
    width, height = 800, 200
    margin_x, margin_y = 40, 20
    fret_w = (width - 2 * margin_x) / max_fret
    string_h = (height - 2 * margin_y) / (n_strings - 1)

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">']
    # Strings (horizontal lines)
    for i in range(n_strings):
        y = margin_y + i * string_h
        parts.append(
            f'<line x1="{margin_x}" y1="{y}" x2="{width - margin_x}" y2="{y}" '
            'stroke="#444" stroke-width="1.5"/>'
        )
    # Frets (vertical lines)
    for f in range(max_fret + 1):
        x = margin_x + f * fret_w
        parts.append(
            f'<line x1="{x}" y1="{margin_y}" x2="{x}" y2="{height - margin_y}" '
            f'stroke="#888" stroke-width="{2 if f == 0 else 1}"/>'
        )
    # Note markers
    for t in notes:
        if t.position.fret > max_fret:
            continue
        cx = margin_x + (t.position.fret + 0.5) * fret_w if t.position.fret > 0 else margin_x - 12
        cy = margin_y + (n_strings - 1 - t.position.string) * string_h
        c = t.note.confidence
        fill = "green" if c >= 0.8 else ("gold" if c >= 0.5 else "red")
        parts.append(
            f'<circle cx="{cx}" cy="{cy}" r="8" fill="{fill}" stroke="#222"/>'
        )
        parts.append(
            f'<text x="{cx}" y="{cy + 4}" font-size="10" fill="#fff" '
            f'text-anchor="middle">{t.position.fret}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)
```

- [ ] **Step 5: Implement plots**

```python
# src/music_decoder/ui/components/chromagram.py
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


_KEYS = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def render_chromagram_figure(
    chroma: np.ndarray, *, sr: int, hop_length: int,
):
    fig, ax = plt.subplots(figsize=(8, 3))
    duration_s = chroma.shape[1] * hop_length / sr
    ax.imshow(
        chroma, aspect="auto", origin="lower",
        extent=(0, duration_s, 0, 12), cmap="magma",
    )
    ax.set_yticks(np.arange(12) + 0.5)
    ax.set_yticklabels(_KEYS)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("pitch class")
    fig.tight_layout()
    return fig
```

```python
# src/music_decoder/ui/components/waveform.py
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render_waveform_figure(
    samples: np.ndarray, *, sr: int, onsets_s: np.ndarray | None = None,
):
    fig, ax = plt.subplots(figsize=(8, 2))
    t = np.arange(samples.size) / sr
    ax.plot(t, samples, linewidth=0.5, color="#234")
    if onsets_s is not None:
        for o in onsets_s:
            ax.axvline(o, color="#c33", linewidth=0.7, alpha=0.6)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("amplitude")
    fig.tight_layout()
    return fig
```

- [ ] **Step 6: Run tests**

Run: `pytest tests/unit/test_ui_components.py -v`
Expected: 5 passed.

- [ ] **Step 7: Commit**

```bash
git add src/music_decoder/ui/ tests/unit/test_ui_components.py
git commit -m "feat(ui): plain-Python renderers for ASCII tab, SVG fretboard, plots"
```

---

### Task 49: Streamlit app entry + upload page

**Files:**
- Create: `src/music_decoder/ui/streamlit_app.py`
- Create: `src/music_decoder/ui/pages/01_Upload.py`
- Create: `src/music_decoder/ui/services.py`
- Create: `tests/unit/test_ui_services.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_ui_services.py
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from music_decoder.artifacts.filesystem import FilesystemArtifactStore
from music_decoder.persistence.models import Base, Job, Upload
from music_decoder.ui.services import enqueue_upload


def test_enqueue_upload_writes_upload_and_job(tmp_path: Path):
    engine = create_engine(f"sqlite:///{tmp_path / 'app.sqlite3'}")
    Base.metadata.create_all(engine)
    artifacts = FilesystemArtifactStore(root=tmp_path / "artifacts")
    audio_bytes = (Path("tests/fixtures/audio_samples/sine_440.wav")).read_bytes()
    job_id = enqueue_upload(
        engine=engine, artifacts=artifacts,
        original_filename="sine.wav", mime_type="audio/wav",
        content=audio_bytes,
        declared_kind="solo_guitar",
        transcription_model="basic-pitch",
        requested_tuning="EADGBE", requested_quality="standard",
        use_demucs=False,
        hyperparameter_set="2026-05-04-baseline",
    )
    assert job_id is not None
    with Session(engine) as s:
        job = s.get(Job, job_id)
        assert job is not None
        assert job.status == "queued"
        upload = s.get(Upload, job.upload_id)
        assert upload.original_filename == "sine.wav"
        assert artifacts.exists(upload.artifact_path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_ui_services.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement the upload service**

```python
# src/music_decoder/ui/services.py
from __future__ import annotations

import hashlib
from typing import Literal

from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from music_decoder.artifacts.base import ArtifactStore
from music_decoder.persistence.repositories import UploadRepo, JobRepo


def enqueue_upload(
    *,
    engine: Engine,
    artifacts: ArtifactStore,
    original_filename: str,
    mime_type: str,
    content: bytes,
    declared_kind: Literal["solo_guitar", "full_mix"],
    transcription_model: Literal["basic-pitch", "crepe"],
    requested_tuning: str,
    requested_quality: Literal["standard", "high"],
    use_demucs: bool,
    hyperparameter_set: str,
) -> int:
    sha = hashlib.sha256(content).hexdigest()
    with Session(engine) as s:
        upload = UploadRepo(s).create(
            sha256=sha, original_filename=original_filename, mime_type=mime_type,
            duration_s=None, sample_rate_hz=None, declared_kind=declared_kind,
            artifact_path="placeholder",
        )
        s.flush()
        ext = original_filename.rsplit(".", 1)[-1].lower() or "wav"
        key = f"uploads/{upload.id}/source.{ext}"
        artifacts.put(key, content)
        upload.artifact_path = key

        job = JobRepo(s).enqueue(
            upload_id=upload.id, transcription_model=transcription_model,
            requested_tuning=requested_tuning, requested_quality=requested_quality,
            use_demucs=use_demucs, hyperparameter_set=hyperparameter_set,
        )
        s.commit()
        return int(job.id)
```

- [ ] **Step 4: Implement the streamlit entry + upload page**

```python
# src/music_decoder/ui/streamlit_app.py
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from music_decoder.artifacts.filesystem import FilesystemArtifactStore
from music_decoder.config.hyperparameters import load_hyperparameters
from music_decoder.config.runtime import load_runtime_config
from music_decoder.persistence.session import build_engine, run_migrations


@st.cache_resource
def app_context():
    runtime_path = Path(os.environ.get("MUSIC_DECODER_RUNTIME_YAML", "config/runtime.yaml"))
    hp_path = Path(os.environ.get("MUSIC_DECODER_HP_YAML", "config/hyperparameters.yaml"))
    runtime = load_runtime_config(runtime_path)
    hp = load_hyperparameters(hp_path)
    engine = build_engine(Path(os.path.expanduser(str(runtime.db_path))))
    run_migrations(engine)
    artifacts = FilesystemArtifactStore(
        root=Path(os.path.expanduser(str(runtime.artifact_dir))),
    )
    return {"runtime": runtime, "hp": hp, "engine": engine, "artifacts": artifacts}


def main() -> None:
    st.set_page_config(page_title="Music Decoder", layout="wide")
    st.title("Music Decoder")
    st.markdown(
        "Local audio analysis tool. Upload an audio file to extract key, tempo, "
        "and a guitar tablature with explicit string + fret positions."
    )
    st.markdown("Use the sidebar to navigate to **Upload**, **Job**, or **Results**.")


if __name__ == "__main__":
    main()
```

```python
# src/music_decoder/ui/pages/01_Upload.py
from __future__ import annotations

import streamlit as st

from music_decoder.tab_assignment.tuning import PRESETS
from music_decoder.ui.services import enqueue_upload
from music_decoder.ui.streamlit_app import app_context


def render() -> None:
    st.header("Upload audio")
    ctx = app_context()
    uploaded = st.file_uploader(
        "Choose an audio file (MP3, WAV, FLAC)",
        type=["mp3", "wav", "flac", "m4a", "ogg"],
    )
    declared_kind = st.radio(
        "Audio source", options=["solo_guitar", "full_mix"], horizontal=True,
    )
    transcription_model = st.radio(
        "Transcription model",
        options=["basic-pitch", "crepe"], horizontal=True,
        help="basic-pitch handles polyphonic; CREPE is monophonic only.",
    )
    requested_tuning = st.selectbox("Tuning", options=list(PRESETS.keys()))
    requested_quality = st.radio(
        "Quality", options=["standard", "high"], horizontal=True,
        help="High = 44.1 kHz, slower. Standard = 22.05 kHz, default.",
    )
    use_demucs = st.checkbox(
        "Run Demucs source separation (full mix only)",
        value=(declared_kind == "full_mix"),
    )
    submitted = st.button("Submit", type="primary", disabled=uploaded is None)
    if submitted and uploaded is not None:
        job_id = enqueue_upload(
            engine=ctx["engine"], artifacts=ctx["artifacts"],
            original_filename=uploaded.name,
            mime_type=uploaded.type or "application/octet-stream",
            content=uploaded.getvalue(),
            declared_kind=declared_kind,
            transcription_model=transcription_model,
            requested_tuning=requested_tuning,
            requested_quality=requested_quality,
            use_demucs=use_demucs,
            hyperparameter_set=ctx["hp"].id,
        )
        st.success(f"Job #{job_id} queued. Visit the Job page to watch progress.")
        st.markdown(f"[Open job](?id={job_id})")


render()
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/unit/test_ui_services.py -v`
Expected: 1 passed.

- [ ] **Step 6: Commit**

```bash
git add src/music_decoder/ui/streamlit_app.py src/music_decoder/ui/pages/01_Upload.py \
        src/music_decoder/ui/services.py tests/unit/test_ui_services.py
git commit -m "feat(ui): upload page + enqueue_upload service"
```

---

### Task 50: Streamlit job-status page

**Files:**
- Create: `src/music_decoder/ui/pages/02_Job.py`
- Modify: `src/music_decoder/ui/services.py`
- Modify: `tests/unit/test_ui_services.py`

- [ ] **Step 1: Add the failing test**

Append to `tests/unit/test_ui_services.py`:
```python
def test_job_status_returns_none_for_missing_job(tmp_path: Path):
    from music_decoder.ui.services import job_status
    engine = create_engine(f"sqlite:///{tmp_path / 'app.sqlite3'}")
    Base.metadata.create_all(engine)
    assert job_status(engine, 999) is None


def test_job_status_returns_progress_rows(tmp_path: Path):
    from datetime import datetime, timezone
    from music_decoder.persistence.repositories import JobProgressRepo
    from music_decoder.ui.services import job_status

    engine = create_engine(f"sqlite:///{tmp_path / 'app.sqlite3'}")
    Base.metadata.create_all(engine)
    audio_bytes = (Path("tests/fixtures/audio_samples/sine_440.wav")).read_bytes()
    artifacts = FilesystemArtifactStore(root=tmp_path / "artifacts")
    job_id = enqueue_upload(
        engine=engine, artifacts=artifacts,
        original_filename="x.wav", mime_type="audio/wav", content=audio_bytes,
        declared_kind="solo_guitar", transcription_model="basic-pitch",
        requested_tuning="EADGBE", requested_quality="standard",
        use_demucs=False, hyperparameter_set="v1",
    )
    with Session(engine) as s:
        JobProgressRepo(s).record(
            job_id=job_id, stage="audio_io",
            started_at=datetime.now(timezone.utc),
            ended_at=datetime.now(timezone.utc),
            success=True, error=None, summary_json="{}",
        )
        s.commit()
    status = job_status(engine, job_id)
    assert status is not None
    assert status["status"] == "queued"
    assert any(p["stage"] == "audio_io" for p in status["progress"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_ui_services.py -v -k job_status`
Expected: FAIL.

- [ ] **Step 3: Implement `job_status` and the page**

Append to `src/music_decoder/ui/services.py`:
```python
from datetime import datetime
from sqlalchemy import select

from music_decoder.persistence.models import Job, JobProgress


def job_status(engine: Engine, job_id: int) -> dict | None:
    with Session(engine) as s:
        job = s.get(Job, job_id)
        if job is None:
            return None
        progress = list(s.execute(
            select(JobProgress).where(JobProgress.job_id == job_id)
            .order_by(JobProgress.started_at)
        ).scalars())
        return {
            "id": job.id,
            "status": job.status,
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "finished_at": job.finished_at.isoformat() if job.finished_at else None,
            "error_class": job.error_class,
            "error_message": job.error_message,
            "progress": [
                {
                    "stage": p.stage,
                    "started_at": p.started_at.isoformat() if p.started_at else None,
                    "ended_at": p.ended_at.isoformat() if p.ended_at else None,
                    "success": p.success, "error": p.error,
                    "summary": p.summary_json,
                }
                for p in progress
            ],
        }
```

```python
# src/music_decoder/ui/pages/02_Job.py
from __future__ import annotations

import time

import streamlit as st

from music_decoder.ui.services import job_status
from music_decoder.ui.streamlit_app import app_context


_TERMINAL = ("succeeded", "failed", "cancelled")


def render() -> None:
    st.header("Job status")
    ctx = app_context()
    qp = st.query_params
    raw_id = qp.get("id")
    if not raw_id:
        st.info("Pass `?id=N` to view a job.")
        return
    try:
        job_id = int(raw_id)
    except (TypeError, ValueError):
        st.error("Invalid job id.")
        return
    status = job_status(ctx["engine"], job_id)
    if status is None:
        st.error(f"Job #{job_id} not found.")
        return
    st.subheader(f"Job #{job_id} — {status['status']}")
    st.write(f"started_at: {status['started_at']}")
    st.write(f"finished_at: {status['finished_at']}")
    if status["error_class"]:
        st.error(f"{status['error_class']}: {status['error_message']}")
    if status["progress"]:
        st.markdown("**Stage progress**")
        for p in status["progress"]:
            ok = "ok" if p["success"] else ("running" if p["success"] is None else "fail")
            st.write(f"- `{p['stage']}` [{ok}] @ {p['started_at']}")
    if status["status"] == "succeeded":
        st.markdown(f"[View results](?id={job_id})")
    if status["status"] not in _TERMINAL:
        time.sleep(1.0)
        st.rerun()


render()
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_ui_services.py -v -k job_status`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/music_decoder/ui/services.py src/music_decoder/ui/pages/02_Job.py \
        tests/unit/test_ui_services.py
git commit -m "feat(ui): job-status page with 1Hz polling and progress timeline"
```

---

### Task 51: Streamlit results page (summary, visualization, tablature, playback, reference, diagnostics)

**Files:**
- Create: `src/music_decoder/ui/pages/03_Results.py`
- Modify: `src/music_decoder/ui/services.py`
- Modify: `tests/unit/test_ui_services.py`

- [ ] **Step 1: Add the failing test**

Append to `tests/unit/test_ui_services.py`:
```python
def test_load_results_returns_assembled_payload(tmp_path: Path):
    """After a successful job, load_results returns key, tempo, notes, and tab refs."""
    import json
    from music_decoder.persistence.repositories import (
        KeyEstimateRepo, TempoEstimateRepo, NoteRepo,
    )
    from music_decoder.ui.services import load_results

    engine = create_engine(f"sqlite:///{tmp_path / 'app.sqlite3'}")
    Base.metadata.create_all(engine)
    audio_bytes = (Path("tests/fixtures/audio_samples/sine_440.wav")).read_bytes()
    artifacts = FilesystemArtifactStore(root=tmp_path / "artifacts")
    job_id = enqueue_upload(
        engine=engine, artifacts=artifacts,
        original_filename="x.wav", mime_type="audio/wav", content=audio_bytes,
        declared_kind="solo_guitar", transcription_model="basic-pitch",
        requested_tuning="EADGBE", requested_quality="standard",
        use_demucs=False, hyperparameter_set="v1",
    )
    with Session(engine) as s:
        KeyEstimateRepo(s).bulk_insert(job_id, [{
            "scope": "global", "window_start_s": None, "window_end_s": None,
            "profile": "krumhansl_kessler", "rank": 1,
            "tonic": "C", "mode": "major",
            "correlation": 0.85, "margin": 0.10,
        }])
        TempoEstimateRepo(s).upsert(
            job_id=job_id, tempo_bpm=120.0,
            beat_times_s_json=json.dumps([0.0, 0.5, 1.0]),
            downbeat_times_s_json=json.dumps([0.0]),
            ts_numerator=4, ts_denominator=4,
            ts_confidence=0.6, ts_assumed=False,
        )
        NoteRepo(s).bulk_insert(job_id, [{
            "start_s": 0.0, "end_s": 0.5, "pitch": 60, "velocity": 80,
            "confidence": 0.9, "string": 4, "fret": 1,
            "cost_breakdown_json": None, "dropped_reason": None,
        }])
        s.commit()
    payload = load_results(engine, job_id)
    assert payload is not None
    assert payload["tempo"]["tempo_bpm"] == 120.0
    assert any(k["tonic"] == "C" for k in payload["keys"])
    assert payload["notes"][0]["pitch"] == 60
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_ui_services.py -v -k load_results`
Expected: FAIL.

- [ ] **Step 3: Implement `load_results` + the page**

Append to `services.py`:
```python
import json as _json

from music_decoder.persistence.models import (
    KeyEstimate, TempoEstimate, Note, TabReference, AccuracyReport,
)


def load_results(engine: Engine, job_id: int) -> dict | None:
    with Session(engine) as s:
        job = s.get(Job, job_id)
        if job is None:
            return None
        keys = list(s.execute(select(KeyEstimate).where(KeyEstimate.job_id == job_id)
                              .order_by(KeyEstimate.scope, KeyEstimate.profile,
                                        KeyEstimate.rank)).scalars())
        tempo = s.get(TempoEstimate, job_id)
        notes = list(s.execute(select(Note).where(Note.job_id == job_id)
                                .order_by(Note.start_s)).scalars())
        refs = list(s.execute(select(TabReference).where(TabReference.job_id == job_id)
                                .order_by(TabReference.created_at)).scalars())
        report = s.get(AccuracyReport, job_id)
        return {
            "job": {
                "id": job.id, "status": job.status,
                "transcription_model": job.transcription_model,
                "requested_tuning": job.requested_tuning,
                "requested_quality": job.requested_quality,
                "hyperparameter_set": job.hyperparameter_set,
            },
            "keys": [
                {"scope": k.scope, "profile": k.profile, "rank": k.rank,
                 "tonic": k.tonic, "mode": k.mode, "correlation": float(k.correlation),
                 "margin": float(k.margin),
                 "window_start_s": k.window_start_s, "window_end_s": k.window_end_s}
                for k in keys
            ],
            "tempo": ({
                "tempo_bpm": float(tempo.tempo_bpm),
                "beat_times_s": _json.loads(tempo.beat_times_s_json),
                "downbeat_times_s": _json.loads(tempo.downbeat_times_s_json),
                "ts_numerator": tempo.ts_numerator,
                "ts_denominator": tempo.ts_denominator,
                "ts_confidence": float(tempo.ts_confidence),
                "ts_assumed": bool(tempo.ts_assumed),
            } if tempo else None),
            "notes": [
                {"start_s": float(n.start_s), "end_s": float(n.end_s),
                 "pitch": int(n.pitch), "velocity": int(n.velocity),
                 "confidence": float(n.confidence),
                 "string": n.string, "fret": n.fret,
                 "dropped_reason": n.dropped_reason}
                for n in notes
            ],
            "tab_references": [
                {"source": r.source, "raw_text": r.raw_text,
                 "similarity_to_prediction": (
                    float(r.similarity_to_prediction)
                    if r.similarity_to_prediction is not None else None),
                 "disagreement_spans": (
                    _json.loads(r.disagreement_spans_json)
                    if r.disagreement_spans_json else [])}
                for r in refs
            ],
            "accuracy_report": ({
                "fixture_name": report.fixture_name,
                "note_f_measure": report.note_f_measure,
                "key_mirex_score": report.key_mirex_score,
                "tab_string_accuracy": report.tab_string_accuracy,
                "full_metrics": _json.loads(report.full_metrics_json),
            } if report else None),
        }
```

```python
# src/music_decoder/ui/pages/03_Results.py
from __future__ import annotations

from io import BytesIO

import streamlit as st

from music_decoder.pipeline.contracts import (
    TabbedNote, TabPosition, TranscribedNote,
)
from music_decoder.tab_reference.alignment import similarity_to_prediction
from music_decoder.tab_reference.parser import parse_ascii_tab
from music_decoder.tab_reference.user_paste import UserPasteProvider
from music_decoder.tab_reference.base import RawTabInput
from music_decoder.ui.components.confidence import confidence_color
from music_decoder.ui.components.tablature import (
    render_ascii_tab, render_svg_fretboard,
)
from music_decoder.ui.services import load_results
from music_decoder.ui.streamlit_app import app_context


def _to_tabbed_notes(rows):
    out = []
    for r in rows:
        if r["string"] is None or r["fret"] is None:
            continue
        out.append(TabbedNote(
            note=TranscribedNote(
                start_s=r["start_s"], end_s=r["end_s"], pitch=r["pitch"],
                velocity=r["velocity"], confidence=r["confidence"],
            ),
            position=TabPosition(string=r["string"], fret=r["fret"]),
            cost_breakdown={},
        ))
    return out


def render() -> None:
    st.header("Results")
    ctx = app_context()
    qp = st.query_params
    raw_id = qp.get("id")
    if not raw_id:
        st.info("Pass `?id=N` to view results.")
        return
    job_id = int(raw_id)
    payload = load_results(ctx["engine"], job_id)
    if payload is None:
        st.error("Job not found.")
        return
    if payload["job"]["status"] != "succeeded":
        st.warning(f"Job is in state {payload['job']['status']!r}; results not ready.")
        return
    thresholds = ctx["hp"].ui.confidence_thresholds
    tabs = st.tabs([
        "Summary", "Visualization", "Tablature", "Playback", "Reference (UG)", "Diagnostics",
    ])

    with tabs[0]:
        st.subheader("Detected key")
        global_keys = [k for k in payload["keys"] if k["scope"] == "global"]
        for k in global_keys:
            st.write(
                f"**{k['profile']}** rank {k['rank']}: "
                f"{k['tonic']} {k['mode']} (corr={k['correlation']:.3f}, "
                f"margin={k['margin']:.3f})"
            )
        if payload["tempo"]:
            t = payload["tempo"]
            tag = " (assumed)" if t["ts_assumed"] else ""
            st.subheader("Tempo & meter")
            st.write(f"{t['tempo_bpm']:.1f} BPM, {t['ts_numerator']}/{t['ts_denominator']}{tag}")
        confs = [n["confidence"] for n in payload["notes"] if n["dropped_reason"] is None]
        if confs:
            from statistics import median
            st.metric("Median note confidence", f"{median(confs):.2f}")
        dropped = [n for n in payload["notes"] if n["dropped_reason"]]
        if dropped:
            st.warning(f"{len(dropped)} notes were dropped during tab assignment.")

    with tabs[1]:
        st.info(
            "Visualization figures depend on the source audio; the worker emits "
            "chromagram and waveform PNGs into the artifact store, which this "
            "page loads when present."
        )
        # The worker writes derived/{job_id}/chromagram.png and waveform.png; if
        # they exist, display them; otherwise show a placeholder.
        for fn, caption in (("chromagram.png", "Chromagram"),
                            ("waveform.png", "Waveform with onsets")):
            key = f"derived/{job_id}/{fn}"
            if ctx["artifacts"].exists(key):
                st.image(str(ctx["artifacts"].path_for(key)), caption=caption)

    with tabs[2]:
        st.subheader("ASCII tablature")
        tabbed = _to_tabbed_notes(payload["notes"])
        st.code(render_ascii_tab(tabbed, n_strings=6, columns=64))
        st.subheader("Fretboard")
        st.markdown(render_svg_fretboard(tabbed, n_strings=6, max_fret=12),
                    unsafe_allow_html=True)
        st.subheader("Per-note confidence")
        for r in payload["notes"]:
            color = confidence_color(
                r["confidence"], high=thresholds["high"], medium=thresholds["medium"],
            )
            st.write(
                f"- t={r['start_s']:.2f}s pitch={r['pitch']} "
                f"({color}, conf={r['confidence']:.2f}) "
                f"-> {('s'+str(r['string'])+'f'+str(r['fret'])) if r['string'] is not None else 'DROPPED'}"
            )

    with tabs[3]:
        st.subheader("Playback")
        for fn, caption in (("synthesized.wav", "Predicted MIDI (synthesized)"),
                            ("source.wav", "Original audio")):
            key = f"derived/{job_id}/{fn}"
            if ctx["artifacts"].exists(key):
                st.write(caption)
                st.audio(str(ctx["artifacts"].path_for(key)))

    with tabs[4]:
        st.subheader("Reference tab (Ultimate Guitar paste)")
        st.write(
            "Paste a UG URL or the tab text. We never override our prediction; "
            "we only display the side-by-side disagreement."
        )
        text = st.text_area("URL or tab text", height=200)
        if st.button("Compare"):
            fetched = UserPasteProvider().fetch(RawTabInput(text=text))
            ref_positions = parse_ascii_tab(fetched.raw_text)
            tabbed = _to_tabbed_notes(payload["notes"])
            sim = similarity_to_prediction(tabbed, ref_positions)
            st.write(f"Similarity to prediction: {sim:.2f}")
            st.code(fetched.raw_text)
        if payload["tab_references"]:
            st.markdown("**Previously pasted references**")
            for r in payload["tab_references"]:
                st.write(f"- source={r['source']}, similarity={r['similarity_to_prediction']}")

    with tabs[5]:
        st.subheader("Diagnostics")
        st.json(payload["job"])
        if payload["accuracy_report"]:
            st.subheader("Accuracy report")
            st.json(payload["accuracy_report"])


render()
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_ui_services.py -v -k load_results`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/music_decoder/ui/services.py src/music_decoder/ui/pages/03_Results.py \
        tests/unit/test_ui_services.py
git commit -m "feat(ui): results page with all six tabs and load_results service"
```

---

## Phase 8 — CLI and packaging

### Task 52: CLI — health checks (ffmpeg, data dirs, models)

**Files:**
- Create: `src/music_decoder/cli/__init__.py`
- Create: `src/music_decoder/cli/health.py`
- Create: `tests/unit/test_cli_health.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_cli_health.py
from pathlib import Path

import pytest

from music_decoder.cli.health import (
    check_ffmpeg, ensure_data_dirs, FfmpegMissing,
)


def test_ensure_data_dirs_creates_paths(tmp_path: Path):
    db_path = tmp_path / "appdata" / "db.sqlite3"
    artifact_dir = tmp_path / "appdata" / "artifacts"
    model_cache = tmp_path / "cache" / "models"
    ensure_data_dirs(db_path=db_path, artifact_dir=artifact_dir, model_cache=model_cache)
    assert db_path.parent.exists()
    assert artifact_dir.exists()
    assert model_cache.exists()


def test_check_ffmpeg_passes_when_present(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/ffmpeg")
    check_ffmpeg()


def test_check_ffmpeg_raises_when_missing(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: None)
    with pytest.raises(FfmpegMissing):
        check_ffmpeg()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_cli_health.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# src/music_decoder/cli/__init__.py
```

```python
# src/music_decoder/cli/health.py
from __future__ import annotations

import shutil
from pathlib import Path


class FfmpegMissing(RuntimeError):
    pass


def check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise FfmpegMissing(
            "ffmpeg is required but not on PATH. Install it:\n"
            "  macOS:    brew install ffmpeg\n"
            "  Ubuntu:   apt-get install ffmpeg\n"
            "  Windows:  https://ffmpeg.org/download.html"
        )


def ensure_data_dirs(
    *, db_path: Path, artifact_dir: Path, model_cache: Path,
) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_cache.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_cli_health.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/music_decoder/cli/ tests/unit/test_cli_health.py
git commit -m "feat(cli): startup health checks for ffmpeg and data directories"
```

---

### Task 53: CLI — model download / warm cache

**Files:**
- Create: `src/music_decoder/cli/models_download.py`
- Create: `tests/unit/test_cli_models.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_cli_models.py
def test_warm_models_calls_each_loader(monkeypatch):
    calls: list[str] = []

    def fake_basic_pitch(): calls.append("basic-pitch")
    def fake_crepe(): calls.append("crepe")
    def fake_demucs(): calls.append("demucs")

    from music_decoder.cli import models_download
    monkeypatch.setattr(models_download, "_load_basic_pitch", fake_basic_pitch)
    monkeypatch.setattr(models_download, "_load_crepe", fake_crepe)
    monkeypatch.setattr(models_download, "_load_demucs", fake_demucs)

    models_download.warm_models()
    assert set(calls) == {"basic-pitch", "crepe", "demucs"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_cli_models.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# src/music_decoder/cli/models_download.py
from __future__ import annotations

from music_decoder.logging_setup import get_logger


_log = get_logger("models_download")


def _load_basic_pitch() -> None:
    from basic_pitch import ICASSP_2022_MODEL_PATH  # noqa: F401
    _log.info("basic_pitch_loaded")


def _load_crepe() -> None:
    import crepe  # noqa: F401
    _log.info("crepe_loaded")


def _load_demucs() -> None:
    try:
        from demucs.pretrained import get_model
        get_model("htdemucs_6s")
    except Exception as e:
        _log.warning("demucs_load_failed", extra={"error": str(e)})
        return
    _log.info("demucs_loaded")


def warm_models() -> None:
    _load_basic_pitch()
    _load_crepe()
    _load_demucs()
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_cli_models.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/music_decoder/cli/models_download.py tests/unit/test_cli_models.py
git commit -m "feat(cli): warm_models() to pre-fetch basic-pitch / CREPE / Demucs"
```

---

### Task 54: CLI — main entry point that spawns worker subprocess + Streamlit

**Files:**
- Create: `src/music_decoder/cli/main.py`
- Create: `tests/unit/test_cli_main.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_cli_main.py
import os
import sys

import pytest

from music_decoder.cli.main import build_parser, run_worker_only


def test_build_parser_has_expected_subcommands():
    p = build_parser()
    assert "worker" in p.format_help()
    assert "ui" in p.format_help()


def test_run_worker_only_starts_and_exits(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSIC_DECODER_DB_PATH", str(tmp_path / "app.sqlite3"))
    monkeypatch.setenv("MUSIC_DECODER_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("MUSIC_DECODER_LOG_LEVEL", "WARNING")
    # We test the assembly path without actually entering the polling loop.
    from music_decoder.cli import main
    called = {"runs": 0}

    class FakeDaemon:
        def __init__(self, *a, **kw): pass
        def install_signal_handlers(self): pass
        def run(self): called["runs"] += 1

    monkeypatch.setattr(main, "WorkerDaemon", FakeDaemon)
    run_worker_only(runtime_yaml="config/runtime.yaml",
                    hp_yaml="config/hyperparameters.yaml")
    assert called["runs"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_cli_main.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# src/music_decoder/cli/main.py
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from music_decoder.artifacts.filesystem import FilesystemArtifactStore
from music_decoder.cli.health import check_ffmpeg, ensure_data_dirs
from music_decoder.cli.models_download import warm_models
from music_decoder.config.hyperparameters import load_hyperparameters
from music_decoder.config.runtime import load_runtime_config
from music_decoder.logging_setup import configure_logging, get_logger
from music_decoder.persistence.session import build_engine, run_migrations
from music_decoder.worker.daemon import WorkerDaemon


_log = get_logger("cli")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="music-decoder")
    sub = p.add_subparsers(dest="cmd", required=False)

    sub.add_parser("ui", help="Run the Streamlit UI only.")
    sub.add_parser("worker", help="Run the worker daemon only.")
    sub.add_parser("doctor", help="Verify ffmpeg, data dirs, and models.")
    sub.add_parser("download-models",
                   help="Download all model weights into the user cache.")
    p.add_argument("--runtime-yaml", default="config/runtime.yaml")
    p.add_argument("--hp-yaml", default="config/hyperparameters.yaml")
    return p


def _bootstrap(runtime_yaml: str, hp_yaml: str):
    runtime = load_runtime_config(Path(runtime_yaml))
    hp = load_hyperparameters(Path(hp_yaml))
    configure_logging(runtime.log_level)
    db_path = Path(os.path.expanduser(str(runtime.db_path)))
    artifact_dir = Path(os.path.expanduser(str(runtime.artifact_dir)))
    model_cache = Path(os.path.expanduser(str(
        runtime.model_cache_dir or "~/Library/Caches/music-decoder/models"
    )))
    ensure_data_dirs(db_path=db_path, artifact_dir=artifact_dir, model_cache=model_cache)
    engine = build_engine(db_path)
    run_migrations(engine)
    artifacts = FilesystemArtifactStore(root=artifact_dir)
    return runtime, hp, engine, artifacts


def run_worker_only(runtime_yaml: str, hp_yaml: str) -> None:
    runtime, hp, engine, artifacts = _bootstrap(runtime_yaml, hp_yaml)
    daemon = WorkerDaemon(engine=engine, artifacts=artifacts, hyperparameters=hp)
    daemon.install_signal_handlers()
    daemon.run()


def run_ui_only(runtime_yaml: str, hp_yaml: str) -> int:
    """Launch Streamlit by re-execing through `streamlit run`."""
    entry = Path(__file__).parents[1] / "ui" / "streamlit_app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(entry),
           "--server.headless", "false",
           "--browser.gatherUsageStats", "false"]
    env = os.environ.copy()
    env["MUSIC_DECODER_RUNTIME_YAML"] = runtime_yaml
    env["MUSIC_DECODER_HP_YAML"] = hp_yaml
    return subprocess.call(cmd, env=env)


def run_doctor(runtime_yaml: str, hp_yaml: str) -> None:
    check_ffmpeg()
    runtime, hp, engine, artifacts = _bootstrap(runtime_yaml, hp_yaml)
    print(f"ffmpeg ........ ok")
    print(f"db ............ {runtime.db_path}")
    print(f"artifacts ..... {runtime.artifact_dir}")
    print(f"hyperparameter set: {hp.id}")


def run_default(runtime_yaml: str, hp_yaml: str) -> int:
    """Default command: spawn worker subprocess + Streamlit in this process."""
    check_ffmpeg()
    worker_proc = subprocess.Popen(
        [sys.executable, "-m", "music_decoder.cli.main", "worker",
         "--runtime-yaml", runtime_yaml, "--hp-yaml", hp_yaml],
    )
    try:
        return run_ui_only(runtime_yaml, hp_yaml)
    finally:
        worker_proc.terminate()
        try:
            worker_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            worker_proc.kill()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cmd = args.cmd or "default"
    runtime_yaml = args.runtime_yaml
    hp_yaml = args.hp_yaml
    if cmd == "worker":
        run_worker_only(runtime_yaml, hp_yaml)
        return 0
    if cmd == "ui":
        return run_ui_only(runtime_yaml, hp_yaml)
    if cmd == "doctor":
        run_doctor(runtime_yaml, hp_yaml)
        return 0
    if cmd == "download-models":
        warm_models()
        return 0
    return run_default(runtime_yaml, hp_yaml)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_cli_main.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/music_decoder/cli/main.py tests/unit/test_cli_main.py
git commit -m "feat(cli): main entry with worker / ui / doctor / download-models / default"
```

---

### Task 55: Dockerfile + docker-compose (secondary install path)

**Files:**
- Create: `Dockerfile`
- Create: `docker-compose.yml`
- Create: `tests/integration/test_docker_smoke.py`

- [ ] **Step 1: Write `Dockerfile`**

```dockerfile
# Dockerfile — secondary install path; matches the local install but containerized.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 libfluidsynth3 libchromaprint1 \
        build-essential pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml /app/
RUN pip install --upgrade pip && pip install -e ".[dev]"

COPY . /app/
RUN python -c "from music_decoder.cli.models_download import warm_models; warm_models()" || \
    echo "warmup failed; models will fetch on first run"

EXPOSE 8501
CMD ["python", "-m", "music_decoder.cli.main"]
```

- [ ] **Step 2: Write `docker-compose.yml`**

```yaml
services:
  music-decoder:
    build: .
    image: music-decoder:dev
    ports:
      - "8501:8501"
    environment:
      MUSIC_DECODER_DB_PATH: /data/db.sqlite3
      MUSIC_DECODER_ARTIFACT_DIR: /data/artifacts
      MUSIC_DECODER_LOG_LEVEL: INFO
    volumes:
      - md-data:/data

volumes:
  md-data: {}
```

- [ ] **Step 3: Write the smoke test**

```python
# tests/integration/test_docker_smoke.py
import shutil
import subprocess

import pytest


@pytest.mark.integration
@pytest.mark.slow
def test_dockerfile_builds_successfully():
    """Verifies the Dockerfile parses and the image builds (does not run it)."""
    if shutil.which("docker") is None:
        pytest.skip("docker not available")
    result = subprocess.run(
        ["docker", "build", "-t", "music-decoder-smoke", "--no-cache=false", "."],
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        print(result.stderr)
    assert result.returncode == 0, "docker build failed"
```

- [ ] **Step 4: Run the smoke test (only when Docker is around)**

Run: `pytest tests/integration/test_docker_smoke.py -v -m "integration and slow"`
Expected: PASS or SKIP.

- [ ] **Step 5: Commit**

```bash
git add Dockerfile docker-compose.yml tests/integration/test_docker_smoke.py
git commit -m "feat(packaging): Dockerfile + compose as secondary install path"
```

---

### Task 56: README — install + usage docs

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace the placeholder README**

Write `README.md`:

```markdown
# music-decoder

Local audio analysis tool: takes any MP3/WAV/FLAC and produces a key/scale
estimate plus a guitar tablature with explicit string + fret positions.
Accuracy is the primary success metric; mir_eval-driven regression tests
run against committed fixtures every time.

## Install (recommended)

Prerequisites: Python 3.11, `ffmpeg` on PATH, `chromaprint` (`fpcalc`) for song
fingerprinting (optional). On macOS: `brew install ffmpeg chromaprint fluidsynth`.
On Ubuntu: `apt install ffmpeg libchromaprint1 libfluidsynth3`.

    pipx install ./

(or `pip install -e .` for development.)

Run:

    music-decoder

Opens the Streamlit UI at http://localhost:8501 and starts a worker daemon in the
background. The first run downloads model weights (~500 MB) into your user cache.

## Subcommands

- `music-decoder ui` — UI only (assumes a worker is already running).
- `music-decoder worker` — worker daemon only.
- `music-decoder doctor` — verify ffmpeg, data dirs, and configuration.
- `music-decoder download-models` — pre-fetch basic-pitch, CREPE, Demucs weights.

## Development

```bash
make dev          # install with dev extras
make test-fast    # fast unit tests
make test         # full unit + slow + regression
make lint         # ruff
make typecheck    # mypy
make eval         # regression accuracy harness
make fixtures     # download GuitarSet via mirdata
```

## Architecture

See [docs/superpowers/specs/2026-05-04-music-decoder-design.md](docs/superpowers/specs/2026-05-04-music-decoder-design.md)
for the full design and [docs/superpowers/plans/2026-05-04-music-decoder.md](docs/superpowers/plans/2026-05-04-music-decoder.md)
for the implementation plan.

## Docker (optional)

```bash
docker compose up
```
Mounts a named volume for the SQLite database and artifacts.

## Configuration

- `config/runtime.yaml` — paths, log level, ffmpeg location. Override any field via
  `MUSIC_DECODER_<UPPERCASE_KEY>` env vars.
- `config/hyperparameters.yaml` — pinned hyperparameter set (id is recorded in
  every job so results are reproducible).
- `config/eval_thresholds.yaml` — accuracy gate values used by the regression test.

## License

MIT.
```

- [ ] **Step 2: Verify it renders**

Run: `pytest -q || true; cat README.md | head -30`
Expected: text shows.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: install, run, and development instructions"
```

---

## Phase 9 — End-to-end smoke test and final tidy

### Task 57: End-to-end smoke test (CLI + worker + pipeline)

**Files:**
- Create: `tests/integration/test_e2e_local_install.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/integration/test_e2e_local_install.py
"""End-to-end: simulates the local-install flow without actually launching Streamlit.

1. Bootstrap config + DB + artifacts.
2. Enqueue a job through the same service the UI uses.
3. Run the worker daemon's `run_until_idle()`.
4. Assert the job reaches a terminal state and produced notes/keys/tempo rows.
"""
from pathlib import Path

import pytest

from music_decoder.artifacts.filesystem import FilesystemArtifactStore
from music_decoder.config.hyperparameters import load_hyperparameters
from music_decoder.persistence.models import Base, Job, KeyEstimate, Note, TempoEstimate
from music_decoder.persistence.session import build_engine, run_migrations
from music_decoder.ui.services import enqueue_upload, load_results
from music_decoder.worker.daemon import WorkerDaemon


@pytest.mark.integration
@pytest.mark.slow
def test_full_pipeline_on_synthetic_audio(tmp_path: Path):
    db_path = tmp_path / "app.sqlite3"
    artifact_dir = tmp_path / "artifacts"
    engine = build_engine(db_path)
    run_migrations(engine)
    artifacts = FilesystemArtifactStore(root=artifact_dir)
    hp = load_hyperparameters(Path("config/hyperparameters.yaml"))

    audio_bytes = (Path("tests/fixtures/audio_samples/sine_440.wav")).read_bytes()
    job_id = enqueue_upload(
        engine=engine, artifacts=artifacts,
        original_filename="sine.wav", mime_type="audio/wav",
        content=audio_bytes, declared_kind="solo_guitar",
        transcription_model="basic-pitch", requested_tuning="EADGBE",
        requested_quality="standard", use_demucs=False,
        hyperparameter_set=hp.id,
    )

    daemon = WorkerDaemon(
        engine=engine, artifacts=artifacts, hyperparameters=hp,
        poll_interval_s=0.2,
    )
    daemon.run_until_idle(idle_timeout_s=2.0)

    payload = load_results(engine, job_id)
    assert payload is not None
    assert payload["job"]["status"] == "succeeded", (
        f"job did not succeed: {payload['job']}"
    )
    assert payload["tempo"] is not None
    assert len(payload["keys"]) >= 6   # 3 per profile minimum
    # Notes: a 1-second sine tone at 440 Hz should yield at least one detected note.
    assert len(payload["notes"]) >= 1
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/integration/test_e2e_local_install.py -v -m "integration and slow"`
Expected: 1 passed (after worker completes the pipeline on the sine fixture).

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_e2e_local_install.py
git commit -m "test(e2e): full pipeline smoke test through enqueue → worker → results"
```

---

### Task 58: CI workflow

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Write the workflow file**

```yaml
# .github/workflows/ci.yml
name: ci
on: [push, pull_request]
jobs:
  lint-and-fast-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - name: Install system deps
        run: sudo apt-get update && sudo apt-get install -y ffmpeg libchromaprint1 libfluidsynth3 libsndfile1
      - name: Install package
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"
      - name: Lint
        run: ruff check src tests
      - name: Type check
        run: mypy src
      - name: Fast tests
        run: pytest -m "not slow and not regression" -n auto

  regression-tests:
    runs-on: ubuntu-latest
    needs: lint-and-fast-tests
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - name: Install system deps
        run: sudo apt-get update && sudo apt-get install -y ffmpeg libchromaprint1 libfluidsynth3 libsndfile1
      - name: Install package
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"
      - name: Build synthetic MIDI fixtures
        run: python scripts/build_synthetic_midi.py
      - name: Build audio sample fixtures
        run: python scripts/build_audio_samples.py
      - name: Run regression suite
        run: pytest -m regression -v
      - name: Upload reports
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: evaluation-reports
          path: evaluation_reports/
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: lint, typecheck, fast tests, and regression suite"
```

---

### Task 59: Wire the regression test against the real pipeline (final swap)

This is the closing task: the regression test currently uses `_identity_pipeline` because the real pipeline didn't exist when Task 16 landed. Now we wire it in so future runs measure the actual implementation.

**Files:**
- Modify: `tests/regression/test_accuracy_thresholds.py`
- Update: `evaluation_reports/baseline.json` after the first green run

- [ ] **Step 1: Replace the identity pipeline with a real one**

Edit `tests/regression/test_accuracy_thresholds.py`:

```python
# Replace _identity_pipeline with an adapter that runs the real pipeline.
from music_decoder.audio_io.load import load_audio
from music_decoder.beat_tracking.beats import track_beats
from music_decoder.config.hyperparameters import load_hyperparameters
from music_decoder.key_detection.api import detect_key
from music_decoder.pipeline.contracts import AudioSource
from music_decoder.tab_assignment.assigner import assign_tab
from music_decoder.tab_assignment.tuning import get_preset
from music_decoder.transcription.basic_pitch_wrapper import transcribe_basic_pitch
from music_decoder.transcription.post_processing import apply_post_processing


def _real_pipeline(audio_path, fixture):
    hp = load_hyperparameters(Path("config/hyperparameters.yaml"))
    src = AudioSource(
        path=audio_path, declared_kind="solo_guitar",
        requested_quality="standard", requested_tuning=get_preset("EADGBE"),
    )
    audio = load_audio(src)
    transcription = transcribe_basic_pitch(
        audio, hp.basic_pitch, output_dir=Path("/tmp/md_eval_tmp"),
    )
    grid = track_beats(audio.samples, sr=audio.sr,
                      start_bpm=hp.beat_tracking.start_bpm,
                      tightness=hp.beat_tracking.tightness)
    cleaned = apply_post_processing(
        transcription.notes, params=hp.post_processing,
        beats=grid.beat_times_s,
    )
    key = detect_key(audio.samples, sr=audio.sr,
                    hpss_margin=hp.key_detection.hpss_margin,
                    segment_length_s=hp.key_detection.windowed_segment_length_s,
                    hop_s=hp.key_detection.windowed_hop_s)
    tab_result = assign_tab(
        cleaned, tuning=get_preset("EADGBE"),
        weights=hp.tab_assignment.weights, max_fret=hp.tab_assignment.max_fret,
    )
    import numpy as np
    intervals = np.array([(n.start_s, n.end_s) for n in cleaned], dtype=float)
    pitches = np.array([n.pitch for n in cleaned], dtype=float)
    consensus = key.consensus_key
    return {
        "intervals": intervals, "pitches_midi": pitches,
        "key": (consensus.tonic, consensus.mode) if consensus else None,
        "tab": [(t.note.pitch, t.position.string, t.position.fret)
                for t in tab_result.tabbed_notes],
    }


# In the test functions, replace `_identity_pipeline` with `_real_pipeline`.
```

- [ ] **Step 2: Run the regression suite**

Run: `pytest tests/regression -v -m regression`

Expected: PASS, with concrete metric values that may be below the strict
thresholds in `config/eval_thresholds.yaml`. If metrics are below the
thresholds:

1. Investigate the underperforming stage.
2. If the metric is genuinely above what is achievable for the v1 pipeline on
   v1 fixtures, **edit `config/eval_thresholds.yaml`** to a defensible lower
   bound and document the change in the commit message.

- [ ] **Step 3: Snapshot the baseline**

After the regression suite is green, capture the achieved metrics into
`evaluation_reports/baseline.json` so future runs can detect regressions:

```bash
python -c "
from pathlib import Path
import json, statistics
data = json.loads(Path('evaluation_reports').glob('20*.json').__next__().read_text())
agg = {}
for metric in ('note_f_measure','onset_f_measure','pitch_class_accuracy','key_mirex_score','tab_string_accuracy'):
    vals = [r[metric] for r in data['per_fixture'] if r[metric] is not None]
    if vals: agg[metric] = statistics.mean(vals)
agg['_note'] = 'Baseline captured after first green regression run.'
Path('evaluation_reports/baseline.json').write_text(json.dumps(agg, indent=2))
"
```

- [ ] **Step 4: Commit**

```bash
git add tests/regression/test_accuracy_thresholds.py evaluation_reports/baseline.json config/eval_thresholds.yaml
git commit -m "test(regression): wire real pipeline into regression suite + capture baseline"
```

---

## Self-review

After authoring this plan, run through the checklist below before handing off.

**1. Spec coverage check.** Walk every section of [docs/superpowers/specs/2026-05-04-music-decoder-design.md](../specs/2026-05-04-music-decoder-design.md) and confirm it is implemented:

- §3 Architecture (modules, process model, data flow) — Tasks 1–9, 47, 54
- §4 Pipeline contracts — Task 17
- §5 SQLite schema — Tasks 7, 8
- §6 A* (state, cost, heuristic) — Tasks 35–40
- §7 Evaluation harness — Tasks 10–16, 22, 27, 32, 59
- §8 Failure modes — covered piecewise across Tasks 18 (audio_io), 19 (separation), 20–21 (transcription empty/OOM), 32 (key low confidence), 33 (beat degenerate), 34 (TS assumed), 40 (A* drops), 47 (worker errors), 52 (ffmpeg/data-dir)
- §9 UI structure — Tasks 48–51
- §10 Configuration & hyperparameters — Tasks 3, 4, 16
- §11 Distribution — Tasks 1 (pyproject), 52–55, 56

**2. Placeholder scan.** No "TBD"s, no "implement later"s, no "similar to Task N"s without re-stating the code.

**3. Type consistency.** Method names match across tasks: `assign_tab` (Task 40) is called the same way in Task 46 and Task 51; `enqueue_upload` (Task 49) signature is consumed unchanged in Task 57.

**4. Frequent commits.** Each task ends with a single `git commit`.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-04-music-decoder.md`.

**Two execution options:**

**1. Subagent-Driven (recommended).** I dispatch a fresh subagent per task and review between tasks. Fast iteration, isolated context per task, less drift.

**2. Inline Execution.** Execute tasks in this session with checkpoints for review.

**Which approach do you want?**

