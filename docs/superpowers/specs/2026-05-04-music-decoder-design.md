# Music Decoder — Design

**Status:** draft, awaiting review
**Date:** 2026-05-04
**Owner:** shubhchak@gmail.com

A locally installable Python application that takes any audio file (MP3/WAV/FLAC) and produces (1) a key/scale estimate with confidence, and (2) guitar tablature with explicit string and fret positions per note. Accuracy is the primary success metric; the UI is the lens, not the goal.

This document is the source of truth for v1 architecture, schema, pipeline contracts, evaluation harness, and failure handling. The companion file `RESEARCH.md` covers technology selection rationale and accuracy expectations.

---

## 1. Goals and non-goals

### Goals

- Accept arbitrary MP3/WAV/FLAC audio (solo guitar, or full mix).
- Produce a transcription of guitar notes with timing, pitch, velocity, and per-note confidence.
- Produce a fingering (string + fret) for each note via cost-based search, parametric over tuning.
- Produce a key estimate (top-3 candidates per profile, plus consensus) using both K-K and Temperley profiles, including windowed/segment-level estimates that surface modulation.
- Produce a tempo and beat grid; infer time signature with a confidence-and-fallback strategy.
- Render results in a Streamlit UI: chromagram, waveform with onsets, ASCII tab, SVG fretboard, side-by-side synthesized vs original audio, optional Ultimate Guitar reconciliation overlay.
- Ship an evaluation harness with mir_eval-based metrics from day one. Regressions fail the build.
- Distribute as a local-install Python package (pipx-installable). No always-on web service.

### Non-goals (v1)

- Multi-user, auth, accounts.
- Public web hosting, TLS, deploy automation.
- Auto-detection of tuning (deferred to v2 — user picks from presets).
- Automated Ultimate Guitar fetching (deferred — v1 supports user-pasted tabs/URLs).
- Modal detection beyond major/minor.
- Real-time streaming input.
- Mobile UI.
- Beyond-six-string instruments (7-string, 12-string).

---

## 2. Locked decisions (from brainstorming)

1. **v1 features**: confidence-coded UI, windowed-K-S modulation detection, side-by-side synthesized vs original playback, time-signature inference (with `4/4 (assumed)` fallback when confidence is low), CREPE wired alongside basic-pitch with a user toggle, no auth.
2. **Ultimate Guitar reconciliation**: optional verification overlay only. User pastes URL or tab text; Chromaprint/AcoustID identifies the song separately. Side-by-side display with disagreements highlighted; never overrides our prediction. Behind a `TabReferenceProvider` interface so automated fetch can be added later.
3. **Tunings**: parametric `Tuning` value object. EADGBE default. Presets: Drop D, Eb (half-step down), D standard, Drop C, DADGAD.
4. **Distribution**: local-install Python package (`pipx install music-decoder`, `pip install -e .` for dev). `music-decoder` CLI starts a worker daemon and Streamlit on localhost. `ffmpeg` is a documented external dependency. Docker is a secondary path for users who want it.
5. **Persistence**: SQLite at the OS-appropriate user data dir (`~/Library/Application Support/music-decoder/db.sqlite3` on macOS; XDG paths on Linux; `%LOCALAPPDATA%` on Windows). Single-user, no auth.
6. **Fixtures**: GuitarSet (mirdata) and synthetic (fluidsynth-rendered MIDI) shipped in v1; manual user-supplied clips drop into `tests/fixtures/manual/` via a documented JSON schema, no code change required.

---

## 3. Architecture

### 3.1 Process model

Two long-lived OS processes started by the `music-decoder` CLI:

- **Streamlit process** — runs the UI on `http://localhost:8501`, performs uploads, writes upload + job rows to SQLite, polls SQLite for status and results. Never executes pipeline code itself.
- **Worker daemon process** — Python process that loads basic-pitch + CREPE + Demucs + Chromaprint models once at startup, polls SQLite at 0.5 Hz for `jobs.status='queued'` rows, executes the pipeline for each one, writes results back. Single-worker by default; configurable via env var (concurrency rarely helps a CPU-bound single-user setup).

Inter-process communication is **the SQLite database itself**. Job rows are the queue. SQLite WAL mode is enabled so the worker can read while the UI writes (and vice versa). This eliminates the need for Redis/Celery and matches the "local install, no services" shape.

### 3.2 Module boundaries

Source layout under `src/music_decoder/`:

| Module | Responsibility |
|---|---|
| `audio_io/` | ffmpeg wrapping, decode/load/resample (22050 default, 44100 high-quality), validation (silence, length, NaN), SHA-256 hashing for dedup. |
| `separation/` | Demucs `htdemucs_6s` wrapper, fail-soft to original audio if separation crashes. |
| `transcription/` | basic-pitch and CREPE wrappers; post-processing filter chain (median filter, min-duration drop, same-pitch merge, rhythmic snap); MIDI assembly via `pretty_midi`. |
| `key_detection/` | `chroma_cqt` with HPSS, K-K and Temperley profile correlations, top-3 + cross-profile consensus, windowed K-S for modulation. |
| `beat_tracking/` | `librosa.beat.beat_track` for tempo and beats; downbeat estimation; time-signature inference from beat-strength autocorrelation with a confidence threshold. |
| `tab_assignment/` | A* fret assigner, `Tuning` value object, presets, hand-crafted unit tests. |
| `tab_reference/` | `TabReferenceProvider` interface; user-paste implementation; Chromaprint/AcoustID song identification; tab-text parser; alignment + disagreement scoring against our prediction. |
| `evaluation/` | mir_eval wrappers (`note_f_measure`, `pitch_class_accuracy`, `key_mirex_score`, `tab_string_accuracy`); fixture loaders (GuitarSet/synthetic/manual); regression runner. |
| `persistence/` | SQLAlchemy models, Alembic migrations, repositories. |
| `pipeline/` | Stage contracts (typed dataclasses), single `process_audio(job_id)` entry point that calls pure stage functions sequentially and emits `StageEvent` records to `job_progress`. |
| `worker/` | Daemon entry point, polling loop, model warmup, SIGTERM handling. |
| `ui/` | Streamlit pages, polling helpers, plot rendering, MIDI synth helpers. |
| `artifacts/` | `ArtifactStore` interface; `FilesystemArtifactStore` for v1 (writes under the user's data dir). |
| `cli/` | `music-decoder` console entry point; manages worker subprocess + Streamlit. |
| `config/`, `logging/` | Settings loader, structured-JSON logger. |

### 3.3 Data flow

1. User uploads in Streamlit → `audio_io` validates → `ArtifactStore` writes to `<artifacts>/uploads/{upload_id}/source.{ext}` → `uploads` row inserted → `jobs` row inserted with `status='queued'`.
2. Worker daemon polls SQLite, picks up the queued job, sets `status='running'`, calls `pipeline.process_audio(job_id)`.
3. Each stage runs as a pure function returning a typed dataclass; between stages the worker writes a `StageEvent` row to `job_progress` (start/end/success/summary). Result dataclasses are persisted to `notes`, `key_estimates`, `tempo_estimates`, optionally `accuracy_reports`.
4. Worker sets `jobs.status='succeeded'` (or `'failed'` with error class/message/traceback).
5. Streamlit polls the `jobs` row at 1 Hz; once terminal, fetches results and renders.

### 3.4 Why a single-job pipeline, not a chain

Model load cost dominates: basic-pitch ~1.5 s, Demucs ~3 s, CREPE ~2 s on CPU. If each stage were a separate job that respawned a worker process, every job would pay reload cost three times. The persistent worker daemon loads models once at startup; `process_audio(job_id)` invokes pure stage functions in-process. We trade per-stage retry granularity for an order-of-magnitude speedup. Full-job retry is sufficient.

---

## 4. Pipeline stage contracts

Each stage is a pure function `run(input: InCls) -> OutCls`. No DB I/O inside stages; the orchestrator reads return values and writes rows. All confidence fields are floats in `[0.0, 1.0]`.

```python
# audio_io
@dataclass(frozen=True)
class AudioSource:
    path: Path
    declared_kind: Literal["solo_guitar", "full_mix"]
    requested_quality: Literal["standard", "high"]
    requested_tuning: Tuning

@dataclass(frozen=True)
class LoadedAudio:
    samples: np.ndarray            # mono float32
    sr: int                        # 22050 or 44100
    duration_s: float
    sha256: str
    source: AudioSource

# separation
@dataclass(frozen=True)
class SeparationResult:
    guitar_samples: np.ndarray | None
    sr: int
    skipped_reason: str | None     # "solo_guitar declared" | "demucs_failed: ..."
    bleed_estimate_db: float | None

# transcription
@dataclass(frozen=True)
class TranscribedNote:
    start_s: float
    end_s: float
    pitch: int                     # MIDI 0-127
    velocity: int                  # 0-127
    confidence: float

@dataclass(frozen=True)
class TranscriptionResult:
    notes: list[TranscribedNote]
    model: Literal["basic-pitch", "crepe"]
    raw_midi_path: Path
    post_midi_path: Path
    hyperparameters: dict
    median_confidence: float

# key_detection
@dataclass(frozen=True)
class KeyEstimate:
    tonic: str                     # "C", "C#", ...
    mode: Literal["major", "minor"]
    profile: Literal["krumhansl_kessler", "temperley"]
    correlation: float
    margin: float                  # corr - second_best_corr

@dataclass(frozen=True)
class KeyDetectionResult:
    global_top3_per_profile: dict[str, list[KeyEstimate]]
    consensus_key: KeyEstimate | None
    windowed_segments: list[tuple[float, float, KeyEstimate]]
    confidence: float              # global, derived from consensus + margin

# beat_tracking
@dataclass(frozen=True)
class BeatGrid:
    tempo_bpm: float
    beat_times_s: np.ndarray
    downbeat_times_s: np.ndarray
    ts_numerator: int              # 4 default
    ts_denominator: int            # 4 default
    ts_confidence: float
    ts_assumed: bool               # true when confidence < threshold

# tab_assignment
@dataclass(frozen=True)
class Tuning:
    name: str
    open_pitches: tuple[int, ...]  # MIDI numbers low-to-high

@dataclass(frozen=True)
class TabPosition:
    string: int                    # 0 = lowest string
    fret: int                      # 0 = open

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
    notes_dropped: list[tuple[TranscribedNote, str]]   # note, reason

# tab_reference
@dataclass(frozen=True)
class TabReferenceMatch:
    source: Literal["user_pasted_url", "user_pasted_text"]
    acoustid: str | None
    raw_text: str
    parsed_positions: list[TabPosition]
    similarity_to_prediction: float | None
    disagreement_spans: list[tuple[float, float]]

# orchestration
@dataclass(frozen=True)
class StageEvent:
    job_id: int
    stage: str
    started_at: datetime
    ended_at: datetime
    success: bool
    error: str | None
    summary: dict
```

The pipeline composition (executes sequentially in v1; ordering shown):

```
LoadedAudio
  → SeparationResult           (skipped if declared_kind == solo_guitar)
  → TranscriptionResult        (basic-pitch | crepe, controlled by job)
  → KeyDetectionResult         (chroma + HPSS + K-K + Temperley + windowed)
  → BeatGrid                   (tempo, beats, downbeats, TS inference)
  → TabAssignmentResult        (uses TranscriptionResult + requested Tuning)
  → TabReferenceMatch          (only when user pastes a UG URL or tab text;
                                runs as a follow-up job, not in the main run)
```

---

## 5. SQLite schema

SQLite-flavored DDL. SQLAlchemy is the actual interface; Alembic manages migrations. WAL mode is enabled at startup. JSON columns are `TEXT` with JSON content (queryable via SQLite's JSON1 extension).

```sql
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE uploads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sha256 TEXT NOT NULL,
    original_filename TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    duration_s REAL,
    sample_rate_hz INTEGER,
    declared_kind TEXT NOT NULL CHECK (declared_kind IN ('solo_guitar','full_mix')),
    artifact_path TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX uploads_sha256_idx ON uploads(sha256);

CREATE TABLE jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    upload_id INTEGER NOT NULL REFERENCES uploads(id) ON DELETE CASCADE,
    status TEXT NOT NULL CHECK (status IN ('queued','running','succeeded','failed','cancelled')),
    transcription_model TEXT NOT NULL CHECK (transcription_model IN ('basic-pitch','crepe')),
    requested_tuning TEXT NOT NULL,
    requested_quality TEXT NOT NULL CHECK (requested_quality IN ('standard','high')),
    use_demucs INTEGER NOT NULL,                 -- 0/1
    hyperparameter_set TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    error_class TEXT,
    error_message TEXT,
    error_traceback TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX jobs_upload_idx ON jobs(upload_id);
CREATE INDEX jobs_queued_idx ON jobs(status) WHERE status IN ('queued','running');

CREATE TABLE job_progress (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    stage TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    success INTEGER,                              -- 0/1, NULL = in flight
    error TEXT,
    summary_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX job_progress_job_idx ON job_progress(job_id, started_at);

CREATE TABLE notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    start_s REAL NOT NULL,
    end_s REAL NOT NULL,
    pitch INTEGER NOT NULL,
    velocity INTEGER NOT NULL,
    confidence REAL NOT NULL,
    string INTEGER,                                -- nullable when tab assignment fails
    fret INTEGER,
    cost_breakdown_json TEXT,
    dropped_reason TEXT
);
CREATE INDEX notes_job_idx ON notes(job_id, start_s);

CREATE TABLE key_estimates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    scope TEXT NOT NULL CHECK (scope IN ('global','window')),
    window_start_s REAL,
    window_end_s REAL,
    profile TEXT NOT NULL CHECK (profile IN ('krumhansl_kessler','temperley')),
    rank INTEGER NOT NULL,                         -- 1=top, 2=second, 3=third
    tonic TEXT NOT NULL,
    mode TEXT NOT NULL CHECK (mode IN ('major','minor')),
    correlation REAL NOT NULL,
    margin REAL NOT NULL
);
CREATE INDEX key_estimates_job_idx ON key_estimates(job_id, scope);

CREATE TABLE tempo_estimates (
    job_id INTEGER PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE,
    tempo_bpm REAL NOT NULL,
    beat_times_s_json TEXT NOT NULL,               -- JSON array of REAL
    downbeat_times_s_json TEXT NOT NULL,
    ts_numerator INTEGER NOT NULL,
    ts_denominator INTEGER NOT NULL,
    ts_confidence REAL NOT NULL,
    ts_assumed INTEGER NOT NULL                    -- 0/1
);

CREATE TABLE tab_references (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    source TEXT NOT NULL,                          -- 'user_pasted_url' | 'user_pasted_text'
    song_acoustid TEXT,
    raw_text TEXT NOT NULL,
    similarity_to_prediction REAL,
    disagreement_spans_json TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE accuracy_reports (
    job_id INTEGER PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE,
    fixture_name TEXT,
    note_f_measure REAL,
    onset_f_measure REAL,
    pitch_class_accuracy REAL,
    key_mirex_score REAL,
    tab_string_accuracy REAL,                      -- only when GT has string info
    self_confidence_summary_json TEXT NOT NULL,
    full_metrics_json TEXT NOT NULL
);
```

---

## 6. A* tab assignment (state space, cost, heuristic)

### 6.1 State space

Order notes by `(start_s, pitch)`. For each note `n_i`, build the candidate set
`C_i = { (s, f) : 0 ≤ s < 6, 0 ≤ f ≤ max_fret, open_pitches[s] + f == n_i.pitch }`
with `max_fret = 22` (pinned in `hyperparameters.yaml`).

For chords (notes with overlapping `[start_s, end_s]` intervals), build a single "chord state" as a tuple of one position per simultaneous note, filtered to combinations with no string collision and span ≤ 5 frets. If more than 6 notes overlap, keep the highest-confidence 6 and drop the rest.

The graph is a layered DAG: `START → C_0 → C_1 → ... → C_{N-1} → END`, with edges between consecutive groups weighted by the transition cost below.

### 6.2 Transition cost

For a transition from previous state `p` to current state `c`. When a state is a chord, its **representative position** is its lowest-fret member; movement and high-fret terms are computed against the representative.

```
c(p → c) =
    w_move          · |c.repr.fret   - p.repr.fret|
  + w_string        · |c.repr.string - p.repr.string|
  + w_span          · max(0, c.repr.fret - hand_anchor)^1.5
  + w_high          · max(0, c.repr.fret - 12)^1.2
  + w_open          · (-1 if c.repr.fret == 0 else 0)
  + w_chord_intra   · chord_span_penalty(c)
  + w_chord_collide · (∞ if c is a chord with a string collision else 0)
```

- `hand_anchor` is the sliding median of the last K=8 played representative frets (encourages position stability across the song).
- `chord_span_penalty(c) = max(0, max_fret_in_chord - min_fret_in_chord - 4)^1.5` (zero for solo notes).

Default weights, pinned in `hyperparameters.yaml`, must be empirically tuned against fixtures:

| Weight | Default |
|---|---|
| `w_move` | 1.0 |
| `w_string` | 0.3 |
| `w_span` | 0.5 |
| `w_high` | 0.4 |
| `w_open` | 0.2 |
| `w_chord_intra` | 0.6 |
| `w_chord_collide` | ∞ |

### 6.3 Heuristic

```
h(state at note i) = Σ_{j > i} w_high · max(0, min_fret(C_j) - 12)^1.2
```

The sum, over all remaining notes, of just the high-fret penalty using the *lowest possible fret* for each remaining note. This is admissible: any actual continuation has fret ≥ `min_fret` per note, and we ignore movement, string-change, span, chord, and open-string terms (all ≥ 0). The bound is tight enough to prune effectively because high-fret penalty often dominates path cost on most songs.

### 6.4 Algorithm

1. Build groups `G_0 ... G_{N-1}` (singleton candidate sets for solo notes; combination sets for chords).
2. Run A* over `(group_index, candidate_index)` using a binary heap on `g + h`, with `g` the accumulated transition cost and `h` the heuristic above.
3. Reconstruct the min-cost path → `TabAssignmentResult`.

### 6.5 Edge cases

- Note with empty `C_i` (out of tuning range) — drop with `dropped_reason='out_of_range_for_tuning'`; the gap stays in the path with no penalty.
- Chord with empty post-filter combination set — drop chord with `'unsatisfiable_chord'`.
- More than 6 simultaneous notes — keep top 6 by confidence, drop the rest.

### 6.6 Hand-crafted unit cases

Tests are written *before* the implementation:

- C-major scale ascending → expected fingering: open / 2 / open / 2 / 3 across A, D, G, B strings; verify position stability.
- G-major chord → expected open-position fingering (3-2-0-0-0-3 across E-A-D-G-B-e).
- Melody crossing 12th fret — verify position-stability heuristic prefers staying in upper position over flipping to open.
- Drop D test — same scale played in Drop D vs EADGBE; verify low D is preferred on the open lowest string in Drop D.

---

## 7. Evaluation harness

### 7.1 Fixture sources

`evaluation/fixtures.py` provides three loaders, each yielding `Fixture(audio_path, ground_truth, source)` records.

- **GuitarSet** via mirdata. 3–5 excerpts. JAMS ground truth includes hexaphonic per-string tab; supports `tab_string_accuracy`.
- **Synthetic** — committed `.mid` files in `tests/fixtures/synthetic/`, rendered to WAV at evaluation time via fluidsynth + a public-domain GeneralUser GS soundfont (committed, license-cleared). Ground truth is MIDI pitches only — no string assignment, so `tab_string_accuracy` is not computed for these.
- **Manual** — user-supplied. Drop a JSON+audio pair into `tests/fixtures/manual/`. When the manual fixture authors include `string` fields, `tab_string_accuracy` is computed; otherwise it is skipped for that fixture. Schema:

```json
{
    "audio": "clip01.wav",
    "tuning": "EADGBE",
    "key": {"tonic": "G", "mode": "major"},
    "tempo_bpm": 120,
    "tab": [
        {"start_s": 0.0, "end_s": 0.5, "pitch": 67, "string": 3, "fret": 0}
    ]
}
```

The loader auto-discovers via glob; `tests/fixtures/manual/.gitkeep` is committed; user clips can be added later with no code change.

### 7.2 Metrics

`evaluation/metrics.py` — pure functions wrapping mir_eval:

- `note_f_measure(predicted, gt, onset_tolerance_s=0.05, pitch_tolerance_cents=50)` — MIREX defaults.
- `onset_f_measure(predicted, gt, tolerance_s=0.05)` — onset-only.
- `pitch_class_accuracy(predicted, gt)` — frame-level pitch-class match rate.
- `key_mirex_score(predicted_key, gt_key)` — MIREX weighted: correct=1.0, perfect-fifth=0.5, relative=0.3, parallel=0.2, else=0.0.
- `tab_string_accuracy(predicted_tabbed, gt_tabbed)` — per-note: correct iff pitch matches *and* string matches; only computable when ground truth has string information.

### 7.3 Runner and reports

`evaluation/runner.py` exposes `run_evaluation(pipeline_fn, fixture_set) -> EvaluationReport`. It iterates fixtures, executes the full pipeline, computes metrics, writes JSON to `evaluation_reports/{timestamp}.json`, and inserts an `accuracy_reports` row when invoked from inside a Celery-equivalent worker job.

### 7.4 Regression test and CI

`tests/test_regression.py` is a pytest test that:

1. Loads the committed fixture set (GuitarSet cache + synthetic + manual).
2. Runs the full harness.
3. Asserts each metric ≥ the pinned threshold from `config/eval_thresholds.yaml`.
4. Compares to `evaluation_reports/baseline.json` (committed). If any metric drops more than `regression_tolerance` (default 0.02) below baseline, the test fails.

Baselines are updated by hand on intentional regressions and committed alongside the changes that caused them. CI runs the regression test inside the pinned Docker image so models and native deps are reproducible.

---

## 8. Failure modes

| Failure | Detection | Handling |
|---|---|---|
| Corrupt audio (ffmpeg nonzero, NaN samples) | `audio_io.load()` validates output | Job fails, `error_class='corrupt_audio'`. UI: "Couldn't decode this file." |
| Silent audio (RMS below threshold for entire duration) | `audio_io` validation | Job fails, `error_class='silent_audio'`. UI: "File appears silent." |
| Very short clip (< 1.0 s) | `audio_io` length check | Job fails, `error_class='clip_too_short'`. UI: "Clip too short for analysis." |
| Demucs OOM or crash | Subprocess error in `separation/` | Skip separation, fall through to original audio with `skipped_reason`. UI banner: "Source separation skipped." |
| Empty MIDI (basic-pitch returns 0 notes) | Post-stage check | If audio confidence is also low → empty result with banner; otherwise continue with empty tab. |
| Out-of-range notes for tuning | `tab_assignment` candidate set empty for note | Drop note with reason; UI lists count + per-note flag. |
| Key profile disagreement, low margins | Compare K-K vs Temperley top-1 | `consensus_key=None`, top-3 per profile reported, UI yellow "Low-confidence key" banner. |
| Beat tracker degenerate (tempo OOR or < 4 beats) | `beat_tracking/` validation | `ts_assumed=true`, rhythmic snap post-filter skipped, UI flags low tempo confidence. |
| TS inference confidence below threshold | Compare to `ts_min_confidence` | `ts_assumed=true`, display "4/4 (assumed)". |
| A* finds no valid path (impossible chord) | A* fails on chord state | Drop chord with `'unsatisfiable_chord'`, continue. |
| Worker OOM | Process exits unexpectedly | Job marked `error_class='out_of_memory'`, no retry. UI: "File too large; try standard quality or shorter clip." |
| SQLite I/O error or disk full | DB exception | Streamlit shows actionable error; worker logs and exits cleanly. |
| Models not yet downloaded | Worker startup check | First run triggers a one-time download with a Streamlit progress banner. |
| `ffmpeg` missing | CLI startup check | `music-decoder` exits with a clear "Please install ffmpeg" message and platform-specific instructions. |
| Chromaprint match fails | `tab_reference/` exception | Skip song-ID step; still accept user-pasted UG tab. |
| User-pasted UG tab unparseable | Parser failure | Show raw text only, `similarity_to_prediction=null`, banner: "Couldn't parse this tab format." |

---

## 9. Streamlit UI structure

Multi-page Streamlit app:

- `pages/01_Upload.py` — file uploader; declared kind (solo guitar / full mix); tuning preset dropdown (EADGBE, Drop D, Eb, D standard, Drop C, DADGAD); transcription model (basic-pitch / CREPE); quality (standard / high); submit.
- `pages/02_Job.py?id=N` — live status polled at 1 Hz from `jobs` + `job_progress`. Stage timeline. Auto-redirect on `succeeded`.
- `pages/03_Results.py?id=N` — main viewer; internal tab strip:
  - **Summary** — detected key (top-3 + cross-profile agreement), tempo + meter ("(assumed)" annotation when applicable), median note confidence, total / dropped note counts.
  - **Visualization** — chromagram heatmap (matplotlib), waveform with note onsets, beat grid markers.
  - **Tablature** — ASCII tab + SVG fretboard. Per-note color: green ≥0.8, yellow ≥0.5, red <0.5 (thresholds pinned). Out-of-range / dropped notes listed beneath.
  - **Playback** — synthesized MIDI WAV (rendered by worker via fluidsynth, cached in artifacts) and original audio side-by-side, two HTML5 `<audio>` elements.
  - **Reference (UG)** — paste URL or tab text; reconciliation runs as a follow-up worker job; side-by-side diff with disagreement spans highlighted. Never overrides the prediction.
  - **Diagnostics** — per-stage timings, hyperparameter set ID, raw vs filtered note counts, accuracy report (when fixture).

Conventions:

- Confidence color helper shared across all per-note UI.
- Page state via query params (shareable URLs); no Streamlit session state for job-specific data.
- Auto-rerun every 1 s while job is `queued|running`; stop polling once terminal.

---

## 10. Configuration and hyperparameter management

Two YAML files under `config/`:

- `config/runtime.yaml` — DB path, artifact dir, log level, fixture path, `ffmpeg` path, model cache dir. Overridable via `MUSIC_DECODER_*` env vars (12-factor).
- `config/hyperparameters.yaml` — pinned hyperparameter set with an `id` (e.g. `2026-05-04-baseline`). Every `jobs` row records `hyperparameter_set`; any result is reproducible by checking out the matching commit.

`hyperparameters.yaml` (excerpted):

```yaml
id: 2026-05-04-baseline
basic_pitch:
  onset_threshold: 0.5
  frame_threshold: 0.3
  minimum_note_length_ms: 58
  minimum_frequency_hz: 32.7      # E1
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
  weights: {w_move: 1.0, w_string: 0.3, w_span: 0.5, w_open: 0.2, w_high: 0.4, w_chord_intra: 0.6}
  max_fret: 22
ui:
  confidence_thresholds: {high: 0.8, medium: 0.5}
evaluation:
  thresholds: {note_f_measure: 0.65, key_mirex_score: 0.75, tab_string_accuracy: 0.55}
  regression_tolerance: 0.02
```

Sweeps via `scripts/sweep.py` over a Cartesian product → `evaluation_reports/sweep-{timestamp}.json`. Promotion is **manual**: edit and commit `hyperparameters.yaml`. No auto-promotion.

---

## 11. Distribution and runtime

### 11.1 Package layout

Standard Python package with `pyproject.toml`. Console entry point `music-decoder` (defined under `[project.scripts]`) maps to `music_decoder.cli:main`.

### 11.2 First run

```
$ pipx install music-decoder        # or: pip install -e . for dev
$ music-decoder
[*] Checking ffmpeg ................ OK
[*] Initializing data dir at ~/Library/Application Support/music-decoder
[*] Running database migrations .... OK
[*] Downloading models (one-time, ~500 MB):
    - basic-pitch ................... OK
    - CREPE (full) .................. OK
    - Demucs htdemucs_6s ............ OK
    - Chromaprint ................... OK
[*] Starting worker daemon (pid 12345)
[*] Starting Streamlit at http://localhost:8501
```

The CLI manages the worker subprocess lifecycle (spawn on start, SIGTERM on shutdown).

### 11.3 Docker (secondary path)

`Dockerfile` and `docker-compose.yml` are provided for users who prefer containers. The compose file defines a single service that runs the same `music-decoder` CLI with a mounted volume for the data dir. CI uses this image.

---

## 12. Open questions and future work

Tracked here, not in v1 scope:

- Auto-detection of tuning from audio.
- Modal detection beyond major/minor (Dorian, Mixolydian, etc.).
- Transformer-based fret assignment (Fretting-Transformer line) as an alternative to A*.
- Automated UG fetch via fingerprint → public APIs (legal review required).
- 7-string and 12-string instruments.
- Multi-track / multi-instrument transcription.
- A "sweep + auto-tune hyperparameters" workflow with Bayesian optimization.

---

## 13. References

- See `RESEARCH.md` in the repo root for technology rationale, accuracy expectations, and the full pipeline-stage literature survey.
- mir_eval: https://github.com/mir-evaluation/mir_eval
- mirdata GuitarSet: https://mirdata.readthedocs.io/en/stable/source/mirdata.html
- Spotify basic-pitch: https://github.com/spotify/basic-pitch
- CREPE: https://github.com/marl/crepe
- Demucs: https://github.com/facebookresearch/demucs
