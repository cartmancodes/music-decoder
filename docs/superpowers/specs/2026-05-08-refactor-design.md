# Music Decoder v2 — Refactor & Refocus Design

**Status:** draft, awaiting review
**Date:** 2026-05-08
**Owner:** shubhchak@gmail.com
**Supersedes:** [2026-05-04-music-decoder-design.md](./2026-05-04-music-decoder-design.md), [2026-05-05-chord-progression-design.md](./2026-05-05-chord-progression-design.md), [2026-05-07-phase-c-design.md](./2026-05-07-phase-c-design.md)

A focused rewrite of the existing music-decoder project around three core
features. The existing codebase has accumulated subsystems (training,
fine-tuning, evaluation harness, persistence, worker daemon, Ultimate Guitar
reconciliation, fixture-authoring CLI) that are not load-bearing for the three
features we want to keep. This document is the new source of truth for v2
architecture.

---

## 1. The three features

1. **Identify chord progression** of any YouTube link or local MP3/WAV/FLAC.
2. **Identify guitar tablature** of any YouTube link or local MP3/WAV/FLAC.
3. **Suggest a music composition** given a scale and a chord progression — full
   simple arrangement: chord accompaniment + melody line + audio render (WAV)
   + ASCII tab + MIDI.

Everything else is removed.

### Non-goals (v2)

- Web hosting, multi-user, auth.
- Persistent job queue / database (no SQLite, no Alembic, no SQLAlchemy).
- Worker daemon, multi-process IPC.
- Model fine-tuning / training pipeline.
- Ultimate Guitar reconciliation, AcoustID fingerprinting, tab-paste parsing.
- GuitarSet downloader, fixture-authoring CLI, manual fixture loader.
- Modal detection beyond major/minor.
- Real-time streaming input.
- Mobile UI, multi-track transcription, 7/12-string instruments.

---

## 2. Locked decisions (from brainstorming)

1. **Composition output** = full simple arrangement (chord comp + melody +
   MIDI + WAV + ASCII tab). Rule-based + 1st-order Markov pitch model. No ML,
   no training. Deterministic given a seed.
2. **Process shape** = CLI-first library, optional Streamlit UI. Single
   in-process synchronous Python API. No worker, no DB, no IPC.
3. **Quality gating** = a slim regression harness against committed synthetic
   fixtures, gating `chord_recognition_score`, `key_mirex_score`, and
   `tab_string_accuracy` on every PR. Frozen thresholds in
   `config/eval_thresholds.yaml`. No GuitarSet, no fixture authoring.
4. **Scale support** at v2: 12 tonics × {major, minor}. Modes (Dorian,
   Mixolydian, etc.) are deferred to v3.
5. **YouTube ingestion** via `yt-dlp` audio-only stream → ffmpeg → WAV cache
   keyed by video ID under the user data dir.
6. **No persistence**: re-runs re-analyze from scratch. Only the YouTube
   download is cached; analysis results are not.
7. **Docker is updated** to drop the SQLite volume and DB env vars (no
   Postgres existed in the repo). Docker remains a secondary install path.

---

## 3. Architecture

### 3.1 Process shape

A single in-process Python library. Two thin frontends both call the library
synchronously.

```
              ┌─────────────────────┐
   CLI ──────►│                     │
              │    music_decoder    │──► artifacts on disk (WAV, MIDI, JSON)
   UI  ──────►│   (sync library)    │
              │                     │
              └─────────────────────┘
                       │
                       ▼
   ffmpeg, yt-dlp, basic-pitch, demucs, madmom, fluidsynth (subprocesses /
   loaded models inside the same process)
```

There is no daemon. There is no SQLite. There is no job queue. The Streamlit
UI imports the library and calls it directly; results live in
`st.session_state`. Long-running stages emit progress through a callback.

### 3.2 Module layout

Source layout under `src/music_decoder/`:

| Module | Responsibility |
|---|---|
| `api.py` | Public synchronous entry points: `analyze()`, `compose()`. |
| `types.py` | Frozen dataclasses shared across modules (`AnalysisResult`, `Composition`, `Scale`, `ChordSymbol`, `KeyEstimate`, `ChordSegment`, `TabbedNote`, `Note`, `VoicedChord`). |
| `errors.py` | Typed exception hierarchy. |
| `progress.py` | `ProgressCallback` protocol + null implementation. |
| `ingest/audio_file.py` | ffmpeg-backed MP3/WAV/FLAC loader; SHA-256 hash; mono float32 at configured SR. |
| `ingest/youtube.py` | `yt-dlp` wrapper; URL detection; cache under `<data_dir>/yt_cache/{video_id}.wav`; transcoded via ffmpeg. |
| `ingest/__init__.py` | `load(source: str \| Path) -> LoadedAudio` — single entry point that detects URL vs path. |
| `dsp/chroma.py` | CQT + HPSS chroma (moved out of `key_detection/chroma.py`). |
| `dsp/beats.py` | librosa-based tempo/beat tracking (was `beat_tracking/beats.py`). |
| `separation/` | Demucs `htdemucs_6s` wrapper (kept; pure function). |
| `transcription/` | basic-pitch wrapper + post-processing. CREPE wrapper, the high-resolution guitar wrapper, and the basic-pitch fine-tuning compat shim are deleted. |
| `chords/` | Was `chord_detection/`. Templates, recognition (Viterbi over template scores), voicings, madmom backend. |
| `key/` | Was `key_detection/`. KS profiles, global estimator. The windowed estimator is kept (it's used in the analysis result for "key changes" hints). |
| `tabs/assigner.py` | A* fret search (was `tab_assignment/`). |
| `tabs/tuning.py` | `Tuning` value object + presets. |
| `tabs/render.py` | ASCII + SVG renderers (moved from `ui/components/tablature.py`). |
| `compose/voicings.py` | `ChordSymbol` + `Tuning` → 1–3 playable voicings. Reuses `chords/voicings.py` as seed. |
| `compose/melody.py` | Diatonic melody generator (rule-based + 1st-order Markov over scale degrees). |
| `compose/arrangement.py` | Combines melody + accompaniment into a `pretty_midi.PrettyMIDI` object. |
| `compose/api.py` | `compose(scale, progression, ...) -> Composition`. |
| `synth/` | Was `midi_synth/`. fluidsynth WAV rendering. |
| `config/runtime.py` | Runtime settings (paths, log level, ffmpeg path) loaded from `config/runtime.yaml` + `MUSIC_DECODER_*` env vars. |
| `config/hyperparameters.py` | Pinned hyperparameter set loader. |
| `cli/main.py` | Click entry point: `analyze`, `compose`, `ui`, `doctor`. |
| `ui/streamlit_app.py` | Single-page Streamlit app with three tabs (Analyze / Tabs / Compose). |
| `logging_setup.py` | Kept. Structured JSON logger. |

**Modules deleted entirely:** `worker/`, `persistence/`, `migrations/`,
`alembic.ini`, `pipeline/`, `artifacts/`, `tab_reference/`, `training/`,
`evaluation/fixtures/guitarset.py`, `evaluation/fixtures/manual.py`,
`evaluation/regression.py` (replaced by a slim integration test in
`tests/integration/`), `evaluation/runner.py`, `cli/fixture.py`,
`cli/models_download.py` (folded into `doctor`),
`transcription/crepe_wrapper.py`,
`transcription/highres_guitar_wrapper.py`,
`transcription/basic_pitch_wrapper.py` is **kept** (only the high-res +
CREPE wrappers go), `ui/pages/` (multi-page flow replaced by single page),
`ui/services.py` (DB-coupled), `scripts/build_audio_samples.py`,
`scripts/build_synthetic_midi.py` (synthetic `.mid` fixtures already
committed), `scripts/convert_guitarset_to_tfrecord.py`,
`scripts/eval_checkpoint.py`, `scripts/finetune_basic_pitch.py`,
`scripts/render_synthetic.py`. **Survivors under `scripts/`:**
`download_soundfont.py` (used by CI to populate the fluidsynth soundfont).

**Survivors under `evaluation/`:** `evaluation/metrics.py` (pure mir_eval
wrappers; ~150 LoC), `evaluation/fixtures/synthetic.py`,
`evaluation/fixtures/base.py`, and the corresponding `__init__.py` files.
Everything else listed above is deleted.

**Files updated:** `pyproject.toml`, `Dockerfile`, `docker-compose.yml`,
`Makefile`, `README.md`, `RESEARCH.md`.

**Modules kept and renamed:** see table above. Renames preserve module-level
public APIs where reasonable to minimize churn inside the kept modules.

### 3.3 Data flow

#### Analyze

```
source: str|Path
  → ingest.load                        (URL → yt-dlp+ffmpeg cache; path → ffmpeg)
  → LoadedAudio
  → separation.run                     (skipped if declared_kind == solo_guitar)
  → SeparationResult
  → dsp.beats.track                    (tempo + beat times)
  → BeatGrid
  → dsp.chroma.compute                 (CQT + HPSS)
  → Chroma
  → key.estimate                       (KS top-3 + consensus + windowed)
  → KeyEstimate
  → chords.recognize                   (madmom backend → ChordSegment list)
  → list[ChordSegment]
  → transcription.run                  (basic-pitch + post-processing)
  → list[TranscribedNote]
  → tabs.assign                        (A*)
  → list[TabbedNote]
  → AnalysisResult
```

`analyze()` returns the full `AnalysisResult`. The orchestration is a single
function in `api.py` (~80 LoC). No DB writes, no progress rows, no job IDs.
Progress is emitted via the optional `ProgressCallback` protocol so the
Streamlit UI can show a progress bar without coupling the library to
Streamlit.

#### Compose

```
(scale, progression, style, tempo, bars, seed)
  → compose.voicings                   (ChordSymbol → VoicedChord)
  → list[VoicedChord]
  → compose.melody                     (Scale + chord tones → list[Note])
  → list[Note]
  → compose.arrangement                (melody + accompaniment → PrettyMIDI)
  → midi_path
  → tabs.render                        (TabbedNote list → ASCII tab)
  → ascii_tab
  → synth.render                       (PrettyMIDI → WAV via fluidsynth)
  → wav_path
  → Composition
```

`compose()` is fully synchronous and deterministic given `seed`. Output paths
are written under `<data_dir>/compositions/{timestamp}-{hash}/` so the user
can keep multiple takes.

### 3.4 Why no persistence

The original v1 persistence layer existed because (a) the worker daemon and
UI were separate processes and needed an IPC channel, and (b) job retries
needed durable state. v2 has a single in-process synchronous API: there is no
IPC, no retry boundary, and the user can re-run `analyze()` deterministically
on the same input. Saving artifacts to disk (audio cache, composition output)
is enough; tracking jobs in a database is not.

The cost of dropping persistence:

- The Streamlit UI cannot resume a previous job after a refresh. The user
  re-clicks Analyze. Acceptable for single-user local tooling.
- We lose the audit trail of past analyses. Acceptable; `analyze()` is fast
  enough (basic-pitch + Demucs ~10–30 s on CPU) and re-running is cheap.

### 3.5 Why a single synchronous API

Models load once per Python process. The CLI loads models on each invocation
(acceptable; CLI users run a handful of commands per session). The Streamlit
UI loads them once when the app starts and caches them via `st.cache_resource`.
Tests load them lazily via `functools.cache`-wrapped factories.

This trades the worker daemon's "persistent process" benefit for radical
simplicity. For single-user local tooling it is the right trade.

---

## 4. Public API

```python
# music_decoder/api.py
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from music_decoder.types import (
    AnalysisResult,
    ChordSymbol,
    Composition,
    Scale,
    Tuning,
)
from music_decoder.progress import ProgressCallback
from music_decoder.tabs.tuning import STANDARD_EADGBE


def analyze(
    source: str | Path,
    *,
    declared_kind: Literal["solo_guitar", "full_mix"] = "full_mix",
    tuning: Tuning = STANDARD_EADGBE,
    use_separation: bool = True,
    progress: ProgressCallback | None = None,
    out_dir: Path | None = None,
) -> AnalysisResult:
    """Audio file or YouTube URL → chord progression + guitar tab + key.

    Synchronous. Deterministic given pinned hyperparameters.
    """


def compose(
    scale: Scale,
    progression: Sequence[ChordSymbol],
    *,
    bars_per_chord: int = 1,
    tempo_bpm: float = 100.0,
    style: Literal["arpeggio", "strum", "fingerstyle"] = "fingerstyle",
    tuning: Tuning = STANDARD_EADGBE,
    seed: int | None = None,
    out_dir: Path | None = None,
) -> Composition:
    """Scale + chord progression → MIDI + WAV + ASCII tab + structured data."""
```

### 4.1 Result types

```python
# music_decoder/types.py
@dataclass(frozen=True)
class Scale:
    tonic: str                                   # "C", "C#", ..., "B"
    mode: Literal["major", "minor"]

@dataclass(frozen=True)
class ChordSymbol:
    """Parsed chord label, e.g. 'Cmaj7' → root=C, quality=maj7."""
    root: str
    quality: Literal[
        "maj", "min", "maj7", "min7", "7", "dim", "sus4", "aug",
    ]
    @classmethod
    def parse(cls, label: str) -> "ChordSymbol": ...
    def to_label(self) -> str: ...

@dataclass(frozen=True)
class ChordSegment:
    start_s: float
    end_s: float
    chord: ChordSymbol
    confidence: float

@dataclass(frozen=True)
class KeyEstimate:
    tonic: str
    mode: Literal["major", "minor"]
    profile: Literal["krumhansl_kessler", "temperley"]
    correlation: float
    margin: float

@dataclass(frozen=True)
class Note:
    start_s: float
    end_s: float
    pitch: int                                   # MIDI 0..127
    velocity: int
    confidence: float

@dataclass(frozen=True)
class TabPosition:
    string: int                                  # 0=lowest
    fret: int                                    # 0=open

@dataclass(frozen=True)
class TabbedNote:
    note: Note
    position: TabPosition

@dataclass(frozen=True)
class Tuning:
    name: str
    open_pitches: tuple[int, ...]                # MIDI numbers, low-to-high

# Presets (defined in music_decoder/tabs/tuning.py):
#   STANDARD_EADGBE, DROP_D, EB_HALF_STEP_DOWN, D_STANDARD, DROP_C, DADGAD

@dataclass(frozen=True)
class VoicedChord:
    chord: ChordSymbol
    positions: tuple[TabPosition, ...]           # one per string; -1=muted

@dataclass(frozen=True)
class AnalysisResult:
    source: str                                  # original URL or path
    audio_path: Path                             # local WAV/MP3 used
    duration_s: float
    sample_rate_hz: int
    key: KeyEstimate
    chord_progression: tuple[ChordSegment, ...]
    tab: tuple[TabbedNote, ...]
    tempo_bpm: float
    beat_times_s: tuple[float, ...]
    metadata: dict                               # hyperparameter id, versions

@dataclass(frozen=True)
class Composition:
    midi_path: Path
    wav_path: Path
    ascii_tab: str
    melody_notes: tuple[Note, ...]
    chord_voicings: tuple[VoicedChord, ...]
    metadata: dict                               # seed, scale, progression, etc.
```

### 4.2 Errors

```python
# music_decoder/errors.py
class MusicDecoderError(Exception): ...
class IngestError(MusicDecoderError): ...
class YouTubeError(IngestError): ...
class CorruptAudioError(IngestError): ...
class SilentAudioError(IngestError): ...
class ClipTooShortError(IngestError): ...
class SeparationError(MusicDecoderError): ...
class TranscriptionError(MusicDecoderError): ...
class ChordRecognitionError(MusicDecoderError): ...
class TabAssignmentError(MusicDecoderError): ...
class CompositionError(MusicDecoderError): ...
class InvalidScaleError(CompositionError): ...
class InvalidProgressionError(CompositionError): ...
```

CLI converts to `click.ClickException`. Streamlit catches at the top of each
tab handler and renders a `st.error(...)` block. Internal modules raise the
specific subclass.

---

## 5. Ingestion

### 5.1 Source detection

```python
# music_decoder/ingest/__init__.py
_YT_RE = re.compile(
    r"^https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([\w-]{11})"
)

def load(source: str | Path) -> LoadedAudio:
    if isinstance(source, str) and _YT_RE.match(source):
        return youtube.fetch_and_load(source)
    return audio_file.load(Path(source))
```

### 5.2 YouTube ingestion

`ingest/youtube.py`:

1. Match URL → extract `video_id`.
2. Check cache: `<data_dir>/yt_cache/{video_id}.wav`. If hit, fall through
   to `audio_file.load`.
3. Otherwise call `yt-dlp` (Python API, not subprocess) with
   `format='bestaudio'`, post-process to wav via the existing ffmpeg
   wrapper, write to cache.
4. Return `LoadedAudio` with `source.path` set to the cached file.

Failure modes raise typed errors (`YouTubeError` with subclasses for private,
age-restricted, region-locked, network). The cache is purely local; there is
no upload, no telemetry, no rate-limit awareness — single-user use is
implicitly assumed.

### 5.3 Local-file ingestion

`ingest/audio_file.py` — kept from existing `audio_io/load.py` with minor
renames. Validates: ffmpeg exit code, NaN samples, RMS > silence threshold,
duration ≥ 1.0 s.

---

## 6. Composition module (NEW)

### 6.1 Voicings

`compose/voicings.py`:

- Input: `ChordSymbol`, `Tuning` (defaults `STANDARD_EADGBE`).
- Output: 1–3 ranked `VoicedChord` candidates.
- Algorithm:
  - Look up canonical fingerings from `chords/voicings.py` (already exists,
    96 chord types covered: 12 roots × 8 qualities after Phase B-3, plus
    a no-chord row → 97 template states overall).
  - Re-voice for the requested tuning when not EADGBE: transpose canonical
    fingering by the open-string-pitch delta, then run the existing A*
    voicing search restricted to a single time-step to find the lowest-cost
    valid fingering.
  - Sort candidates by total fret span + min fret (preferring lower
    positions).

### 6.2 Melody generator

`compose/melody.py`:

- Input: `Scale`, progression, tempo, bars-per-chord, seed.
- Output: `list[Note]`, all in scale.
- Algorithm:
  1. Build the scale's pitch-class set (e.g. C major = {0, 2, 4, 5, 7, 9, 11}).
  2. For each chord segment in the progression:
     - Compute chord-tone set (e.g. Cmaj7 = {0, 4, 7, 11}).
     - Choose `notes_per_bar` (default 4 — eighth notes-ish) per
       `bars_per_chord` bar.
     - For each beat, sample a pitch with weighted distribution:
       - Strong beats (downbeat, beat 3): chord tone with prob 0.7,
         scale tone with prob 0.3.
       - Weak beats: chord tone with prob 0.3, scale tone with prob 0.7.
       - Avoid notes outside scale.
     - 1st-order Markov pitch transition: prefer intervals ≤ 4 semitones from
       the previous pitch. Implemented as a precomputed transition matrix
       over scale degrees, sampled with `numpy.random.Generator(seed)`.
  3. Assign each note `start_s`, `end_s` from the tempo grid.
  4. Velocity: 80 on strong beats, 70 on weak beats.

Determinism: a `numpy.random.Generator(seed)` is the only source of
randomness. Same seed → same melody.

### 6.3 Arrangement

`compose/arrangement.py`:

- Build a `pretty_midi.PrettyMIDI` with two instruments:
  - **Melody track** — program 25 (Steel Acoustic Guitar). Notes from
    `compose/melody.py`.
  - **Accompaniment track** — program 25. Pattern depends on `style`:
    - `arpeggio` — play chord-tone sequence ascending at eighth-note interval.
    - `strum` — block chord on beats 1 and 3 (4/4 assumed).
    - `fingerstyle` — bass note on 1, treble cluster on 2, bass on 3, treble
      on 4.
- Render to MIDI file at `<out_dir>/{ts}-{hash}/composition.mid`.
- Render to WAV via `synth/fluidsynth_wrapper.py` (kept from existing).
- Build ASCII tab from melody + accompaniment positions using
  `tabs/render.py`.

### 6.4 Hand-crafted unit cases (test-first)

- C major + I–vi–ii–V (Cmaj7, Am7, Dm7, G7) at 100 bpm, seed=42 →
  - All melody notes in C major.
  - Strong beats are chord tones at least 70% of the time (statistical
    check over the deterministic output).
  - Voicings cover the four chords with playable fret spans (≤ 5).
  - WAV file is non-silent and has the expected duration.
- A minor + Am–F–C–G at 80 bpm, seed=7 → analogous checks for minor.
- Drop D + D5 power chord progression → voicings use the open low D string.

---

## 7. CLI surface

```
$ music-decoder analyze <file-or-url>
    [--tuning EADGBE | "Drop D" | "Eb" | "D standard" | "Drop C" | DADGAD]
    [--no-separation]
    [--solo-guitar]
    [--out DIR]
    [--format pretty|json]

$ music-decoder compose
    --scale C:major
    --progression "Cmaj7 Am7 Dm7 G7"
    [--style arpeggio|strum|fingerstyle]
    [--tempo 100]
    [--bars 1]
    [--seed 42]
    [--tuning EADGBE]
    [--out DIR]
    [--format pretty|json]

$ music-decoder ui                       # launches Streamlit
$ music-decoder doctor                   # checks ffmpeg, fluidsynth, models
```

Click-based. The `--format json` mode prints a JSON document equivalent to
the public API result; `pretty` prints a human-readable summary plus paths to
artifacts.

`doctor` checks: ffmpeg on PATH, fluidsynth available, soundfont present,
basic-pitch model available, demucs model available, madmom data present.
Exit nonzero on any miss with a clear remediation message.

---

## 8. Streamlit UI

Single-page app at `src/music_decoder/ui/streamlit_app.py`. Three tabs:

- **Analyze** — input box (file uploader or URL paste, mutually exclusive),
  tuning preset, "solo guitar" / "full mix" toggle, "skip Demucs" toggle.
  Submit calls `analyze()` with a progress callback wired to a Streamlit
  progress bar. Renders:
  - Detected key + tempo.
  - Chord progression labels timeline (existing renderer in
    `ui/components/chord_progression.py`, kept and trimmed).
  - ASCII tab + SVG fretboard (existing renderer in
    `ui/components/tablature.py`, kept and moved).
  - Original audio playback (HTML5 `<audio>`).
- **Compose** — scale picker (12 tonics × {major, minor}), progression input
  (free text "Cmaj7 Am7 Dm7 G7"), style/tempo/bars/seed widgets. Submit
  calls `compose()`. Renders:
  - WAV playback.
  - ASCII tab.
  - Voicings as small SVG diagrams.
  - Download buttons for MIDI and WAV.
- **About** — version, hyperparameter set ID, environment summary.

State held in `st.session_state`. Models loaded once via
`@st.cache_resource`. No DB, no polling.

---

## 9. Quality gating

### 9.1 Test pyramid

```
tests/
├── conftest.py
├── unit/
│   ├── test_ingest_audio_file.py
│   ├── test_ingest_youtube.py             # mocked yt-dlp
│   ├── test_dsp_chroma.py
│   ├── test_dsp_beats.py
│   ├── test_chords_templates.py           # kept
│   ├── test_chords_recognize.py           # kept
│   ├── test_chords_voicings.py            # kept
│   ├── test_chords_madmom_compat.py       # kept
│   ├── test_chords_madmom_backend.py      # kept
│   ├── test_key_global.py                 # kept (renamed)
│   ├── test_key_chroma.py                 # kept (renamed)
│   ├── test_tabs_assigner.py              # kept (renamed)
│   ├── test_tabs_candidates.py            # kept
│   ├── test_tabs_heuristic.py             # kept
│   ├── test_tabs_astar.py                 # kept
│   ├── test_tabs_tuning.py                # kept
│   ├── test_tabs_render.py                # NEW (was UI test)
│   ├── test_transcription_post.py         # kept
│   ├── test_compose_voicings.py           # NEW
│   ├── test_compose_melody.py             # NEW
│   ├── test_compose_arrangement.py        # NEW
│   ├── test_synth.py                      # NEW (sanity check fluidsynth call)
│   ├── test_api_analyze.py                # NEW (mocked stages)
│   ├── test_api_compose.py                # NEW
│   ├── test_cli_main.py                   # kept, rewritten
│   └── test_config.py                     # kept (merge runtime + hyperparams)
└── integration/
    ├── test_analyze_e2e.py                # synthetic fixture, full pipeline
    ├── test_compose_e2e.py                # full pipeline
    └── test_regression.py                 # accuracy gate (see 9.2)
```

Tests are pure-Python. Tests that need a soundfont skip with a clear message
when fluidsynth/soundfont is not installed; CI installs both.

### 9.2 Regression test

`tests/integration/test_regression.py`:

```python
@pytest.mark.regression
def test_chord_recognition_score(synthetic_fixture):
    result = analyze(synthetic_fixture.audio_path, declared_kind="solo_guitar")
    score = chord_recognition_score(result.chord_progression, synthetic_fixture.gt)
    assert score >= thresholds.chord_recognition_score   # 0.60

@pytest.mark.regression
def test_key_mirex_score(synthetic_fixture):
    result = analyze(synthetic_fixture.audio_path, declared_kind="solo_guitar")
    score = key_mirex_score(result.key, synthetic_fixture.gt.key)
    assert score >= thresholds.key_mirex_score           # 0.75

@pytest.mark.regression
def test_tab_string_accuracy(synthetic_fixture):
    result = analyze(synthetic_fixture.audio_path, declared_kind="solo_guitar")
    score = tab_string_accuracy(result.tab, synthetic_fixture.gt.tab)
    assert score >= thresholds.tab_string_accuracy       # 0.55
```

Thresholds in `config/eval_thresholds.yaml`. Fixtures: the existing
`tests/fixtures/synthetic/*.mid` files (rendered to WAV at test time via
fluidsynth), kept from Phase B-5. No GuitarSet, no manual fixtures.

`evaluation/metrics.py` is kept (~150 LoC, pure mir_eval wrappers). Everything
else under `evaluation/` is deleted.

### 9.3 CI

Single GitHub Actions job:

1. Install Python 3.11.
2. Install system deps (`ffmpeg libsndfile1 libfluidsynth3`).
3. `pip install -e ".[dev]"`.
4. `make lint typecheck test`.
5. `make test-regression` (runs `pytest -m regression`).

Models are downloaded on first run; CI uses a cached model dir keyed by
basic-pitch + demucs + madmom versions.

---

## 10. Configuration

Two YAML files under `config/`:

### 10.1 `config/runtime.yaml`

```yaml
data_dir: ~/.music-decoder            # platformdirs default; override per-OS
log_level: INFO
ffmpeg_path: ffmpeg
fluidsynth_soundfont: GeneralUser-GS.sf2
sample_rate_hz: 22050
youtube_cache_dir: ${data_dir}/yt_cache
composition_out_dir: ${data_dir}/compositions
```

Override any field via `MUSIC_DECODER_<UPPERCASE_KEY>` env vars.

### 10.2 `config/hyperparameters.yaml`

Pinned hyperparameter set with an ID (`2026-05-08-v2-baseline`). Recorded in
every `AnalysisResult.metadata` and `Composition.metadata`. Sections:

```yaml
id: 2026-05-08-v2-baseline
basic_pitch:
  onset_threshold: 0.5
  frame_threshold: 0.3
  minimum_note_length_ms: 58
  minimum_frequency_hz: 32.7
  maximum_frequency_hz: 2000
post_processing:
  median_filter_window: 5
  min_note_duration_s: 0.05
  same_pitch_merge_gap_s: 0.05
  rhythmic_snap_confidence_threshold: 0.7
key_detection:
  hpss_margin: 1.0
  windowed_segment_length_s: 8.0
  windowed_hop_s: 2.0
beat_tracking:
  start_bpm: 120
  tightness: 100
chord_detection:
  backend: madmom_deep_chroma            # falls back to template_hmm
  qualities: [maj, min, maj7, min7, 7, dim, sus4, aug]
  min_segment_duration_s: 0.4
tab_assignment:
  weights:
    w_move: 1.0
    w_string: 0.3
    w_span: 0.5
    w_open: 0.2
    w_high: 0.4
    w_chord_intra: 0.6
  max_fret: 22
composition:
  notes_per_bar: 4
  strong_beat_chord_tone_prob: 0.7
  weak_beat_chord_tone_prob: 0.3
  markov_max_interval_semitones: 4
  velocity_strong: 80
  velocity_weak: 70
```

### 10.3 `config/eval_thresholds.yaml`

```yaml
chord_recognition_score: 0.60
key_mirex_score: 0.75
tab_string_accuracy: 0.55
```

Sweeps and hyperparameter promotion are out of scope for v2.

---

## 11. Distribution

### 11.1 Package layout

Standard Python package (`pyproject.toml`, hatchling). Console entry point
`music-decoder` defined under `[project.scripts]` maps to
`music_decoder.cli.main:main`.

### 11.2 Dependencies

**Add:** `yt-dlp`.

**Drop:** `sqlalchemy`, `alembic`, `mirdata`, `pyacoustid`, `tensorflow`
(was a training dep), `freezegun` (was used in DB tests), `crepe` (optional
extra), basic-pitch fine-tuning extras.

**Keep:** `streamlit`, `librosa`, `basic-pitch`, `demucs`, `pretty_midi`,
`music21`, `mir_eval`, `pyfluidsynth`, `matplotlib`, `numpy`, `scipy`,
`pydantic`, `pyyaml`, `click`, `platformdirs`, `madmom`, `pytest`,
`pytest-cov`, `pytest-xdist`, `ruff`, `mypy`, `types-PyYAML`.

### 11.3 Docker

`docker-compose.yml` is updated:

```yaml
services:
  music-decoder:
    build: .
    image: music-decoder:dev
    ports:
      - "8501:8501"
    environment:
      MUSIC_DECODER_DATA_DIR: /data
      MUSIC_DECODER_LOG_LEVEL: INFO
    volumes:
      - md-data:/data       # only used for yt_cache + compositions

volumes:
  md-data: {}
```

`MUSIC_DECODER_DB_PATH` and `MUSIC_DECODER_ARTIFACT_DIR` are removed. The
volume is kept (the YouTube cache and composition output benefit from
persistence across container restarts) but is no longer used for SQLite.

`Dockerfile`: drop `libchromaprint1` (no AcoustID), drop `build-essential`
and `pkg-config` if unused after removing training deps; otherwise keep.
Drop the model warmup line that references `cli.models_download` (folded into
`doctor`); replace with a `doctor`-based warmup or remove and let the first
analyze pay the download.

### 11.4 First run

```
$ pipx install ./                     # or: pip install -e .  (development)
$ music-decoder doctor
[*] Checking ffmpeg ............ OK
[*] Checking fluidsynth ........ OK
[*] Soundfont present .......... OK (GeneralUser-GS.sf2)
[*] Initializing data dir ...... ~/.music-decoder
[*] Downloading models (one-time, ~400 MB):
    - basic-pitch ............... OK
    - Demucs htdemucs_6s ........ OK
    - madmom DeepChroma ......... OK
[*] All checks passed.

$ music-decoder analyze "https://youtu.be/dQw4w9WgXcQ" --solo-guitar
$ music-decoder compose --scale C:major --progression "Cmaj7 Am7 Dm7 G7"
$ music-decoder ui
```

---

## 12. Failure modes

| Failure | Detection | Handling |
|---|---|---|
| Corrupt audio | ffmpeg nonzero, NaN samples | `CorruptAudioError`. CLI: nonzero exit + "Couldn't decode this file." UI: red banner. |
| Silent audio (RMS below threshold) | `audio_file.load` validation | `SilentAudioError`. |
| Clip too short (< 1.0 s) | length check | `ClipTooShortError`. |
| Demucs OOM or crash | subprocess error | log warning, fall through to original audio. UI banner: "Source separation skipped." |
| YouTube private/age-restricted | yt-dlp DownloadError class | `YouTubeError` subclass; CLI prints actionable message; UI red banner. |
| YouTube network failure | yt-dlp DownloadError | `YouTubeError` with retry-once on transient. |
| Empty MIDI (basic-pitch returns 0 notes) | post-stage check | empty `tab` in `AnalysisResult`. UI banner: "No notes detected." |
| Out-of-range notes for tuning | A* candidate set empty for note | drop note with reason; result lists count. |
| A* finds no valid path (unsatisfiable chord) | A* fails | drop chord, continue. |
| Invalid scale string | `Scale.parse` validation | `InvalidScaleError`. |
| Invalid chord symbol | `ChordSymbol.parse` validation | `InvalidProgressionError`. |
| Unknown style | enum validation | CLI argument error. |
| ffmpeg missing | `doctor` check | exit nonzero with platform-specific install command. |
| fluidsynth missing | `doctor` check | exit nonzero. |
| Soundfont missing | startup check inside `synth/` | clear error with path the user can populate. |
| Models not yet downloaded | first-call lazy load | one-time download with a stderr progress line; UI shows progress bar via callback. |

---

## 13. Migration & cleanup steps

The implementation plan (next document) will cover this in detail. At a
high level:

1. Delete unused modules and tests in one big commit.
2. Move and rename surviving modules. Keep their public APIs stable enough to
   minimize churn in tests.
3. Write `api.py`, `types.py`, `errors.py`, `progress.py`.
4. Rewrite `cli/main.py` from scratch (Click; subcommands).
5. Write `ingest/youtube.py` + `ingest/__init__.py`.
6. Write `compose/{voicings,melody,arrangement,api}.py`.
7. Rewrite `ui/streamlit_app.py` as single-page.
8. Update `pyproject.toml`, `Dockerfile`, `docker-compose.yml`, `Makefile`,
   `README.md`.
9. Run linter, typechecker, full unit suite, regression suite.

---

## 14. Open questions / future work

- **Auto scale detection inside `compose()`** — let the user pass only a
  progression and infer scale from chord roots. Deferred.
- **Modal scales** beyond major/minor (Dorian, Mixolydian, ...). Deferred to
  v3.
- **Chord progression suggester** — given a key, suggest progressions. Out
  of v2 scope.
- **Audio export of analyze results** — render the detected MIDI as WAV for
  side-by-side playback. Cheap to add later; deferred.
- **Multi-track output** — separate WAV stems for melody vs accompaniment.
  Deferred.
- **Caching of analysis results** — currently re-runs from scratch. Could
  add a content-hash-keyed disk cache of `AnalysisResult` if the use pattern
  changes. Deferred.

---

## 15. References

- Existing v1 design (superseded): [2026-05-04-music-decoder-design.md](./2026-05-04-music-decoder-design.md)
- Existing chord-progression design (superseded): [2026-05-05-chord-progression-design.md](./2026-05-05-chord-progression-design.md)
- Existing Phase C design (superseded): [2026-05-07-phase-c-design.md](./2026-05-07-phase-c-design.md)
- mir_eval: https://github.com/mir-evaluation/mir_eval
- Spotify basic-pitch: https://github.com/spotify/basic-pitch
- Demucs: https://github.com/facebookresearch/demucs
- madmom: https://github.com/CPJKU/madmom
- yt-dlp: https://github.com/yt-dlp/yt-dlp
- fluidsynth: https://www.fluidsynth.org/
