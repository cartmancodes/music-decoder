# Music Decoder — Technical Architecture

Other docs: **Technical (this file)** · [Overview](architecture-overview.md) · [Usage](usage.md)

A senior-engineer-grade reference for the v2 codebase. Audience: a Python
engineer joining the project who needs to make changes. This doc is meant to
let you jump straight to the source — every non-trivial section links to a
file under `src/`.

If you only want to know what the tool does, read [Overview](architecture-overview.md).
If you want to install and run it, read [Usage](usage.md).

---

## 1. Process model

A single in-process synchronous Python library. Two thin frontends call it:

- A Click-based CLI ([cli/main.py](../src/music_decoder/cli/main.py)).
- A single-page Streamlit UI ([ui/streamlit_app.py](../src/music_decoder/ui/streamlit_app.py)).

There is no daemon, no job queue, no database, no IPC. Models load lazily
the first time a stage runs in the current Python process; the CLI re-loads
on each invocation, the Streamlit UI caches them via `@st.cache_resource`,
and tests load lazily through pytest fixtures.

```text
   ┌──────────────┐
   │  CLI (Click) │──┐
   └──────────────┘  │
                     ▼
              ┌──────────────────────┐         disk artifacts
              │   music_decoder      │ ──────► (yt cache, MIDI, WAV, JSON)
              │   sync library API   │
              └──────────────────────┘
                     ▲
   ┌──────────────┐  │
   │  Streamlit   │──┘
   └──────────────┘
```

The library is fully synchronous. Long-running stages emit progress through
an optional callback so the UI can render a progress bar without coupling
the library to Streamlit. See [progress.py](../src/music_decoder/progress.py).

Determinism: given the pinned hyperparameter file
([`config/hyperparameters.yaml`](../config/hyperparameters.yaml)) and a fixed
input, `analyze()` is deterministic. `compose()` is deterministic given a
seed.

---

## 2. Module layout

Source layout under [src/music_decoder/](../src/music_decoder/):

- [api.py](../src/music_decoder/api.py) — public synchronous entry points: `analyze()`, `compose()`.
- [types.py](../src/music_decoder/types.py) — frozen dataclasses shared across modules.
- [errors.py](../src/music_decoder/errors.py) — typed exception hierarchy.
- [progress.py](../src/music_decoder/progress.py) — `ProgressCallback` protocol + null implementation.
- [logging_setup.py](../src/music_decoder/logging_setup.py) — structured JSON logger.
- [ingest/](../src/music_decoder/ingest/) — local-file and YouTube ingestion.
  - [audio_file.py](../src/music_decoder/ingest/audio_file.py) — ffmpeg-backed loader; SHA-256 hash; mono float32 at configured SR.
  - [ffmpeg.py](../src/music_decoder/ingest/ffmpeg.py) — `find_ffmpeg`, `decode_to_wav`.
  - [youtube.py](../src/music_decoder/ingest/youtube.py) — yt-dlp wrapper; URL detection; per-video-id WAV cache.
- [separation/](../src/music_decoder/separation/) — Demucs `htdemucs_6s` wrapper. The single guitar stem is what later stages consume.
- [dsp/](../src/music_decoder/dsp/) — primitives.
  - [chroma.py](../src/music_decoder/dsp/chroma.py) — CQT chroma after harmonic-percussive separation.
  - [beats.py](../src/music_decoder/dsp/beats.py) — librosa beat tracking.
  - [time_signature.py](../src/music_decoder/dsp/time_signature.py) — meter inference (kept; not wired into the public pipeline yet).
- [key/](../src/music_decoder/key/) — Krumhansl-Kessler / Temperley key estimation.
- [chords/](../src/music_decoder/chords/) — chord recognition (templates + 97-state Viterbi, optional madmom backend).
- [transcription/](../src/music_decoder/transcription/) — basic-pitch wrapper + post-processing.
- [tabs/](../src/music_decoder/tabs/) — A* fret search + ASCII/SVG renderers.
- [compose/](../src/music_decoder/compose/) — voicings, melody generator, arrangement, public `compose()`.
- [synth/](../src/music_decoder/synth/) — MIDI to WAV via fluidsynth or sine fallback.
- [config/](../src/music_decoder/config/) — runtime + hyperparameter loaders.
- [evaluation/](../src/music_decoder/evaluation/) — mir_eval wrappers + synthetic fixture loader.
- [cli/](../src/music_decoder/cli/) — Click entry point.
- [ui/](../src/music_decoder/ui/) — Streamlit single-page app + display components.

The IO/ML/DSP/rule-based boundaries:

- **IO**: [ingest/](../src/music_decoder/ingest/), [synth/](../src/music_decoder/synth/), parts of [config/](../src/music_decoder/config/) (YAML loading), the YouTube cache, file writing in [compose/api.py](../src/music_decoder/compose/api.py).
- **ML (pre-trained, no training in v2)**: [separation/demucs.py](../src/music_decoder/separation/demucs.py) (Demucs htdemucs_6s), [transcription/basic_pitch_wrapper.py](../src/music_decoder/transcription/basic_pitch_wrapper.py) (Spotify basic-pitch), [chords/backends/madmom_deep_chroma.py](../src/music_decoder/chords/backends/madmom_deep_chroma.py) (madmom DeepChromaChordRecognitionProcessor).
- **DSP**: [dsp/](../src/music_decoder/dsp/), [key/](../src/music_decoder/key/), [chords/recognize.py](../src/music_decoder/chords/recognize.py).
- **Rule-based**: [tabs/](../src/music_decoder/tabs/), [compose/](../src/music_decoder/compose/), [chords/templates.py](../src/music_decoder/chords/templates.py), [chords/voicings.py](../src/music_decoder/chords/voicings.py).

---

## 3. Public API

The single source of truth is [api.py](../src/music_decoder/api.py); the
shapes it returns are defined in [types.py](../src/music_decoder/types.py).

### 3.1 Functions

```python
def analyze(
    source: str | Path,
    *,
    declared_kind: Literal["solo_guitar", "full_mix"] = "full_mix",
    tuning: Tuning = STANDARD_EADGBE,
    use_separation: bool = True,
    progress: ProgressCallback | None = None,
) -> AnalysisResult: ...


def compose(
    scale: Scale,
    progression: Sequence[ChordSymbol],
    *,
    bars_per_chord: int = 1,
    tempo_bpm: float = 100.0,
    style: Literal["arpeggio", "strum", "fingerstyle"] = "fingerstyle",
    tuning: Tuning = STANDARD_EADGBE,
    seed: int | None = None,
    out_dir: Path,
) -> Composition: ...
```

Notes:

- `analyze()` does not write any artifacts to disk other than the YouTube
  cache that [ingest/youtube.py](../src/music_decoder/ingest/youtube.py)
  maintains.
- `compose.out_dir` is required (the implementation in
  [compose/api.py](../src/music_decoder/compose/api.py) treats it as a
  required keyword argument; the README and [Usage](usage.md) reflect this).

### 3.2 Result types

All in [types.py](../src/music_decoder/types.py). Frozen dataclasses; safe
to share across threads; printable in JSON via the CLI's `_serialize`
helper.

- `Scale(tonic: str, mode: "major"|"minor")` — also has `Scale.parse("C:major")`.
- `ChordSymbol(root: str, quality: ChordQuality)` where `ChordQuality` is one
  of `"maj" | "min" | "maj7" | "min7" | "7" | "dim" | "sus4" | "aug"`. Has
  `ChordSymbol.parse("Cmaj7")` and `to_label() -> "Cmaj7"`.
- `ChordSegment(start_s, end_s, root, quality, confidence)`. The string
  fields `root` / `quality` are the canonical recognizer output (no-chord
  uses `root="N"`, `quality=""`); `.chord` lazily wraps to `ChordSymbol`.
- `KeyEstimate(tonic, mode, profile, correlation, margin)`. `profile` is
  one of `"krumhansl_kessler" | "temperley"`.
- `Note = TranscribedNote(start_s, end_s, pitch, velocity, confidence)`.
- `TabPosition(string, fret)` — `string` is 0-indexed low-to-high; `fret`
  is 0 for open, `-1` to indicate muted (used in `VoicedChord`).
- `TabbedNote(note, position, cost_breakdown)`.
- `Tuning(name, open_pitches: tuple[int, ...])` — open-string MIDI pitches
  low-to-high. Presets in [tabs/tuning.py](../src/music_decoder/tabs/tuning.py).
- `VoicedChord(chord, positions: tuple[TabPosition, ...])` — one position
  per string; `fret = -1` means muted.
- `AnalysisResult(source, audio_path, duration_s, sample_rate_hz, key,
  chord_progression, tab, tempo_bpm, beat_times_s, metadata)`.
- `Composition(midi_path, wav_path, ascii_tab, melody_notes,
  chord_voicings, metadata)`.

`metadata` on both result types includes the loaded hyperparameter-set ID
(`AnalysisResult.metadata["hyperparameter_set"]`). `Composition.metadata`
also pins `scale`, `progression`, `bars_per_chord`, `tempo_bpm`, `style`,
`seed`, `tuning`.

### 3.3 Errors

Hierarchy lives in [errors.py](../src/music_decoder/errors.py):

- `MusicDecoderError` — base.
  - `IngestError` → `YouTubeError`, `CorruptAudioError`, `SilentAudioError`, `ClipTooShortError`.
  - `SeparationError`, `TranscriptionError`, `ChordRecognitionError`,
    `TabAssignmentError`, `KeyDetectionError`, `BeatTrackingError`,
    `SynthesisError`.
  - `CompositionError` → `InvalidScaleError`, `InvalidProgressionError`.

The CLI converts every `MusicDecoderError` to `click.ClickException` (see
[cli/main.py](../src/music_decoder/cli/main.py)). The Streamlit UI catches
at the top of each tab handler and renders `st.error(...)`.

---

## 4. Data flow: `analyze()`

End-to-end pipeline (orchestrated in [api.py](../src/music_decoder/api.py)):

```text
source: str | Path
  │
  ▼
ingest.load                    [ingest/__init__.py]
  │   YouTube URL → yt_dlp + ffmpeg → cache file
  │   Local path  → ffmpeg → mono float32 @ 22050 Hz
  ▼
LoadedAudio (samples, sr, duration_s, sha256, source)
  │
  ▼  (skipped if declared_kind="solo_guitar" or use_separation=False)
separation.run_separation      [separation/demucs.py]
  │   Demucs htdemucs_6s, returns the "guitar" stem
  │   On any error, logs + falls through to original samples
  ▼
samples, sr (possibly Demucs-isolated)
  │
  ├─► dsp.track_beats          [dsp/beats.py]   →  BeatGrid (tempo + beats)
  │
  ├─► dsp.compute_chroma       [dsp/chroma.py]  →  (12, T) chroma matrix
  │       │
  │       └─► key.estimate_key [key/__init__.py] →  KeyEstimate
  │
  ├─► chords.recognize_chords  [chords/__init__.py]
  │       beat-syncs chroma, runs template_hmm or madmom backend
  │       → tuple[ChordSegment, ...]
  │
  └─► transcription.transcribe [transcription/__init__.py]
          basic-pitch CoreML/ONNX/TF model on the same samples
          → tuple[Note, ...]
              │
              ▼
          tabs.assign_tabs     [tabs/__init__.py]
              A* fret search with weighted transition cost
              → tuple[TabbedNote, ...]
  │
  ▼
AnalysisResult
```

Between stages, samples and sr are passed by reference (no copies). The
orchestrator function is intentionally short — fewer than 100 lines — so
you can read it top to bottom in one sitting.

### 4.1 Ingest

[ingest/__init__.py](../src/music_decoder/ingest/__init__.py) `.load(source)`
detects YouTube URLs via the regex
`^https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([\w-]{11})` and
dispatches:

- **YouTube path**: [youtube.py](../src/music_decoder/ingest/youtube.py)
  reads `<data_dir>/yt_cache/<video_id>.wav` if present; otherwise calls
  yt-dlp's Python API with `format="bestaudio/best"` and an
  `FFmpegExtractAudio` post-processor (`preferredcodec="wav"`,
  `preferredquality="0"`). On a yt-dlp `DownloadError` it raises
  [`YouTubeError`](../src/music_decoder/errors.py).
- **Local path**: [audio_file.py](../src/music_decoder/ingest/audio_file.py)
  invokes ffmpeg via [ffmpeg.py](../src/music_decoder/ingest/ffmpeg.py)
  (`-ac 1 -ar <SR> -acodec pcm_s16le`) into a temp WAV, reads it, and
  validates: `np.isfinite` (NaN/Inf check), `duration ≥ 1.0 s`,
  `RMS ≥ 1e-4`. The file is hashed (SHA-256, 1 MiB chunks) for caching
  invariants.

The default sample rate is `SR_STANDARD = 22050`. The high-resolution path
(`SR_HIGH = 44100`) exists in `audio_file.py` but the public API does not
expose it in v2.

### 4.2 Separation

[separation/run_separation()](../src/music_decoder/separation/__init__.py)
runs Demucs `htdemucs_6s` (a 6-stem model: drums, bass, other, vocals,
piano, guitar). Internally it:

1. Loads `demucs.pretrained.get_model("htdemucs_6s")` and moves it to CUDA
   if available.
2. Resamples to the model's native rate, fakes stereo by repeating the
   mono input, and calls `demucs.apply.apply_model` with `split=True,
   overlap=0.25`.
3. Picks the index of `"guitar"` in `model.sources` and returns the mean
   of its two channels, resampled back to the input sample rate and
   trimmed to the original sample count.

Any backend exception is wrapped as `SeparationError`; the `analyze()`
orchestrator catches it and falls back to the original samples
(annotated in [api.py](../src/music_decoder/api.py)).

Separation is skipped entirely when `declared_kind="solo_guitar"` or
`use_separation=False`.

### 4.3 Beat tracking

[dsp/beats.py](../src/music_decoder/dsp/beats.py) wraps
`librosa.beat.beat_track` with `start_bpm=120, tightness=100`. If the input
is essentially silent (`RMS < 1e-5`) it returns an empty BeatGrid rather
than letting librosa raise. Downbeats are a crude `beats[::4]`. The
returned `BeatGrid.ts_*` fields default to 4/4 with `ts_assumed=True`.

The richer time-signature inference in
[dsp/time_signature.py](../src/music_decoder/dsp/time_signature.py) (a
beat-strength autocorrelation peak picker over the candidate set
`(3, 4, 6, 5, 7)`) is not currently wired into the public pipeline.

### 4.4 Chroma

[dsp/chroma.py](../src/music_decoder/dsp/chroma.py)
`compute_chroma_with_hpss(samples, sr, hpss_margin)`:

1. `librosa.effects.hpss(samples, margin=hpss_margin)` to keep the
   harmonic component only (drums removed).
2. `librosa.feature.chroma_cqt` on the harmonic signal — CQT-based chroma
   is more robust to inharmonicity than STFT chroma.
3. Per-frame max normalization to bring each column into `[0, 1]`.

Returns shape `(12, T)`. The public adapter in
[dsp/__init__.py](../src/music_decoder/dsp/__init__.py) hard-codes
`hpss_margin=8.0` (more aggressive harmonic isolation than the YAML
default of 1.0).

### 4.5 Key detection

[key/](../src/music_decoder/key/) implements Krumhansl-Schmuckler-style key
finding.

`compute_chroma_with_hpss` produces a `(12, T)` matrix; the public adapter
reduces it to a 12-dim pitch-class distribution via `chroma.mean(axis=1)`.

[key/ks.py](../src/music_decoder/key/ks.py) `correlate_against_profiles`
takes the 12-dim PCD and a profile (`krumhansl_kessler` or `temperley`,
each with major + minor variants from
[key/profiles.py](../src/music_decoder/key/profiles.py)). For every tonic
`i ∈ [0..11]` and every mode it computes Pearson correlation against the
PCD and returns the 24 candidates ranked by correlation. Each candidate
includes a `margin = best_corr - second_corr`.

[key/global_estimator.py](../src/music_decoder/key/global_estimator.py)
runs `top_k=3` for both profiles and produces a consensus:

- If both profiles' top picks agree on `(tonic, mode)`, that pair is the
  consensus. Confidence = `clip((kk_margin + temp_margin) / 2 + 0.5, 0, 1)`.
- Otherwise consensus is `None` and confidence is half that.

The public adapter [key/estimate_key()](../src/music_decoder/key/__init__.py)
returns the consensus when present; otherwise it falls back to the
`krumhansl_kessler` top-1 estimate. If no estimates were produced at all
(empty chroma), it raises `KeyDetectionError`.

The windowed estimator [key/windowed.py](../src/music_decoder/key/windowed.py)
slides a `segment_length_s` window with `hop_s` step across the chroma
matrix and returns per-segment top-1 KK estimates. It is exposed through
[key/api.py](../src/music_decoder/key/api.py) `detect_key()` for callers
that want a richer `KeyDetectionResult` with `windowed_segments`. The
public `analyze()` only uses the global consensus; windowed results are
available to anyone using `key.api.detect_key` directly.

### 4.6 Chord recognition

[chords/](../src/music_decoder/chords/) is the most complex DSP module.

`recognize_chords(samples, sr, beat_grid)` (in
[chords/__init__.py](../src/music_decoder/chords/__init__.py)) is the
`analyze()` adapter. It:

1. Computes chroma at the default hop length (`hop_length=512`) using
   `compute_chroma_with_hpss(hpss_margin=8.0)`.
2. Builds a [`ChordDetectionParams`](../src/music_decoder/config/hyperparameters.py)
   with `qualities=QUALITIES` (8 qualities), `hmm_self_transition_prob=0.9`,
   `no_chord_threshold=0.3`, `min_segment_duration_s=0.25`, and a
   `backend` value read at call time from
   [`config/hyperparameters.yaml`](../config/hyperparameters.yaml)
   (defaults to `madmom_deep_chroma`; the dispatcher in
   [chords/api.py](../src/music_decoder/chords/api.py) falls back to
   `template_hmm` when madmom is unimportable).
3. If the supplied `BeatGrid` has fewer than 4 beats (silence, very short
   clips, sustained chords), it swaps in a synthetic uniform 4-beat grid
   spanning the audio so the recognizer can still emit at least one
   segment.
4. Calls `detect_chords(...)` (the dispatcher in
   [chords/api.py](../src/music_decoder/chords/api.py)).
5. As a final safety net, if the backend returns `skipped_reason` and an
   empty segment list, emits a single `("N", "no-chord")` segment over
   the full duration so downstream code never sees zero segments.

#### 4.6.1 Templates

[chords/templates.py](../src/music_decoder/chords/templates.py) defines:

- 12 roots × 8 qualities = 96 binary chord templates (length-12 vectors
  with 1.0 at active pitch classes).
- 1 no-chord template (`[1/12] * 12`, uniform).
- `QUALITIES = ("maj", "min", "7", "maj7", "min7", "dim", "sus4", "aug")`.
- `_QUALITY_INTERVALS`: `maj=(0,4,7)`, `min=(0,3,7)`, `7=(0,4,7,10)`,
  `maj7=(0,4,7,11)`, `min7=(0,3,7,10)`, `dim=(0,3,6)`, `sus4=(0,5,7)`,
  `aug=(0,4,8)`.
- Total state space: **97 states** (96 chord + 1 no-chord) after the
  Phase B-3 vocabulary expansion (4 → 8 qualities).
- Stable `label_index(root, quality) -> int` and round-trip
  `chord_label` / `label_to_root_quality` helpers used by the recognizer
  and by the JAMS-label parser.

#### 4.6.2 template_hmm backend

[chords/backends/template_hmm.py](../src/music_decoder/chords/backends/template_hmm.py)
implements the always-available DSP-only backend.
[chords/recognize.py](../src/music_decoder/chords/recognize.py) provides the
primitives:

1. **Beat-sync chroma** (`beat_sync_chroma`): for each beat-to-beat
   interval, average the chroma columns whose frame index falls inside.
   Returns shape `(12, num_beats - 1)`.
2. **Score** (`score_beats`): cosine similarity between every beat
   column and every of the 97 templates → `(num_beats, 97)`. Beats whose
   chroma is all-zero get a zero row (they can never match).
3. **No-chord override**: in the backend, if `max_per_beat <
   no_chord_threshold` for a beat, that beat's row is wiped and only the
   no-chord state gets `1.0`. This forces silent / weak beats to emit `N`.
4. **Viterbi smoothing** (`viterbi_smooth`): standard log-space Viterbi
   over 97 states with a uniform stay/switch transition. Self-transition
   probability is `params.hmm_self_transition_prob` (the adapter pins
   0.9; the YAML doesn't currently set it). Switch probability is
   `(1 - p_self) / 96`. The implementation tracks a single
   "best-of-others" candidate for efficiency, with a fallback for the
   pathological case where the best-of-others happens to be `j` itself.
5. **Segment merge** (`merge_segments`): collapses runs of the same
   state into `ChordSegment` records, drops segments shorter than
   `min_segment_duration_s` by absorbing them into the predecessor, and
   re-coalesces newly-adjacent same-state runs. Confidence is the mean
   of per-beat scores for the chosen state across the segment.

#### 4.6.3 madmom backend

[chords/backends/madmom_deep_chroma.py](../src/music_decoder/chords/backends/madmom_deep_chroma.py)
wraps madmom's `DeepChromaProcessor` + `DeepChromaChordRecognitionProcessor`.
This pre-trained CNN-based pipeline operates directly on the audio file
(not on our pre-computed chroma) and emits `(start_s, end_s, jams_label)`
triples.

- The backend lazily imports madmom on first use. It first calls into
  [chords/madmom_compat.py](../src/music_decoder/chords/madmom_compat.py),
  which patches `collections.MutableSequence` (and friends) plus the
  removed numpy aliases `np.float`, `np.int`, `np.bool`. madmom 0.16.1
  predates Python 3.10 / NumPy 1.20 and these shims are required.
- JAMS labels go through
  [chords/labels.py](../src/music_decoder/chords/labels.py)
  `parse_jams_chord_label`. The `_QUALITY_MAP` does conservative
  downgrades (`min9 → min7`, `13 → 7`, `dim7 → dim`, `+ → aug`,
  `sus2 → sus4`). Unsupported qualities (e.g. `5`, `alt`) return `None`
  and are silently dropped.
- The dispatcher [chords/api.py](../src/music_decoder/chords/api.py)
  selects backends based on `params.backend`. Choosing
  `"madmom_deep_chroma"` falls back to `template_hmm` if madmom can't be
  imported. Choosing anything else uses `template_hmm`. The public
  `analyze()` adapter materializes a temporary WAV from its in-memory
  samples when madmom is requested so the file-based madmom pipeline
  has something to read.

### 4.7 Transcription

[transcription/transcribe()](../src/music_decoder/transcription/__init__.py)
wraps Spotify's basic-pitch model. The wrapper in
[basic_pitch_wrapper.py](../src/music_decoder/transcription/basic_pitch_wrapper.py)
writes the (mono float32) samples to a temp 16-bit WAV, prefers a CoreML
or ONNX variant of `ICASSP_2022_MODEL_PATH` if present (avoids TF version
issues on macOS), and calls `basic_pitch.inference.predict` with the
hyperparameters from [`BasicPitchParams`](../src/music_decoder/config/hyperparameters.py)
— in v2: `onset_threshold=0.5`, `frame_threshold=0.3`,
`minimum_note_length_ms=58`. The frequency bounds
(`minimum_frequency_hz`, `maximum_frequency_hz`) are read at call time
from [`config/hyperparameters.yaml`](../config/hyperparameters.yaml). The
adapter logs a one-line warning when the YAML values fall outside a
guitar-friendly window (< 50 Hz or > 3000 Hz) but still respects them;
when the YAML cannot be loaded it falls back to 65 Hz / 2093 Hz
(low E2 / C7).

The wrapper writes both `raw_basic_pitch.mid` and `post_basic_pitch.mid`
to its `output_dir`. The post-MIDI is currently a copy; the post-processing
in [transcription/post_processing.py](../src/music_decoder/transcription/post_processing.py)
is exposed as a separate function (`apply_post_processing`) and is not
wired into the public `analyze()` flow — its operations are:

- `drop_short_notes(min_duration_s=0.05)`.
- `merge_same_pitch(gap_s=0.05)` — coalesce same-pitch notes within the
  gap, taking max velocity and weighted-mean confidence.
- `median_filter_pitch_contour(window=5)` over a numerical pitch contour
  (helper, not currently called from the pipeline).
- `snap_to_beats(beats, confidence_threshold=0.7, max_snap_s=0.05)` —
  shifts high-confidence note onsets to the nearest beat if within
  `max_snap_s`.

The transcription layer returns `tuple[Note, ...]` to the orchestrator.

### 4.8 Tab assignment

[tabs/assign_tabs()](../src/music_decoder/tabs/__init__.py) is the public
adapter. It calls
[tabs/assigner.py](../src/music_decoder/tabs/assigner.py) `assign_tab`
with default weights:

```python
{"w_move": 1.0, "w_string": 0.3, "w_span": 0.5,
 "w_high": 0.4, "w_open": 0.2, "w_chord_intra": 0.6}
```

and `max_fret=22`.

#### 4.8.1 Grouping into chord states

`_group_simultaneous` packs notes whose intervals overlap into chord
groups. Two notes belong to the same group if any subsequent note's
`start_s` falls inside the current group's max `end_s`. This is
intentionally conservative — slightly overlapping legato notes get
grouped, which is acceptable because the per-note transition cost is
zero inside a group.

Chords with > 6 notes are clamped to the 6 highest-confidence notes; the
overflow is dropped with reason `chord_too_dense_capped_to_6`.

#### 4.8.2 Candidate generation

[tabs/candidates.py](../src/music_decoder/tabs/candidates.py):

- `note_candidates(pitch, tuning, max_fret)`: every `(string, fret)` pair
  for which `tuning.open_pitches[string] + fret == pitch` and
  `0 ≤ fret ≤ max_fret`.
- `chord_combinations(pitches, tuning, max_fret)`: cartesian product of
  per-pitch candidates, filtered to (a) no two notes on the same string
  and (b) `max(fretted) - min(fretted) ≤ 5` (the constant
  `_MAX_CHORD_SPAN`). Returns the surviving combinations.

A note with no candidates is dropped with reason
`out_of_range_for_tuning`. A chord with no satisfiable combination is
dropped with reason `unsatisfiable_chord`.

#### 4.8.3 A* search

[tabs/astar.py](../src/music_decoder/tabs/astar.py) treats the sequence
of groups as a layered graph: nodes are `(group_index, candidate_index,
recent_anchor_frets)`. The hand-anchor history holds the last
`hand_anchor_window` (default 2) representative frets so that each
candidate's "stretch" cost is computed against a rolling median of the
hand position.

The cost between successive states (in
[tabs/cost.py](../src/music_decoder/tabs/cost.py)
`transition_cost(prev, curr, weights, hand_anchor)`):

```text
move        = w_move   * max(0, curr.fret - prev.fret)
string_jump = w_string * |curr.string - prev.string|
span        = w_span   * max(0, curr.fret - hand_anchor) ** 1.5
high_fret   = w_high   * max(0, curr.fret - 12) ** 1.2
open_bonus  = w_open   * (-1.0 if curr.fret == 0 else 0.0)
total       = move + string_jump + span + high_fret + open_bonus
```

The "representative" fret of a chord state is the lowest non-negative
fret among its members. Per-state extras:

- `chord_collides`: two notes on the same string → cost `inf` (filtered).
- `chord_span_penalty`: `(span - 4) ** 1.5` once the fretted span exceeds
  4. Multiplied by `w_chord_intra` (0.6) and added to the state cost.

The heuristic in [tabs/heuristic.py](../src/music_decoder/tabs/heuristic.py)
`remaining_high_fret_penalty` is admissible: for each remaining group it
takes the minimum fret across all candidates and applies only the
`high_fret` term. Since every term in the actual transition cost is
non-negative and the high-fret term is only raised by going higher (not
lower), the lower bound is safe.

The search uses `heapq` priority queue, expanding the `(group, candidate)`
state with the lowest `g + h`. It terminates when popping a state with
`group_index = n - 1`. If the heap empties without reaching the last
layer, every note in every group is dropped with reason `no_valid_path`.

#### 4.8.4 Tunings

[tabs/tuning.py](../src/music_decoder/tabs/tuning.py) re-exports
`Tuning` from [types.py](../src/music_decoder/types.py) and defines the
preset table:

- `STANDARD_EADGBE`: `(40, 45, 50, 55, 59, 64)` (E2 A2 D3 G3 B3 E4).
- `DROP_D`: `(38, 45, 50, 55, 59, 64)` (low E → D).
- `EB_HALF_STEP_DOWN`: `(39, 44, 49, 54, 58, 63)`.
- `D_STANDARD`: `(38, 43, 48, 53, 57, 62)`.
- `DROP_C`: `(36, 43, 48, 53, 57, 62)`.
- `DADGAD`: `(38, 45, 50, 55, 57, 62)`.

`get_preset(name)` returns by string key (used by the CLI and Streamlit).

#### 4.8.5 Tab rendering

[tabs/render.py](../src/music_decoder/tabs/render.py):

- `render_ascii_tab(notes, num_strings=6, columns=64)`: produces a
  6-line ASCII tab with E/A/D/G/B/e labels. Notes are placed at column
  `int((start_s / end_time) * (columns - 4))`. Multi-digit fret numbers
  occupy multiple columns.
- `render_svg_fretboard(notes, num_strings=6, max_fret=12)`: emits raw
  SVG with horizontal strings and vertical fret bars; markers are
  colored by confidence (green ≥ 0.8, gold ≥ 0.5, red < 0.5). Used by
  the Streamlit UI.

Both accept `num_strings` (v2) and `n_strings` (legacy) for backward
compatibility.

---

## 5. Data flow: `compose()`

Implemented in [compose/api.py](../src/music_decoder/compose/api.py):

```text
(scale, progression, style, tempo, bars_per_chord, seed)
  │
  ▼
_validate(scale, progression)
  │   Checks mode ∈ {major, minor}; progression non-empty.
  │   Otherwise raises InvalidScaleError / InvalidProgressionError.
  ▼
compose.voicings.voicings_for(chord, tuning) for each chord
  │   Returns 1-3 ranked VoicedChord candidates per chord.
  │   The orchestrator picks index [0] (the lowest-fret playable voicing).
  ▼
compose.melody.generate_melody(scale, progression, tempo, seed)
  │   Returns list[Note] spanning the entire progression.
  ▼
work = out_dir / f"{ts}-{payload_hash}"        # per-call directory
  │   payload_hash = sha1(scale|progression|tempo|style|seed)[:8]
  ▼
compose.arrangement.build_midi(...)            → composition.mid
  │
  ▼
synth.render_wav(...)                          → composition.wav
  │
  ▼
_ascii_tab_for(melody, voicings, tuning)       → ASCII string
  │
  ▼
Composition(midi_path, wav_path, ascii_tab, melody_notes, chord_voicings, metadata)
```

`compose()` is fully synchronous. Determinism: the only source of
randomness is `numpy.random.default_rng(seed)` inside the melody
generator; same seed → same melody. The output directory layout is
content-addressed (`{timestamp}-{first 8 hex of payload SHA-1}`) so the
caller can keep multiple takes side by side.

### 5.1 Voicings

[compose/voicings.py](../src/music_decoder/compose/voicings.py):

1. Look up the canonical EADGBE voicing in
   [chords/voicings.py](../src/music_decoder/chords/voicings.py) — a
   hand-curated table of 96 entries (12 roots × 8 qualities) plus `N`.
   Each entry is a 6-tuple `(E, A, D, G, B, e)` with `-1` = muted, `0` = open,
   positive ints = fret numbers.
2. **Retune** to the requested tuning by `_retune`:
   - For each fretted string: `new_fret = canonical_fret +
     (eadgbe_open_pitch - target_open_pitch)`. If a fret would go
     negative the candidate is filtered out by `_is_playable`.
   - For muted strings: if the new tuning's open string happens to be a
     chord tone (e.g. low D in Drop D for a D chord), un-mute it as an
     open-string addition.
3. **Playability gate** `_is_playable`: at least one fretted note,
   `min(fret) ≥ 0`, `max(fret) ≤ 22`, `max - min ≤ 5`.
4. **Fallback**: if every retuned candidate is unplayable, fabricate a
   sparse voicing of the chord's root pitch class on the lowest 1-3
   strings in the target tuning.
5. **Sort** by `(max(fret), sum(frets))` — prefer lower position, then
   tighter packing.
6. Return the top 3 candidates.

### 5.2 Melody generator

[compose/melody.py](../src/music_decoder/compose/melody.py)
`generate_melody` builds a list of `Note` for the whole progression:

1. Compute the scale's pitch-class set: major = `(0, 2, 4, 5, 7, 9, 11)`,
   minor (natural) = `(0, 2, 3, 5, 7, 8, 10)`, both rotated by the tonic's
   pitch class.
2. Enumerate all MIDI pitches in `[pitch_low=60, pitch_high=84]` (C4 to
   C6) whose pitch class is in the scale → `scale_pitches`.
3. For each chord segment in the progression, for each bar
   (`bars_per_chord` of them), for each beat (`notes_per_bar=4`):
   - Determine if the beat is "strong" (`nidx % 2 == 0` for 4 notes per
     bar — beats 1 and 3).
   - Sample `use_chord_tone ~ Bernoulli(p_strong=0.7 / p_weak=0.3)`.
   - Pool = chord-tone scale pitches if `use_chord_tone` else all scale
     pitches.
   - **Markov constraint**: prefer pitches within
     `markov_max_interval_semitones=4` of `prev_pitch`. If no candidates
     within that window, fall back to the full pool.
   - Pick uniformly from the filtered pool with `rng.choice`.
   - Velocity: 80 strong, 70 weak.
   - Note duration: `seconds_per_note * 0.95` (small gap).
4. Update `prev_pitch` after each note.

The "Markov" part is a hard interval cap, not a learned transition matrix:
it biases toward stepwise/melodic motion rather than wide leaps.

### 5.3 Arrangement

[compose/arrangement.py](../src/music_decoder/compose/arrangement.py)
`build_midi` writes a two-track `pretty_midi.PrettyMIDI`:

- **Melody track** — program `25` (Acoustic Steel Guitar). One
  `pretty_midi.Note` per generated `Note`.
- **Accompaniment track** — same program. Pattern depends on `style`:
  - `strum`: block chord on beats 1 and 3, sustained ~1.9 beats.
  - `arpeggio`: ascending then descending sequence over the voicing's
    pitches across one bar (8-ish eighth notes), each note 95% of the
    inter-note step.
  - `fingerstyle`: bass note on beats 1 and 3, treble cluster (everything
    above the bass) on beats 2 and 4, each ~95% of a beat.

Accompaniment pitches come from `_voiced_pitches(voicing, tuning) =
sorted(open_pitches[s] + fret for fretted strings)`.

### 5.4 ASCII tab for the composition

`_ascii_tab_for` greedy-assigns each melody note to the highest string
that produces a non-negative fret ≤ 22, then calls
[render_ascii_tab](../src/music_decoder/tabs/render.py). The ASCII tab is
an approximate visual reference; the MIDI is the source of truth.

### 5.5 Synthesis

[synth/render_wav](../src/music_decoder/synth/__init__.py) wraps
[synth/fluidsynth_wrapper.py](../src/music_decoder/synth/fluidsynth_wrapper.py)
`synthesize_midi_to_wav` with auto-selection: it picks
**`SynthBackend.FLUIDSYNTH`** when both
`RuntimeConfig.fluidsynth_soundfont` resolves to an existing `.sf2` and
the `fluidsynth` Python module is importable; otherwise it logs a
single warning line and falls back to `SynthBackend.SINE`
(`pretty_midi.PrettyMIDI.synthesize(fs=sr)`, no external deps). Callers
can force a specific backend via the optional `backend` keyword. The
output is peak-normalized to 0.9 and written as 16-bit PCM at
`sr=22050` Hz.

---

## 6. Configuration

Two YAML files under [config/](../config/):

### 6.1 `config/runtime.yaml`

Loaded by
[config/runtime.py](../src/music_decoder/config/runtime.py)
`load_runtime_config()`. Fields:

- `data_dir` (default `~/.music-decoder`).
- `log_level` (default `INFO`).
- `ffmpeg_path` (default `ffmpeg` on `$PATH`).
- `fluidsynth_soundfont` (default `GeneralUser-GS.sf2`).
- `sample_rate_hz` (default `22050`).
- `youtube_cache_dir` (default `${data_dir}/yt_cache`; `${...}`
  references resolve from sibling keys via a 4-pass interpolation).
- `composition_out_dir` (default `${data_dir}/compositions`).

Every field can be overridden via `MUSIC_DECODER_<UPPERCASE>` env vars
(e.g. `MUSIC_DECODER_DATA_DIR=/tmp/md-test`). Required keys:
`youtube_cache_dir`, `composition_out_dir`, `log_level`. Missing
required keys raise at load time.

### 6.2 `config/hyperparameters.yaml`

Loaded by
[config/hyperparameters.py](../src/music_decoder/config/hyperparameters.py)
`load_hyperparameters(path)` into a frozen
[`HyperparameterSet`](../src/music_decoder/config/hyperparameters.py).
The `id` field (currently `2026-05-08-v2-baseline`) is stamped into
every `AnalysisResult.metadata["hyperparameter_set"]`. Sections:

- `basic_pitch` — onset/frame thresholds, min note length, freq bounds.
- `post_processing` — note-merging gap, snap threshold, median filter
  window.
- `key_detection` — HPSS margin, windowed segment length / hop.
- `beat_tracking` — `start_bpm`, `tightness`.
- `chord_detection` — `backend` (`madmom_deep_chroma` or
  `template_hmm`), `qualities` list (must equal templates `QUALITIES`
  or `detect_chords` raises), `min_segment_duration_s`.
- `tab_assignment` — six A* weights and `max_fret`.
- `composition` — notes per bar, strong/weak chord-tone probabilities,
  Markov interval cap, strong/weak velocities.

Note: most adapter functions read their hyperparameters from this YAML
at call time. As of the v2 reconciliation, the public adapters do
respect:

- `chord_detection.backend` — picked up by
  [chords/\_\_init\_\_.py](../src/music_decoder/chords/__init__.py).
- `basic_pitch.minimum_frequency_hz` /
  `basic_pitch.maximum_frequency_hz` — picked up by
  [transcription/\_\_init\_\_.py](../src/music_decoder/transcription/__init__.py).

A few sub-keys remain hard-coded in the adapter modules — notably the
`tab_assignment.weights` block and `post_processing.median_filter_window`,
plus the chord-detection threshold / self-transition probabilities. Those
defaults are intentional for v2 and are listed alongside their adapters
in §4.x; we track the gap as future work in spec §14.

### 6.3 `config/eval_thresholds.yaml`

Three numbers consumed only by
[tests/integration/test_regression.py](../tests/integration/test_regression.py):

- `chord_recognition_score: 0.60`
- `key_mirex_score: 0.75`
- `tab_string_accuracy: 0.55`

---

## 7. CLI

[cli/main.py](../src/music_decoder/cli/main.py) — Click group `main` with
four subcommands:

- `analyze <source>`
  - `--tuning EADGBE | "Drop D" | "Drop-D" | Eb | "D standard" | "Drop C" | DADGAD`
  - `--solo-guitar / --full-mix` (default `--full-mix`).
  - `--no-separation` (boolean flag).
  - `--format pretty | json`.
- `compose`
  - `--scale "C:major"` (required).
  - `--progression "Cmaj7 Am7 Dm7 G7"` (required, space-separated).
  - `--style arpeggio | strum | fingerstyle` (default `fingerstyle`).
  - `--tempo FLOAT` (default `100.0`).
  - `--bars INT` (default `1`).
  - `--seed INT` (optional).
  - `--tuning ...` (same set as `analyze`).
  - `--out DIR` (required).
  - `--format pretty | json`.
- `ui` — launches Streamlit via `subprocess` on
  `ui/streamlit_app.py` with `--server.headless false
  --browser.gatherUsageStats false`.
- `doctor` — runs three checks: `ffmpeg` on `PATH`, importable
  `fluidsynth` Python module, soundfont file exists at
  `RuntimeConfig.fluidsynth_soundfont` (only checked if absolute). Exits
  non-zero on any fail.

The CLI converts `MusicDecoderError` into `click.ClickException` so the
process exits with a clean non-zero status and a one-line message.
`json` mode walks the dataclass tree via `_serialize` (recursive
`asdict` + `Path → str` + `tuple → list` coercion).

---

## 8. Streamlit UI

[ui/streamlit_app.py](../src/music_decoder/ui/streamlit_app.py) is a
single-page app with three tabs:

- **Analyze** — file uploader / YouTube URL text box, tuning dropdown,
  declared-kind radio, "Use Demucs separation" checkbox, "Analyze" button.
  On submit, calls `analyze()` with a progress callback bound to a
  `st.progress` bar. Shows: detected key + tempo, the chord progression
  rendered through
  [`ui/components/chord_progression.py`](../src/music_decoder/ui/components/chord_progression.py),
  ASCII tab via `render_ascii_tab`, and HTML5 audio playback of the
  source.
- **Compose** — tonic dropdown (12 notes), mode dropdown (`major`/
  `minor`), free-text progression box (default `"Cmaj7 Am7 Dm7 G7"`),
  style dropdown, tempo slider 40-220, bars slider 1-4, optional seed
  text box, tuning dropdown. On submit, calls `compose()` and renders
  WAV playback, ASCII tab, and download buttons for the MIDI and WAV.
- **About** — version (`music_decoder.__version__`) and the loaded
  hyperparameter-set ID.

Errors are caught at the top of each tab handler (`MusicDecoderError`
in Analyze; `MusicDecoderError | ValueError` in Compose) and rendered
via `st.error(str(e))`.

State lives in `st.session_state`. Models are loaded once per
Streamlit session (the underlying lazy imports inside the library are
cached by Python's import system; the UI doesn't decorate with
`@st.cache_resource` explicitly because the lazy imports already do
the work).

---

## 9. Failure handling

Source: [errors.py](../src/music_decoder/errors.py) and the per-stage
modules. Summary:

- **Corrupt audio / NaNs / Inf** → `CorruptAudioError` from
  `audio_file._validate`.
- **Silent audio** (`RMS < 1e-4`) → `SilentAudioError`.
- **Clip < 1.0 s** → `ClipTooShortError`.
- **YouTube download failure** (private, region-locked, network) →
  `YouTubeError` wrapping `yt_dlp.utils.DownloadError`.
- **Demucs error** → `SeparationError`; `analyze()` catches and falls
  back to original samples (logged but does not raise).
- **No key estimable** (empty chroma) → `KeyDetectionError` from
  `key.estimate_key`.
- **Degenerate beat grid** (< 4 beats) → the `chords` adapter swaps in
  a synthetic 4-beat grid spanning the audio. If that still yields no
  segments, a single `("N", "no-chord")` segment is emitted as a
  safety net.
- **A* finds no path** → all notes in all groups are dropped with
  reason `no_valid_path`; `tab` is empty.
- **Out-of-range note for tuning** → dropped with reason
  `out_of_range_for_tuning`.
- **Unsatisfiable chord** → all notes in the group dropped with reason
  `unsatisfiable_chord`.
- **Chord with > 6 notes** → highest-confidence 6 kept; rest dropped
  with reason `chord_too_dense_capped_to_6`.
- **Invalid `Scale` / `ChordSymbol` parse** → `ValueError`; the CLI
  upgrades to `ClickException`. `compose()` validates and raises
  `InvalidScaleError` / `InvalidProgressionError`.
- **No canonical voicing for chord** → `InvalidProgressionError` (the
  voicings table covers every supported quality, so this only fires if
  the chord symbol has been mutated past the parser).
- **ffmpeg / fluidsynth / soundfont missing** → caught by
  `music-decoder doctor` (exit non-zero) and surfaced lazily by the
  underlying call site otherwise.

The Streamlit UI catches any `MusicDecoderError` at the tab boundary
and renders it as `st.error(...)`; the underlying stack trace appears
in the terminal where Streamlit was launched.

---

## 10. Dependency graph

Top-level imports (transitively pruned):

```text
api.py
 ├─ chords (recognize_chords)
 │   ├─ chords.api.detect_chords (lazy)
 │   ├─ chords.templates / recognize / labels
 │   └─ chords.backends.{template_hmm, madmom_deep_chroma}
 │       └─ chords.madmom_compat (preflight)
 ├─ compose.api.compose
 │   ├─ compose.voicings → chords.voicings
 │   ├─ compose.melody
 │   ├─ compose.arrangement → pretty_midi
 │   ├─ synth.render_wav → synth.fluidsynth_wrapper → pretty_midi/scipy
 │   └─ tabs.render.render_ascii_tab
 ├─ dsp (compute_chroma, track_beats)
 │   ├─ dsp.chroma → librosa
 │   └─ dsp.beats → librosa
 ├─ ingest (load)
 │   ├─ ingest.audio_file → ffmpeg subprocess
 │   └─ ingest.youtube → yt_dlp
 ├─ key (estimate_key)
 │   └─ key.global_estimator → key.ks → key.profiles
 ├─ separation.run_separation (lazy)
 │   └─ separation.demucs → demucs/torch/librosa
 ├─ tabs.assign_tabs
 │   └─ tabs.assigner → tabs.{astar, candidates, cost, heuristic, tuning}
 ├─ transcription.transcribe (lazy)
 │   └─ transcription.basic_pitch_wrapper → basic_pitch / TF / CoreML
 ├─ types (frozen dataclasses)
 ├─ errors
 └─ progress
```

Heavy imports (Demucs/torch, basic-pitch/TF, madmom) are deferred to
first-use to keep `import music_decoder` cheap. The UI subprocess
re-imports through Streamlit so it pays the initial load only at app
start.

---

## 11. Testing & regression

### 11.1 Layout

Under [tests/](../tests/):

- `unit/` — pure-Python tests for individual modules.
- `integration/` — three e2e suites:
  - [test_analyze_e2e.py](../tests/integration/test_analyze_e2e.py)
    runs the full pipeline against each committed synthetic fixture,
    asserts result shape (non-zero duration, ≥ 1 chord, ≥ 1 tab note).
  - [test_compose_e2e.py](../tests/integration/test_compose_e2e.py)
    runs `compose("C:major", "Cmaj7 Am7 Dm7 G7", seed=11)` and
    asserts MIDI and WAV files exist, two instruments, all melody
    pitches in C major.
  - [test_regression.py](../tests/integration/test_regression.py)
    parametrized over each synthetic fixture; gates
    `chord_recognition_score`, `key_mirex_score`, and
    `tab_string_accuracy` against `config/eval_thresholds.yaml`.
- `regression/` — placeholder for future committed-result snapshots.
- `fixtures/synthetic/` — committed `.mid` fixtures (`c_major_scale.mid`,
  `g_major_chord.mid`). The first run renders them to `.wav` via
  `pretty_midi.fluidsynth` if `tests/fixtures/synthetic/soundfont/*.sf2`
  exists, else via the sine synth. Render output is cached on disk
  alongside the MIDI with a `.synth_marker` invalidation key.

### 11.2 Markers

`pyproject.toml` `[tool.pytest.ini_options]` declares three markers:
`slow`, `regression`, `integration`. The default `make test` runs unit
+ integration; `make regression` runs only `-m regression`.

### 11.3 Metrics

[evaluation/metrics.py](../src/music_decoder/evaluation/metrics.py)
implements:

- `chord_recognition_score`: MIREX-style frame-level (10 ms grid,
  configurable). Per frame: `1.0` for exact `(root, quality)` match,
  `0.5` for matching root with different quality, `0.0` otherwise.
- `key_mirex_score`: 1.0 exact, 0.5 for fifth-related, 0.3 for
  relative major/minor, 0.2 for parallel major/minor, else 0.0.
- `tab_string_accuracy`: in time-aligned mode, calls
  `mir_eval.transcription.match_notes` to find onset+pitch matches,
  then reports the fraction whose `string` index agrees. Falls back
  to legacy index-aligned mode when intervals are not supplied.
- `note_f_measure` / `onset_f_measure` / `pitch_class_accuracy` —
  thin wrappers around `mir_eval.transcription` and a custom 100 Hz
  pitch-class frame accuracy.

### 11.4 CI

A single GitHub Actions job (per the design doc):

1. Install Python 3.11.
2. Install system deps: `ffmpeg libsndfile1 libfluidsynth3`.
3. `pip install -e ".[dev]"`.
4. `make lint typecheck test`.
5. `make regression`.

Models are downloaded on first run; the CI cache keys are basic-pitch
+ Demucs + madmom versions.

---

## 12. Distribution

- Build backend: `hatchling >= 1.21` (declared in
  [pyproject.toml](../pyproject.toml)).
- Console entry point: `music-decoder = "music_decoder.cli.main:main"`.
- Python: `>= 3.11, < 3.13`.
- Packaging surface: `src/music_decoder/` (single namespace).
- Docker: [Dockerfile](../Dockerfile) installs ffmpeg, libsndfile1,
  libfluidsynth3 + build tools, then `pip install -e ".[dev]"`. The
  default CMD is `python -m music_decoder.cli.main ui`.
- Compose: [docker-compose.yml](../docker-compose.yml) maps
  `8501:8501` and mounts a named volume `md-data:/data` for the
  YouTube cache and composition output. `MUSIC_DECODER_DATA_DIR=/data`
  is set in the environment.

There is no SQLite, no Alembic, no Postgres dependency. The legacy
`migrations/` directory still exists at the repo root but is not
imported by the v2 codebase.

---

## 13. Conventions

- All shared shapes are frozen dataclasses in
  [types.py](../src/music_decoder/types.py); modules construct, never
  mutate.
- Every long-running stage takes an optional `ProgressCallback` and
  emits at start (`fraction=0.0`) and end (`fraction=1.0`). The library
  doesn't emit fractions in between because the underlying ML models
  are opaque; if you wire one in, keep its emissions monotonic.
- Module-level adapters under each subpackage's `__init__.py` are the
  stable surface. Internals (e.g.
  [chords/recognize.py](../src/music_decoder/chords/recognize.py)) may
  change.
- Lazy imports for heavy ML modules (`demucs`, `basic_pitch`, `madmom`)
  go inside the function that needs them, not at module top.
- Logging via [logging_setup.py](../src/music_decoder/logging_setup.py)
  `get_logger("...")` — JSON output, structured `extra={...}`. No bare
  `print` in library code.

---

## 14. Where to make common changes

- **Add a new tuning preset**: edit
  [tabs/tuning.py](../src/music_decoder/tabs/tuning.py) `PRESETS` and
  add to the `_TUNINGS` map in [cli/main.py](../src/music_decoder/cli/main.py)
  and [ui/streamlit_app.py](../src/music_decoder/ui/streamlit_app.py).
- **Add a new chord quality**: extend `QUALITIES` and
  `_QUALITY_INTERVALS` in
  [chords/templates.py](../src/music_decoder/chords/templates.py),
  add a row to every root in
  [chords/voicings.py](../src/music_decoder/chords/voicings.py), update
  `_QUALITY_PATTERNS` and `_QUALITY_TO_LABEL` in
  [types.py](../src/music_decoder/types.py), update
  `_QUALITY_MAP` in
  [chords/labels.py](../src/music_decoder/chords/labels.py), and update
  the `_CHORD_PCS` tables in
  [compose/voicings.py](../src/music_decoder/compose/voicings.py) and
  [compose/melody.py](../src/music_decoder/compose/melody.py).
- **Tweak A* weights**: edit `_DEFAULT_WEIGHTS` in
  [tabs/__init__.py](../src/music_decoder/tabs/__init__.py).
  `config/hyperparameters.yaml` records the canonical values for
  reproducibility but the adapter pins its own copy.
- **Switch chord backend**: edit
  [`config/hyperparameters.yaml`](../config/hyperparameters.yaml)
  `chord_detection.backend` (currently `madmom_deep_chroma`; the
  alternative is `template_hmm`). The public `analyze()` adapter reads
  this at call time.
- **Use fluidsynth output**: ensure
  `RuntimeConfig.fluidsynth_soundfont` resolves to an existing `.sf2`
  file (run `scripts/download_soundfont.py` for a public-domain one)
  and that `pip install pyfluidsynth` succeeds.
  [synth/__init__.py](../src/music_decoder/synth/__init__.py)
  `render_wav` auto-selects fluidsynth when both conditions hold and
  falls back to the sine synth otherwise. Callers can force a specific
  backend via the optional `backend=` keyword.
- **Change the synthetic regression fixtures**: drop a `.mid` under
  [tests/fixtures/synthetic/](../tests/fixtures/synthetic/) and add
  ground truth in
  [evaluation/fixtures/synthetic.py](../src/music_decoder/evaluation/fixtures/synthetic.py)
  `_SYNTHETIC_GT`. Without a GT entry the fixture is skipped silently.

---

## 15. Open work

- **Auto scale detection inside `compose()`** — infer scale from the
  progression. Deferred.
- **Modal scales** beyond major/minor (Dorian, Mixolydian, Phrygian,
  Lydian). Deferred.
- **Chord progression suggester** — given a key, generate candidate
  progressions. Out of scope.
- **Audio export of `analyze()` results** — render the detected MIDI
  back to WAV for side-by-side playback. Cheap to add.
- **Multi-track output** — separate WAV stems for melody vs
  accompaniment. Cheap to add.
- **Caching of analysis results** — currently re-runs from scratch.
  Could add a content-hash-keyed disk cache of `AnalysisResult`.

---

## 16. References

- v2 design doc:
  [`docs/superpowers/specs/2026-05-08-refactor-design.md`](superpowers/specs/2026-05-08-refactor-design.md)
- mir_eval: <https://github.com/mir-evaluation/mir_eval>
- Spotify basic-pitch: <https://github.com/spotify/basic-pitch>
- Facebook Demucs: <https://github.com/facebookresearch/demucs>
- madmom: <https://github.com/CPJKU/madmom>
- yt-dlp: <https://github.com/yt-dlp/yt-dlp>
- FluidSynth: <https://www.fluidsynth.org/>
