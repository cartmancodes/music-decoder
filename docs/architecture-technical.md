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
- [assets.py](../src/music_decoder/assets.py) — model/asset provisioning: bundled soundfont, soundfont download + validation, Demucs weight cache check/fetch (used by `fetch-models`, `doctor`, `setup.sh`).
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
  ├─► dsp.track_beats          [dsp/__init__.py] → BeatGrid (madmom RNN+DBN; librosa fallback)
  │
  ├─► dsp.compute_chroma       [dsp/chroma.py]  →  (12, T) chroma matrix
  │
  ├─► chords.recognize_chords  [chords/__init__.py]
  │       madmom deep-chroma (default) or template_hmm backend
  │       → tuple[ChordSegment, ...]
  │
  ├─► key.estimate_key         [key/__init__.py]
  │       fusion of KK + Temperley profiles (chroma), madmom CNN (audio)
  │       and the chord progression → KeyEstimate
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

The adapter [dsp/track_beats()](../src/music_decoder/dsp/__init__.py)
picks a backend from `beat_tracking.backend` in
[`config/hyperparameters.yaml`](../config/hyperparameters.yaml):

- **`madmom`** — [dsp/beats_madmom.py](../src/music_decoder/dsp/beats_madmom.py)
  runs madmom's `RNNDownBeatProcessor` + `DBNDownBeatTrackingProcessor`
  (Böck, Krebs & Widmer, ISMIR 2016) with `beats_per_bar=[3, 4]`, so it
  returns real downbeats and a 3/4-vs-4/4 decision (`ts_assumed=False`).
  Tempo is `60 / median(inter-beat interval)`. madmom 0.16.1's DBN calls
  `np.asarray` on ragged data, which NumPy ≥ 1.24 rejects;
  [chords/madmom_compat.py](../src/music_decoder/chords/madmom_compat.py)
  `patch_downbeats_numpy()` scopes a ragged-tolerant `asarray` to that one
  madmom module. If madmom raises or finds fewer than 4 beats the adapter
  falls back to librosa.
- **`librosa`** — [dsp/beats.py](../src/music_decoder/dsp/beats.py) wraps
  `librosa.beat.beat_track` with `start_bpm=120, tightness=100`. Downbeats
  are a crude `beats[::4]`, 4/4 with `ts_assumed=True`.

If the input is essentially silent (`RMS < 1e-5`) the librosa path returns
an empty BeatGrid rather than letting librosa raise.

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
`estimate_key(chroma, *, samples=None, sr=None, chords=None)` first checks
`key_detection.backend` (v3 default: `fusion`).

With **`fusion`** ([key/fusion.py](../src/music_decoder/key/fusion.py)) it
scores all 24 keys with up to four independent cues, z-normalizes each and
sums them: Krumhansl-Kessler and Temperley correlations with the mean
chroma, madmom's CNN key probabilities (when audio is given), and a
duration-weighted diatonic fit of the recognized chord progression (when
chords are given — which is why `analyze()` now runs chord recognition
before key detection). The cues fail differently (profiles confuse relative
keys, the CNN fifths, chord fits modes), so the sum beats every single cue:
GuitarSet dev 0.685 MIREX-weighted vs 0.560 for v2's profile consensus.
Unavailable or flat cues are skipped; if none is usable the profile path
below runs. The result has `profile="fusion"`, `correlation` = the
winner's mean z-score, `margin` = its lead over the runner-up per cue.

With `cnn` (and raw samples supplied, as
`analyze()` does) it runs madmom's `CNNKeyRecognitionProcessor`
(Korzeniowski & Widmer, ISMIR 2018) via
[key/cnn.py](../src/music_decoder/key/cnn.py) and returns a
`KeyEstimate(profile="cnn", correlation=top probability,
margin=top1 − top2)`; any madmom failure falls back to the profile path.
With `profile` it returns the consensus when present; otherwise it falls
back to the `krumhansl_kessler` top-1 estimate. If no estimates were produced at all
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
- `MadmomCNNBackend` (same file) swaps the feature extractor for madmom's
  fully-convolutional `CNNChordFeatureProcessor` + `CRFChordRecognitionProcessor`
  (Korzeniowski & Widmer, MLSP 2016); selected with `"madmom_cnn"`.
- The dispatcher [chords/api.py](../src/music_decoder/chords/api.py)
  selects backends based on `params.backend`. Choosing
  `"madmom_deep_chroma"` or `"madmom_cnn"` falls back to `template_hmm` if
  madmom can't be imported **or** if the madmom backend returns no segments
  at runtime (init/inference failure) — previously such a failure turned
  the whole song into a single `N`. Choosing anything else uses
  `template_hmm`. The public
  `analyze()` adapter materializes a temporary WAV from its in-memory
  samples (via [dsp/tempwav.py](../src/music_decoder/dsp/tempwav.py)
  `temp_wav`, shared with the key and beat madmom paths) when a madmom
  backend is requested so the file-based madmom pipeline has something to
  read.

### 4.7 Transcription

[transcription/transcribe()](../src/music_decoder/transcription/__init__.py)
wraps Spotify's basic-pitch model in two steps:

1. **Inference** —
   [basic_pitch_wrapper.run_model()](../src/music_decoder/transcription/basic_pitch_wrapper.py)
   writes the samples to a temp WAV and runs the network once, preferring
   a CoreML or ONNX variant of `ICASSP_2022_MODEL_PATH` (avoids TF version
   issues on macOS). It returns basic-pitch's raw `note` / `onset` /
   `contour` posteriorgrams.
2. **Decoding** — `decode_model_output()` turns the posteriorgrams into
   notes with `basic_pitch.note_creation.model_output_to_notes`, using
   every parameter from the `basic_pitch` section of
   [`config/hyperparameters.yaml`](../config/hyperparameters.yaml):
   `onset_threshold`, `frame_threshold`, `minimum_note_length_ms`,
   `minimum_frequency_hz`, `maximum_frequency_hz`, `melodia_trick`
   (`resolve_params()`; warns when the bounds look non-guitar but still
   respects them; falls back to basic-pitch's defaults with 65–2093 Hz
   bounds if the YAML can't be loaded). Notes come back sorted by
   `(onset, pitch)`.

Splitting the two lets the benchmark cache network outputs and re-decode
them during threshold sweeps (§11.5). When `post_processing.enabled` is
true, decoding then applies
[post_processing.py](../src/music_decoder/transcription/post_processing.py)
`merge_same_pitch(gap_s=same_pitch_merge_gap_s)` and
`drop_short_notes(min_duration_s=min_note_duration_s)`. The legacy
`transcribe_basic_pitch()` (writes raw MIDI to an output directory) is kept
for callers that want MIDI files.

### 4.8 Tab assignment

[tabs/assign_tabs()](../src/music_decoder/tabs/__init__.py) is the public
adapter. It calls
[tabs/assigner.py](../src/music_decoder/tabs/assigner.py) `assign_tab`
with the weights and `max_fret` from `tab_assignment` in
[`config/hyperparameters.yaml`](../config/hyperparameters.yaml) (explicit
`weights=` / `max_fret=` arguments override them). The code fallback, used
only when the YAML can't be loaded, is:

```python
{"w_move": 0.5, "w_string": 0.15, "w_span": 0.5,
 "w_high": 1.6, "w_open": 0.05, "w_chord_intra": 0.3}
```

and `max_fret=22`.

#### 4.8.1 Grouping into chord states

Notes no string can play in the current tuning are dropped first, one by
one, with reason `out_of_range_for_tuning` — so a single sub-range artifact
can't make an otherwise playable chord unsatisfiable.

`_group_simultaneous` then groups notes **by onset proximity**: a note
joins the current group when its onset is within 50 ms
(`_CHORD_ONSET_WINDOW_S`) of the group's first onset, which covers strums.
A note struck while an earlier one is still ringing starts a new group. (v2
grouped by interval overlap, which fused re-attacks of the same pitch into
one unsatisfiable "chord"; GuitarSet dev string accuracy 0.584 → 0.614 on
ground-truth notes, 0.550 → 0.587 end-to-end.)

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
move        = w_move   * |curr.fret - prev.fret|
string_jump = w_string * |curr.string - prev.string|
span        = w_span   * max(0, curr.fret - hand_anchor) ** 1.5
high_fret   = w_high   * max(0, curr.fret - 12) ** 1.2
open_bonus  = w_open   * (-1.0 if curr.fret == 0 else 0.0)
total       = move + string_jump + span + high_fret + open_bonus
```

Hand movement is symmetric (v2 charged only upward moves, so sliding down
the neck was free). The "representative" fret of a chord state is the
lowest *fretted* (non-zero) fret among its members — open strings don't
pin the fretting hand — or 0 if every member is open. Both changes were
chosen by measurement on GuitarSet (§11.5); a variant that made open
strings movement-free was measured and rejected. Per-state extras:

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

Implemented in [compose/api.py](../src/music_decoder/compose/api.py). All the
musical work happens in the pure `arrange()`, which reads no files or
configuration. `compose()` adds only file I/O:

```text
(scale, progression, style, tempo, bars_per_chord, seed, tuning)
  │
  ▼
arrange(...)                                   # pure: no files, no YAML
  │ _validate(scale, progression, bars_per_chord)
  │   mode ∈ {major, minor}; progression non-empty; bars_per_chord ≥ 1.
  │   Otherwise raises InvalidScaleError / InvalidProgressionError.
  │ compose.voicings.voicings_for(chord, tuning)[0] for each chord
  │ _melody_floor(voicing) for each chord     # keep the tune above the third/seventh
  │ compose.melody.generate_melody(..., chord_floors=...)
  │ accompany(voicings, melody, ...)           # melody-aware accompaniment
  │ _melody_tab(melody, tuning)                # A* with pinned weights + fallback
  ▼
Arrangement(melody, voicings, accompaniment, melody_tab)
  │
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
compose.tabtext.render_composition_tab(...)    → ASCII string
  │
  ▼
Composition(midi_path, wav_path, ascii_tab, melody_notes, chord_voicings, metadata)
```

`compose()` is fully synchronous. Determinism: the only source of randomness
is `numpy.random.default_rng(seed)` inside the melody generator. It drives
both the rhythm plan and the pitch choices, so the same seed gives the same
music. `metadata["composer_version"]` (currently `2`) is bumped whenever the
composer changes what a given seed produces. Seeds from before version 2
(the uniform quarter-note generator) give different melodies now. The melody
tab's A\* weights are pinned in `compose/api.py` (`COMPOSE_TAB_WEIGHTS`), so
editing `config/hyperparameters.yaml` or setting
`MUSIC_DECODER_HYPERPARAMETERS` changes `analyze()` but never compose
output. The output directory layout is content-addressed (`{timestamp}-{first
8 hex of payload SHA-1}`), so callers can keep several takes side by side.

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
`generate_melody` builds the list of `Note` for the whole progression in two
layers. The pitch-class helpers it uses (scale, chord tones, essential tones,
chord scale) live in [compose/harmony.py](../src/music_decoder/compose/harmony.py).

**Rhythm and phrases**, from [compose/rhythm.py](../src/music_decoder/compose/rhythm.py):

1. `phrase_layout(n_bars)` fixes the phrase structure from the bar count
   alone. The piece is cut into 4-bar phrases. A remainder of 2 or 3 bars
   becomes a shorter last phrase, and a remainder of 1 turns the last 4 into
   3 + 2. The layout also gives each phrase's cadence bar and the motif
   *echo pairs*: bar 3 echoes bar 1 in a 4-bar antecedent, and a consequent
   phrase echoes the antecedent bar for bar. The metrics read the same
   layout.
2. `plan_bars(n_bars, rng)` gives every bar a rhythm (eighth-note values that
   sum to 8) from `BODY_RHYTHMS` (11 one-bar templates: quarters, eighth
   pairs, dotted figures, half notes). An echoing bar reuses its source bar's
   rhythm. Every phrase ends on a `CADENCE_RHYTHMS` bar (last note ≥ a half
   note).

**Pitches**, chosen one note at a time:

1. **Chord-scale pool.** `chord_scale_pcs(scale, chord)` spells the chord
   as a stack of thirds over the scale (root, third, fifth, seventh; a sus4
   has a fourth in place of the third). Each out-of-scale chord tone then
   replaces the scale degree it occupies. An out-of-scale root takes the
   spelling that alters the fewest semitones, with ties going to the
   flat-side spelling. Examples:
   - E7 in A minor gives A harmonic minor (G→G#).
   - D7 in C major replaces F with F# and keeps G.
   - B7 in C major replaces D and F with D# and F# and keeps G, the third of
     the Em it resolves to.
   - A major in C replaces C with C#.

   Every other degree is kept. Over diatonic chords the pool is just the
   scale. The spelling is a heuristic for chromatic chords without a clear
   function, not a claim of a uniquely correct scale.
2. **Register.** Candidates are the pool's pitches in
   `[pitch_low=62, pitch_high=79]` (D4–G5). Over each chord they are also
   kept at or above `chord_floors[i]`, a minor third above the voicing's
   highest identity tone (for an open D chord, F#4 + 3 = A4). The floor
   yields to musical necessity: it is dropped for a bar if it would leave
   fewer than 5 candidates, no pitch within an octave of the previous note,
   or, in the last bar, no reachable cadence note.
3. **Final target.** In the last bar the final note is fixed in advance: the
   tonic if the final chord contains it, otherwise the chord root, nearest
   the previous pitch. That is always within an octave, because the window
   spans more than an octave and the floor yields when needed. Every note of
   the bar must also stay within an octave of the target, so no move,
   including the forced final one, exceeds 12 semitones.
4. **Score** each candidate with `score_pitch` (weights are module constants):
   - interval size (steps preferred; repeats, wide leaps and the tritone
     penalised; more than an octave excluded)
   - chord fit by metric weight (chord tones strongly preferred on beats 1
     and 3 and on long notes; quarter-or-longer notes a semitone off a chord
     tone discouraged)
   - leap recovery (an opposite-direction step after a leap of a fourth or
     more)
   - **leading-tone resolution** (the leading tone steps up to the tonic
     when the harmony moves to a tonic-containing chord)
   - motif echo (same direction and size as the echoed bar's interval)
   - register gravity toward `tessitura_center=70`
   - a pull toward the final target in the last bar
   - a penalty on non-chord tones that end a phrase
5. **Sample** from the softmax of the scores with the seeded generator.

Velocity: 80 on beat 1, 75 on beat 3, 70 elsewhere, and 80 on the final note.
Each note sounds for 95% of its slot.

`notes_per_bar=N` (an integer) keeps the legacy uniform grid of N equal
notes per bar, still with the scorer and the final-note target. `compose()`
always uses the phrase planner.

### 5.3 Arrangement

[compose/arrangement.py](../src/music_decoder/compose/arrangement.py)
`build_midi` writes a two-track `pretty_midi.PrettyMIDI`:

- **Melody track:** program `25` (Acoustic Steel Guitar), one
  `pretty_midi.Note` per generated `Note`.
- **Accompaniment track:** same program. It is built by
  `_accompaniment_notes(..., melody=melody)` on a straight 4/4 grid:
  - `arpeggio`: 8 eighths per bar. The bass goes on the downbeat, then seven
    notes cycle up and back down the treble strings.
  - `strum`: strokes on eighths 0, 2, 3, 5, 6, 7 ("D · D U · U D U"). Down
    strokes play the voicing low→high with 8 ms between strings. Up strokes
    play the top 3 strings high→low. Each stroke rings until the next one.
  - `fingerstyle`: alternating bass. The root goes on beat 1 and the
    second-lowest voicing note on beat 3 (when the voicing has ≥ 4 notes).
    The treble pinch plays on beats 2 and 4. It leaves out the
    alternate-bass string, unless that string is the chord's only source
    of an essential tone (for example the third of open C), in which
    case it stays in the pinch.
  - **Final bar (every style):** one down stroke (10 ms per string) that
    rings to the end of the bar.
- **Melody-aware thinning** (`_thin_below`): at each attack, voicing pitches
  that come within a minor third of the lowest melody note sounding over
  that attack are dropped. Two rules override this:
  - The bass is always kept.
  - Each **essential** tone (third, seventh, altered fifth or suspended
    fourth, from `harmony.essential_pcs`) is put back at its lowest voicing
    pitch unless a kept note has it, or the melody note still sounding at
    the end of the attack's window does. A passing melody note earlier in
    the window does not count, because the accompaniment rings on under the
    notes that follow. For the final ringing chord that window is the whole
    bar, so the harmony under the held cadence note is always complete.

  So the accompaniment never drops an identity tone that the chord's shape
  provides. Combined with the melody floors in §5.2, the tune stays on top
  of almost every attack. Where a floor has to yield, a restored identity
  tone can briefly meet a passing melody note; identity takes priority over
  clearance. (A few canonical shapes are
  themselves imperfect. The C#dim shape has no G and adds A#. These shape
  defects are reported separately as S1 and left for a future voicing
  change.)

Accompaniment pitches come from `_voiced_pitches(voicing, tuning) =
sorted(open_pitches[s] + fret for fretted strings)`.

### 5.4 ASCII tab for the composition

`arrange()` fingers the melody with the A\* search that `analyze()` also uses
(`music_decoder.tabs.assigner.assign_tab`). The weights and fret cap are
pinned in `compose/api.py` rather than read from YAML. A note A\* cannot
place falls back to its lowest-fret position, so no note is lost.

[compose/tabtext.py](../src/music_decoder/compose/tabtext.py)
`render_composition_tab` prints two labelled blocks:

- **`Melody`:** one column per note, in playing order, with bars separated
  by `|` and four bars per line. No note can overwrite another.
- **`Chords`:** each chord's shape under its name (`x` = muted string).

String labels follow the tuning (`E A D G B e`, `D A D G A d`, …). Rhythm is
not drawn; the MIDI file is the source of truth for timing. (The shared
`tabs.render.render_ascii_tab` used by `analyze` is unchanged.)

### 5.4.1 Musicality metrics and evaluation

[compose/metrics.py](../src/music_decoder/compose/metrics.py) holds pure,
typed metric functions:

- melody: strong-beat chord tones, step / repeat / large-leap ratios, pooled
  leap recovery, ambitus, final resolution and length, distinct durations
  and IOI entropy, motif echo coverage and contour agreement, phrase-end
  length, exposed chromatic clashes, leading-tone resolution
- accompaniment: grid alignment, melody above the accompaniment and exposed
  semitone clashes (both judged over each melody note's whole sounding
  interval), final-chord ring, essential-tone coverage per bar, and cadence
  identity (the full harmony under the held final note)
- tab: hand-position shifts
- voicings: nearest-pitch proximity (a proximity guard, not a voice-leading
  score) and shape defects (essential tones missing from, or foreign tones in,
  each canonical shape)

[evaluation/compose_musicality.py](../src/music_decoder/evaluation/compose_musicality.py)
defines two held-out input matrices (seeds 1000+ were never used for
tuning):

- a 400-melody validation matrix: 8 keys, both modes, 10 progressions, 1–2
  bars per chord, × 3 styles
- a 110-melody edge matrix: 3/5/7/9 bars, a single chord, secondary
  dominants including B7→Em in C, a borrowed chord, final suspended chords

It summarises every metric overall and per mode and bar length.
`scripts/compose_musicality_report.py` prints that summary.
`tests/fixtures/compose/musicality_baseline.json` is the same summary for
the pre-change composer, measured on identical inputs.
[tests/unit/test_compose_musicality.py](../tests/unit/test_compose_musicality.py)
asserts two things:

- absolute targets
- that every subgroup improves on that baseline

The definitions, numbers and reasons are in
[the design spec](superpowers/specs/2026-11-06-compose-improvements-design.md).

### 5.5 Synthesis

[synth/render_wav](../src/music_decoder/synth/__init__.py) wraps
[synth/fluidsynth_wrapper.py](../src/music_decoder/synth/fluidsynth_wrapper.py)
`synthesize_midi_to_wav` with auto-selection: it picks
**`SynthBackend.FLUIDSYNTH`** when the `fluidsynth` Python module is
importable and a soundfont resolves — first
`RuntimeConfig.fluidsynth_soundfont` (relative paths against `data_dir`,
then `tests/fixtures/synthetic/soundfont/`), else pretty_midi's bundled
`TimGM6mb.sf2` ([assets.bundled_soundfont](../src/music_decoder/assets.py)),
so a fresh install needs no soundfont download. Otherwise it logs a single
warning line and falls back to `SynthBackend.SINE`
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

The adapters read this YAML at call time, so editing it changes the next
`analyze()` without a code change. `compose()` reads none of it: the composer's
parameters, including its A* tab weights, are pinned in code and versioned by
`metadata["composer_version"]`. Set `MUSIC_DECODER_HYPERPARAMETERS` to
point at a different file (the benchmark uses this to compare
configurations). Keys the public pipeline honours:

- `key_detection.backend` — `cnn` | `profile` ([key/\_\_init\_\_.py](../src/music_decoder/key/__init__.py)).
- `beat_tracking.backend` — `madmom` | `librosa` ([dsp/\_\_init\_\_.py](../src/music_decoder/dsp/__init__.py)).
- `chord_detection.backend` — `madmom_cnn` | `madmom_deep_chroma` | `template_hmm`
  ([chords/\_\_init\_\_.py](../src/music_decoder/chords/__init__.py)).
- every `basic_pitch` key, including `melodia_trick`, and
  `post_processing.enabled` ([transcription/\_\_init\_\_.py](../src/music_decoder/transcription/__init__.py)).
- `tab_assignment.weights` / `max_fret` ([tabs/\_\_init\_\_.py](../src/music_decoder/tabs/__init__.py)).

The v3 values were chosen on the GuitarSet dev split (§11.5) and each
carries a comment with the measurement behind it. Still hard-coded: the
chord-detection threshold / self-transition probability for `template_hmm`
and the 50 ms chord onset window in
[tabs/assigner.py](../src/music_decoder/tabs/assigner.py).

### 6.3 `config/eval_thresholds.yaml`

Three numbers consumed only by
[tests/integration/test_regression.py](../tests/integration/test_regression.py):

- `chord_recognition_score: 0.60`
- `key_mirex_score: 0.75`
- `tab_string_accuracy: 0.55`

---

## 7. CLI

[cli/main.py](../src/music_decoder/cli/main.py) — Click group `main` with
five subcommands:

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
- `doctor` — checks `ffmpeg` on `PATH`, importable `fluidsynth` Python
  module, and that a soundfont resolves (prints whether it is the
  installed or the bundled one); reports whether the Demucs weights are
  cached (informational — not a failure). Exits non-zero on any fail.
- `fetch-models [--no-demucs] [--no-soundfont]` — idempotently
  pre-downloads the Demucs `htdemucs_6s` weights (~52 MB, torch hub cache)
  and the GeneralUser GS soundfont (~30 MB, to the configured
  `fluidsynth_soundfont` path) via [assets.py](../src/music_decoder/assets.py).
  Downloads go to a `.part` file and are validated (SF2 `RIFF…sfbk`
  header) before being moved into place. `setup.sh` runs it by default
  (`--no-models` / `--no-soundfont` opt out).

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
  underlying call site otherwise. A missing configured soundfont is not
  an error: synthesis uses pretty_midi's bundled one.
- **Demucs weights not cached** → downloaded on the first full-mix
  analysis (or ahead of time by `music-decoder fetch-models`).

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
  `g_major_chord.mid`). The first run renders them to `.wav` with
  fluidsynth using `tests/fixtures/synthetic/soundfont/*.sf2` if present,
  else pretty_midi's bundled `TimGM6mb.sf2`; sine synthesis only if no
  soundfont exists at all. Render output is cached on disk
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

basic-pitch and madmom ship their weights in their wheels; only Demucs
downloads (on first full-mix run, or via `music-decoder fetch-models`).

---

### 11.5 GuitarSet accuracy benchmark

[scripts/benchmark_guitarset.py](../scripts/benchmark_guitarset.py) scores
the real pipeline stages on GuitarSet (360 annotated solo-guitar
recordings; expected at `tests/fixtures/guitarset/`, git-ignored; loader in
[evaluation/guitarset.py](../src/music_decoder/evaluation/guitarset.py)).
Dev split = players 00–04 (300 tracks, used for every tuning decision);
test split = player 05 (60 tracks, reported only). Audio = `audio_mono-mic`.

| stage | metric |
|---|---|
| `key` | MIREX weighted key score |
| `beats` | `mir_eval` beat F-measure (±70 ms, first 5 s trimmed) |
| `chords` | MIREX `majmin` duration-weighted recall vs lead-sheet chords |
| `notes` | onset-only note P / R / F (50 ms, ±50 cents) |
| `tab_gt` | string accuracy of `assign_tabs` on ground-truth notes |
| `tab_e2e` | string accuracy of `assign_tabs` on transcribed notes |

```bash
python scripts/benchmark_guitarset.py --split dev --stages key,chords --by-style
python scripts/benchmark_guitarset.py --split test --json out/bench/test.json
python scripts/benchmark_guitarset.py --split dev --sweep-notes   # basic-pitch decoding
python scripts/benchmark_guitarset.py --split dev --sweep-tabs    # A* weights
```

basic-pitch's raw network output is cached in `out/bench-cache/bp/`, so
decoding sweeps don't re-run the network. Results and decisions:
[docs/reports/2026-10-08-guitarset-benchmark.md](reports/2026-10-08-guitarset-benchmark.md).

## 12. Distribution

- Build backend: `hatchling >= 1.21` (declared in
  [pyproject.toml](../pyproject.toml)).
- Console entry point: `music-decoder = "music_decoder.cli.main:main"`.
- Python: `>= 3.11, < 3.13`.
- Packaging surface: `src/music_decoder/` (single namespace).
- Install: `setup.sh` (dev clones), `pipx install ./` (end users),
  or `pip install -e ".[dev]"` in a virtualenv. There is no container
  image; the tool runs directly on the host. `setup.sh` pre-downloads the
  models; with pip/pipx, run `music-decoder fetch-models` once.

There is no SQLite, no Alembic, no Postgres dependency, and no
persistence layer. The v2 pipeline is stateless: the only on-disk
state is flat-file caching (YouTube WAVs under `<data_dir>/yt_cache/`
and composition output folders).

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
- **Tweak A* weights**: edit `tab_assignment.weights` in
  [`config/hyperparameters.yaml`](../config/hyperparameters.yaml) (re-tune
  with `scripts/benchmark_guitarset.py --sweep-tabs`). `_DEFAULT_WEIGHTS` in
  [tabs/__init__.py](../src/music_decoder/tabs/__init__.py) is only the
  fallback when the YAML can't be loaded; keep it in sync.
- **Check an accuracy change**: run the dev split of
  `scripts/benchmark_guitarset.py` before and after (§11.5); report the
  test split only once the decision is made.
- **Switch chord backend**: edit
  [`config/hyperparameters.yaml`](../config/hyperparameters.yaml)
  `chord_detection.backend` (currently `madmom_deep_chroma`; the
  alternative is `template_hmm`). The public `analyze()` adapter reads
  this at call time.
- **Use a different soundfont**: point `RuntimeConfig.fluidsynth_soundfont`
  at any `.sf2` (or `python scripts/download_soundfont.py --url … --dest …`).
  [synth/__init__.py](../src/music_decoder/synth/__init__.py)
  `render_wav` uses fluidsynth whenever `pyfluidsynth` imports (falling
  back to the bundled TimGM6mb when no configured soundfont exists) and
  the sine synth otherwise. Callers can force a specific
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
