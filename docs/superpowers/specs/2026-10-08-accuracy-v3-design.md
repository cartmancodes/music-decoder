# Accuracy v3 — evidence-driven upgrade of the analyze() pipeline

**Date:** 2026-10-08
**Status:** approved under `/goal` directive (autonomous session — the user asked
for the work to proceed without pausing; decisions below are recorded for
review rather than gated on chat approval).

## 1. Intent

**What the user asked for:** "research online and update our music decoder
logic/code as per best practices to achieve higher accuracy."

**Understanding:** Raise the measured accuracy of `analyze()` — key, chord
progression, note transcription, tablature, and beat grid — using techniques
that are established best practice in MIR as of 2026, without breaking the
frozen public contract (`tests/unit/test_public_contract.py`).

**Assumptions (not stated by the user):**

- Accuracy is judged on real recordings, not the two sine-synth fixtures.
  GuitarSet (360 annotated real guitar recordings, already on disk at
  `tests/fixtures/guitarset/`, git-ignored) is the benchmark.
- New dependencies are acceptable only if already installed or trivially
  pip-installable; madmom (already a dependency) is preferred over new heavy
  models.
- CPU-only Mac runtime must stay practical (seconds, not minutes, per clip
  beyond what basic-pitch/Demucs already cost).

**Success criteria:** on the held-out GuitarSet test fold (player `05`, 60
tracks, never used for tuning), every stage scores ≥ its v2 baseline and the
stages we change score strictly higher. Numbers are recorded in
`docs/reports/2026-10-08-guitarset-benchmark.md`.

## 2. Research summary (best practices that apply here)

| Stage | v2 today | Best practice found | Source |
|---|---|---|---|
| Evaluation | Two sine-synth fixtures; custom metrics | Benchmark on real annotated data with `mir_eval` standard metrics (key weighted score, chord `majmin`, onset-only note F at 50 ms, beat F at 70 ms); tune on a dev split, report on a held-out fold | mir_eval docs; GAPS (Riley et al., ISMIR 2024) protocol |
| Key | Krumhansl/Temperley correlation on global mean chroma | CNN key classifier (Korzeniowski & Widmer, ISMIR 2018, "genre-agnostic key classification"), shipped in madmom as `CNNKeyRecognitionProcessor` | arXiv 1808.05340; madmom docs |
| Beats | `librosa.beat.beat_track` (onset envelope + DP), downbeats = `beats[::4]` | RNN activation + DBN decoding (`RNNDownBeatProcessor` + `DBNDownBeatTrackingProcessor`) gives beats *and* real downbeats/meter; Beat This! (ISMIR 2024) is newer SOTA but adds a new torch package | madmom docs; Foscarin et al. 2024 |
| Chords | madmom DeepChroma + CRF | madmom also ships the fully-convolutional CNN feature + CRF model (Korzeniowski & Widmer 2016, arXiv 1612.05082), reported stronger than deep-chroma; pick by measurement | madmom docs |
| Notes | basic-pitch with library defaults (onset 0.5, frame 0.3) | Basic Pitch on GuitarSet is recall-heavy (P 54.6 / R 85.0 / F 66.1 per GAPS paper) → tune thresholds with a coarse-to-fine sweep (frame threshold first, then onset), apply note post-processing | GAPS paper Table 3; arXiv 2509.12712 tuning procedure |
| Tabs | A* with asymmetric move cost (moving *down* the neck is free) | Symmetric hand-movement cost (Burlet & Fujinaga A*-Guitar); weights fit on real fingerings (GuitarSet hexaphonic string labels) | RESEARCH.md; arXiv 2510.10619 |

**Considered and deferred (recorded, not built):**

- *MuScriptor* (open-weight multi-instrument transformer, ISMIR 2026) and the
  *GAPS / Riley high-resolution guitar* models (~0.86–0.88 onset F on
  GuitarSet): best published numbers, but heavy new dependencies, weights
  availability unclear (GAPS), and too new to validate on macOS CPU here.
  Revisit as a pluggable transcription backend.
- *Beat This!* — MIT-licensed SOTA beat tracker; deferred because madmom's
  DBN is already installed. Easy follow-up if the benchmark shows beats are
  the weak link.
- *Fretting-Transformer / MIDI-to-Tab* — no released checkpoints usable here.

## 3. Approaches considered

1. **Measure-then-tune (chosen).** Build a GuitarSet benchmark harness first,
   then make each stage change behind a config switch and keep it only if the
   dev split improves; confirm on the held-out fold. Low risk, every claim
   backed by a number, no new dependencies.
2. **Model swap.** Replace basic-pitch / chords / beats with the newest
   published models. Highest ceiling, but new heavy deps, unverifiable on this
   machine, and no way to show it helped without approach 1's harness anyway.
3. **Parameter tweaks from literature only.** Cheap, but unmeasured — the
   previous roadmap (May 2026) proposed sweeps that never landed for exactly
   this reason.

## 4. Design

### 4.1 Benchmark harness (new)

- `src/music_decoder/evaluation/guitarset.py` — dependency-free JAMS reader
  (JAMS is JSON). `GuitarSetTrack` exposes: `track_id`, `player`, `audio_path`
  (default `audio_mono-mic`), `notes` (onset, offset, midi pitch, string
  index 0=low E), `chords` (instructed leadsheet labels, Harte syntax),
  `key` (tonic, mode), `beats`, `downbeats`. `iter_tracks(root, players=...)`
  and `split(...)` → dev = players 00–04, test = player 05.
- `src/music_decoder/evaluation/metrics.py` — add standard-metric helpers:
  `key_weighted_score` (mir_eval.key), `chord_majmin_score` (mir_eval.chord,
  duration-weighted), `note_onset_f` (sorted intervals, 50 ms, onset-only),
  `beat_f_measure` (mir_eval.beat). Existing functions stay unchanged.
- `scripts/benchmark_guitarset.py` — runs selected stages on a split, caches
  expensive per-track outputs under `out/bench-cache/` (basic-pitch raw model
  output as `.npz`, so threshold sweeps re-decode without re-running the
  network), prints a table, and writes JSON. `--sweep-notes` performs the
  coarse-to-fine threshold search on dev. `--stage` selects key / beats /
  chords / notes / tabs.

### 4.2 Stage changes (each behind `config/hyperparameters.yaml`)

- **Key:** new `key_detection.backend: cnn | profile`. `cnn` runs madmom's
  `CNNKeyRecognitionProcessor` on a temp WAV and converts the 24-way softmax
  to `KeyEstimate(profile="cnn", correlation=max prob, margin=top1-top2)`.
  Falls back to the existing profile consensus on any madmom failure.
  `estimate_key(chroma, *, samples=None, sr=None)` gains keyword-only args;
  `analyze()` passes them. `KeyEstimate.profile` literal widened to include
  `"cnn"`.
- **Beats:** new `beat_tracking.backend: madmom | librosa`. `madmom` uses
  `RNNDownBeatProcessor` + `DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4])`
  → beat times, true downbeats, numerator, `ts_assumed=False`. Tempo = 60 /
  median IBI. Librosa path unchanged and remains the fallback.
- **Chords:** new backend value `madmom_cnn` (CNN features + CRF). Default
  chosen by dev-split `majmin`. Existing backends untouched.
- **Notes:** thresholds (`onset`, `frame`, `minimum_note_length_ms`,
  `melodia_trick`) read from YAML (closing the hard-coding gap noted in the
  technical doc §6.2) and set to the dev-split optimum. Post-processing
  (`drop_short_notes`, `merge_same_pitch`) wired in only if it improves dev F.
- **Tabs:** `transition_cost` move term becomes symmetric
  `w_move * |Δfret|` (bug: moving down the neck currently costs nothing).
  Weights read from YAML `tab_assignment.weights`; re-tuned with a small grid
  on dev using ground-truth notes (isolates the assigner from transcription
  errors). The open-string bonus term is unchanged.
- **Shared helper:** `dsp/tempwav.py` `temp_wav(samples, sr)` context manager,
  reused by the chord, key, and beat madmom paths (replaces the inline copy in
  `chords/__init__.py`).

### 4.3 Config

`config/hyperparameters.yaml` id → `2026-10-08-v3-guitarset-tuned`; new keys
`key_detection.backend`, `beat_tracking.backend`, `basic_pitch.melodia_trick`.
`HyperparameterSet` dataclasses gain matching optional fields with defaults so
older YAML files still load.

### 4.4 Error handling

Every new madmom path is wrapped: on import/inference failure it logs one
structured warning and falls back to the v2 implementation, so `analyze()`
never fails because of an optional model.

### 4.5 Testing

- Unit tests (TDD) for: JAMS parsing on a tiny inline JAMS dict; new metric
  helpers against hand-computed values; symmetric move cost; key CNN output
  conversion (mocked processor); madmom beat conversion (mocked processor);
  YAML loading of new keys with defaults; fallback paths when madmom raises.
- Integration: existing `tests/integration` suites must still pass.
- Benchmark: dev numbers drive defaults; test-fold numbers are reported once
  at the end.

### 4.6 Out of scope

New transcription models, full-mix (Demucs) benchmarking (no annotated
full-mix guitar data on disk), UI changes, extended chord qualities beyond
what backends emit.
