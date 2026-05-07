# Chord Recognition + Complete Tablature — Design

**Status:** draft, awaiting review
**Date:** 2026-05-05
**Owner:** shubhchak@gmail.com
**Builds on:** [2026-05-04-music-decoder-design.md](2026-05-04-music-decoder-design.md)

Adds two user-visible capabilities to music-decoder v1:

1. **Chord recognition** — detect a sequence of chord segments (`Am - F - C - G`) aligned to the beat grid, persisted alongside the existing per-note tablature.
2. **Complete tablature view** — chord labels rendered above the ASCII tab, organised by measure; a grid of chord-diagram SVGs showing canonical fingerings; a dedicated "Chord Progression" tab in the Streamlit UI.

The chord-recognition stage is the new core; everything else hangs off it.

---

## 1. Goals and non-goals

### Goals

- Produce a time-ordered list of chord segments from any successfully-loaded audio.
- Cover the chord vocabulary used in ~95% of popular music: 12 roots × 4 qualities (`maj`, `min`, `7`, `maj7`) = 48 chords plus a `N` ("no chord") state.
- Persist segments in SQLite alongside the existing transcription artefacts; expose them through the Streamlit UI.
- Use the existing chroma + beat tracking outputs without recomputing them.
- Add a regression metric (chord recognition score) and a baseline; gate future changes against it.

### Non-goals (this feature, v1)

- Chord inversions / slash chords (`C/E`, `G/B`).
- Extended qualities: `sus2`, `sus4`, `dim`, `aug`, `9`, `add9`, etc.
- Chord-aware fingering — A* tab assignment does not consult the detected chord. (Deferred; the tab assigner stays as-is.)
- Editing / overriding detected chords in the UI.
- Chord-progression-driven harmonic analysis (Roman numerals, modal interchange).

---

## 2. Algorithm

Beat-synchronous template matching with HMM smoothing — the standard academic baseline (Cho et al.; Mauch & Dixon).

### 2.1 Chromagram (reused)

The pipeline already computes `compute_chroma_with_hpss(samples, sr, hpss_margin)` for `detect_key`. The orchestrator hoists that computation upstream so both `detect_key` and the new `detect_chords` consume the same `(12, T)` matrix.

### 2.2 Beat-synchronous chroma

For each pair of adjacent beats `(b_i, b_{i+1})` from `BeatGrid.beat_times_s`, average the chroma columns whose center timestamp falls inside `[b_i, b_{i+1})`. Output: a `(12, num_beats - 1)` matrix `C_beat`.

If `len(beat_times_s) < 4` the stage is skipped with `skipped_reason="degenerate_beat_grid"`.

### 2.3 Templates

48 binary templates, one per `(root, quality)`:

| Quality | Pattern (relative to root, semitones) |
|---|---|
| `maj`   | `{0, 4, 7}` |
| `min`   | `{0, 3, 7}` |
| `7`     | `{0, 4, 7, 10}` |
| `maj7`  | `{0, 4, 7, 11}` |

Plus a `N` (no-chord) template — uniform `1/12` across all pitch classes.

Each template is a length-12 vector `t` with 1.0 at active pitch classes and 0.0 elsewhere; templates and beat-chroma columns are L2-normalised before scoring.

### 2.4 Per-beat scoring

For each beat column `c_i`, compute `s_{i,k} = cos(c_i, t_k)` for each of the 49 templates (48 chords + N). The argmax `k* = argmax_k s_{i,k}` is the noisy per-beat prediction.

If `max_k s_{i,k} < no_chord_threshold` (default 0.15), force the prediction to `N`. This catches silent/percussion-dominated beats.

### 2.5 HMM smoothing

A single-state-per-chord HMM with 49 states. Initial distribution: uniform. Emission probability of state `k` at beat `i`: `s_{i,k}` (re-normalised to sum to 1 across `k`). Transition matrix `A`:

```
A[k][k]   = p_self                        # default 0.7
A[k][k']  = (1 - p_self) / 48             # for any k' ≠ k
```

Decoding: standard Viterbi (work in log-space, add `1e-12` to emissions before log to avoid `-inf`). If Viterbi fails for any reason (numerical underflow despite log-space, empty input), fall back to the per-beat argmax sequence and log a warning.

### 2.6 Segment merging

Walk the smoothed sequence; collapse runs of consecutive identical chords into single `ChordSegment(start_s, end_s, root, quality, confidence)` records. `confidence` = mean of the per-beat similarities for the merged beats.

Drop segments shorter than `min_segment_duration_s` (default 0.25 s) by absorbing them into the preceding segment.

---

## 3. Module structure

```
src/music_decoder/chord_detection/
  __init__.py
  templates.py     # 48 chord templates, (root, quality) ↔ index helpers, ROOTS, QUALITIES
  recognize.py     # beat-sync chroma → per-beat scores → Viterbi → segments
  api.py           # public detect_chords(chroma, beat_grid, params) -> ChordRecognitionResult

src/music_decoder/ui/components/chord_progression.py
                   # render_chord_diagram_svg, render_chord_progression_text,
                   # render_chord_labels_above_tab
```

`recognize.py` is the only module with non-trivial logic; the others are thin.

---

## 4. Data model

### 4.1 Pipeline contracts

Added to `pipeline/contracts.py`:

```python
@dataclass(frozen=True)
class ChordSegment:
    start_s: float
    end_s: float
    root: str            # "C", "C#", "D", ..., "B", or "N"
    quality: str         # "maj", "min", "7", "maj7", or ""  (empty for "N")
    confidence: float    # mean cosine similarity over the segment

@dataclass(frozen=True)
class ChordRecognitionResult:
    segments: list[ChordSegment]
    median_confidence: float
    skipped_reason: str | None
```

### 4.2 SQLite schema (new Alembic migration `0002_chord_segments.py`)

```sql
CREATE TABLE chord_segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    start_s REAL NOT NULL,
    end_s REAL NOT NULL,
    root TEXT NOT NULL,
    quality TEXT NOT NULL,
    confidence REAL NOT NULL
);
CREATE INDEX chord_segments_job_idx ON chord_segments(job_id, start_s);
```

Plus a matching `ChordSegment` SQLAlchemy model in `persistence/models.py` and a `ChordSegmentRepo` in `persistence/repositories.py` with `bulk_insert(job_id, rows)` and `list_for_job(job_id)`.

### 4.3 Ground truth

`GroundTruth` dataclass in `evaluation/fixtures/base.py` gains an optional field:

```python
chord_segments: list[tuple[float, float, str, str]] | None = None
# (start_s, end_s, root, quality)
```

GuitarSet's `track.chords` annotation populates this when present (parsed from JAMS via mirdata). The synthetic and manual fixture loaders leave it `None` (synthetic clips are short/non-harmonic; manual fixtures can opt in by adding a `"chords"` array to the JSON drop-in schema — that's a pure additive change to the fixture format).

---

## 5. Pipeline integration

The orchestrator gains one new stage between `beat_tracking` and `post_processing`:

```
audio_io → separation → transcription → key_detection → beat_tracking
       → chord_detection (new)        → post_processing → tab_assignment
       → midi_synth → visualizations
```

`chord_detection` consumes:
- The chroma matrix from `key_detection` (hoisted into a local variable in the orchestrator so it's computed once and passed to both stages — minor refactor of existing code).
- The `BeatGrid` from `beat_tracking`.
- `hyperparameters.chord_detection` (new section).

It emits a `StageEvent` with `summary = {"segments": N, "median_conf": ...}` or `summary = {"skipped_reason": "degenerate_beat_grid"}` when the stage is skipped.

Persistence: `ChordSegmentRepo.bulk_insert(job_id, rows)` after the stage emits.

---

## 6. UI integration

### 6.1 New tab: Chord Progression

In `ui/pages/03_Results.py`, the existing tab strip `[Summary, Visualization, Tablature, Playback, Reference, Diagnostics]` becomes `[Summary, Visualization, Tablature, Chord Progression, Playback, Reference, Diagnostics]` — Chord Progression slots between Tablature and Playback.

Tab content:
- Time-ordered chord sequence rendered as text: `| Am | F | C | G | (etc) |` with start timestamps for each segment.
- A grid of chord-diagram SVGs (one per unique chord that appears in the song), labeled with chord name + how many times it appears.
- Per-segment confidence colour-coded via the existing `confidence_color(value, high, medium)` helper.
- Banner when `chord_recognition_result.skipped_reason` is set: "Chord progression unavailable — beat grid too short."

### 6.2 Tablature tab — chord labels above tab

The existing ASCII tab renderer in `ui/components/tablature.py` becomes a richer "labelled tab":

- A new helper `render_chord_labels_above_tab(tabbed_notes, chord_segments) -> str` produces a header row whose chord labels are positioned above the column where each chord change occurs.
- The Tablature tab calls both `render_ascii_tab` and the new chord-label header, joining them with a newline.

### 6.3 Diagnostics tab — chord stage stats

Add one row: `chord_detection: median_conf=0.71, n_segments=42`.

### 6.4 New component module

`ui/components/chord_progression.py` exposes:

- `render_chord_diagram_svg(root: str, quality: str) -> str` — emits a small SVG fretboard box (similar to the existing `render_svg_fretboard` but constrained to ~120×100 px and rendering a single canonical voicing). The voicing comes from a static lookup table of 48 standard fingerings (open-position chords for common keys + barre voicings for the rest). The 48 voicings are committed in `chord_detection/voicings.py` as a constant dict — no UI logic needed.
- `render_chord_progression_text(segments: list[ChordSegment]) -> str` — pipe-delimited textual summary.
- `render_chord_labels_above_tab(tabbed_notes, segments) -> str` — described above.

---

## 7. Accuracy harness

### 7.1 New metric

`chord_recognition_score(predicted: list[ChordSegment], truth: list[tuple[float,float,str,str]]) -> float`:

- Build a 10 ms frame grid spanning the union of predicted and truth time spans.
- For each frame, score 1.0 if predicted `(root, quality)` matches truth exactly; 0.5 if root matches but quality differs; 0.0 otherwise.
- Return the time-weighted mean (equivalent to the MIREX MajMin score for our 4-quality vocabulary).

The function lives in `evaluation/metrics.py` next to the existing metric wrappers. Pure function; no mir_eval dependency (mir_eval's chord submodule has its own vocabulary mapping that doesn't quite line up with our 4-quality choice — implementing the score directly is simpler than translating).

### 7.2 Regression test

Add to `tests/regression/test_accuracy_thresholds.py`:

- `_real_pipeline` returns `chord_segments` in its prediction dict.
- `run_evaluation` invokes `chord_recognition_score` when both `prediction["chord_segments"]` and `gt.chord_segments` are present.
- New aggregate metric in `EvaluationReport` and `FixtureMetrics`.
- Threshold in `config/eval_thresholds.yaml`: `chord_recognition_score: 0.40` (defensible v1 floor; RESEARCH.md baselines for chord recognition on raw polyphonic guitar typically land 50–70%).

After the first green regression run, capture the achieved value into `evaluation_reports/baseline.json`.

### 7.3 Unit tests

- 48 templates have correct binary patterns (e.g., `template("C", "maj") == (1,0,0,0,1,0,0,1,0,0,0,0)`).
- Pure C-major triad chroma vector argmaxes to `("C", "maj")`.
- Per-beat argmax that flickers between `("C", "maj")` and `("F", "maj")` smooths to a single segment under HMM.
- Three identical-chord beats followed by two different beats merges to two segments.
- Segments shorter than `min_segment_duration_s` get absorbed.
- `chord_recognition_score` returns 1.0 for identical predicted + truth; 0.5 for root-match-quality-mismatch (e.g., predict `("C","maj")`, truth `("C","7")`).

---

## 8. Failure modes

| Failure | Detection | Handling |
|---|---|---|
| Beat grid `< 4` entries | `len(beat_times_s) < 4` | Skip stage; `skipped_reason="degenerate_beat_grid"`; no rows persisted; UI banner |
| Chroma all zeros (silence after HPSS) | `chroma.sum(axis=0).max() == 0` | All beats emit `N`; UI shows a single `N` segment |
| All max similarities below `no_chord_threshold` | per-beat check | All beats labeled `N` |
| Audio shorter than 1 beat after sync | `C_beat.shape[1] == 0` | Skip stage with `skipped_reason="audio_too_short_for_chord_window"` |
| HMM Viterbi underflow | exception | Fall back to per-beat argmax; log warning; still produce segments |
| GuitarSet chord annotation missing | `track.chords is None` | Set `gt.chord_segments=None`; metric returns null for that fixture |
| User-toggled `use_demucs=False` on full mix | upstream skip | Chord recognition still runs; quality may degrade — UI flags `median_conf` if low |

---

## 9. Configuration

Add to `config/hyperparameters.yaml`:

```yaml
chord_detection:
  qualities: [maj, min, "7", maj7]
  hmm_self_transition_prob: 0.7
  no_chord_threshold: 0.15
  min_segment_duration_s: 0.25
```

Mirrored in `HyperparameterSet` with a new `ChordDetectionParams` dataclass in `config/hyperparameters.py`.

---

## 10. Implementation ordering (for the writing-plans phase)

Anticipated task ordering — kept as a sketch here; the writing-plans skill will produce the canonical decomposed plan:

1. `ChordDetectionParams` + YAML loader extension.
2. Templates module + unit tests (binary patterns, normalisation).
3. Beat-sync function + unit test.
4. Per-beat scorer + HMM Viterbi + unit tests (flickering and merging).
5. Public `detect_chords` API + unit test on synthetic chord-progression chroma.
6. `ChordSegment` dataclass + Alembic migration + SQLAlchemy model + repo + repo tests.
7. Orchestrator: hoist chroma, add `chord_detection` stage, persist segments. Integration test.
8. Voicings table for 48 chord diagrams.
9. Chord-diagram SVG renderer + unit test (assert SVG opens/closes, contains expected fret marks for `("C", "maj")`).
10. Chord progression text renderer + unit test.
11. Chord labels above tab renderer + unit test.
12. UI: extend Results page with new "Chord Progression" tab; update Tablature tab to call the new label helper.
13. `chord_recognition_score` metric + unit tests.
14. GuitarSet loader: parse chord annotations; populate `gt.chord_segments`.
15. Regression test: extend `_real_pipeline` to return `chord_segments`; add threshold to YAML; capture baseline.
16. End-to-end integration test: run the worker daemon on the synthetic fixture and assert at least one chord segment row was persisted.

---

## 11. Open questions

These are documented but deliberately unresolved in v1; tracked as future work:

- **Chord-aware tab assignment.** Letting A* prefer fingerings that match the detected chord shape (e.g., bias toward standard chord-grip positions when transcribing strumming) is a meaningful accuracy win but a non-trivial change to the cost function and a separate accuracy story. Out of v1 scope.
- **Inversions and slash chords.** Adding 3 inversions per quality bumps templates from 48 to ~144 and complicates chord-diagram lookup. Possible v2.
- **Section detection.** Recognising verse/chorus repetition from the chord progression (and rendering the tab with section labels) is interesting but its own feature.
- **Editable chords in UI.** Letting users override a detected chord and re-render. Useful for polishing exports.

---

## 12. References

- Cho, T., & Bello, J.P. (2014). On the relative importance of individual components of chord recognition systems.
- Mauch, M., & Dixon, S. (2010). Approximate Note Transcription for the Improved Identification of Difficult Chords.
- MIREX Chord Estimation task — vocabulary and scoring conventions.
- GuitarSet: Xi et al. (2018), and the JAMS chord namespace.
