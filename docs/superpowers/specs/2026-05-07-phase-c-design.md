# Phase C — Big Bets

**Status:** in implementation
**Date:** 2026-05-07
**Builds on:** [docs/superpowers/research/2026-05-05-accuracy-improvement-roadmap.md](../research/2026-05-05-accuracy-improvement-roadmap.md)

Phase C of the accuracy roadmap: three big-effort items aimed at lifting accuracy beyond what tuning + algorithm-swap (Phases A and B) can achieve.

| Item | Status | Why now |
|---|---|---|
| **C-4 Madmom deep chord recognition** | doing | pip-installable, verified import + 0.1s/22s on bossa nova; biggest immediate ROI |
| **C-2 Manual fixture authoring tool** | doing | unblocks the user from labelling ground truth without code; cheap to build |
| **C-3 basic-pitch fine-tuning harness** | scaffold + docs | training takes hours; build the harness here, run later |
| **C-1 Fretting-Transformer** | watching | code not yet released |
| **C-5 TART end-to-end audio→tab** | watching | code not yet released |

---

## C-4. Deep chord recognition via Madmom

### Why

Current chroma_cqt + 49-state Viterbi scores **0.055** on the GuitarSet test corpus (4 of 5 fixtures: 0.0; one: 0.275). Per the roadmap, ~30% of that is real algorithmic weakness; the rest is GuitarSet's vocabulary outside our 4-quality template set. Madmom's `DeepChromaChordRecognitionProcessor` is trained on the McGill Billboard dataset, supports a wider chord vocabulary natively, and is reported to achieve 0.78–0.82 MIREX MajMin in the literature.

### Architecture

The chord-detection module gains a `Backend` protocol with two concrete implementations:

```
src/music_decoder/chord_detection/
  templates.py            # unchanged
  recognize.py            # unchanged
  voicings.py             # unchanged
  api.py                  # gains backend selection
  backends/
    __init__.py
    base.py               # ChordBackend protocol
    template_hmm.py       # existing logic (extracted from api.py)
    madmom_deep_chroma.py # new
  madmom_compat.py        # one-time monkey-patch shim
```

**`ChordBackend` protocol:**

```python
class ChordBackend(Protocol):
    def detect(
        self,
        *,
        chroma: np.ndarray,
        sr: int,
        hop_length: int,
        beat_grid: BeatGrid,
        params: ChordDetectionParams,
        audio_path: Path | None,    # madmom needs raw audio path
    ) -> ChordRecognitionResult:
        ...
```

The `audio_path` is optional because `template_hmm` doesn't need it — it operates on the chroma alone — but `madmom_deep_chroma` reads the audio directly through its own pipeline.

**`api.detect_chords(...)` becomes a dispatcher:**

```python
def detect_chords(
    *, chroma, sr, hop_length, beat_grid, params, audio_path=None,
) -> ChordRecognitionResult:
    if params.qualities != list(QUALITIES):
        raise ValueError(...)   # unchanged config-drift guard
    if params.backend == "madmom_deep_chroma":
        return MadmomDeepChromaBackend().detect(
            chroma=chroma, sr=sr, hop_length=hop_length,
            beat_grid=beat_grid, params=params, audio_path=audio_path,
        )
    return TemplateHmmBackend().detect(
        chroma=chroma, sr=sr, hop_length=hop_length,
        beat_grid=beat_grid, params=params, audio_path=None,
    )
```

**Config:** `ChordDetectionParams` gains a `backend: Literal["template_hmm", "madmom_deep_chroma"]` field. Default: `"madmom_deep_chroma"` if available, falling back to `"template_hmm"` if madmom import fails.

### Madmom-specific details

**Compat shim.** Madmom 0.16.1 has Python 3.10+ and NumPy 1.20+ incompatibilities (`collections.MutableSequence`, `np.float`). We apply a one-time monkey-patch at import time inside `madmom_compat.py`:

```python
# madmom_compat.py — must be imported BEFORE any madmom import
import collections
import collections.abc as cabc
for name in ("MutableSequence", "MutableMapping", "Mapping",
             "Sequence", "Iterable", "Callable"):
    if not hasattr(collections, name) and hasattr(cabc, name):
        setattr(collections, name, getattr(cabc, name))
import numpy as np
for name, repl in (("float", float), ("int", int), ("bool", bool)):
    if not hasattr(np, name):
        setattr(np, name, repl)
```

Every consumer of madmom imports `madmom_compat` first. We do NOT install a patched fork — easier to maintain a small shim than diverge from upstream.

**Label parsing.** Madmom returns chord segments as `np.recarray`-like `(start, end, label)` triples. The label format is JAMS-style: `"D#:maj"`, `"F:min"`, `"C#:7"`, `"N"` for no chord. Parser maps:

| Madmom label | Our (root, quality) |
|---|---|
| `"C:maj"` | `("C", "maj")` |
| `"A:min"` | `("A", "min")` |
| `"G:7"` | `("G", "7")` |
| `"C:maj7"` | `("C", "maj7")` |
| `"N"` | `("N", "")` |
| `"D:min7"` | `("D", "min")` *(downgrade for our 4-quality vocabulary)* |
| `"F:sus4"` | `None` *(skip; falls through to nearest segment)* |

The parser is shared with the existing GuitarSet `_parse_jams_chord_label` — extract into `chord_detection/labels.py` so both call sites use the same rules.

### Failure modes

| Failure | Detection | Handling |
|---|---|---|
| madmom not installed | ImportError on backend init | log warning, fall back to template_hmm |
| Compat shim fails (newer Python deprecates collections shimming) | ImportError on madmom load | same fallback |
| Audio file path missing | `audio_path is None` | fall back to template_hmm with a logged warning |
| madmom produces 0 segments (silent audio) | empty result | return `ChordRecognitionResult(skipped_reason="no_chord_segments")` |
| madmom outputs `N` for entire audio | all segments are no-chord | preserve as-is; UI shows N |

### Tests

**Unit (`tests/unit/test_chord_detection_madmom_backend.py`):**

- `test_madmom_compat_shim_idempotent` — calling the shim twice doesn't error.
- `test_madmom_backend_emits_segments_on_real_audio` (slow) — runs on bossa nova fixture, asserts ≥4 segments and median confidence > 0.0.
- `test_madmom_backend_falls_back_when_audio_path_missing` — without audio_path, raises a clean exception or returns a skipped_reason result.
- `test_madmom_backend_label_parsing` — mock the madmom processor with a fake list of (start, end, label) tuples, assert the (root, quality) parsing for `"C:maj"`, `"A:min"`, `"G:7"`, `"D:min7"` (downgrade), `"F:sus4"` (drop).

**Regression:** Re-run the existing accuracy threshold suite with both backends. Capture a new baseline that records both backends' aggregate scores side by side. Floor for `chord_recognition_score` raised from 0.05 → at least 0.20 (the madmom number we expect).

### Acceptance criteria

- All existing tests still pass.
- `chord_recognition_score` aggregate on the GuitarSet+synthetic corpus moves from 0.055 (template_hmm baseline) to ≥ 0.25 (madmom deep chroma).
- Per-fixture: at least 3 of the 5 GuitarSet tracks score ≥ 0.20 (vs current: 4 of 5 score 0.0).
- Pipeline still runs end-to-end with `backend: template_hmm` (regression-protect the fallback).

---

## C-2. Manual fixture authoring CLI

### Why

The manual fixture format already exists (`tests/fixtures/manual/<name>.json`). The user has to hand-author the JSON. There's no validator, no template generator, no helper to convert audio + chord progression text into the canonical schema. A small CLI removes the friction.

### What it does

```bash
music-decoder fixture-init path/to/audio.wav --name "my_clip"
```

Creates `tests/fixtures/manual/my_clip.json` pre-populated with:

- `audio` field set to the relative path
- `tuning` defaulted to `"EADGBE"` (configurable via `--tuning`)
- `key`, `tempo_bpm`, `tab`, `chords` fields stubbed with example structure + a comment
- audio metadata pre-filled where derivable: duration (via librosa), peak amplitude check, silence detection

The user then opens the JSON in their editor and fills in the actual ground truth for tabs/chords. A second CLI subcommand validates:

```bash
music-decoder fixture-validate tests/fixtures/manual/my_clip.json
```

Walks the schema and reports:
- ✓ all required fields present
- ✓ tuning is a known preset
- ✓ tab times are within audio duration
- ✓ tab pitches are valid MIDI in [0, 127]
- ✓ if `string` and `fret` fields are present, the math works (`open_pitch + fret == pitch`)
- ✓ chord roots are valid pitch classes
- ✓ chord qualities are in our supported set

### Architecture

```
src/music_decoder/cli/
  fixture.py              # new — both init and validate subcommands
src/music_decoder/cli/main.py    # +2 subparsers: fixture-init, fixture-validate
```

Reuses the existing manual fixture loader for parsing, and a fresh validator that enumerates schema rules. Validation is data-driven (a list of `(predicate, error_message)` pairs) so adding new rules is one line.

### Failure modes

| Failure | Handling |
|---|---|
| Audio file doesn't exist | print error, exit 2 |
| Output JSON file already exists | refuse to overwrite unless `--force` |
| librosa can't read audio | print ffmpeg suggestion, exit 2 |
| Validation finds errors | print each with file:line, exit 1; print summary count |

### Tests

- `tests/unit/test_cli_fixture_init.py`: writes a temp WAV, invokes `fixture_init(...)` programmatically, asserts the JSON is created with expected stub fields.
- `tests/unit/test_cli_fixture_validate.py`: parameterised over a hand-rolled fixture-with-known-errors and asserts each error gets surfaced.

### Acceptance criteria

- `music-decoder fixture-init` creates a valid stub from a real WAV.
- `music-decoder fixture-validate` rejects 8 categories of errors (missing fields, wrong tuning, out-of-range times, invalid pitches, math mismatches, invalid chord roots, invalid chord qualities, audio-file-missing).
- The CLI is documented in README.md.

---

## C-3. basic-pitch fine-tuning harness scaffold

### Why

Current basic-pitch is the off-the-shelf checkpoint trained on a heterogeneous mix (GuitarSet + MAESTRO + MedleyDB + Slakh + iKala). Fine-tuning on GuitarSet alone is a documented +5–15 F-measure win (per RESEARCH.md and Riley et al. 2024). Full training takes hours of CPU; the goal of this phase is to land the *infrastructure* — train script, validation harness, checkpoint loader — and document the workflow so the user can run it later.

### Scope (this phase)

- ✅ `scripts/finetune_basic_pitch.py` — CLI script that drives basic-pitch's training loop on a configurable dataset spec
- ✅ Dataset spec format (YAML) describing track lists, train/val splits, audio paths, JAMS paths
- ✅ Validation harness: load a candidate checkpoint, run on the regression fixtures, report F-measure delta vs. baseline
- ✅ `scripts/eval_checkpoint.py` — given a checkpoint path, report accuracy metrics
- ✅ Documentation in [docs/training.md](../../training.md) describing the workflow
- ❌ Actually running the full training (deferred — takes hours; user runs when ready)
- ❌ Distributing the resulting checkpoint (deferred until trained)

### Architecture

```
scripts/
  finetune_basic_pitch.py      # CLI entry point
  eval_checkpoint.py           # CLI entry point
src/music_decoder/training/
  __init__.py
  dataset.py                   # GuitarSet + JAMS → basic-pitch tensor pairs
  finetune.py                  # training loop wrapper (calls basic-pitch's TF train code)
  evaluate.py                  # checkpoint evaluation
  config.py                    # FineTuneConfig dataclass + YAML loader
configs/
  finetune_guitarset.yaml      # default training spec
docs/
  training.md                  # workflow doc (audio prep → training → eval → swap-in)
```

basic-pitch's training code is in [spotify/basic-pitch](https://github.com/spotify/basic-pitch). Their `basic_pitch.training` module exposes `train()`. Our wrapper builds the dataset in basic-pitch's expected format from GuitarSet's audio + JAMS, calls their `train()`, and drops the resulting checkpoint into the model cache directory. Our `BasicPitchWrapper` in `transcription/` already supports loading a checkpoint by path — extend it to accept a `checkpoint_path` parameter that overrides the default ICASSP_2022_MODEL_PATH.

### Failure modes

| Failure | Handling |
|---|---|
| basic-pitch's training submodule has incompatible API | document the error in training.md, provide a workaround pinned version |
| GuitarSet is too small to train from scratch | fine-tuning ONLY (no scratch); document this clearly |
| GPU not available | document CPU-only training (slower, still works) |
| Out-of-memory | reduce batch size in config |

### Tests

- `tests/unit/test_training_dataset.py`: build the dataset spec from a 1-track mock, assert the output tensors have the right shapes.
- `tests/unit/test_training_evaluate.py`: pass a known checkpoint path, assert the eval CLI reports the right format.

These tests don't actually train anything — they verify the harness wiring.

### Acceptance criteria

- `python scripts/finetune_basic_pitch.py --config configs/finetune_guitarset.yaml --dry-run` validates the config + reports estimated wall-clock without training.
- `python scripts/eval_checkpoint.py --checkpoint <path>` runs the regression suite against the candidate checkpoint and reports per-fixture and aggregate metrics.
- Workflow documented in `docs/training.md`.

---

## Implementation order

1. **C-4 first** — biggest immediate user-visible win, fully implementable in this session.
2. **C-2 second** — small CLI; unblocks the user from authoring manual fixtures.
3. **C-3 third** — scaffolding only, since training itself is deferred.

Each item lands as a self-contained set of commits. The regression suite gates merging (no metric regresses below baseline). The Phase C design itself is one commit; each item gets its own implementation plan + commits.

---

## Open questions / risks

- **madmom maintenance.** Last release was 0.16.1 in 2018. The compat shim works today on Python 3.11 + NumPy 1.26; the next major NumPy or Python release could break it again. Mitigation: the template_hmm fallback ensures the pipeline never depends entirely on madmom.
- **License.** Madmom is BSD; our project is MIT — compatible.
- **Threshold movement.** Raising `chord_recognition_score` from 0.05 → 0.25 is fine for floor protection but the *baseline* should record the actual achieved value so future regressions are caught at the new level.
- **Dataset size for C-3.** 5 GuitarSet tracks is too few for fine-tuning. The C-3 config should default to using the FULL GuitarSet cache (~360 tracks) for training and the 5-track regression subset for held-out validation.
