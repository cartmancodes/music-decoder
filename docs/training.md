# Fine-tuning basic-pitch on GuitarSet (Phase C-3)

This doc walks through the workflow for fine-tuning Spotify's basic-pitch
checkpoint on GuitarSet (or your own audio + JAMS pairs). The expected gain
is +5–15 F-measure points on guitar-specific transcription per RESEARCH.md
and Riley et al. 2024 ([arXiv 2402.15258](https://arxiv.org/abs/2402.15258)).

After Phase C-3 the entire pipeline is wired and reproducible end-to-end.
Running training itself is still up to you because CPU runs take hours; on
this Mac the actual training command is gated by sandboxing of TF's
just-in-time-compiled cache writes — run it from your own shell.

---

## Prerequisites (one-time)

1. **GuitarSet cache.** Required.

   ```bash
   make fixtures   # downloads ~3 GB of audio + JAMS via mirdata
   ```

2. **System dependencies.** ffmpeg + sox + fluidsynth. Already installed if
   you've been running the v1 pipeline + Phase B-5.

   ```bash
   brew install ffmpeg sox fluid-synth chromaprint    # macOS
   sudo apt install ffmpeg sox libfluidsynth3 libchromaprint1   # Debian/Ubuntu
   ```

3. **Python deps for training.** basic-pitch's training pipeline isn't part
   of our default install (it adds ~300 MB of Apache Beam, etc.).

   ```bash
   pip install apache-beam sox tf-keras
   ```

4. **`tf-keras` is critical.** basic-pitch's training code uses Keras-2-style
   symbolic tensor ops that don't work under Keras 3 (TF 2.16+). The
   `TF_USE_LEGACY_KERAS=1` environment variable + the `tf-keras` package
   provide the Keras-2 compatibility shim. Our `basic_pitch_compat.py`
   patches the remaining `input_shape.rank` calls.

---

## Step-by-step

### 1. Validate the config

```bash
python scripts/finetune_basic_pitch.py --config configs/finetune_guitarset.yaml --dry-run
```

This:
- Loads `configs/finetune_guitarset.yaml`
- Enumerates audio + JAMS pairs in the train and val datasets
- Reports estimated wall-clock time
- Verifies every listed track has both files on disk
- Exits without touching any model

If the dry-run prints zero pairs found, your `cache_dir` or `track_ids` are
wrong. Fix the config and re-run.

### 2. Convert GuitarSet → TFRecord (Phase C-3)

basic-pitch's training expects TFRecord files in a specific layout. Use our
Beam-free converter:

```bash
python scripts/convert_guitarset_to_tfrecord.py \
    --source tests/fixtures/guitarset \
    --destination /tmp/bp_tfrecords \
    --train-percent 0.85 --validation-percent 0.10 --seed 42
```

Output layout (matches what `basic_pitch.train` looks for):

```
/tmp/bp_tfrecords/guitarset/splits/train/*.tfrecord
/tmp/bp_tfrecords/guitarset/splits/validation/*.tfrecord
/tmp/bp_tfrecords/guitarset/splits/test/*.tfrecord
```

For a smoke test, add `--limit 30` to convert just 30 tracks
(takes ~30 s).

### 3. Run training

When you're ready to invest the wall-clock:

```bash
TF_USE_LEGACY_KERAS=1 python scripts/finetune_basic_pitch.py \
    --config configs/finetune_guitarset.yaml \
    --tfrecord-source /tmp/bp_tfrecords \
    --execute
```

The harness:
- Imports `music_decoder.training.basic_pitch_compat` (applies the
  TF 2.21 / Keras 3 compatibility shim — `NormalizedLog.build` and
  `Stft.build` no longer crash on tuple `input_shape`).
- Calls `basic_pitch.train.main(...)` with our dataset spec.
- Writes checkpoints under `output_checkpoint_dir/<UTC-timestamp>/`,
  with `model.best/` being the best-validation checkpoint and
  `checkpoints/model.{epoch:02d}/` being per-epoch snapshots.

For a quick demonstration run (a few minutes):

```bash
TF_USE_LEGACY_KERAS=1 python scripts/finetune_basic_pitch.py \
    --config configs/finetune_guitarset.yaml \
    --tfrecord-source /tmp/bp_tfrecords \
    --execute \
    --steps-per-epoch 3 \
    --validation-steps 1
```

For a real training run (hours on CPU, ~1 hr on Apple Silicon Metal):

```bash
TF_USE_LEGACY_KERAS=1 python scripts/finetune_basic_pitch.py \
    --config configs/finetune_guitarset.yaml \
    --tfrecord-source /tmp/bp_tfrecords \
    --execute \
    --steps-per-epoch 100 \
    --validation-steps 10
```

(Edit `configs/finetune_guitarset.yaml` first to change `epochs:` from 5 to
something more substantial — Riley used 50.)

### 4. Evaluate the candidate checkpoint

```bash
python scripts/eval_checkpoint.py \
    --checkpoint ~/Library/Caches/music-decoder/checkpoints/<id>/<timestamp>/model.best \
    --report-out evaluation_reports/eval_finetune_v1.json
```

This runs the regression pipeline against the held-out 5-track subset
plus the synthetic fixtures and reports per-fixture + aggregate metrics.
Compare against `evaluation_reports/baseline.json`:

| Metric                      | Stock v1 | Candidate | Delta | Promote? |
|----------------------------|----------|-----------|-------|----------|
| note_f_measure              | 0.6315   | …         | …     | ≥ +0.03   |
| pitch_class_accuracy        | 0.6013   | …         | …     | ≥ +0.05   |
| onset_f_measure             | 0.7304   | …         | …     | no regression |

### 5. Promote the checkpoint

Update the model loader to use the candidate. The current
`transcription/basic_pitch_wrapper.py` uses the auto-discovered stock
checkpoint; extending it to accept a custom checkpoint path is a one-liner:

```python
def transcribe_basic_pitch(audio, params, *, output_dir, checkpoint_path=None):
    model_path = checkpoint_path or _auto_discover_stock_checkpoint()
    ...
```

Wire `checkpoint_path` through `HyperparameterSet.basic_pitch.checkpoint`
and the orchestrator. Capture the new baseline, commit the YAML + the
baseline JSON.

For the `highres-guitar` (Kong et al.) backend, `checkpoint_path` is
already a parameter on `transcribe_highres_guitar` — same workflow but
swap in a Kong-architecture checkpoint instead.

---

## Expected gains (baseline)

Per RESEARCH.md and Riley et al.:

- Stock basic-pitch on GuitarSet: ~0.55–0.70 F-measure
- Domain-adapted (Riley): ~0.85 F-measure (zero-shot)

Our v1 baseline (stock, on 5-track subset): **0.6315**. A successful
fine-tune should land in the **0.70–0.80** range.

If your fine-tune scores lower than baseline, possible causes:
- Train/val leakage (track in both lists) — check `configs/finetune_guitarset.yaml`
- Insufficient training data (a few dozen tracks isn't enough)
- Learning rate too high — basic-pitch's pre-trained weights are sensitive
- Catastrophic forgetting (model loses non-guitar generalization; check
  pitch_class_accuracy on the synthetic c_major_scale fixture)

---

## What changed in Phase C-3 (vs. the prior scaffold)

Before:
- `execute_training()` raised `NotImplementedError`
- Workflow doc told you to manually clone spotify/basic-pitch and figure
  out the data pipeline yourself
- No way to convert GuitarSet to the format the trainer expects

After:
- `scripts/convert_guitarset_to_tfrecord.py` does the conversion
  Beam-free in a single subprocess (no `apache-beam` flakiness around
  multi-Python-interpreter `sys.path`)
- `music_decoder/training/basic_pitch_compat.py` patches the TF 2.21 /
  Keras 3 incompat in `basic_pitch.layers.signal` so the model builds
  cleanly under our existing TF
- `execute_training()` actually invokes `basic_pitch.train.main(...)`
  with the right env, dataset spec, and prerequisites
- Tests cover the prerequisite-missing path (`TF_USE_LEGACY_KERAS=1`
  enforcement) and the shim's idempotency

---

## Open questions / known limitations

- **Apple Silicon Metal training is unverified.** TF Metal may or may not
  work with the legacy Keras path; if Metal training crashes, fall back
  to CPU.
- **basic-pitch's training code is sensitive to TF version.** Tested on
  TF 2.21.0 with `tf-keras 2.21.0` + `TF_USE_LEGACY_KERAS=1`. If TF 2.22+
  ships before basic-pitch updates, the shim may need extending.
- **Catastrophic forgetting.** The fine-tune workflow doesn't currently
  load the stock checkpoint as a starting point — `basic_pitch.train`
  trains from scratch. To genuinely *fine-tune* (resume from stock
  weights), you'd need to add `model.load_weights(...)` in basic-pitch's
  trainer between `models.model(...)` and `model.compile(...)`. Filing
  an upstream feature request is the cleanest path.

---

## References

- [basic-pitch (Spotify, 2022)](https://github.com/spotify/basic-pitch)
- [GuitarSet dataset](https://github.com/marl/GuitarSet)
- [GAPS (Riley et al. 2024)](https://aim-qmul.github.io/GAPS/)
- [High-resolution guitar transcription (Riley & Edwards 2024)](https://arxiv.org/abs/2402.15258)
- [Phase C design](superpowers/specs/2026-05-07-phase-c-design.md)
- [Accuracy roadmap](superpowers/research/2026-05-05-accuracy-improvement-roadmap.md)
