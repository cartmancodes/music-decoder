# Fine-tuning basic-pitch on GuitarSet (Phase C-3)

This doc walks through the workflow for fine-tuning Spotify's basic-pitch
checkpoint on GuitarSet (or your own audio + JAMS pairs). The expected gain
is +5–15 F-measure points on guitar-specific transcription per RESEARCH.md
and Riley et al. 2024 ([arXiv 2402.15258](https://arxiv.org/abs/2402.15258)).

The harness is **scaffolding only** — running training is not automated.
Training takes hours of CPU time (or 30-60 minutes of GPU); the C-3 design
ships the infrastructure, you run training when ready.

---

## Prerequisites

1. Full GuitarSet cache. The 5-track regression subset is enough for
   validation but won't train a meaningful model.

   ```bash
   make fixtures   # downloads all 360 GuitarSet tracks (~3 GB)
   ```

2. ffmpeg (already required for v1):

   ```bash
   brew install ffmpeg          # macOS
   sudo apt install ffmpeg      # Debian/Ubuntu
   ```

3. basic-pitch's training package. The pip-installed `basic-pitch` ships
   inference-only by default; training requires the source repository:

   ```bash
   git clone https://github.com/spotify/basic-pitch.git ~/code/basic-pitch
   cd ~/code/basic-pitch
   pip install -e ".[train]"   # if the [train] extra exists in the version you cloned
   ```

   Note: at the time of writing, basic-pitch's training pipeline lives in
   `basic_pitch/training/` and exposes a `train()` entrypoint, but the
   exact API has shifted between releases. Check the current `README.md`
   in the cloned repo.

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

### 2. Expand the track list

The shipped `configs/finetune_guitarset.yaml` has only ~10 tracks under
`train.track_ids` as a placeholder. To actually train, replace this with the
full GuitarSet track list (excluding the 5 validation tracks):

```bash
python -c "
import mirdata
ds = mirdata.initialize('guitarset', data_home='tests/fixtures/guitarset')
val = {'00_BN1-129-Eb_comp', '00_BN1-129-Eb_solo',
       '00_Funk1-114-Ab_comp', '00_Jazz1-200-B_comp', '00_Rock1-130-A_comp'}
for t in ds.track_ids:
    if t not in val:
        print(f'    - {t}')
" >> configs/finetune_guitarset.train.tracks.yaml
```

Then merge that into the config under `train.track_ids`. (Not automated;
this is the kind of thing you eyeball once.)

### 3. Run training

When you're ready to invest the wall-clock:

```bash
python scripts/finetune_basic_pitch.py --config configs/finetune_guitarset.yaml --execute
```

The harness will currently raise `NotImplementedError` with a pointer back
here, because actual basic-pitch training execution is not wired up in the
v1 scaffold. To proceed:

#### Manual workflow until execution is wired

1. Convert the dataset spec to basic-pitch's expected format:
   ```python
   from music_decoder.training.dataset import enumerate_pairs
   from music_decoder.training.config import load_finetune_config
   cfg = load_finetune_config(Path('configs/finetune_guitarset.yaml'))
   pairs = list(enumerate_pairs(cfg.train))
   # write pairs to whatever basic-pitch's training API consumes (TFRecord,
   # CSV manifest, etc., depending on version).
   ```

2. Run basic-pitch's training driver from the cloned repo:
   ```bash
   cd ~/code/basic-pitch
   python -m basic_pitch.training.train \
       --train-dataset /path/to/manifest.csv \
       --output ~/Library/Caches/music-decoder/checkpoints/2026-05-07-guitarset-finetune-v1.npz \
       --epochs 5 --batch-size 4 --lr 1e-4
   ```

   Adjust to whatever basic-pitch's current training entrypoint looks like.

3. The output checkpoint goes into `output_checkpoint_dir` from the config.

### 4. Evaluate the candidate checkpoint

```bash
python scripts/eval_checkpoint.py \
    --checkpoint ~/Library/Caches/music-decoder/checkpoints/2026-05-07-guitarset-finetune-v1.npz \
    --report-out evaluation_reports/eval_finetune_v1.json
```

This runs the regression pipeline against the held-out 5-track subset
plus the synthetic fixtures and reports per-fixture + aggregate metrics.
Compare against `evaluation_reports/baseline.json`:

| Metric                      | Stock | Candidate | Delta | Promote? |
|----------------------------|-------|-----------|-------|----------|
| note_f_measure              | 0.6315 | …         | …     | ≥ +0.03   |
| pitch_class_accuracy        | 0.6013 | …         | …     | ≥ +0.05   |
| onset_f_measure             | 0.7304 | …         | …     | no regression |

If the candidate beats the baseline outside the regression tolerance
(0.02), promote it: copy the checkpoint to a stable path, update
`config/runtime.yaml`'s `model_cache_dir` to point at the new file, run
the regression suite to capture a fresh baseline, and commit.

### 5. Promote the checkpoint

Update the model loader to use the candidate. The current
`transcription/basic_pitch_wrapper.py` uses
`ICASSP_2022_MODEL_PATH` directly; extending it to accept a custom
checkpoint path is a one-liner:

```python
def transcribe_basic_pitch(audio, params, *, output_dir, checkpoint_path=None):
    model_path = checkpoint_path or ICASSP_2022_MODEL_PATH
    ...
```

Wire `checkpoint_path` through `HyperparameterSet.basic_pitch.checkpoint`
and the orchestrator. Capture the new baseline, commit the YAML + the
baseline JSON.

---

## Expected gains (baseline)

Per RESEARCH.md and Riley et al.:

- Stock basic-pitch on GuitarSet: ~0.55–0.70 F-measure
- Domain-adapted (Riley): ~0.85 F-measure (zero-shot)

Our v1 baseline (stock, on 5-track subset): **0.6315**. A successful
fine-tune should land in the **0.70–0.80** range.

If your fine-tune scores lower than baseline, possible causes:
- Train/val leakage (track in both lists)
- Insufficient training data (a few dozen tracks isn't enough)
- Learning rate too high — basic-pitch's pre-trained weights are sensitive
- Catastrophic forgetting (model loses non-guitar generalization; check
  pitch_class_accuracy on the synthetic c_major_scale fixture)

---

## Open questions

- **basic-pitch's training package compatibility** — the install
  instructions above are aspirational; the pip-installed `basic-pitch`
  may not include the `[train]` extra in all versions. Check the upstream
  README before assuming the API.
- **Custom-checkpoint loading in `transcribe_basic_pitch`** — currently
  hard-coded to the stock checkpoint. The wiring task is small but is
  not in the C-3 scaffold; it's a follow-up.
- **GAPS dataset** — Riley's GAPS provides 14 hours of classical guitar
  with frame-aligned MIDI. If you have time to absorb a second dataset,
  GAPS + GuitarSet should land closer to the 0.85 SOTA number.

---

## References

- [basic-pitch (Spotify, 2022)](https://github.com/spotify/basic-pitch)
- [GuitarSet dataset](https://github.com/marl/GuitarSet)
- [GAPS (Riley et al. 2024)](https://aim-qmul.github.io/GAPS/)
- [High-resolution guitar transcription (Riley & Edwards 2024)](https://arxiv.org/abs/2402.15258)
- [Phase C design](superpowers/specs/2026-05-07-phase-c-design.md)
- [Accuracy roadmap](superpowers/research/2026-05-05-accuracy-improvement-roadmap.md)
