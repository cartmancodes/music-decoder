# Test audio for the tuner spike

No audio is ever committed here. Two tiers, two folders, both git-ignored:

| Folder | What may go in | What it may be used for | Never |
|---|---|---|---|
| `data/local/` | Anything **you** choose for your own quick checks, including clips you downloaded yourself. Source and legal risk are your call. | Private listening and smoke tests on your own machine: `tune_render`, `tune_eval --clips data/local` | Committed, uploaded, shared with contractors or beta users, used for training, used as CI/golden fixtures, used in demos, marketing or presets, or quoted in reports by name |
| `data/corpus/` | Only **cleared** audio: your own recordings, consented sessions (consent form from plan T0.8), royalty-free acapella packs whose licence allows it, CC0 | The official benchmark (plan T1.6), listening tests, golden renders (via the private bucket), later training if the consent / licence covers it | Anything without a provenance row in `manifest.csv` |

Rules for `data/corpus/`:
- Every file has a row in `data/corpus/manifest.csv` (template:
  `manifest.example.csv`). Each row includes source, licence or consent ID,
  and allowed uses. No row means the file isn't used.
- The audio lives in the private bucket; only the manifest gets committed
  once the suite repo exists.

## Running on clips

```bash
prototypes/tune/build/tune_eval --clips prototypes/tune/data/local --key A --scale minor --out prototypes/tune/out/local
```

```bash
prototypes/tune/build/tune_render prototypes/tune/data/local/take1.wav prototypes/tune/out/take1_tuned.wav --key A --scale minor --retune 0
```

- `--clips` reads every `.wav` in the folder.
- An optional `manifest.csv` (`file,key,scale[,low_hz,high_hz]`) sets
  the key per clip; otherwise `--key` / `--scale` apply.
- Real clips have no ground truth, so it reports how close voiced frames
  sit to the scale before and after tuning, plus the splice rate and CPU.
  Listening is still the real test.
- **Best material: dry, unprocessed vocal takes.** Released songs are
  already tuned and mixed, so they show little about the tuner, and
  separating the vocal out adds artifacts of its own.
