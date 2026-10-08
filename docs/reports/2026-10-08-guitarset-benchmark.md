# GuitarSet benchmark — v2 baseline vs v3 (2026-10-08)

Design: [docs/superpowers/specs/2026-10-08-accuracy-v3-design.md](../superpowers/specs/2026-10-08-accuracy-v3-design.md) ·
Harness: [scripts/benchmark_guitarset.py](../../scripts/benchmark_guitarset.py) ·
Config: [config/hyperparameters.yaml](../../config/hyperparameters.yaml) (`2026-10-08-v3-guitarset-tuned`)

## Headline

Held-out GuitarSet test split (player 05, 60 recordings, never used for any
tuning decision), `analyze(declared_kind="solo_guitar")` stages:

| stage | metric | v2 | v3 | Δ |
|---|---|---|---|---|
| notes | onset-only note F (50 ms) | 0.788 | **0.860** | +7.2 |
|  | precision / recall | 0.698 / 0.931 | 0.881 / 0.863 | |
| tab (end-to-end) | string accuracy on transcribed notes | 0.499 | **0.601** | +10.2 |
| tab (fingering only) | string accuracy on ground-truth notes | 0.537 | **0.600** | +6.3 |
| key | MIREX weighted score | 0.555 | **0.650** | +9.5 |
| beats | beat F-measure (±70 ms) | 0.438 | **0.479** | +4.1 |
| chords | MIREX `majmin` | 0.385 | 0.385 | 0 (backend unchanged) |

Every stage we changed is higher on the held-out split; the unchanged stage
(chords) is identical.

## Method

- **Data.** GuitarSet: 360 solo acoustic-guitar recordings (6 players × 5
  styles × 2 progressions × comp/solo) with hexaphonic note annotations
  (pitch, onset, offset, *string*), lead-sheet chords, key and beats.
  Audio: `audio_mono-mic`.
- **Splits.** Dev = players 00–04 (300 tracks) for every decision; test =
  player 05 (60 tracks), run once for v2 and once for the final v3 config.
- **Metrics.** Standard `mir_eval` implementations (see
  [architecture-technical.md §11.5](../architecture-technical.md)). Note F is
  the onset-only protocol used by the GAPS / GuitarSet literature. Tab
  accuracy is the fraction of onset+pitch-matched notes placed on the
  annotated string.
- **Practice.** Each change was implemented behind a config switch,
  measured on dev against the v2 number, and kept only if it improved it.
  Rejected variants are listed below.

## Dev-split decisions (n = 300)

| change | v2 → v3 (dev) | kept? |
|---|---|---|
| Beats: madmom RNN + DBN downbeat tracker instead of librosa | 0.501 → 0.704 | yes |
| Key: fuse KK + Temperley profiles, madmom CNN, chord-progression fit | 0.560 → 0.685 | yes |
| ↳ CNN key classifier alone | 0.560 → 0.579 | superseded by fusion |
| ↳ fusion with CNN-confidence gating (τ 0.2–0.6) | 0.657–0.671 | no |
| Chords: madmom CNN+CRF instead of deep-chroma | 0.469 → 0.462 (comp +1.6, solo −3.0) | no |
| Chords: snap boundaries to beats | +0.001 | no (noise) |
| Notes: basic-pitch decoding sweep → onset 0.6, frame 0.5, 58 ms, melodia off | 0.746 → 0.811 | yes |
| Notes: same-pitch merge / short-note drop post-processing | −11 pts recall | no |
| Notes: min frequency 32.7 → 60 Hz | ±0.0001 | yes (removes impossible sub-guitar notes) |
| Tabs: symmetric hand-move cost (v2 made moving down the neck free) | 0.528 → 0.539 | yes |
| Tabs: chord hand position = lowest *fretted* note, not an open string | +2.3 (combined 0.568) | yes |
| Tabs: open strings movement-free | −5.3 | no |
| Tabs: A* weight coordinate descent | 0.568 → 0.584 | yes |
| Tabs: group chords by onset (50 ms) instead of interval overlap | 0.584 → 0.614 (e2e 0.550 → 0.587) | yes |
| Tabs: hand-anchor window 3 instead of 2 | +0.004 | no (noise) |

v3 on the full dev split (one run, final config): beats 0.704 · chords 0.469
· key 0.685 · notes 0.811 (P 0.862 / R 0.781) · tab e2e 0.587 · tab gt 0.614.

## Bugs found along the way

- **Free downward hand movement** in the A* cost (`max(0, Δfret)`): moving
  from fret 9 to fret 2 cost nothing.
- **One out-of-range note poisoned a whole chord.** A sub-guitar artifact
  (e.g. MIDI 31) inside a chord group made the group unsatisfiable, dropping
  every note; and overlap grouping fused re-attacks of the same pitch into
  one impossible "chord". Together these made
  `tests/integration/test_analyze_e2e.py[g_major_chord]` fail on `master`
  (empty tab); it passes now.
- **madmom downbeat tracker crashed on NumPy ≥ 1.24** (ragged
  `np.asarray`); fixed with a shim scoped to that madmom module.
- **A madmom chord failure turned the whole song into `N`**; it now falls
  back to the template HMM.
- `analyze()` metadata read the hyperparameter id from a cwd-relative path.

## Per-style caveats (test split, n = 6 per row — noisy)

- **Beats** improved overall and on every comping style (Rock-comp
  0.926 → 0.992, SS-comp 0.455 → 0.630) but dropped on three *solo* groups
  (Funk-solo 0.383 → 0.234, Rock-solo 0.320 → 0.257, SS-solo 0.365 → 0.304).
  On dev the madmom tracker was better on every group, including solos, so
  this looks like player-05 variance, but sparse single-line solos remain
  the beat tracker's weak spot.
- **Key** gained most on comping (Funk-comp 0.200 → 0.750, Jazz-comp
  0.100 → 0.367) and dropped on Jazz-, Rock- and SS-solo. Single-line
  solos give every cue less to work with.
- **Chords** on solo recordings score ~0.1–0.2 for every backend: a melody
  line without accompaniment does not state the lead-sheet harmony.
- The dev → test gap is largest for beats (+20 vs +4 pts) and key (+12.5 vs
  +9.5).

## Scope limits

- Solo guitar only. The Demucs full-mix path is unchanged and unmeasured (no
  annotated full-mix guitar data on disk).
- Synthetic regression fixtures are rendered with sine tones because the
  soundfont mirrors were unreachable from this machine; the four-cue key
  fusion was chosen partly because it keeps the sine-rendered
  `c_major_scale` key gate passing (the CNN alone misreads it) while also
  scoring higher on dev.

## Not adopted (worth revisiting)

- **Transcription models** with higher published GuitarSet scores than
  basic-pitch: Riley et al. high-resolution guitar (onset F ≈ 0.87
  zero-shot), the GAPS model (≈ 0.86–0.88; weights not confirmed public),
  and MuScriptor (open-weight multi-instrument transformer, ISMIR 2026).
  Each would slot in behind `transcription.decode_model_output` /
  `run_model`. The benchmark's `notes` stage can score them directly.
- **Beat This!** (ISMIR 2024, MIT): newer beat tracker without DBN
  post-processing; a candidate for the solo-guitar beat weakness above.
- **Learned MIDI → tab** (Fretting-Transformer, MIDI-to-Tab): no usable
  checkpoints today. The `tab_gt` stage is the right gate for them.

## Reproduce

```bash
python scripts/benchmark_guitarset.py --split dev  --by-style --workers 6
python scripts/benchmark_guitarset.py --split test --by-style --workers 6
# compare a config variant without editing the live YAML:
MUSIC_DECODER_HYPERPARAMETERS=/path/to/variant.yaml python scripts/benchmark_guitarset.py --split dev --stages key
```

Runtime on an Apple-silicon Mac: about 5 minutes for the test split and
about 25 minutes for dev with 6 workers. basic-pitch network outputs are
cached under `out/bench-cache/bp/`.
