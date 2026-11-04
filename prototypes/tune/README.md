# VOX tuner spike (milestone M1)

Offline C++20 test bed for VOX's Track-mode tuner
([spec §6](../../docs/superpowers/specs/2026-11-04-vox-design.md#6-tune),
[task plan M1](../../docs/superpowers/plans/2026-11-04-vox-p0-p1.md#m1-tuner-spike)).
Not shipped. The code is written to real-time rules (no allocation after
`prepare()`, per-sample processing, block-size independent) so it can move
into `rapsuite/libs/rapcore/pitch` unchanged once that repo exists.

## What's here

| Part | File | Notes |
|---|---|---|
| Pitch detection | `include/tune/detector.hpp` | cycfi/q bitstream autocorrelation (MIT) + voicing gate + per-period NCCF tracker + fast voicing end and decimated YIN onset acquisition |
| Note decision | `include/tune/scale.hpp` | 8 scale types, key parsing, hysteresis quantizer |
| Retune curve | `include/tune/correction.hpp` | Critically damped 2nd-order smoother; `retune_ms` = time to ~90% |
| Shifter | `include/tune/splice_shifter.hpp` | Period-synchronous resampling with correlation-refined ±1-period splices and 1.33 ms crossfades; 2.0 ms reported latency |
| Formants | `include/tune/formant.hpp`, `include/tune/lpc.hpp` | Frequency-warped LPC whiten → shift → re-colour (exact inverse, no added latency); `FormantMode` Auto / Preserve / Follow |
| Tuner | `include/tune/tuner.hpp` | Wires the above; mono |
| Evaluation | `src/synth.cpp`, `src/yin_offline.cpp`, `src/metrics.cpp` | Synthetic voices with known pitch; an independent offline YIN as judge; pitch-error and HF-artifact metrics |
| Tools | `tools/tune_eval.cpp`, `tools/tune_render.cpp` | Benchmark table (accuracy, onset settle, HF energy, CPU); `--clips` for real WAV folders; render real WAV files |
| Test audio | `data/README.md` | Local vs Corpus tiers; no audio is ever committed |
| Tests | `tests/` | doctest (from cycfi/infra); 30 cases incl. exact-latency null test, block-size invariance, no-allocation checks |

Not yet built (see the task plan): Throat, humanize /
flex / vibrato controls, HQ (PSOLA) mode, stereo-linked processing, SIMD /
decimated tracker.

## Build and run

Needs CMake ≥ 3.25, a C++20 compiler (Clang 16+ or GCC 13+), and network
access on first configure (FetchContent pulls pinned cycfi/q and cycfi/infra).

```bash
cmake -S prototypes/tune -B prototypes/tune/build -G Ninja -DCMAKE_BUILD_TYPE=Release
```

```bash
cmake --build prototypes/tune/build
```

```bash
prototypes/tune/build/tune_tests
```

```bash
prototypes/tune/build/tune_eval --out prototypes/tune/out/eval48
```

```bash
prototypes/tune/build/tune_render vocal.wav vocal_tuned.wav --key A --scale minor --retune 0 --trace trace.csv
```

`tune_render` options: `--transpose` semitones, `--formant auto|preserve|follow`, `--key` (C, C#, Db, …), `--scale` (major, minor,
harmonic_minor, melodic_minor, major_pentatonic, minor_pentatonic, blues,
chromatic), `--retune` ms (0 = hard-tune), `--amount` 0–1, `--low` / `--high`
Hz detection range, `--trace` CSV at 200 Hz.

Results and findings: [spike report](../../docs/reports/2026-11-04-tune-spike-v0.md).

## Licences

Our code: same as this repo. Dependencies: cycfi/q (MIT, LICENSE file) and
cycfi/infra (MIT per file headers; the repo has no LICENSE file), both
pinned by commit in `CMakeLists.txt`. doctest (MIT) comes with infra. No GPL
code was read or used (clean-room rule, spec §6.4).
