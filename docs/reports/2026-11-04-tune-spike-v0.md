# VOX tuner spike v0: first measurements

Date: 2026-11-04 · Code: [`prototypes/tune`](../../prototypes/tune) ·
Spec: [VOX §6](../superpowers/specs/2026-11-04-vox-design.md#6-tune) ·
Plan: [M1](../superpowers/plans/2026-11-04-vox-p0-p1.md#m1-tuner-spike)

## Summary

The Track-mode architecture meets the latency target and corrects pitch
accurately on synthetic voices, **including vibrato and rap-speed
syllables**:
- reported latency **2.0 ms** at 44.1–192 kHz, verified by an exact null
  test;
- median residual error **≤ 1 cent** in every scenario;
- no gross (> 50 cent) or octave errors;
- no added high-frequency splice energy;
- about **0.6–1.5% of one core** on this test machine.

**Verdict: provisional GO for the ≤ 3 ms Track-mode design.** (v0.1 adds the onset fix below: notes now settle in 0–12 ms median.) It isn't
final until it passes on real rap vocals and in a listening session (plan
tasks T1.6–T1.8). Synthetic signals can't tell us how splices sound on
breathy, noisy, fast real voices.

## Setup

- Signals: additive "voice" (harmonics to 8 kHz, vowel /a/ formant
  envelope, breath noise) with known f0, plus band-passed noise for
  consonants. Five scenarios (`src/synth.cpp`), A natural minor, notes
  detuned by known offsets so the expected target is unambiguous.
- Tuner: retune 0 ms (hard-tune) and 50 ms; default detector range
  70–800 Hz.
- Judge: independent offline YIN on the output (not the tuner's own
  detector), aligned by the reported latency. The first 60 ms after each
  onset and the last 30 ms of each segment are not scored.
- Machine: this cloud container (x86-64, 4 vCPU, unknown model), Clang 18
  `-O2` Release build. Timings are single runs, so treat them as indicative.

## Results (48 kHz)

| Scenario | Input median / p95 (cents) | Hard-tune median / p95 | Within 10 c | > 50 c | Retune 50 ms median | Splices / s | ns / sample |
|---|---|---|---|---|---|---|---|
| steady_detuned (A2–E3, ±40 c) | 27.5 / 40.0 | **0.0 / 0.2** | 100% | 0% | 0.0 | 2.9 | 184 |
| steady_high (A3–E4) | 27.5 / 40.0 | **0.9 / 7.0** | 99.7% | 0% | 1.1 | 4.1 | 131 |
| vibrato (5.5 Hz, ±30 c) | 17.8 / 39.3 | **0.8 / 3.3** | 100% | 0% | 16.8 (vibrato kept, as intended) | 3.4 | 179 |
| fast_flow (70–140 ms syllables) | 20.0 / 34.3 | **0.2 / 1.0** | 100% | 0% | 3.8 | 5.3 | 181 |
| low_voice (E2–A2, 82–110 Hz) | 30.0 / 35.0 | **0.1 / 0.4** | 100% | 0% | 0.1 | 1.6 | 241 |

- 44.1 kHz and 96 kHz give the same accuracy (within 0.1 cent).
- **HF energy above 12 kHz** relative to total in voiced regions is
  −29 to −35 dB in the input and 1–2 dB *lower* in the output in every
  scenario, so the splices add no measurable clicks. The slight drop is
  the low-pass effect of cubic interpolation.
- **CPU:** 130–240 ns/sample means 0.6–1.2% of one core at 48 kHz on this
  machine. The budget is the whole VOX chain ≤ 3% on an M1, so the tuner
  should get to ≤ 1% after the optimisation task (T1.9).

All unit tests pass (17 at v0; 30 after v0.2) with Clang 18 and GCC 13, and under GCC
AddressSanitizer + UndefinedBehaviorSanitizer. They include:
- exact delay at ratio 1;
- frequency accuracy and click-free output at constant ratios;
- bounds on the delay wander;
- bit-identical output for block sizes 1 / 7 / 64 / 512 / 4096;
- **zero heap allocations** during processing and parameter changes;
- hard-tune accuracy thresholds.

### Onset settle time (T1.2, added later on 2026-11-04)

The settle time is the time from onset until the output stays within
±10 c of the target for ≥ 20 ms. It uses the YIN judge at a 2 ms hop,
minus half its window as bias.

| Scenario | Hard-tune median / p90 (ms) | Retune 50 ms median / p90 (ms) | Unsettled (hard / 50 ms) |
|---|---|---|---|
| steady_detuned | 17 / 24 | 21 / 35 | 0/6 · 0/6 |
| steady_high | 22 / 33 | 23 / 47 | 0/6 · 0/6 |
| vibrato | 20 / 24 | 41 / 106 | 0/4 · 0/4 |
| fast_flow | 9.7 / 18 | 16 / 46 | 0/28 · 2/28 |
| low_voice | 18 / 24 | 18 / 46 | 0/5 · 0/5 |

Settling is dominated by detection lag at onsets. The higher register
(22 ms) misses the "≤ 2 periods + 5 ms" target (≈ 13 ms at 250 Hz).
q's `predict_frequency()` at onsets is the next step. Fast-flow syllables
settle in ≈ 10 ms, which matters most for rap.

### Onset fix (v0.1, later on 2026-11-04)

**Diagnosis.** Instrumenting the detector at every onset showed the
problem wasn't slow acquisition. **Voicing carried over the 25–60 ms
consonant gaps**, so most notes started already "voiced" with the
previous note's period, or garbage (e.g. 427 Hz for a 258 Hz note). The
note then settled only once q re-anchored. Using q's `predict_frequency()`
(the idea in the plan) **doesn't help**: it reached ±30 c in 15–21 ms
on average, slower than q's regular estimate (7–12 ms).

**Changes** (`include/tune/detector.hpp`):
1. **Fast voicing end:** if the tracker's best correlation stays below 0.5
   for 2 updates (≈ 2.7 ms), voicing ends and q is reset. A successful
   track also keeps the hold alive.
2. **Fast acquisition:** while unvoiced but above the gate, a YIN search
   (4 ms window, all lags) runs every update **on a 4× decimated copy**
   (≈ 16× cheaper). Candidates are refined at full rate, and two agreeing
   candidates start voicing.
3. **Confirmed q proposals:** from unvoiced, q's estimate must show NCCF ≥
   0.8 at its period (stops voicing on consonant noise).
4. **Guarded re-anchor:** while voiced, q may move the tracker only if its
   period explains the signal at least as well as the current one. This
   removed bad re-anchors right after a reset, which had made the
   fast-flow tail worse.
5. Tracker window cap as a duration (5.33 ms), not 256 samples, which
   fixed a low-voice regression at 96 kHz.

**Result (hard-tune, 48 kHz; median / p90 settle):**

| Scenario | v0 | v0.1 | Accuracy v0.1 (median / p95 c) |
|---|---|---|---|
| steady_detuned | 17 / 24 ms | **3 / 10 ms** | 0.0 / 0.1 |
| steady_high | 22 / 33 ms | **0 / 7 ms** | 0.0 / 0.3 (was 0.9 / 7.0) |
| vibrato | 20 / 24 ms | **0 / 8 ms** | 0.7 / 3.1 |
| fast_flow | 9.7 / 18 ms | **1.2 / 6.1 ms** | 0.3 / 0.9 |
| low_voice | 18 / 24 ms | **12 / 14 ms** | 0.1 / 0.5 |

- "0 ms" means settled within the judge's own half-window (≈ 9 ms), so it
  is at or below what the metric can resolve.
- 44.1 and 96 kHz match within ±1 ms and ±0.1 c.
- CPU is unchanged at 150–290 ns/sample. Without decimation it was up to
  905 ns, because acquisition runs on every consonant and breath.
- Tests: 23 cases pass with Clang 18 and GCC 13, and under ASan + UBSan.
  New detector tests check:
  - every note's first voiced pitch is fresh (within 50 c of the true f0)
    and none is still voiced at its onset;
  - voicing drops inside consonant gaps;
  - no allocation, including during acquisition.
- **Plan target "≤ 2 periods + 5 ms" is now met in every scenario except
  low_voice.** There it's 12 ms against ≈ 10 ms at 100 Hz, because the
  acquisition needs window + one period of the new note. A 2.5 ms
  acquisition window is the next thing to try if real vocals show it
  matters.

### Formant preservation (v0.2, plan T1.3)

**What it is:** whiten the input with an LPC envelope filter, shift the
flat residual, then re-apply the envelope of the input moment being read
(spec §6.5). It adds no latency, because analysis is causal. Whitening and
re-colouring stay exact inverses while the coefficients change, so with
correction off the tuner is still an exact delay (tested, error < 1e-4).

**What it took** (each step measured; harness in `tune_eval`'s formant
table and `tests/test_formant.cpp`):
1. **Plain LPC failed.** At order 16 and 48 kHz, F1 under +3 st moved to
   842–852 Hz, almost as far as no preservation at all (880–898 Hz).
   Even with ideal fixed coefficients, LPC under-modelled F1/F2 by
   ≈ 8–9 dB: its poles spread evenly up to 24 kHz. Pre-emphasis didn't
   help, and order 48 helped but was too expensive.
2. **Frequency-warped LPC fixed it.** All-pass warping with λ = 0.756
   (Bark-like at 48 kHz) at order 20 puts resolution where formants are:
   - the whitening filter is a warped FIR;
   - the warped all-pole re-colour filter's delay-free loop is solved in
     closed form per sample;
   - reflection coefficients are interpolated and converted to direct form
     every 16 samples;
   - the hop went from 5 to 10 ms to halve the analysis cost.
3. **Splices are chosen on the original waveform, not the residual.** The
   flat, noise-like residual made poor splice decisions. That doubled
   fast-flow onset settle; with the fix it matches the no-chain value.
4. **Noise floor 3e-2 and lag window 100 Hz** stop high voices' harmonics
   being modelled as formants, which had delayed a 300 Hz onset to 36 ms.
   This costs some envelope detail (sweep below).
5. **Auto formant mode (default):** preserve only when |transpose| ≥ 0.5 st.
   Hard-tune corrections (≤ ~50 c) move formants ≤ 3%, so the chain's
   small pitch cost isn't worth paying there. Preserve and Follow remain
   selectable.

**Results (steady notes, F1 at the 0 st baseline ≈ 735–755 Hz):**

| Transpose | Follow (no chain) F1 | Preserve F1, Gaussian voice @ 48 / 44.1 / 96 kHz | Preserve F1, resonator voice @ 48 kHz |
|---|---|---|---|
| −5 st | 554 Hz (−25%) | 751 / 749 / 734 | 757 |
| −3 st | 622 Hz (−16%) | 749 / 725 / 713 | 756 |
| +3 st | 880 Hz (+19%) | 750 / 751 / 769 | 770 |
| +5 st | 918 Hz (+24%) | 761 / 757 / 771 | 779 |

- **F1 stays within ≈ 4% everywhere** (FR-TUNE-6 requires < 5%), against
  16–25% without the chain. The tests check < 5% on both voice models at
  −3, +3 and +5 st.
- **Envelope error (LSD) vs the original vowel:** Preserve 5.6–7.0 dB
  against Follow 10–14 dB (Gaussian voice). With no noise floor or lag
  window it was 2.3–4.3 dB, but 300 Hz onsets then took up to 36 ms.
  Shifting *down* is inherently harder: new harmonics sample the envelope
  between the old ones, where sharp resonances can't be observed.
- **Pitch cost of Preserve mode** (Auto avoids it at transpose 0):
  - vibrato hard-tune: 2.3 / 5.3 c (median / p95), against 0.7 / 3.1 c
    without the chain;
  - vibrato onset settle: 8 / 13 ms, against 0 / 8 ms.

  Likely cause: the whitening filter's phase response near formants turns
  vibrato's frequency sweep into a small residual phase modulation (≈ 2–3
  c), below pitch JND. Other scenarios are unchanged. Pinned in tests.
- **CPU:** + ≈ 110 ns/sample (+20%) when the chain is active.
- **Not done yet:** Throat (warp the re-colour envelope) and an onset-aware
  analysis frame (the envelope needs ≈ 35 ms of the new note to fully
  update).

### Real-clip mode

`tune_eval --clips DIR` scores WAV folders without ground truth: how close
voiced frames sit to the scale before and after tuning. It was checked on
synthetic WAVs (input 16.5 → output 0.4 cents median for fast_flow).
Official numbers come only from the Corpus tier (master plan §10.1).

## Findings (fed back into the spec)

1. **Detector lag ruined hard-tune vibrato until we added a tracker.** On
   our signals, cycfi/q reports a new estimate about every two periods,
   then median-filters it. Hard-tune then left vibrato *worse* than the
   input (26.5 c median against 17.8 c in). A per-period normalised
   autocorrelation tracker took it to **0.8 c (33× better)** and fast flow
   from 5.2 to 0.2 c:
   - it searches ±4% around the last period every max(1.33 ms, ¼ period);
   - its window is capped at 256 samples;
   - q re-anchors it when they disagree by > 6%.

   q stays the anchor for onsets, note changes and octave safety.
2. **The voicing hold must cover q's update gap.** A 12 ms hold made
   voicing flicker mid-note at 110 Hz. Each flicker sent the read head
   back to rest with an unaligned jump. Fix: hold ≥ 3.5 periods, and
   return to rest only after 20 ms of continuous unvoiced input.
3. **Timing wander is inherent and needs a listening check.** While
   correcting, the read head's delay moves within [latency, latency + one
   input period]:
   - up to about 14 ms extra at 70 Hz, and 5 ms at 200 Hz;
   - it can't be compensated by the host (it isn't constant);
   - consonants re-align to the reported latency once 20 ms of unvoiced
     input has passed.

   Whether this is audible as timing smear on fast flows is the main open
   question for the listening session.
4. **The pitch ratio can change continuously.** The spec said ratio
   changes happen "only at splice boundaries". That isn't needed for a
   resampling shifter (a per-sample ratio is just playback speed), and
   doing it that way would add lag. Spec corrected.
5. **cycfi/infra licence verified:** every header says "Distributed under
   the MIT License" (the repo has no LICENSE file). This settles the open
   item from the DSP research.

## Limits of this result

- Synthetic voices only. Real rap vocals have breath, noise, creak, fry,
  distortion and room sound. **T1.6** (real vocal set) and **T1.8**
  (listening) decide the M1 go/no-go.
- `fast_flow` scores only 80 frames, because syllables are short relative
  to the 60 ms settle exclusion. An onset-settle metric (time to within
  10 c after onset) is needed (**T1.2**).
- Not yet implemented:
  - formant preservation (since added in v0.2, see above);
  - humanize, flex and vibrato controls;
  - HQ mode;
  - stereo;
  - RTSan (Clang 18 here predates it; it runs in the suite's CI with
    Clang 20+).

## Reproduce

```bash
cmake -S prototypes/tune -B prototypes/tune/build -G Ninja -DCMAKE_BUILD_TYPE=Release
```

```bash
cmake --build prototypes/tune/build && prototypes/tune/build/tune_tests
```

```bash
prototypes/tune/build/tune_eval --out prototypes/tune/out/eval48
```
