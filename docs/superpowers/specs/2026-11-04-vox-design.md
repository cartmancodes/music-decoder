# VOX: rap vocal chain plugin, design spec

**Date:** 2026-11-04
**Status:** draft for review (updated 2026-11-04 with tuner spike v0 findings)
**Parent:** [Rapper suite master plan](../plans/2026-11-04-rapper-suite-master-plan.md)
(§4 VOX; decisions D1 closed-commercial, D2 VOX first, D3 macOS + Windows)

Requirement IDs (`FR-`, `NFR-`) are referenced by the test plan (§15) and
the task plan that will follow this spec. Values marked **[tune]** are
starting points to be set by measurement or listening tests; values marked
**[UV]** are facts not yet verified against a primary source.

---

## 1. Intent

**Goal:** a single plugin that takes a raw rap vocal to a release-ready
sound while the artist is tracking, with pitch correction that **follows
the beat's key without the user knowing the key**.

**Success criteria (launch gate, month 8):**
1. Track-mode latency ≤ 3 ms reported at every sample rate (NFR-1).
2. Blind tests: VOX is not significantly worse than the reference chain
   (AutoTune 2026 + a stock chain) for correction quality, and is
   preferred or equal for the hard-tune effect (§15.4).
3. Auto-Key picks the correct scale (pitch-class set) on ≥ 92% of the rap
   beat test set (FR-KEY-6).
4. Passes pluginval strictness 10, `auval`, RTSan and the full host matrix
   (§15).

### 1.1 Non-goals for v1
Graphical note editing (Melodyne-style; also patent-sensitive, see master
plan §13), MIDI **out**, polyphonic correction, ARA, AUv3 / iOS, Windows on
ARM, Linux builds, cloud or AI features, built-in voice changers or voice
cloning.

### 1.2 Products from this codebase

| Build | Plugin ID | Contents |
|---|---|---|
| **VOX** | own code (TBD) | Everything in this spec |
| **VOX Lite** (free) | separate ID, so Lite sessions never "upgrade" silently | Tune in Track mode (manual key, Auto-Key, retune speed, major / minor scales only), one-knob **Polish** macro (de-ess + comp + EQ), output. No HQ mode, doubler, throws or presets beyond 3 |

Lite is a compile-time feature set (`VOX_LITE=1`), not a runtime unlock,
so a cracked Lite never contains the paid DSP.

---

## 2. Users and scenarios

| ID | Scenario | Must work |
|---|---|---|
| S1 | Rapper tracking in FL Studio, hearing hard-tune live through headphones at 128-sample buffer | Track mode, retune 0, ≤ 3 ms, no artifacts on fast flows |
| S2 | Same rapper doesn't know the beat's key | Drop beat file on VOX, or route the beat to the sidechain and press **Listen** → key set in < 5 s |
| S3 | Engineer mixing in Pro Tools, natural correction on a melodic hook | HQ mode, retune 20–60 ms, humanize, flex, formant preserved |
| S4 | Ad-libs on a separate track | "Ad-lib" preset: band-pass, saturation, throw delay |
| S5 | Producer in Ableton wants vocal throws on the last word of each bar | Throw = Auto (last word), tempo-synced to host |
| S6 | Logic user drives the target notes from a MIDI melody | MIDI input sets target notes (AU music effect, `aumf`) |
| S7 | Session reopened months later | Identical sound: state and preset recall, render parity |

---

## 3. Formats, hosts, platforms

- **Formats:** VST3, AU v2 (type **`aumf`** music effect so Logic can route
  MIDI to it **[UV: verify in Logic]**), CLAP (via clap-juce-extensions),
  AAX (after PACE approval), Standalone (for demos and testing, with
  audio-input monitoring).
- **Platforms (D3):** macOS 12+ universal binary (arm64 + x86_64); Windows
  10/11 x64.
- **Host matrix (each release):** FL Studio, Ableton Live, Logic Pro, Pro
  Tools, Reaper, Studio One, Cubase, all at their current versions on both
  OSes (Logic is macOS only).
- **Bus layouts:**

  | Main in | Main out | Sidechain (optional) |
  |---|---|---|
  | mono | mono | – / stereo |
  | mono | stereo (default; doubler, throws and width are stereo) | – / stereo |
  | stereo | stereo | – / stereo |

  With stereo input, detection, gain computation and splice positions use
  the mid signal and are applied identically to L and R, keeping the
  channels phase-coherent.
- **MIDI in:** target notes (FR-TUNE-8). **MIDI out:** v1.1.
- **Sample rates:** 44.1, 48, 88.2, 96, 176.4, 192 kHz. **Block sizes:**
  1–8192, variable and odd sizes (pluginval exercises these).

---

## 4. Signal architecture

```
               ┌───────────── sidechain (beat) ───► key capture buffer (Listen)
main in ─► trim ─► HPF ─► TUNE ─► DE-ESS ─► COMP1 ─► EQ ─► SAT ─► COMP2 ─┬─► DOUBLER ─┐
                                                                         │            ├─► WIDTH ─► OUT gain ─► (safety clip) ─► main out
                                                                         ├─► DELAY throw ┤
                                                                         └─► REVERB throw┘
                         dry (latency-matched) ──────────────────────────────► MIX ──────┘
```

- **Order:** fixed in v1 except that **SAT can be placed pre- or
  post-COMP1** (`sat.position`). Reordering arbitrary blocks is a v1.1
  candidate.
- **Processing core** is mono up to SAT; DOUBLER, throws and WIDTH make
  stereo.
- **Global mix** (`out.mix`, 0–100%) blends the latency-matched dry signal
  with the processed one.
- **Internal sub-block:** up to 32 samples, for parameter smoothing and
  splice scheduling. The host block is split as needed; behaviour is
  identical for any host block size (NFR-6).
- **Oversampling** is used **only inside SAT** (2× polyphase IIR,
  minimum-phase), as an isolated region. Its latency counts toward the mode
  budget.

---

## 5. Latency model

| Mode | Reported latency (fixed) | Composition at 48 kHz |
|---|---|---|
| **Track** | **2.5 ms** = `round(0.0025 × fs)` (120 samples at 48 kHz) | shifter fragment 64 + zero-crossing lookahead 32 + SAT oversampling (≈ 2–4 **[tune]**) + padding to 120 |
| **HQ** | **20 ms** = `round(0.020 × fs)` (960 samples) | PSOLA lookahead (1 period at 70 Hz ≈ 14.3 ms) + de-esser lookahead 1 ms + SAT + padding |

Rules:
- **FR-LAT-1:** reported latency depends only on mode and sample rate,
  never on which blocks are on or bypassed. Bypassed blocks are replaced by
  delay so the timing never moves.
- **FR-LAT-2:** host bypass (`getBypassParameter`) outputs the dry signal
  delayed by the current reported latency.
- **FR-LAT-3:** changing mode calls `setLatencySamples` from the message
  thread and notifies the host. The UI warns "Host re-syncs timing; change
  modes while stopped".
- **FR-LAT-4:** no quality switch for offline render (`isNonRealtime()`).
  Render parity (NFR-7) wins over extra quality.
- Pitch-detection lag (about 2 periods at note onsets) is **not** latency.
  It affects how fast correction engages, and is hidden by the detector's
  attack prediction (§6.1).
- **Timing wander (spike finding 3):** while correcting, the shifter's
  read-head delay moves within [reported latency, reported latency + one
  input pitch period]. That's up to about 14 ms extra at 70 Hz and 5 ms at
  200 Hz. It's inherent to time-domain shifting and not compensable by the
  host. Unvoiced input re-aligns to the reported latency after 20 ms.
  Audibility on fast flows is a gate criterion of the M1 listening session
  (plan T1.8).

---

## 6. TUNE

### 6.1 Pitch detection
- **Algorithm:** bitstream autocorrelation (BACF) from **cycfi/q (MIT)**,
  with its periodicity confidence and `predict_frequency()` for early
  estimates during attacks. q reports a new estimate roughly every 2
  periods and median-filters it, which is too laggy for hard-tune on
  vibrato (spike finding 1). So:
- **Period tracker** between q's estimates:
  - normalised autocorrelation, searched ±4% around the last period every
    max(1.33 ms, ¼ period), window capped at 256 samples;
  - q re-anchors it when they differ by > 6% (note changes, octave
    safety).

  In the spike this cut hard-tune vibrato error from 26.5 to 0.8 cents
  median.
- **Voicing hold** ≥ max(12 ms, 3.5 periods), so voicing doesn't flicker
  in q's update gaps (spike finding 2).
- **Onset handling (spike v0.1):**
  - **Voicing end:** voicing ends when the tracker's correlation is < 0.5
    for 2 updates (≈ 2.7 ms), and q is reset, so consonant gaps break
    voicing.
  - **Acquisition:** a decimated (≈ 12 kHz) YIN search runs over all lags
    while unvoiced and above the gate, refined at full rate; two agreeing
    candidates start voicing.
  - **Checks on q:** its proposals from unvoiced need NCCF ≥ 0.8. While
    voiced, it re-anchors only if its period explains the signal at least
    as well as the current track.
  - `predict_frequency()` was measured and is **not** used: it's slower
    than q's regular estimate on our signals.
  - Result: hard-tune settle median 0–12 ms, p90 6–14 ms across scenarios
    (spike report).
- **Range:** set by **Voice** (`tune.voice`): Bass 55–600 Hz, Low 70–800
  Hz, Mid 90–1000 Hz, High 120–1400 Hz. Default Low **[tune]**.
- **Voicing gate:** voiced when confidence ≥ `θc` (0.8 **[tune]**) and
  spectral flatness of the last 5 ms < `θf` **[tune]**. Unvoiced frames
  (sibilants, breaths, plosives) are never corrected; the shift ratio
  glides back to 1.0 over 5 ms.
- **Octave guard:** reject an estimate that jumps by about ×2 or ×0.5
  versus the 20 ms median unless it holds for ≥ 3 periods; a 3-point median
  over period estimates.
- **FR-TUNE-1:** octave errors ≤ 0.5% of voiced frames on the rap test
  corpus.
- **FR-TUNE-2:** gross pitch error (> 50 cents) ≤ 2% of voiced frames after
  the onset settle time.

### 6.2 Target note decision
- **Pitch set:** from **Key** (tonic, 12 values) + **Scale**: Major,
  Natural minor, Harmonic minor, Melodic minor, Major pentatonic, Minor
  pentatonic, Blues, Chromatic, Custom (12 note toggles).
- **Per-note "remove"** toggles (exclude a scale degree) and **"bypass"**
  toggles (pass that pitch class through uncorrected), like the
  established workflow.
- **Quantizer:** nearest allowed pitch in cents, with **hysteresis** of
  ±`h` cents (15 **[tune]**) beyond the decision boundary so notes near a
  boundary don't flip back and forth.
- **FR-TUNE-3:** no target flipping within one sung note for input
  wandering ±40 cents around a boundary (test with synthetic glides).

### 6.3 Correction curve (all in cents)
`correction(t) = amount × shape(target − detected)`, smoothed:

| Control | ID | Range | Definition |
|---|---|---|---|
| Retune speed | `tune.retune_ms` | 0–400 ms, default 20 | Time constant of a critically damped 2nd-order smoother on the correction. **0 = step change** (hard-tune) |
| Humanize | `tune.humanize` | 0–100%, default 0 | Retune time grows with how long the current note has been held: `τ_eff = τ + humanize · 120 ms · min(held/300 ms, 1)`. Additive, so it also works at retune 0. The clock restarts on every new note, so short syllables stay snappy (spike v0.3) |
| Flex | `tune.flex` | 0–100%, default 0 (off) | Correction is weighted by the distance d between the note centre and the target: window W = 100 c at flex → 0 down to 15 c at flex 1; full correction for d ≤ W/2, fading to none at d ≥ W. Scoops and slides pass through; correction engages as the voice approaches the note. High flex corrects only notes already close |
| Vibrato | `tune.vibrato` | 0–200%, default **0** | Detected pitch is split into a centre (2 Hz 2nd-order low-pass, reset on each new note) and the rest. Output pitch → target + gain × rest. 0 = classic flat hard-tune, 100% = the singer's own vibrato kept around the corrected note, 200% = exaggerated. The "rest" also holds the singer's faster inflections, not only 4–8 Hz vibrato; restricting it to a band-pass is an open option (spike v0.3) |
| Amount | `tune.amount` | 0–100% | Scales the final correction |
| Transpose | `tune.transpose` | −12…+12 st | Added after correction (formant follows `tune.formant_follow`) |

- **FR-TUNE-4:** with retune 0, the output pitch after the onset settle
  time stays within ±3 cents of the target on a steady synthetic tone.
- **FR-TUNE-5:** with retune 0, the step between two target notes completes
  within one splice (≤ 1.33 ms at 48 kHz in Track).

### 6.4 Shifter, Track mode
- **Method:** period-synchronous resampling. Audio is read from a short
  ring buffer at rate `r = f_target / f_detected`. When the read pointer
  drifts by about a period, it jumps ±1 period at a point of low waveform
  difference (search within the 32-sample lookahead), with a 64-sample
  equal-power crossfade.
- **Clean-room rule:** zita-AT1 / x42-autotune (GPL) is known to use this
  family of methods. **Nobody on the DSP team reads that code**; work only
  from public descriptions and papers. The design record lives in
  `docs/design/tune-shifter.md` in the suite repo.
- **Ratio updates** are continuous (per sample). For a resampling shifter
  the ratio is just playback speed, so restricting changes to splice
  boundaries would only add lag (spike finding 4). Splices happen when the
  delay leaves [latency, latency + period]. The jump distance is refined by
  a short correlation search (±3% of the period, 0.67 ms window) and
  crossfaded over 1.33 ms.
- **Prototype:** `prototypes/tune` (reported latency 2.0 ms; results in
  `docs/reports/2026-11-04-tune-spike-v0.md`).
- **Range:** optimised for corrections within ±3 st; transpose up to ±12 st
  is allowed, with lower quality accepted.

### 6.5 Formant handling (Track and HQ)
Updated by tuner spike v0.2; measurements in the spike report.
- **Method:** frequency-warped LPC.
  - all-pass warping λ = Bark-like for the sample rate (0.756 at 48 kHz),
    order 20;
  - analysis on the **pre-shift** input: 20 ms Hann window, **10 ms hop**,
    past samples only (no added latency);
  - noise floor 3e-2 and lag window 100 Hz, so harmonics of high voices
    aren't modelled as formants;
  - reflection coefficients interpolated between frames (stable), converted
    to direct form every 16 samples.
- **Chain:** whiten (warped FIR A(D(z))) → shift → re-colour (warped
  all-pole 1/A(D(z)), delay-free loop solved in closed form) using the
  coefficients of the input moment being read. The pair is an exact inverse
  even with time-varying coefficients, so latency and the ratio-1 null test
  are unaffected.
- **Splice points** are chosen on the original waveform, not the whitened
  residual.
- **Mode** (`tune.formant_mode`):
  - **Auto (default):** preserve when |transpose| ≥ 0.5 st, otherwise
    don't. Hard-tune corrections move formants ≤ 3%, and Preserve costs
    ≈ 1.6 c extra vibrato residual and ≈ 8 ms onset settle on vibrato
    notes;
  - **Preserve;**
  - **Follow pitch.**
- **Throat** (`tune.throat`, −100…+100 → ±3 st of envelope warp): planned
  as a different warping factor or envelope warp at re-colour (not yet
  built).
- **FR-TUNE-6:** with Preserve active and a ±3 st (and +5 st) shift, F1
  moves < 5% compared with the unshifted output. Spike: ≤ ≈ 4% at 44.1 /
  48 / 96 kHz on both synthetic voice models; Follow moves it 16–25%.

### 6.6 Shifter, HQ mode
- **TD-PSOLA:** pitch marks from detector periods refined by peak picking;
  two-period Hann grains; synthesis marks spaced at the target period;
  lookahead of one period at the 70 Hz floor (Voice = Bass is clamped to 70
  Hz in HQ).
- Uses the same LPC formant chain for Throat; PSOLA itself already
  preserves formants.
- Unvoiced: grains copied through at ratio 1.

### 6.7 Key sources (`tune.key_source`)

| Source | Behaviour |
|---|---|
| **Manual** | User picks tonic + scale |
| **Auto** (default) | Result of Listen or file drop (§7); user can override |
| **MIDI** | Held MIDI notes define the allowed pitch set (chord mode) or the exact target (target mode, `tune.midi_mode`) |

- **FR-TUNE-7:** key and scale changes take effect at the next note
  decision, not mid-note.
- **FR-TUNE-8:** MIDI target mode: the target pitch equals the held MIDI
  note; with no notes held, fall back to the scale.

---

## 7. Auto-Key

### 7.1 Inputs
1. **Listen:** the sidechain (beat) is captured into a pre-allocated
   30 s ring buffer **[tune]** while Listen is armed. The audio thread only
   copies samples; no allocation happens on it.
2. **Drop a beat file** onto the key strip (WAV / AIFF / FLAC / MP3),
   decoded on a worker thread with JUCE audio formats. This works in every
   host without sidechain routing, so it's the primary path in FL Studio,
   where routing is fiddly.
3. **Booth / rapd** (later): the key arrives over IPC.

### 7.2 Algorithm (C++ port of our Python reference)
1. Mono, resample to 22.05 kHz (r8brain), STFT 4096 / hop 1024.
2. Percussion suppression: HPSS median filtering (17-frame time ×
   17-bin frequency **[tune]**), keeping the harmonic part.
3. Log-frequency mapping to 12 pitch classes from 55 Hz to 5 kHz, with
   harmonic down-weighting; mean chroma over the analysed span (the
   loudest 20 s if longer).
4. Correlate against Krumhansl-Kessler and Temperley profiles for all 24
   keys; z-normalise each and sum (the `fusion` approach without the
   madmom CNN cue, which is non-commercial).
5. Output the top-3 keys with confidence and the **scale pitch-class set**.
   Relative major / minor share a pitch set, so a confusion between them
   doesn't change tuning. The UI still shows the more likely tonic.

- **Reference implementation:** `music_decoder` backend
  `key_detection.backend: fusion_nocnn` (master plan §17 task 2). The C++
  port is validated by a parity test (FR-KEY-5).

### 7.3 Requirements
- **FR-KEY-1:** analysis runs only on a worker thread, never blocking audio
  or UI; takes < 3 s for 30 s of audio on the reference CPUs.
- **FR-KEY-2:** the result applies only when the user confirms it ("Use C♯
  minor?"). Auto-apply is an option, off by default.
- **FR-KEY-3:** the result, confidence, source and timestamp are stored in
  plugin state (S7).
- **FR-KEY-4:** if confidence is below a threshold **[tune]**, show the
  top 3 and don't auto-apply.
- **FR-KEY-5:** the C++ port agrees with the Python reference top-1 key on
  ≥ 98% of 200 parity files.
- **FR-KEY-6:** scale pitch-class set correct on ≥ 92% of the **rap beat
  key set** (master plan §10); exact key ≥ 85% (MIREX-weighted). If missed,
  add a chord-fit cue (port of our template-HMM chords) before considering
  an own small CNN.

---

## 8. Chain blocks

All filters are trapezoidal SVF (implemented from Simper's derivation) or
RBJ cookbook designs. All gain changes are smoothed (20 ms, multiplicative
**[tune]**).

### 8.1 Input and HPF
`in.trim` −24…+24 dB; `hpf.freq` 20–400 Hz (default 80), `hpf.slope` 12 /
24 dB per octave.

### 8.2 DE-ESS
- Detector: band-pass `deess.freq` 4–10 kHz (default 6.5), Q fixed **[tune]**,
  RMS 1 ms attack / 40 ms release **[tune]**.
- **Split-band** (default): dynamic bell / high-shelf reduction above the
  detector band; **Wideband**: full-band gain reduction.
- `deess.threshold` −60…0 dB, `deess.range` 0–20 dB (max reduction),
  `deess.listen` (solo the detector band).
- Lookahead 1 ms in HQ only (counted in the HQ budget).

### 8.3 COMP1 (fast, FET-style character)
Feed-forward, log-domain gain computer with soft knee. `comp1.threshold`
−60…0 dB, `comp1.ratio` {2, 4, 8, 12, 20, All}, `comp1.attack` 0.02–10 ms,
`comp1.release` 20–1000 ms, `comp1.knee` 0–12 dB, `comp1.makeup` auto /
manual, `comp1.mix` 0–100% (parallel compression). Character: subtle
asymmetric saturation in the gain stage **[tune]**.

### 8.4 EQ
Low shelf (40–400 Hz), Bell 1 (100 Hz–2 kHz), Bell 2 (1–8 kHz), High shelf
"Air" (6–20 kHz), each ±18 dB (Q 0.3–6 on the bells), plus **Tilt** (±6 dB
around 1 kHz). Optional cramping compensation above fs/4 (matched
bilinear).

### 8.5 SAT
- Modes: Tape (soft, asymmetric), Tube (even-heavy), Hard (clip), all via
  **first-order ADAA** waveshapers inside **2× polyphase IIR
  oversampling**. `sat.drive` 0–24 dB, `sat.tone` (tilt after saturation),
  `sat.mix` 0–100%, `sat.position` pre / post COMP1, auto gain-compensation.
- **NFR-9:** aliasing products < −90 dBFS for a full-scale 20 Hz–20 kHz
  sine sweep at drive 12 dB, 48 kHz (master plan §2).

### 8.6 COMP2 (glue, opto-style)
Program-dependent release (fast-then-slow, two time constants). Parameters
`comp2.threshold`, `comp2.ratio` (1.5–4), `comp2.release_char`, `comp2.mix`.

### 8.7 DOUBLER
2 or 4 voices (`dbl.voices`); per-voice modulated delay 8–35 ms
(`dbl.time`), detune 0–20 cents (`dbl.detune`, via delay modulation, LFO
0.1–0.6 Hz with random drift), spread 0–100%, level, high-pass on doubles.
Uses signalsmith-dsp delay lines (MIT). Mode "Tight" (no detune, 8–12 ms)
for stacked ad-libs.

### 8.8 Delay throw
Tempo-synced to host BPM (fallback: manual BPM): 1/4, 1/8, 1/8 dotted,
1/4 triplet, 1/16 or ms. Controls: feedback 0–95%, HP / LP filters in the
loop, ping-pong, **ducking** (the dry signal sidechains the wet), level.
**Throw trigger** (`thr.delay_mode`):
- **Always**;
- **Button**: momentary `thr.delay_send` parameter, automatable;
- **Auto (last word)**: a phrase end (voiced → silence > `gap` ms, default
  120 **[tune]**) opens the send for the last 300 ms of voiced audio, which
  is held in a ring buffer.

### 8.9 Reverb throw
8-line FDN with early reflections, based on Signalsmith Basics Reverb
(MIT). Size, decay 0.3–8 s, pre-delay 0–200 ms, damping, width, level;
same throw modes as the delay.

### 8.10 WIDTH and OUT
M/S width 0–200% (mono-compatible, bass below 150 Hz kept mono),
`out.gain` −24…+12 dB, `out.mix`, **Auto gain-match** (A/B fairness:
matches short-term loudness of output to input when on), optional **safety
soft clip** at 0 dBFS (no lookahead, off by default).

### 8.11 Metering (UI only)
Input / output peak + RMS, per-dynamics gain reduction, LUFS-short
(libebur128 in histogram mode on a **meter thread**, never the audio
thread). Tune display data: see §11.3.

---

## 9. Parameters

- IDs are **permanent** once a version ships (JUCE `ParameterID{id,
  version}`). The table above (§6, §8) is the source; the full list
  (≈ 75 parameters) is generated into `params.json` and the TypeScript
  types from one C++ definition file, so UI and DSP can't drift apart.
- Every continuous parameter is automatable and smoothed; enums switch at
  sub-block boundaries with a 5 ms crossfade where audible (e.g.
  `comp1.ratio`).
- **Macros:**
  - **Polish** (Lite and full): de-ess threshold + COMP1 threshold + EQ
    Air, mapped by curves **[tune]**.
  - **Tune character**: retune + humanize + flex.
- **NFR-10:** automating any parameter at audio rate produces no zipper
  noise or clicks (> −80 dBFS difference from a smoothed reference).

---

## 10. State, presets, A/B

- **State:** APVTS tree + `vox` metadata (schema version, mode, key result,
  UI size, preset name, A/B slots) serialised as XML inside the host chunk.
  - **Migrations:** `migrate_vN_to_vN+1` functions with tests. Unknown
    parameters are ignored and missing ones take defaults.
  - **FR-STATE-1:** a session saved by version N loads in N+k with an
    identical render (NFR-7).
- **Presets:** JSON
  `{schema, product, version, name, author, tags[], style, params{id: value}}`.
  - Factory presets are embedded and read-only. User presets live in
    `~/Documents/Rapsuite/VOX/Presets` and
    `%USERPROFILE%\Documents\Rapsuite\VOX\Presets`.
  - Import and export by file drag.
  - A preset loads **without changing the key** unless "Load key with
    preset" is on.
- **Factory set:** the 8 presets in master plan §4.4, plus "Init".
- **A/B:** two complete parameter states, plus copy A→B.
- **Undo / redo:** 100 parameter gestures (UndoManager on APVTS).

---

## 11. UI

### 11.1 Layout
Default 1100 × 700 logical px, resizable 880 × 560 to 2200 × 1400, HiDPI.
Three zones:
1. **Tune strip:**
   - key / scale selector, plus the big **Auto-Key** area ("Drop beat or
     press Listen");
   - retune and humanize knobs, mode switch (Track / HQ) with the latency
     readout;
   - live **pitch trace** (detected vs target over the last 4 s, with note
     lanes).
2. **Chain strip:** one tile per block, in signal order, with an on/off
   toggle, one macro knob and a gain-reduction or activity light. Clicking
   a tile expands its detail panel.
3. **Output:** meters, width, mix, output gain, A/B, preset browser.

### 11.2 Technology
- React + TypeScript, built with Vite into a resource bundle served by
  JUCE's `WebBrowserComponent` resource provider (no network).
- `@juce-framework/webview` npm package for the bridge, with
  `WebSliderRelay` / `WebToggleButtonRelay` / `WebComboBoxRelay` bound to
  APVTS.
- Shared `ui-kit` package (knobs, meters, pitch display, preset browser).
- **Windows:** WebView2 statically linked
  (`JUCE_USE_WIN_WEBVIEW2_WITH_STATIC_LINKING`), with
  `withUserDataFolder()` pointing to a writable per-user folder.
- **FR-UI-1 (fallback):** if WebView2 can't start, show a native JUCE
  fallback UI with key, scale, retune, mode, mix, output and preset
  selection, so the plugin is never unusable.

### 11.3 Data flow
- The audio thread pushes tune frames `(t, f0_in, f0_target, confidence,
  voiced)` at 200 Hz, plus meter frames, into lock-free SPSC FIFOs
  (farbot).
- The message thread drains them and emits batched events to JS at 60 Hz
  (tune) and 30 Hz (meters), only while the editor is visible.
- **FR-UI-2:** opening or closing the editor never affects audio (no
  allocation or locks shared with the audio thread).

### 11.4 Keyboard focus
- **FR-UI-3:** after any mouse interaction outside a text field, the
  plugin returns keyboard focus to the host. Spacebar must toggle the host
  transport in every host in the matrix; this is an explicit test case.
- Only text inputs (preset name, search) take focus, and they release it
  on Enter, Escape or blur.

### 11.5 Accessibility and copy
- Keyboard navigation inside the plugin when focused, visible focus rings,
  4.5:1 contrast, tooltips on every control, screen-reader labels.
- The UI never uses the word "Auto-Tune".

---

## 12. Performance and real-time safety

| ID | Requirement |
|---|---|
| **NFR-1** | Track-mode latency ≤ 3 ms at every supported sample rate (target 2.5 ms) |
| NFR-2 | CPU, all blocks on, Track mode, 48 kHz / 64-sample buffer: ≤ 3% of one core on Apple M1 (P-core) and on a Ryzen 5 5600 / Core i5-12400 class CPU; HQ ≤ 5% **[tune]** |
| NFR-3 | 16 instances at 48 kHz / 64 samples run without dropouts on the reference machines |
| NFR-4 | Memory ≤ 50 MB native heap per instance (WebView runtime excluded) |
| NFR-5 | Plugin instantiate ≤ 200 ms; state restore ≤ 50 ms; editor open ≤ 700 ms cold, ≤ 300 ms warm |
| NFR-6 | Output identical for any host block size (bit-exact across block sizes 1, 7, 64, 512, 4096 in the test harness) |
| NFR-7 | Offline render nulls against real-time playback (< −120 dBFS) |
| NFR-8 | No allocation, lock, syscall, logging or I/O on the audio thread: RTSan clean (macOS / Linux test builds), pluginval `--rtcheck` clean |
| NFR-9 | Aliasing (§8.5) |
| NFR-10 | Automation without zipper noise (§9) |
| NFR-11 | Denormal-safe (FTZ / DAZ + `ScopedNoDenormals`); silence in → silence out within 10 s, CPU unchanged on silence |
| NFR-12 | No NaN / Inf ever leaves the plugin: a guard detects them, resets the DSP state and logs once to the crash or diagnostics channel |

Coding rules: all buffers allocated in `prepareToPlay`; SPSC FIFOs for
cross-thread data; `std::atomic` parameters; no `std::function`
allocation, exceptions or virtual dispatch in the inner loops; SIMD
via JUCE `dsp::SIMDRegister` or xsimd where profiling shows a need.

---

## 13. Licensing, activation, privacy

- **Store / licences:** Moonbase (merchant of record).
  - **14-day full trial** with no account needed to start.
  - Activation online, or offline by file; licence stored as a signed token
    (Ed25519), revalidated every 30 days with a 30-day grace period.
- **Unlicensed or expired:** the plugin **passes audio through
  unprocessed**, with a banner. It never injects noise or silence into
  users' sessions.
- **VOX Lite:** free, no account; optional email for updates.
- **Privacy:** no audio ever leaves the machine. Crash reports via
  sentry-native / Crashpad are **opt-in** on first launch, filtered to our
  modules, and never use process-wide crash handlers inside hosts. There
  is no other telemetry in the plugin.

---

## 14. Code structure (in the `rapsuite` monorepo)

```
libs/rapcore/
  dsp/       svf.h, biquad.h, envelope.h, compressor.h, deesser.h, saturator_adaa.h,
             oversampling.h, delayline.h, fdn_reverb.h, smoother.h, lpc.h, lsf.h
  pitch/     bacf_detector.h (wraps cycfi/q), voicing.h, note_decider.h,
             correction_curve.h, splice_shifter.h, psola_shifter.h, formant.h
  theory/    scales.h, key_estimator.h (KK+Temperley fusion), chroma.h, hpss.h
  rt/        spsc_fifo.h (farbot), nan_guard.h, denormals.h
  licensing/ token.h, moonbase_client.h
plugins/vox/
  VoxProcessor.{h,cpp}   bus layouts, latency, state, parameter tree
  VoxEngine.{h,cpp}      block graph, sub-block scheduler, latency padding
  VoxEditor.{h,cpp}      WebView host, relays, focus handling, native fallback
  params/params.def      single source → C++ APVTS layout + params.json + TS types
  presets/factory/*.json
ui/vox/                  React app (Vite)
tests/vox/               unit, golden, measurement, parity
```

Third-party (all compatible with D1): cycfi/q (MIT), signalsmith-dsp and
Basics (MIT), r8brain (MIT), libebur128 (MIT), farbot (MIT), JUCE
(commercial), clap-juce-extensions (MIT), moonbase-cpp (MIT),
sentry-native (MIT). CI licence scan blocks anything else.

---

## 15. Verification

### 15.1 Automated (every PR)

| Test | Covers |
|---|---|
| Unit (Catch2): SVF / biquad responses vs analytic; compressor static curve; de-esser band; LPC → LSF round trip; scale quantizer and hysteresis; correction-curve step responses; key estimator on synthetic chroma | FR-TUNE-3/4/5, FR-KEY-* |
| Synthetic tuner tests: detuned sine / sawtooth / formant-synth vowels with known f0 paths (steps, glides, vibrato) → residual error, settle time, octave errors | FR-TUNE-1/2/4/5/6 |
| Golden renders: 20 reference vocals × factory presets × {44.1, 48, 96 kHz} → null vs stored WAV (per-preset threshold) | Regression, NFR-7 |
| Block-size invariance: same input at block sizes 1, 7, 64, 512, 4096, variable | NFR-6 |
| Latency audit: impulse through every mode × SR × bypass combo; measured delay == reported | FR-LAT-1/2, NFR-1 |
| Aliasing sweep, NaN / denormal torture (silence, DC, full-scale noise, Inf injection) | NFR-9/11/12 |
| State: save → load → render equality; migration tests for every schema bump | FR-STATE-1 |
| C++ vs Python key parity (200 files) | FR-KEY-5 |
| RTSan build (macOS / Linux) running the render harness | NFR-8 |
| pluginval strictness 10 (+ `--rtcheck`), Steinberg validator, `auval -v aumf` | Format compliance |
| Licence scan | D1 |

### 15.2 Nightly
- CPU benchmarks (NFR-2/3) on self-hosted M1 and Windows reference boxes.
- Memory and instantiate / open timings (NFR-4/5).
- Corpus accuracy (Auto-Key FR-KEY-6; tuner metrics on the rap corpus).

### 15.3 Manual host matrix (each beta / RC)
For each of the 7 hosts on each OS, check: scan / load; mono and stereo
tracks; automation record and playback; state recall after restart;
latency compensation (loopback null); sample-rate change mid-session;
offline bounce parity; **spacebar after knob click**; sidechain Listen
routing; beat-file drop; MIDI target mode (Logic `aumf`, Ableton, FL,
Reaper); Lite and full side by side; trial expiry behaviour.

### 15.4 Listening tests
- **Correction quality (MUSHRA, ITU-R BS.1534-3):**
  - Stimuli: 12 in-tune rap / melodic vocal excerpts from the corpus,
    **detuned by known synthetic curves**. The original is the true
    hidden reference.
  - Conditions: VOX Track, VOX HQ, the competitor at matched settings, and
    a low anchor.
  - ≥ 12 listeners (engineers + rappers).
  - Pass: VOX Track / HQ not significantly worse than the competitor
    (p < 0.05).
- **Hard-tune character (paired preference):** retune 0 on 12 melodic-rap
  excerpts, VOX vs the competitor, ≥ 20 rappers and producers. Pass: VOX
  preferred or no significant difference.
- **Chain (ABX + preference):** factory presets vs an engineer-built
  reference chain on 10 raw vocals.
- Run #1 at beta (month 6), #2 at RC (month 7.5).

### 15.5 Objective tuner metrics (rap corpus, detuned set)
Median residual error ≤ 5 cents on voiced frames after settle; octave
errors ≤ 0.5%; log-spectral distance to the original (in-tune) take ≤ the
competitor's + 10% **[tune]**.

---

## 16. Milestones (months from kickoff)

| ID | When | Deliverable | Exit |
|---|---|---|---|
| M0 | 0–1.5 | Suite monorepo, CI with signing, `hello-gain` passes host matrix (master plan P0) | Green CI on macOS + Windows |
| **M1 Tuner spike** | wk 3–8 | Offline C++ test bed: BACF + splice shifter + LPC formant; renders + objective metrics on the detuned set; first internal listening | **Go/no-go on the ≤ 3 ms target.** Fallback: 5 ms budget with longer crossfades, then short-lookahead PSOLA |
| M2 DSP alpha | month 4 | All blocks real-time in the plugin; parameters v0 frozen; golden renders; RTSan clean | §15.1 green |
| M3 Feature complete | month 5.5 | React UI, Auto-Key (C++ port + parity), presets, A/B, MIDI targets, native fallback UI, Lite build | Host matrix pass on 3 hosts |
| M4 Beta | month 6 | Moonbase licensing, installers, notarization / signing, crash reporting; 50–100 beta users; MUSHRA #1 | No P1 bugs open for 1 week |
| M5 RC | month 7.5 | AAX signed, full host matrix, MUSHRA #2, FTO search done, trademark cleared | §1 success criteria met |
| Launch | month 8 | VOX + VOX Lite | – |

### 16.1 Work breakdown (engineer-weeks, rough)

| Epic | DSP | Plugin / app | Front end | ML / R&D | QA |
|---|---|---|---|---|---|
| Tuner (detector, decision, curve, Track + HQ shifters, formant) | 12 | 1 | – | 1 (corpus metrics) | 2 |
| Chain blocks (HPF, de-ess, comps, EQ, SAT, doubler, throws, width) | 8 | 1 | – | – | 1 |
| Auto-Key (Python `fusion_nocnn`, C++ port, parity, file decode) | 2 | 1 | – | 3 | 1 |
| Processor shell (buses, latency, state, presets, params codegen, MIDI) | 1 | 5 | – | – | 1 |
| UI (ui-kit, VOX app, relays, pitch trace, focus, fallback) | – | 2 | 10 | – | 1 |
| Licensing, installers, signing, crash reporting | – | 3 | 1 | – | 1 |
| Test infrastructure (golden, latency, sweeps, RTSan, benchmarks) | 2 | 3 | – | 1 | 2 |
| Listening tests + host matrix + beta | – | 1 | – | 1 | 6 |
| **Total ≈ 74** | 25 | 17 | 11 | 6 | 15 |

The window from M0 to launch is about 28 weeks. Front end (11), plugin
(17), ML (6) and QA (15) fit comfortably. **DSP is the bottleneck: 25
weeks of work for one engineer leaves almost no contingency, and the tuner
is the critical path.** Mitigation: hire a contract DSP engineer for the
chain blocks (8 weeks) from month 2, so the lead DSP engineer stays on the
tuner. With that, about 20% contingency remains.

---

## 17. Risks and open questions

| Risk / question | Plan |
|---|---|
| Splice shifter artifacts on fast, breathy rap at ≤ 3 ms | M1 go/no-go; fallback ladder in M1; HQ mode for mixing |
| Detector lag (~2 periods) audible on low voices | `predict_frequency`, the Voice setting, and listening tests on low male voices |
| Auto-Key fails on sparse beats (808 + hats only) | 808 pitch is strong evidence; weight sub-bass chroma **[tune]**; top-3 UI; manual override |
| WebView focus bugs in some hosts | Native fallback UI; FR-UI-3 test from M0 with `hello-gain` |
| Logic `aumf` MIDI routing behaviour | Verify in M0 with a stub plugin |
| Competitor quality bar keeps rising (AutoTune Advanced < 1 ms) | Differentiate on Auto-Key + rap chain + price; keep Track mode improving after launch |
| Patents on vibrato or humanize behaviours | FTO search scope includes §6.3 behaviours; design-around options documented |
| Q: should Track mode allow a 0-latency "monitor" variant (detection only, no lookahead)? | Decide after M1 measurements |
| Q: user-reorderable chain in v1? | Default no (v1.1); revisit after beta feedback |

---

## 18. References

Master plan §4, §11–13 and its source list. Key items for this spec:
[cycfi/q pitch](https://github.com/cycfi/q/blob/develop/docs/modules/ROOT/pages/reference/pitch.adoc),
[sevagh/pitch-detection](https://github.com/sevagh/pitch-detection),
[Signalsmith DSP](https://github.com/Signalsmith-Audio/dsp),
[Signalsmith Basics](https://github.com/Signalsmith-Audio/basics),
[r8brain](https://github.com/avaneev/r8brain-free-src),
[libebur128](https://github.com/jiixyj/libebur128),
[farbot](https://github.com/hogliux/farbot),
[rtsan](https://github.com/realtime-sanitizer/rtsan),
[pluginval](https://github.com/Tracktion/pluginval),
[clap-juce-extensions](https://github.com/free-audio/clap-juce-extensions),
[JUCE change list](https://github.com/juce-framework/JUCE/blob/master/CHANGE_LIST.md),
[WebView focus issue](https://forum.juce.com/t/fixing-webview-keyboard-focus-issue/63987),
[Moonbase](https://moonbase.sh/docs/fees),
[sentry-native](https://github.com/getsentry/sentry-native),
[ADAA (Vicanek)](https://www.vicanek.de/articles/AADistortion.pdf),
[Auto-Tune manual (feature semantics)](https://www.barryrudolph.com/recall/manuals/auto-tune_artist.pdf),
[AutoTune Advanced latency claim](https://www.mixonline.com/technology/news-products/autotune-advanced-pitch-correction-plug-in-debuts),
[US 5,973,252 (expired)](https://patents.google.com/patent/US5973252),
ITU-R BS.1534-3 (MUSHRA).
