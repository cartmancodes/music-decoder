# Rapper Production Suite: Master Plan

Date: 2026-11-04 · Status: proposed · Owner: TBD

Builds on (and where they disagree, overrides):
[rapper-production-tool](../../research/2026-11-04-rapper-production-tool.md) →
[production-quality bar](../../research/2026-11-04-rapper-tool-production-quality.md).
Research inputs: five parallel research tracks run on 2026-11-04 (real-time
DSP, plugin release engineering, desktop ML deployment, beats and flow,
market and legal). Facts marked **[UV]** could not be checked against a
primary source and must be confirmed before anyone relies on them.

---

## 0. Summary

**What we build:** a rap-specific production suite that runs on the
artist's own machine:

| Product | What it is | Ships |
|---|---|---|
| **VOX** | Rap vocal chain plugin: ≤ 3 ms tuning that follows the beat's key, de-esser, compressor, EQ, saturation, doubler, throws, presets | Month 8 |
| **VOX Lite** | Free hard-tune + one-knob chain (lead magnet) | Month 8 |
| **BEAT** | Drum + 808 generator: style rules → MIDI, built-in sampler with commissioned kits, drag-to-DAW | Month 11 |
| **Booth** | Standalone recording app for rappers without a DAW: beat in, record, comp takes, VOX/BEAT/MASTER built in | Month 15 |
| **Pen & Flow** | Rhyme notebook + flow analysis (lyrics aligned to the beat grid, pocket metrics); inside Booth | Month 15 |
| **MASTER** | Loudness-targeted mastering chain with true-peak limiter and reference matching | Month 17 |

**How:** JUCE 9 C++ plugins (VST3 / AU / CLAP / AAX / Standalone) with
React UIs in JUCE's WebView, plus **one C++ ML helper process** (ONNX
Runtime + whisper.cpp) shared by every plugin and the app. **Python
(this repo) becomes the R&D, training and evaluation workbench.** It is not
shipped to users.

**Positioning:** "The rap studio in one window, offline." The gaps we
target (market research): tuning that **finds the key from the beat**, a
chain built for rap *tracking*, private offline AI, a desktop booth for
rappers without a DAW, and loud-rap mastering presets.

**Headline risks found in this research round** (each has a mitigation
below):
1. Several models we planned to use are **non-commercial or have unclear
   licences**: madmom models (CC BY-NC-SA, used by this repo today),
   PC-NSF-HiFiGAN (CC BY-NC-SA), and the community RoFormer separation
   weights.
2. The market leader's tuner (AutoTune Advanced, Sep 2026) claims
   **< 1 ms** live latency, so our tuner target tightens from ≤ 5 ms to
   **≤ 3 ms**.
3. Retail sample-pack licences **forbid bundling** samples in software.
   Drum kits must be commissioned under buyout contracts.
4. The EU AI Act Art. 50 marking duties apply to generated audio (the beat
   generator), but likely not to tuning or mixing.

**Decisions taken by default in this plan** (change them in §15):
closed-source commercial licensing (the safe superset: nothing here stops
us open-sourcing later); plugins first; macOS (Apple Silicon + Intel) and
Windows 10/11; iOS deferred.

---

## 1. Users and jobs

| Persona | Setup | Jobs to be done | Products |
|---|---|---|---|
| **Bedroom rapper** | Laptop, USB mic or interface, YouTube type beats, often no DAW or a free one | Record over a beat and sound "industry"; tune to the beat without knowing its key; finish a song | VOX Lite → Booth, VOX, MASTER |
| **Rapper-producer** | FL Studio (assumed dominant, **not supported by survey data**) or Ableton | Fast beats in a given style, 808s that hit, vocal chain in one window | BEAT, VOX |
| **Engineer** | Pro Tools / Logic, pro plugins | Fast rap vocal chain, low-latency tracking tune, reliable recall | VOX (AAX), MASTER |

DAW share data is weak (pro surveys put FL at 1–13%; nothing is
hip-hop-specific). **Action, month 1:** run our own survey of ≥ 300
respondents through YouTube producer creators before locking UX
priorities and the order of format testing.

---

## 2. Quality bar (updated with this round's findings)

| Area | Requirement | Verified by |
|---|---|---|
| Tracking tune latency | **≤ 3 ms reported** (target ≈ 2 ms) at 48 kHz; note-onset settle ≤ 2 pitch periods | Loopback null test per DAW; `getLatencySamples` audit |
| HQ tune mode | PSOLA mode ≤ 15 ms reported latency; formant-preserving | Listening panel |
| Monitoring path | Round trip ≤ 10 ms at 64–128-sample buffers (ASIO / CoreAudio / WASAPI) | Latency rig |
| Take alignment | ±1 sample after loopback calibration | Automated loopback test |
| Take safety | Streams to disk while recording; kill-mid-take recovery | Fault-injection test |
| CPU | VOX ≤ 3% of one Apple M1 performance core at 48 kHz / 64 samples (Track mode); BEAT ≤ 2%; no glitches with 16 instances | Benchmark job |
| Real-time safety | Zero allocations, locks or syscalls on the audio thread | RTSan in CI (macOS / Linux), pluginval `--rtcheck` |
| Signal path | 32-bit float; 44.1–192 kHz; sample-rate conversion via r8brain (MIT) | Unit tests |
| Non-linear stages | ADAA + ≥ 2–4× oversampling; aliasing products < −90 dBFS on a full-scale sine sweep | Sweep test |
| Loudness | **ITU-R BS.1770-5** (current; the earlier doc said -4), EBU R128 v5; true peak via ≥ 4× oversampling (≥ 192 kHz) | libebur128 cross-check |
| Render parity | Offline bounce nulls < −120 dBFS against real-time playback | Golden render test |
| Plugin validation | pluginval strictness 10, Steinberg VST3 validator, `auval` clean | CI gate |
| Host matrix | FL Studio, Ableton Live, Logic, Pro Tools, Reaper, Studio One, Cubase on macOS + Windows | Manual matrix each release |
| Sound | Commissioned 24-bit kits with velocity layers and round robin; synthesized 808 engine | Producer panel |
| Perceived quality | Not worse than the reference chain in blind MUSHRA (p < 0.05) with ≥ 12 engineers and rappers | Listening test protocol |

---

## 3. System architecture

```
┌──────────────────────── User machine ───────────────────────────────┐
│                                                                      │
│  DAW process (FL, Ableton, Logic, Pro Tools, …)                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                               │
│  │  VOX     │ │  BEAT    │ │  MASTER  │  JUCE 9 plugins               │
│  │ C++ DSP  │ │ C++ DSP  │ │ C++ DSP  │  React UI (WebView)           │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘  real-time only, no big ML    │
│       └────────────┼────────────┘                                    │
│                    │ local socket / named pipe (file paths, JSON)    │
│  ┌─────────────────▼──────────────────────────────┐                  │
│  │  rapd: ML helper process (C++, one per user)    │                  │
│  │  ONNX Runtime (CoreML EP / DirectML→WinML / CUDA)│                 │
│  │  whisper.cpp (Metal / CUDA / Vulkan)             │                 │
│  │  job queue · cancel · progress · model manager   │                 │
│  └─────────────────▲──────────────────────────────┘                  │
│                    │                                                 │
│  ┌─────────────────┴──────────────────────────────┐                  │
│  │  Booth app (JUCE 9 + Tracktion Engine, React UI) │                 │
│  │  hosts VOX/BEAT/MASTER internally + 3rd-party    │                 │
│  └────────────────────────────────────────────────┘                  │
└──────────────────────────────────────────────────────────────────────┘
        │ (later, opt-in)                     ▲ licence activation
┌───────▼────────────────────────┐   ┌────────┴──────────────┐
│ Cloud: FastAPI · Postgres ·    │   │ Moonbase (store,      │
│ S3 · GPU workers (Docker)      │   │ licences, MoR)        │
│ sync, collab, ACE-Step jobs    │   └───────────────────────┘
└────────────────────────────────┘

R&D (this repo, Python): prototyping, training, ONNX export, evaluation,
benchmark reports. Nothing here ships to users.
```

### 3.1 Principles
- **What the user hears while tracking is C++ and allocation-free.** ML
  never runs on the audio thread and never runs inside the DAW's process
  beyond tiny CPU models on a worker thread (see §3.3).
- **One helper process (`rapd`)** loads models once for all plugin
  instances. A crash in it can't take down the DAW. Plugins pass **file
  paths**, not audio buffers. If `rapd` is unreachable, features degrade
  ("Open in Booth"), and the audio thread never waits on IPC.
- **Shared core library (`rapcore`)** holds DSP, music theory, project
  format, IPC client and licence check. It is linked into every plugin,
  `rapd` and Booth.
- **Determinism:** every generator is seeded and every ML job is cached by
  `(input hash, model id, params)`, matching music-decoder's current
  conventions.

### 3.2 Repositories

```
rapsuite/                         (new monorepo, private)
├── cmake/  CMakeLists.txt        pamplejuce-based, JUCE 9 via CPM
├── libs/rapcore/                 C++23: dsp/, theory/, project/, ipc/, licensing/
├── plugins/vox/  beat/  master/  JUCE plugin targets (+ beat_aumi for Logic)
├── apps/booth/                   Tracktion Engine app
├── apps/rapd/                    ML helper (ONNX Runtime, whisper.cpp)
├── ui/                           pnpm workspace: shared React design system,
│   ├── packages/ui-kit/          knobs, meters, keyboard, waveform, grid
│   ├── vox/ beat/ master/ booth/ Vite apps → embedded via resource provider
├── tests/                        Catch2 unit, golden renders, measurement
├── tools/                        latency rig, render CLI, preset builder
└── .github/workflows/            build · test · pluginval · sign · notarize · package

music-decoder/ (this repo)        R&D: prototypes, training, ONNX export, eval
```

### 3.3 Where things run

| Workload | Runs in | Runtime |
|---|---|---|
| Tuning, dynamics, EQ, saturation, reverb, delay, sampler, 808, limiter | Plugin audio thread | C++ (`rapcore::dsp`) |
| Key detection from beat ("Listen" button) | Plugin worker thread | C++ port of our KK + Temperley + chord-fit fusion (no madmom) |
| Beat / downbeat tracking | `rapd` | Beat This! (MIT) ONNX |
| Stem separation | `rapd` | Our own RoFormer weights → ONNX (STFT / iSTFT in C++) |
| Vocal f0 (offline editor) | `rapd` | FCPE (MIT) ONNX; RMVPE pending data review |
| Formant-preserving resynthesis | `rapd` | DSP first (LPC / cepstral envelope); own-trained neural vocoder later |
| Lyrics alignment | `rapd` | wav2vec2 CTC (MIT English model) ONNX + our aligner |
| Lyrics transcription (draft) | `rapd` | whisper.cpp large-v3-turbo q5 / q8 + DTW alignment heads |
| Note transcription (melody → MIDI) | `rapd` | basic-pitch (Apache-2.0) ONNX |
| Beat-idea generation (optional) | Cloud GPU first; local later | ACE-Step 1.5 (MIT) in Python |

### 3.4 IPC contract (`rapd` v1)
- Transport: Unix domain socket (macOS) / named pipe (Windows), framed
  JSON-RPC 2.0, protocol version in the handshake.
- Methods: `analyze.key|tempo|beats`, `separate`, `pitch.track`,
  `align.lyrics`, `transcribe.lyrics`, `flow.analyze`, `render.retune`,
  `job.status|cancel`, `models.list|install|remove`.
- Every job: `{job_id, inputs: [abs paths], params, model_id}` returns
  progress events, then `{outputs: [abs paths], meta}`. Results are cached
  under `~/Library/Application Support/Rapsuite/cache` (macOS) or
  `%LOCALAPPDATA%\Rapsuite\cache` (Windows).
- Models are downloaded on first use, checksummed (SHA-256), signed
  manifest, never bundled in the installer.

### 3.5 Project format (Booth)
Project folder: `project.json` (versioned schema: tempo map, key, tracks,
clips, takes and comp regions, plugin states, lyric doc), `audio/`
(immutable WAV/CAF recordings, never rewritten), `renders/`, `cache/`.
Autosave every 10 s plus on every edit, with crash-recovery journal.

---

## 4. VOX (rap vocal chain)

### 4.1 Signal flow

```
in → input trim → HPF (SVF) → TUNE → de-esser → comp 1 (fast, FET-style)
   → EQ (4-band SVF + tilt) → saturation (ADAA, 2–4× OS) → comp 2 (glue)
   → doubler (2–4 voices) → sends: delay throw · reverb throw → width → out
```

Order can be changed per preset. Every block has an independent bypass.

### 4.2 TUNE engine
- **Detection:** cycfi/q **bitstream autocorrelation** (MIT, per-sample
  updates, `predict_frequency()` during the attack, confidence output) plus
  our hysteresis note-decision layer. Cross-check against YIN / MPM from
  sevagh/pitch-detection (MIT) in tests.
- **Shifting, Track mode (default):** clean-room **period-synchronous
  resampling with ±1-period splices + crossfade** (method described by
  zita-AT1, which is GPL, so **implement from the description, never read
  or copy its code into our repo**). Formant control by **causal LPC
  whitening → shift → re-colour** (order 16–24); "Throat" warps the
  envelope. Budget: 64-sample fragment (1.3 ms) + 32-sample zero-cross
  lookahead (0.7 ms) ≈ **2 ms reported**.
- **Shifting, HQ mode:** TD-PSOLA with pitch marks, about one max period
  of lookahead (~14 ms at a 70 Hz floor), reported via
  `setLatencySamples`. Fixed latency per mode; a mode switch shows a "host
  will re-sync" notice.
- **Controls → algorithm:**

  | Control | Implementation |
  |---|---|
  | Key / scale | Allowed pitch classes; **Auto-Key**: "Listen" analyses the beat (sidechain input or Booth project) with the C++ key estimator; manual override; MIDI target notes |
  | Retune speed | Critically damped 2nd-order smoother on the correction curve (cents); 0 ms = step change (hard-tune); ratio changes only at splice boundaries |
  | Humanize | Retune time grows with note duration |
  | Flex | Correction amount by distance from target; scoops and slides beyond the threshold pass through |
  | Vibrato | f0 split into centre (< 3 Hz low-pass) and vibrato band (4–8 Hz) with its own gain |
  | Formant / Throat | LPC envelope keep / warp |
  | MIDI out | Detected notes → MIDI (e.g. to double a melody) |

- **Never use the word "Auto-Tune"** in product or marketing (Antares
  trademark). Use "Tune".

### 4.3 Other blocks

| Block | Design | Source |
|---|---|---|
| De-esser | Split-band sidechain (5–9 kHz band-pass detector) → dynamic SVF bell / high shelf; listen-to-sidechain; 0 ms (optional 1 ms lookahead) | Own |
| Compressors | Feed-forward log-domain (Giannoulis/Massberg/Reiss topology); FET-style fast character + opto-style glue | Own, Signalsmith Basics (MIT) as reference |
| EQ | Trapezoidal SVF (implement from Simper's maths) + RBJ cookbook shapes | Own |
| Saturation | ADAA waveshapers (tape / tube / hard clip) with 2–4× oversampling (JUCE `dsp::Oversampling` IIR, or HIIR) | Own + JUCE |
| Doubler | 2–4 voices, ±5–15 cent detune via modulated delay, 10–30 ms offsets, pan spread | signalsmith-dsp delay lines (MIT) |
| Throws | Tempo-synced delay (1/4, 1/8 dotted, 1/4 triplet) with filter + ducking; reverb send (FDN) triggered by automation or last-word detection | Signalsmith Basics Reverb (MIT) as base |
| Meters | Input / GR / output, LUFS-S | libebur128 (MIT) in histogram mode on a meter thread |

**Do not use:** chowdsp_utils GPL modules (compressor, eq, filters,
reverb, waveshapers, …). Use only its BSD modules (core, math, simd,
parameters, presets). Airwindows (MIT) is allowed as a reference.

### 4.4 Presets (shipped)
Clean modern · Melodic trap (hard tune, retune 0) · Drill (dark, aggressive
comp) · Boom-bap (no tune, warm) · Ad-lib (band-pass, high throw) · Doubles
· Phone / lo-fi · Whisper / ASMR. Artist preset packs come later and are a
marketing channel.

### 4.5 UI
One window, three zones: **Tune** (key strip with Auto-Key button, retune
speed knob, live pitch trace), **Chain** (horizontal block strip, one
"macro" knob per block, expandable details), **Output** (meters, LUFS,
mix). React + `@juce-framework/webview` (npm, JUCE 9.0.1+) with
`WebSliderRelay` / `WebToggleButtonRelay` / `WebComboBoxRelay` bound to
APVTS; meters pushed at 30–60 Hz batched. **Keyboard-focus forwarding to
the host must pass the host matrix** (known WebView issue; spacebar must
still reach the transport).

### 4.6 VOX exit criteria
- [ ] Reported latency ≤ 3 ms in Track mode across 44.1/48/96 kHz
- [ ] CPU and RT-safety budgets met (§2)
- [ ] MUSHRA vs reference chain (AutoTune 2026 + stock chain): not worse, p < 0.05
- [ ] Host matrix pass incl. spacebar / transport focus test
- [ ] Auto-Key accuracy ≥ 85% (MIREX-weighted) on our rap-beat key set (§10)

---

## 5. BEAT (drum + 808 generator)

> Correction to the earlier docs: `compose/` in this repo is guitar-only
> and holds **no drum or 808 rules to port**. BEAT's rule engine is new
> work. Prototype it in Python here first (fast iteration with producers),
> then port to C++.

### 5.1 Style engine
Per style: a **probability grid** per instrument per step (960 PPQ;
1/16 = 240 ticks, 1/24 = 160, 1/32 = 120, 1/48 = 80), swing amount and
swing targets, **roll vocabulary** (1/32, 1/24, 1/48, 1/64, pitch-ramped
and velocity-ramped) with placement policy (pre-snare, phrase end), a
counter-snare policy, and 808 rules (root from chord, octave-jump
probability, **slide budget per 4 bars**, slide intervals {m3, P4, 8va},
overlap length). Variation over 2-bar and 4-bar cycles; "complexity" and
"energy" macros map onto the probabilities.

| Style | BPM | Grid / swing | Signature | 808 |
|---|---|---|---|---|
| Trap | 130–150 (≈ 140), half-time | Straight 1/16, mixed-resolution hats | Snare / clap on 3 (half-time); 1/32 and triplet rolls before snares | Root-following, 80–150 ms glides, octave jumps, long sustain, saturation |
| UK drill | 138–145 | Triplet-feel, swung hats | Moving **counter-snares** across 2-bar cycle; 2-2-1 / tresillo hat gaps | **Slides** 2–3 per 4 bars (m3 / P4), in hat gaps |
| NY drill | 140–150 | As UK | Harder 3-snare, more counters | 808 carries melody |
| Boom-bap | 85–100 | **MPC swing 54–62% (58% classic)** on off-16ths | Kick 1 and the "and" of 3, snare 2 and 4 | Usually sampled bass or short sub |
| Plugg / pluggnb | 130–160 (felt 65–75) | Straight | Sparse; restrained rolls | Melodic, sliding, locked to hats |
| Rage | 140–165 | Straight | Busy hats | **808 as lead**: heavy mid-band distortion |
| Jersey club | 130–145 | Straight 1/16 | **5-kick bar at 16ths 1, 5, 9, 12, 15**; own "squeak" sound (not the Trillville sample) | Light / none |
| Afrobeats-adjacent | 100–115 | 12/8 feel / swung 16ths, 3-3-2 | Rim / clave patterns, constant shaker, congas | Log-drum answers kick (amapiano) |

Every row is validated with ≥ 3 producers per style before C++ port.

### 5.2 808 engine and sampler
- **Synth 808:** sine + pitch envelope (2–4 st over 20–70 ms), amp
  envelope, optional click layer; phase reset on new notes but **not** on
  legato or glide; 1–3 ms attack ramp; few-ms voice-steal crossfade.
  Glide in log-frequency, constant-time (drill default) or constant-rate.
  Split-band drive (sub < 100–120 Hz stays clean), ADAA + oversampling.
  Auto-tunes to project key.
- **Sampler:** pads with tune (±24 st + cents), ADSR, start offset, reverse,
  pan; **choke groups**; velocity layers + round robin or random; mono /
  legato 808 pad; per-pad multi-out; hat-roll pitch ramp; humanize; kit
  browser with preview.
- **Kits:** **6–8 commissioned kits** (one per style + 808 multisamples)
  under written buyout / work-for-hire contracts granting perpetual,
  worldwide, sublicensable rights to embed in software, with a "no
  third-party samples" warranty and indemnity. CC0 filler only (no
  CC-BY-SA). Samples packed and encrypted; user EULA "use in music, not
  resale"; provenance log per sample.

### 5.3 DAW integration
- MIDI out per host: FL (wrapper port routing), Ableton (track routing),
  **Logic needs a separate `aumi` MIDI FX build**, Pro Tools (AAX MIDI
  node). VST3 CC output is unreliable, so use notes only.
- **Drag-to-DAW:** write a temp `.mid` / `.wav` with a unique name per drag
  → `performExternalDragDropOfFiles`. Test in every host (Bitwig and Logic
  on Sequoia have reported issues).
- Chord input: type a progression, or "Listen" to a loop / sample (key +
  chords via `rapd`).

### 5.4 Variation model (phase 5, optional)
Rules first. Later, a small VAE or tiny transformer on 16th / 48th grids
trained on **our own commissioned, style-labelled pattern corpus**
(≥ 500 bars per style from hired producers); GMD / E-GMD (CC BY 4.0) only
for pre-training and humanize, with attribution. **MIDI-GPT weights are
non-commercial, so don't use them.**

### 5.5 BEAT exit criteria
- [ ] Producer blind test: generated patterns rated "usable as-is or with
      light edits" ≥ 70% per style
- [ ] Drag-to-DAW and MIDI out verified in the host matrix, incl. Logic `aumi`
- [ ] 808 glide click-free (null-spectrum test on glide transitions)

---

## 6. MASTER

Chain: input → dynamic EQ (4 bands) → multiband compressor (3-band,
Linkwitz-Riley) → stereo width (bass mono below a set frequency) →
soft clipper (ADAA, oversampled) → **true-peak limiter** (1–5 ms
lookahead, gain computed on ≥ 4–8× oversampled peaks, hold + Hann attack,
0.1–0.3 dB safety margin) → dither (TPDF, 16-bit export).

- Targets: Streaming (−14 LUFS-I, ≤ −1 dBTP), Loud rap (−8 to −6 LUFS-I,
  ≤ −1 dBTP), Club, Custom. Live loudness via libebur128 (histogram mode,
  meter thread).
- **Reference match:** our own implementation of spectrum / RMS / width
  matching (Matchering is GPL-3, so it's only a conceptual reference).
- Exit: measured LUFS within ±0.5 LU and TP never over target on the 50-song
  test set; MUSHRA not worse than Ozone 12 Standard presets.

---

## 7. Booth (standalone app)

- **Engine:** Tracktion Engine (commercial licence) for recording,
  lock-free disk writing, plugin delay compensation, **take comping
  (`CompManager`)**, automation, plugin hosting, render. It has **no UI**,
  so the timeline is ours (React).
- **Flow:** drop a beat (WAV / MP3 / YouTube link only for personal
  practice, labelled as such) → `rapd` detects tempo / key / sections →
  arm → count-in → loop record into take lanes → comp → VOX on vocal
  bus → MASTER → export (24-bit WAV, 16-bit dithered, MP3 320, stems).
- **Latency calibration wizard:** loopback click measurement per device
  and buffer size, stored per device; ±1 sample.
- **Pen & Flow** panels live here (§8).
- Exit: 8-hour soak test, kill-mid-take recovery, ≤ 10 ms monitoring on
  reference interfaces (Focusrite Scarlett, UA Volt, Audient iD, MOTU M2).

---

## 8. Pen & Flow

### 8.1 Rhyme engine (`rapcore::theory::rhyme`, prototyped in Python here)
1. Tokenize → **G2P chain:** CMUdict raw file (BSD-2; **avoid the GPL
   `cmudict` PyPI wrapper**) → slang / AAVE override lexicon (user-editable,
   learns from corrections) → neural G2P for out-of-vocabulary words
   (DeepPhonemizer MIT or g2p_en Apache-2.0, exported to ONNX). Several
   pronunciations per word, plus rule-generated dialect variants
   (/aɪ/→/aː/, -ing→-in', final cluster reduction, th-fronting, r-lessness)
   **[UV: rules from memory; validate with a linguist]**.
2. Syllabify (maximal onset) + stress.
3. Score rhymes by **weighted alignment of syllable rimes**: vowel identity
   dominates; codas compared by phonetic features. Weights are learned
   Hirjee & Brown-style from a rhyme-pair corpus. Detect multisyllabic
   chains with local alignment across word boundaries. Label each rhyme as
   perfect, slant, assonance or consonance, and as end or internal.
4. Metrics: rhyme density (Malmi et al.), multisyllabic rate, internal
   rhymes per bar, syllables per bar, rhyme-scheme graph.

UI: live colour-coding as you type, stress dots, a 16-step-per-bar
syllable grid, beat player synced to the text, rhyme finder that includes
slang. Note: the Datamuse API's free tier is non-commercial, so build our
own index.

### 8.2 Flow analysis
- **Primary path: forced alignment of the user's own lyrics** (rappers
  have them). ASR is only a draft, because expected WER on rap is about
  15–30% (> 30% on fast or ad-lib-heavy verses) and there is no
  peer-reviewed rap benchmark.
- Pipeline: vocal stem (or dry take) → wav2vec2 CTC alignment (English
  model MIT; **fr/de/es/it VoxPopuli aligners are CC BY-NC, so English
  only at launch**) → syllable onsets (refine word timings with an onset
  detector) → Beat This! beat / downbeat grid.
- Metrics: syllables per beat / bar; slot per syllable (1/16 or 1/24);
  **microtiming offset** (ms and % of a 16th; mean = laid-back or pushed,
  SD = tightness); pocket score (circular consistency; our own
  definition); triplet vs duplex share; stresses and rhymes on 2 and 4;
  phrase starts; Ohriner-style groove classes.
- Benchmarks: MCFlow (licence **[UV]**), JamendoLyrics-style alignment
  metrics (mean absolute error, % of words within 0.3 s) on our rap corpus.
- Exit: median word-onset error ≤ 50 ms and ≥ 90% within 0.3 s on the rap
  test split; flow metrics stable across takes (test-retest).

---

## 9. ML model plan (licence-cleared path)

| Capability | Dev / prototype (R&D only) | Shipping model | Licence status | Action |
|---|---|---|---|---|
| Beats / downbeats | madmom (current repo) | **Beat This!** | MIT code + weights | Swap in R&D now; ONNX export; benchmark |
| Key | Fusion incl. madmom CNN | **KK + Temperley + chord fit** in C++ (our code) | Ours | Measure accuracy drop without CNN on rap set; optional own small CNN later |
| Chords | madmom deep chroma | Our `template_hmm` backend, then own small CNN | Ours | Benchmark on rap loops |
| Stem separation | Community RoFormer weights | **Own RoFormer** trained with ZFTurbo MSST (MIT) on licensed stems | Community weights GPL-tagged or unlicensed; htdemucs weights "not clear" | Buy or licence multitrack data; train; ONNX (STFT outside graph) |
| Vocal f0 | RMVPE, FCPE | **FCPE** | MIT; training data not stated | Legal review; RMVPE only after data-provenance review |
| Formant-preserving resynthesis | PC-NSF-HiFiGAN | **DSP (LPC / cepstral)** in v1; own-trained NSF-HiFiGAN in phase 5 | openvpi weights **CC BY-NC-SA** | Train on licensed singing / rap corpus |
| Lyrics ASR | WhisperX | **whisper.cpp large-v3-turbo** | MIT | Add DTW alignment-head preset for turbo manually |
| Lyrics alignment | WhisperX aligner | **wav2vec2 base 960h (English)** ONNX | MIT | Fine-tune on rap corpus |
| Melody → MIDI | basic-pitch | **basic-pitch ONNX** | Apache-2.0 | – |
| Beat ideas | ACE-Step 1.5 | **ACE-Step 1.5** (cloud first) | **MIT** (earlier doc said Apache: corrected); weights MIT per card **[UV]**; training-data provenance unaddressed | Legal review before launch; watermark outputs (§13) |

Runtimes: ONNX Runtime (MIT) with CoreML EP on Mac and DirectML now,
moving to **Windows ML** (Microsoft's new focus) on Windows 11 24H2+,
plus CUDA EP optional; whisper.cpp (MIT). **No Python in shipped
products.** ACE-Step stays Python on cloud GPUs (Docker) in v1.

Expected runtimes for a 3-minute song (estimates; P0 must measure):
separation 20–60 s on RTX 3060, 1–3 min on M-series, 5–15 min on CPU;
f0 and alignment seconds; Whisper turbo 7–25× real time on Apple silicon.

---

## 10. Data strategy

| Dataset | Purpose | Size target | How we get it | Rough cost **[estimate]** |
|---|---|---|---|---|
| **Rap vocal corpus** | Tune listening tests, flow / alignment benchmark, aligner fine-tune, vocoder training | 100+ artists, ≥ 20 h dry vocals with lyrics, beat stems, tempo / key | Paid sessions with written consent (ML training + evaluation, no voice cloning), diverse regions and styles; hand-label syllable onsets for a 2 h test split | $30–60k |
| **Rap-beat key / tempo set** | Auto-Key and tempo accuracy | 500 beats | Commission + licence from producers; label key / BPM / downbeats | $5–10k |
| **Separation training stems** | Own RoFormer | ≥ 500 multitrack songs (more is better) | Licence multitracks from labels / libraries / producers; every source contract allows ML training | $20–100k+ (biggest unknown) |
| **Pattern corpus** | BEAT validation + variation model | ≥ 500 bars per style | Hire producers to program MIDI | $10–20k |
| **Drum kits** | BEAT sampler | 6–8 kits | Buyout contracts | $15–60k ($2–10k per kit) |

Rules: every item has a signed contract stored with a data card
(source, licence, consent scope, demographics); a held-out test split
nobody tunes on (same discipline as the GuitarSet v3 work); and no
YouTube-scraped audio in any training set.

---

## 11. Quality assurance

1. **Unit (Catch2):** DSP blocks against analytic references (filter
   responses, compressor static curves, LUFS reference signals from EBU
   Tech 3341 / 3342).
2. **Golden renders:** fixed input + preset → render → null against stored
   reference (threshold per block); runs on every PR.
3. **Measurement suite:** aliasing sweeps, latency reports versus measured
   loopback, true-peak overs, CPU benchmarks.
4. **Real-time safety:** RTSan (`-fsanitize=realtime`, upstream Clang 20+)
   on macOS / Linux test builds; pluginval 2 `--rtcheck`; there's no RTSan
   on Windows, so rely on the code rules (no allocation, locks or I/O on
   the audio thread; SPSC FIFOs via farbot / readerwriterqueue).
5. **Validation gates:** pluginval strictness 10 (includes the Steinberg
   validator in v2), `auval` for aufx / aumu / aumi.
6. **Host matrix** (manual, per release): 7 DAWs × 2 OS; checklist: load /
   scan, automation, state recall, latency compensation, sample-rate
   change, offline bounce parity, keyboard focus, drag-to-DAW, MIDI out.
7. **Listening tests:** MUSHRA / ABX protocol (blind, randomized, hidden
   reference + anchor), ≥ 12 listeners (engineers + rappers), run
   pre-beta and pre-launch for each product.
8. **Accuracy benchmarks** (Python, this repo): Auto-Key, tempo, flow
   alignment, rhyme detection; reports in `docs/reports/` like the
   GuitarSet report.
9. **Beta program:** 50–100 users per product, telemetry opt-in (crash +
   feature usage only, never audio), weekly builds.

---

## 12. Release engineering

| Item | Choice | Cost **[from research; verify]** |
|---|---|---|
| Framework | **JUCE 9.0.x** (9.0.5, 2026-11-02; JUCE 8 is security-only) | Indie $40/user/mo or $800 perpetual (≤ $300k revenue); Pro $175/user/mo; Starter free ≤ $20k |
| Template | pamplejuce (JUCE 9, CMake 3.25+, C++23, CPM, Catch2) | $0 |
| Formats | VST3 (SDK 3.8 **MIT**), AU v2 (+ `aumi` for BEAT), **CLAP via clap-juce-extensions** (not native in JUCE 9), Standalone; **AAX** after PACE approval | $0 + iLok ≈ $50 |
| ASIO (Booth, Windows) | Steinberg ASIO licence (the GPL option isn't viable for closed source) | $0, agreement required |
| Booth engine | Tracktion Engine commercial (separate from JUCE) | Indie $35/seat/mo (≤ $200k) or Pro 1 $50 (≤ $400k), 12-month minimum |
| macOS signing | Developer ID Application + Installer, hardened runtime, notarytool, staple; pkg + DMG | $99/yr |
| Windows signing | **Azure Artifact Signing** (org in US/CA/EU/UK) or OV cert (≤ 460-day validity since 2026-03) | $9.99/mo or $150–300/yr |
| Installers | pkgbuild / productbuild; Inno Setup | $0 |
| Store / licensing | **Moonbase** (merchant of record, JUCE activation module, offline activation, trials) | 5% + €1 per sale |
| Copy protection | Signed offline tokens (Ed25519) with grace period; **no iLok in v1** | – |
| Crash reporting | sentry-native + Crashpad, opt-in; filter to our modules (we live in the DAW's process); never install process-wide handlers | Free tier, then ~$26/mo **[UV]** |
| Updates | In-app update check (signed manifest), installer download | – |
| CI | GitHub Actions macOS + Windows + Linux (tests); macOS minutes are expensive, so cache builds | ~$100–400/mo **[estimate]** |

AAX path: Avid developer account → Pro Tools Developer (loads unsigned) →
request PACE Eden signing tools (screen recording of plugin in Pro Tools
Developer) → wraptool in CI. Allow 2–6 weeks; start in month 2.

---

## 13. Legal and compliance

| Topic | Requirement | Owner / when |
|---|---|---|
| Framework licences | JUCE commercial, Tracktion commercial, ASIO agreement, Moonbase | Before first public build |
| Third-party code | Licence allowlist: MIT / BSD / Apache / ISC / zlib / BSL / WTFPL; MPL only unmodified; **GPL / LGPL / AGPL / NC banned** in shipped code; automated licence scan (e.g. ScanCode) in CI | Month 1 |
| Clean-room rule | GPL references (zita-AT1, Rubber Band, chowdsp GPL modules, Matchering) may be read for ideas only in a separate process, documented; never copied | Ongoing |
| Models | Shipping table in §9; data cards; no NC or unclear weights | Each release |
| **EU AI Act Art. 50** | Applies from 2026-08-02 (grace to 2026-12-02 only for systems already on the market; **sources disagree, check the Official Journal**). Generated audio (BEAT AI ideas, ACE-Step) needs a machine-readable mark + detection; follow the Code of Practice (final 2026-06-10). Tuning, mixing and separation are likely exempt as "assistive editing" **[legal confirm]** | Before any generative feature ships |
| Voice | **No celebrity or third-party voice cloning.** ELVIS Act (TN) can reach tool providers; NO FAKES Act passed Senate Judiciary 2026-06-18 but stalled ~2026-10-07 (not law) | Policy now |
| Copyright guidance | US Copyright Office Part 2 (2025-01-29): purely AI output isn't protectable; human contribution is. Tell users; keep provenance logs for generated material | Docs at launch |
| Gen-AI litigation | UMG / Sony v. Suno and Sony v. Udio active; GEMA beat Suno in Munich (2026-07-31, not final). Use only models with defensible training data | Before ACE-Step launch |
| Patents | US 5,973,252 (Auto-Tune core) **expired**. Celemony US 8,022,286 (note-object editing) likely in force to ~2029, which matters for a graphical note editor. Antares vibrato engine patent-pending. **Freedom-to-operate search** (pitch, formant, stretch, separation) before VOX launch | Month 3–6 |
| Trademarks | Never "Auto-Tune"; clear our product names (VOX / BEAT / Booth are generic, so pick distinctive marks) | Month 2 |
| Privacy (GDPR / CCPA) | Local by default; cloud opt-in; voice is biometric only if used for identification; data processing agreements with processors; retention limits; EU hosting or SCCs | Before cloud features |
| Content contracts | Kit buyouts, corpus consent forms, producer pattern contracts | Before collection |
| User EULA | Kit sound licence ("use in music, not resale"), AI-output terms, no stripping of marks | Launch |

Budget **[estimate]**: $25–60k legal (FTO search, contracts, AI Act
review, EULA / privacy).

---

## 14. Team, budget, roadmap

### 14.1 Team

| Role | FTE | Owns |
|---|---|---|
| Lead DSP engineer (C++) | 1 | TUNE, dynamics, saturation, 808, limiter |
| Plugin / app engineer (C++, JUCE / Tracktion) | 1 | Plugin shells, WebView bridge, Booth engine, `rapd`, CI |
| Front-end engineer (React / TS) | 1 | ui-kit, all plugin and Booth UIs |
| ML engineer (Python → ONNX) | 1 | R&D repo, training, export, benchmarks, data pipeline |
| Product designer | 0.5 | UX, visual design, preset UX |
| QA + producer relations | 0.5 → 1 | Host matrix, beta, listening panels, kit / producer contracts |
| Contractors | – | Sound designers, producers (patterns), rappers (corpus), legal |

### 14.2 Year-1 non-salary budget **[estimates]**
Data and content $80–250k (the separation training data dominates
uncertainty) · legal $25–60k · tools and licences $5–10k · GPU training
$5–20k · DAW licences + test interfaces $3–5k · marketing launch
$10–30k.

### 14.3 Phased roadmap (months from kickoff)

| Phase | Months | Deliverables | Exit gate |
|---|---|---|---|
| **P0 Foundations** | 0–1.5 | Monorepo on pamplejuce + JUCE 9; CI (build, Catch2, pluginval 10, auval, RTSan, signing, notarize, installers); `rapcore` skeleton; WebView bridge + ui-kit; latency rig; golden-render harness; licence scanner; DAW survey launched; legal kickoff (FTO, contracts); corpus consent forms | Empty "hello gain" plugin passes the full CI + host matrix |
| **P1 VOX** | 1.5–8 | TUNE (Track + HQ), all chain blocks, presets, Auto-Key (C++ key estimator), UI; VOX Lite; AAX approval; Moonbase integration; beta at month 6; MUSHRA ×2 | §4.6 criteria; launch month 8 |
| **P2 BEAT** | 5–11 | Python prototype of style engine with producers (from month 5) → C++ engine, 808 synth, sampler, commissioned kits, MIDI out incl. `aumi`, drag-to-DAW | §5.5 criteria; launch month 11 |
| **P3 Booth + rapd + Pen & Flow** | 8–15 | `rapd` (ONNX Runtime, whisper.cpp, model manager); Tracktion-based recording, comping, calibration wizard; rhyme engine; lyrics alignment + flow metrics; separation with own weights (training starts month 6) | §7 and §8.2 criteria; launch month 15 |
| **P4 MASTER** | 12–17 | Mastering chain, true-peak limiter, targets, reference match | §6 criteria; launch month 17 |
| **P5 Generative + cloud + mobile** | 15+ | ACE-Step ideas (cloud, watermarked), variation model, own neural vocoder for HQ tuning, opt-in sync, iOS AUv3 evaluation | Legal sign-off on Art. 50 + data provenance |

Bundle pricing: VOX $129 (intro $79), BEAT $99 ($59), MASTER $79–99,
**bundle $249 perpetual or $9.99/mo / $99/yr** (rent-to-own via Splice
optional), Booth free to record with paid export / AI tools in the bundle.
Channels: own store (Moonbase), Plugin Boutique, YouTube producer
affiliates (20–30%), artist preset packs. Plan around Black Friday each
year.

---

## 15. Decisions log and open questions

| # | Decision | Default in this plan | Who / when |
|---|---|---|---|
| D1 | Licensing model | Closed commercial (bans GPL / NC) | You, before P0 ends |
| D2 | Product order | VOX → BEAT → Booth → MASTER | You, P0 |
| D3 | Platforms | macOS + Windows; iOS in P5 | You, P0 |
| D4 | AAX at launch | Yes if PACE approval lands by month 7, else +1 month | Plugin engineer |
| D5 | Separation weights | Train our own (needs data budget) vs license a vendor SDK | ML engineer + you, month 3 |
| D6 | Cloud | None until P5 | You, month 12 |
| D7 | Brand names | Placeholder VOX / BEAT / Booth / MASTER | Designer + legal, month 2 |

Open questions to settle early: budget ceiling for data (D5 depends on
it); whether ACE-Step training-data provenance is acceptable; whether to
seek a commercial licence for madmom or openvpi weights instead of
replacing them.

---

## 16. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Tuner doesn't reach ≤ 3 ms with acceptable quality | Med | High | Two modes (Track / HQ); early prototype in month 2 with listening test; fallback 5 ms target |
| No affordable licensed separation data | Med | Med | Ship separation later (P3+), or license a vendor SDK; core products don't depend on it |
| WebView focus / host bugs | Med | Med | Host matrix from P0 with a dummy plugin; native JUCE fallback for critical controls |
| Patent claim (note editing, vibrato) | Low–Med | High | FTO search before launch; avoid polyphonic note-object editing; design-around notes |
| Market saturation (Antares, iZotope, BandLab free) | High | Med | Rap-specific differentiators (Auto-Key from beat, flow analysis, booth); aggressive VOX Lite free tier; creator marketing |
| Native Instruments / iZotope ownership churn changes pricing | Med | Low | Monitor; opportunity for switchers |
| AI-regulation drift (EU AI Act guidance, US bills) | Med | Med | Generative features isolated, cloud-side, watermark-ready |
| Key accuracy drops without madmom CNN | Med | Med | Measure in P0 on rap-beat set; train own small CNN if needed |
| Scope creep into a full DAW | Med | High | Booth is "record + comp + chain" only; no MIDI editing, no mixer beyond essentials in v1 |

---

## 17. What changes in this repo (music-decoder)

This repo becomes the **R&D workbench**. Concrete first tasks (each gets
its own task-by-task plan in `docs/superpowers/plans/`):

1. **Beat This! backend** behind `beat_tracking.backend` (keep madmom as
   R&D-only option); benchmark versus madmom on GuitarSet and the rap-beat
   set.
2. **Key without madmom:** `key_detection.backend: fusion_nocnn`; measure
   the drop; decide on own CNN.
3. **`music_decoder.rhymes`:** G2P chain + rime alignment scorer +
   metrics; unit tests on curated multisyllabic examples.
4. **`music_decoder.beatgen`:** Python style engine (§5.1) emitting MIDI,
   used in producer review sessions before the C++ port.
5. **`music_decoder.flow`:** forced alignment + flow metrics on the beat
   grid; evaluation harness and report format.
6. **ONNX export tooling** (`scripts/export_onnx.py`) and a parity test
   (PyTorch versus ONNX outputs) for each shipped model.
7. **Licence audit** of the current dependency tree (`pip-licenses` +
   model files) recorded in `docs/reports/`.

### 17.1 First two weeks (checklist)
- [ ] Confirm D1–D3 (§15)
- [ ] Create `rapsuite` monorepo from pamplejuce; pin JUCE 9.0.5; CI green on macOS / Windows
- [ ] Apple Developer + Azure Artifact Signing accounts; Avid developer account
- [ ] JUCE Indie licence; Moonbase account; Tracktion licence quote
- [ ] Engage IP counsel: FTO scope, kit buyout template, corpus consent form
- [ ] Draft the DAW / workflow survey; line up 3 YouTube creators to distribute it
- [ ] Prototype spike: q BACF detector + period-splice shifter in a test harness; measure latency and listen
- [ ] In this repo: task 1 (Beat This!) and task 7 (licence audit)

---

## Sources

**DSP:** [cycfi/q pitch docs](https://github.com/cycfi/q/blob/develop/docs/modules/ROOT/pages/reference/pitch.adoc) · [sevagh/pitch-detection](https://github.com/sevagh/pitch-detection) · [JUCE forum: pitch latency](https://forum.juce.com/t/lowest-latency-real-time-pitch-detection/51741) · [fat1.lv2 retuner (GPL, reference only)](https://github.com/x42/fat1.lv2/blob/master/src/retuner.cc) · [x42-autotune](https://x42-plugins.com/x42/x42-autotune) · [Signalsmith Stretch](https://github.com/Signalsmith-Audio/signalsmith-stretch) · [Signalsmith DSP](https://github.com/Signalsmith-Audio/dsp) · [Signalsmith Basics](https://github.com/Signalsmith-Audio/basics) · [stftPitchShift](https://github.com/jurihock/stftPitchShift) · [Röbel true envelope](https://typeset.io/pdf/efficient-spectral-envelope-estimation-and-its-application-30w5wf4s19.pdf) · [Airwindows](https://github.com/airwindows/airwindows) · [chowdsp_utils](https://github.com/Chowdhury-DSP/chowdsp_utils) · [HIIR mirror](https://github.com/unevens/hiir) · [r8brain](https://github.com/avaneev/r8brain-free-src) · [libebur128](https://github.com/jiixyj/libebur128) · [ITU-R BS.1770-5](https://www.itu.int/rec/R-REC-BS.1770-5-202311-I) · [EBU R128](https://tech.ebu.ch/publications/r128) · [rtsan](https://github.com/realtime-sanitizer/rtsan) · [farbot](https://github.com/hogliux/farbot) · [readerwriterqueue](https://github.com/cameron314/readerwriterqueue) · [ADAA (Vicanek)](https://www.vicanek.de/articles/AADistortion.pdf) · [AutoTune Advanced latency](https://www.mixonline.com/technology/news-products/autotune-advanced-pitch-correction-plug-in-debuts) · [Auto-Tune manual](https://www.barryrudolph.com/recall/manuals/auto-tune_artist.pdf)

**Plugin engineering:** [JUCE change list](https://github.com/juce-framework/JUCE/blob/master/CHANGE_LIST.md) · [JUCE 9 pricing post](https://forum.juce.com/t/juce-9-no-pricing-or-eula-changes/69000) · [JUCE 9 licence](https://juce.com/legal/juce-9-licence/) · [JUCE 9 release](https://forum.juce.com/t/juce-9-is-available-now/69175) · [VST 3.8 MIT press release](https://ocl-steinberg-live.steinberg.net/_storage/asset/819253/storage/master/Press%20Release%20-%202025-10-29%20-%20VST%203.8%20-%20EN.pdf) · [clap-juce-extensions](https://github.com/free-audio/clap-juce-extensions) · [Avid AAX](https://developer.avid.com/aax/) · [pamplejuce](https://github.com/sudara/pamplejuce) · [pluginval](https://github.com/Tracktion/pluginval) · [Azure Artifact Signing pricing](https://azure.microsoft.com/pricing/details/artifact-signing/) · [code-signing validity change](https://www.globalsign.com/en/blog/code-signing-validity-changes) · [Moonbase fees](https://moonbase.sh/docs/fees) · [moonbase-cpp](https://github.com/Moonbase-sh/moonbase-cpp) · [PACE pricing](https://paceap.com/products/ilok-software-licensing/pricing-5/) · [sentry-native](https://github.com/getsentry/sentry-native) · [WebView focus issue](https://forum.juce.com/t/fixing-webview-keyboard-focus-issue/63987) · [JUCE WebView tutorial](https://github.com/JanWilczek/juce-webview-tutorial) · [MIDI drag-and-drop](https://forum.juce.com/t/can-one-drag-and-drop-midi-from-a-juce-plug-in-to-the-daw-timeline/27816) · [FL MIDI Out](https://www.image-line.com/fl-studio-learning/fl-studio-online-manual/html/plugins/MIDI%20Out.htm) · [Tracktion Engine](https://github.com/Tracktion/tracktion_engine) · [Tracktion agreement](https://engine.tracktion.com/agreement)

**ML:** [Beat This!](https://github.com/CPJKU/beat_this) · [madmom licence](https://github.com/CPJKU/madmom/blob/main/LICENSE) · [openvpi vocoders](https://github.com/openvpi/vocoders/releases) · [ZFTurbo MSST](https://github.com/ZFTurbo/Music-Source-Separation-Training) · [BS-RoFormer (lucidrains)](https://github.com/lucidrains/BS-RoFormer) · [Mel-RoFormer ONNX export](https://huggingface.co/audiojs/mel-roformer) · [demucs](https://github.com/adefossez/demucs) · [RMVPE](https://github.com/Dream-High/RMVPE) · [FCPE paper](https://arxiv.org/html/2509.15140v1) · [whisper.cpp](https://github.com/ggml-org/whisper.cpp) · [whisper.cpp DTW turbo issue](https://github.com/ggml-org/whisper.cpp/issues/2480) · [basic-pitch](https://github.com/spotify/basic-pitch) · [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) · [ACE-Step 1.5 paper](https://arxiv.org/html/2602.00744v1) · [DirectML → Windows ML](https://learn.microsoft.com/windows/ai/directml) · [mlx-audio-separator](https://pypi.org/project/mlx-audio-separator/) · [Jam-ALT](https://huggingface.co/datasets/audioshake/jam-alt) · [rap-bench](https://github.com/ACNoonan/rap-bench)

**Beats, rhyme, flow:** [MusicRadar mixed-resolution hats](https://www.musicradar.com/how-to/how-to-program-mixed-resolution-trap-style-hi-hat-patterns) · [audeobox trap workflow](https://www.audeobox.com/learn/mpc-software/modern-trap-workflow/) · [cedarsound drill counter-snares](https://www.cedarsoundstudios.com/blogs/news/drill-drum-patterns-101-counter-snares-hi-hat-rolls-and-bounce) · [audeobox FL drill](https://www.audeobox.com/learn/fl-studio/how-to-make-drill-beats-in-fl-studio/) · [audeobox MPC swing](https://www.audeobox.com/learn/mpc-software/mpc-drum-programming/) · [Reason plugg](https://www.reasonstudios.com/news/post/how-to-make-plugg-beats) · [rage beats](https://musicproductionwiki.com/articles/how-to-make-rage-beats) · [jersey club (orphiq)](https://orphiq.com/resources/what-is-jersey-club) · [NI jersey club](https://www.native-instruments.com/blogs/blog/jersey-club) · [afrobeats in FL](https://www.audeobox.com/learn/fl-studio/how-to-make-afrobeats-in-fl-studio/) · [808 design](https://violetrecording.com/how-to-design-an-808/) · [Splice licensing FAQ](https://intercom.help/splice-b7a2d5b83f2e/en/articles/8652642-splice-sounds-licensing-faq) · [JUCE forum: bundling samples](https://forum.juce.com/t/bundling-3rd-party-drum-samples-along-with-a-plugin/45239) · [Groove MIDI Dataset](https://magenta.tensorflow.org/datasets/groove) · [MIDI-GPT](https://github.com/Metacreation-Lab/MIDI-GPT) · [Hirjee & Brown](https://doi.org/10.18061/1811/48548) · [DopeLearning](https://arxiv.org/abs/1505.04771) · [Adams on flow](https://mtosmt.org/issues/mto.09.15.5/mto.09.15.5.adams.php) · [Ohriner, Flow](https://academic.oup.com/book/37383) · [TISMIR vocal microtiming](https://transactions.ismir.net/articles/278) · [RapPad](https://rappad.co)

**Market and legal:** [Production Expert DAW survey 2025](https://www.production-expert.com/production-expert-1/2025-daw-survey-the-results) · [BandLab 100M users](https://musicbusinessworldwide.com/music-making-app-bandlab-surpasses-100-million-users) · [AutoTune 2026](https://www.musicradar.com/music-tech/weve-taken-what-artists-love-about-autotune-and-rebuilt-it-for-a-new-generation-of-creators-antares-announces-autotune-2026-and-discontinues-artist-and-access-editions) · [Antares pricing](https://www.antarestech.com/blog/autotune-evolves-new-pricing-structure) · [Ozone 12](https://www.soundonsound.com/news/izotope-ozone-12-has-arrived) · [NI / inMusic](https://mixmag.net/read/native-instruments-acquired-by-inmusic-following-insolvency-news) · [Slate All Access](https://slatedigital.com/all-access-pass/) · [EU AI-content Code of Practice](https://digital-strategy.ec.europa.eu/en/news/commission-publishes-code-practice-marking-and-labelling-ai-generated-content) · [NO FAKES status](https://www.dailyjournal.com/article/395055-no-fakes-act-dead-for-now-as-cruz-objects-to-unanimous-consent-request) · [ELVIS Act analysis](https://www.proskauer.com/blog/the-king-is-back-in-the-digital-era-the-elvis-act-generative-ai-and-right-of-publicity) · [USCO Part 2](https://www.copyright.gov/newsnet/2025/1060.html) · [Sony v. Udio](https://www.musicbusinessworldwide.com/sony-music-denied-bid-to-add-30k-recordings-to-its-udio-lawsuit/) · [US 5,973,252](https://patents.google.com/patent/US5973252) · [US 8,022,286](https://patents.google.com/patent/US8022286) · [AutoTune Advanced patent-pending](https://www.musicbusinessworldwide.com/antares-launches-autotune-advanced-a-pitch-correction-plugin-built-to-leave-a-singers-vibrato-alone/)
