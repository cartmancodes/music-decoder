# Rapper production tool: what "production quality" requires

Date: 2026-11-04 · Status: research / proposal ·
Follows [2026-11-04-rapper-production-tool.md](2026-11-04-rapper-production-tool.md)

> **Superseded by the
> [master plan](../superpowers/plans/2026-11-04-rapper-suite-master-plan.md).**
> Corrections from later research: target JUCE 9 (not 8); tuner latency
> target ≤ 3 ms (not 5); BS.1770-5 (not -4); ACE-Step 1.5 is MIT;
> PC-NSF-HiFiGAN and madmom model weights are non-commercial
> (CC BY-NC-SA); `compose/` has no drum rules to port; the ML sidecar is a
> C++ ONNX Runtime / whisper.cpp helper, not shipped Python.

The first plan picked tools for a quick offline prototype (PSOLA auto-tune,
browser recording, a soundfont for drums, Matchering). Those work for a demo
but **would not pass with an engineer or a signed artist**. This note sets a
measurable quality bar, marks which earlier picks fall short of it, and
proposes an architecture that can meet it.

---

## 1. The bar, in numbers

What "sounds and behaves like a pro studio tool" means, stated so each item
can be tested in CI or in a listening panel:

| Area | Requirement |
|---|---|
| Signal path | 32-bit float internally; 48 kHz session default (44.1/88.2/96 supported); high-quality sample-rate conversion (soxr VHQ or r8brain) |
| Export | 24-bit WAV, 16-bit WAV with TPDF dither, 320 kbps MP3 / AAC; stems with identical length and start time |
| Tracking latency | **≤ 10 ms round trip** while monitoring *through* effects (rappers track hearing their auto-tune); 64–128-sample buffers via ASIO / CoreAudio / WASAPI exclusive |
| Take alignment | Latency compensation accurate to **±1 sample**, measured with a loopback test, not guessed |
| Take safety | Audio streams to disk while recording; crash recovery; **no lost take, ever** |
| Loudness | ITU-R BS.1770-4 integrated LUFS; true-peak meter with ≥ 4× oversampling; presets for streaming (−14 LUFS, ≤ −1 dBTP) and "loud rap master" (≈ −8 to −6 LUFS) |
| Non-linear DSP | Saturation, clipper and 808 distortion oversampled ≥ 4× (no audible aliasing); linear stages pass null tests |
| Render parity | Offline bounce is bit-identical, or nulls below −120 dBFS, against real-time playback |
| Pitch correction | Low-latency mode for tracking (hard-tune ≤ ~5 ms added latency); offline note editing that **preserves formants**; no "chipmunk" or phasey artifacts on a blind listening panel |
| Time and pitch stretch | At least as good as Rubber Band R3 on vocals and loops |
| Sounds | Licensed 24-bit drum kits with velocity layers and round-robins; tuned 808s with glide; **no General MIDI soundfont in final renders** (fine for preview) |
| Interop | Hosts VST3 / AU (CLAP nice to have) so users' Auto-Tune, FabFilter or Waves plugins work; MIDI and stem drag-and-drop into FL Studio, Ableton, Logic and Pro Tools |

The most important point: **pros don't give up their plugins.** A tool that
can't host Auto-Tune Pro or FabFilter, or can't run *inside* FL Studio, won't
be adopted however good its own DSP is.

---

## 2. Earlier picks, re-graded against the bar

| Earlier pick | Verdict | Production replacement |
|---|---|---|
| Browser recording (AudioWorklet, ~25–50 ms) | ❌ above the ≤ 10 ms target for monitoring through effects | Native audio engine (JUCE, §3). The browser stays a companion for writing and rough demos (direct monitoring + compensation) |
| PSOLA auto-tune (pYIN + `psola`) | ❌ offline only, artifacts, no formant control | **Real-time:** our own low-latency corrector in C++ (x42/zita-AT1 shows the method; its own docs say it's for small corrections and has no formant handling, so it's a reference, not the product). **Offline graphical:** RMVPE or FCPE pitch → note editing → PC-NSF-HiFiGAN, a formant-preserving vocoder trained on 79 h of 44.1 kHz singing |
| `pedalboard` vocal chain | ⚠️ good DSP, but Python and GPL-3 | Native C++ DSP in the engine; keep pedalboard for prototyping and offline test renders |
| Matchering 2.0 | ⚠️ reference matching only, few controls, GPL-3 | Custom chain: dynamic EQ → multiband comp → clipper → true-peak limiter (oversampled); keep reference-track *matching* as one assist inside it |
| Fluidsynth + GM soundfont | ❌ for final output | Licensed sample kits + a purpose-built 808 voice (sample or synth, glide, drive) |
| Streamlit UI | ❌ for a product | React UI hosted in JUCE 8's WebView, which keeps your React skills on the front end |
| Rubber Band (GPL / commercial) | ✅ quality, ⚠️ licence | **Signalsmith Stretch (MIT)** as default (informal tests rate it at or above R3; verify on our material); Rubber Band R3 under a commercial licence as an option |
| RoFormer separation | ✅ best open option (~12 dB SDR vocals) | Keep it, but label output "for sampling and remixing". Artifacts remain audible, so it is not release-grade for acapellas |
| ACE-Step for beats | ⚠️ v1.5's own paper places it **between Suno v4.5 and v5** in blind tests; Suno still leads on polish | Use it for **ideas and loops**, then separate into stems and let users replace drums with pro samples. Core beat-making stays symbolic (MIDI) + pro samples, which is both higher quality and editable |
| Postgres for everything | ⚠️ wrong place for a desktop app's projects | Local project folder (audio + JSON/SQLite); **Postgres + object storage + Docker only for cloud** (accounts, sync, collaboration, beat store, GPU jobs) |

---

## 3. Architecture for production

### Recommendation: build a plugin suite + standalone app, not a full DAW

Rap production mostly happens in FL Studio, with Ableton, Logic and Pro
Tools next. Building a DAW that beats those takes years. Building **JUCE
plugins** gives VST3 / AU / AAX **and** a standalone app from one codebase,
running inside the DAWs people already use, so the work goes into the
rap-specific parts.

```
┌─────────────── JUCE 8 (C++) — one codebase ───────────────┐
│  Targets: VST3 · AU · AAX · Standalone                     │
│                                                            │
│  1. VOX     — rap vocal chain: low-latency tune, de-ess,   │
│               comp, EQ, saturation, delay/reverb throws,   │
│               doubler, ad-lib presets                      │
│  2. BEAT    — MIDI-out drum + 808 generator (style, chords,│
│               hat rolls, glides) + built-in sampler        │
│  3. MASTER  — loudness-targeted master chain + TP limiter  │
│                                                            │
│  UI: React + TypeScript in JUCE WebView                    │
│  DSP: C++ (oversampling, SIMD), Signalsmith Stretch        │
└─────────────┬──────────────────────────────────────────────┘
              │ localhost gRPC/HTTP (async jobs, file handles)
┌─────────────▼──────── Python ML sidecar (local) ──────────┐
│ music_decoder core: key · chords · beats/BPM (existing)    │
│ + separation (RoFormer via audio-separator)                │
│ + offline tuning (RMVPE/FCPE → PC-NSF-HiFiGAN)             │
│ + flow analysis (WhisperX + beat grid), rhyme engine       │
│ + optional ACE-Step 1.5 idea generator (GPU)               │
│ Packaged with uv/PyInstaller; models via fetch-models      │
└─────────────┬──────────────────────────────────────────────┘
              │ (optional, later)
┌─────────────▼──────── Cloud (Docker) ─────────────────────┐
│ FastAPI · Postgres · S3/MinIO · GPU workers                │
│ accounts, project sync, collaboration, beat/kit store      │
└────────────────────────────────────────────────────────────┘
```

- **Real-time work stays in C++** (anything a rapper hears while
  tracking). **Heavy or offline ML stays in Python**, reusing this repo.
  Results come back as rendered audio clips or MIDI.
- The **standalone app** is the "booth" for rappers without a DAW:
  record, comp takes, apply VOX, and drop in a BEAT. For multitrack
  recording, comping, automation and plugin hosting, **Tracktion Engine**
  (built on JUCE, GPL / commercial) saves a large amount of work compared
  with writing our own.
- Real-time key detection inside VOX (so auto-tune follows the beat) needs
  a C++ port of the fused key estimator; until then the sidecar analyses
  the beat once and sends the key to the plugin.

### Licensing: one decision to make up front

| If the product is… | Then |
|---|---|
| **Open source (AGPL/GPL)** | JUCE under AGPLv3, Tracktion Engine GPL, pedalboard / Matchering / Rubber Band GPL all fine. Free, but the whole app must be released under a compatible licence |
| **Closed / commercial** | JUCE commercial licence (Starter is free below a trailing-12-month revenue cap; check juce.com for current numbers), Tracktion Engine commercial (per-seat tiers), **no GPL code shipped** (reimplement what pedalboard / Matchering do in C++, use Signalsmith Stretch or licence Rubber Band). Check model-weight licences one by one (ACE-Step Apache-2.0 ✅; RMVPE, FCPE and PC-NSF-HiFiGAN weights must be checked before shipping) |

The music-decoder Python core is MIT, so it is fine either way.

---

## 4. Quality assurance (how we prove it)

1. **Golden-render tests:** fixed inputs → render → compare with stored
   reference (null test threshold); runs in CI for every DSP change.
2. **Measurement tests:** LUFS and true peak per BS.1770-4 on exports;
   aliasing sweep (sine sweep through non-linear stages, check the
   spectrum above Nyquist fold); latency loopback rig on macOS and Windows
   interfaces.
3. **Real-time safety:** no allocations or locks on the audio thread
   (checked with a RealtimeSanitizer-style tool); plugin validation with
   `pluginval` at strictness 10 and Steinberg's validator; tested in FL
   Studio, Ableton, Logic and Pro Tools.
4. **Listening panel:** blind ABX / MUSHRA against reference chains
   (e.g. Auto-Tune Pro + stock chain) with engineers and rappers. Ship a
   feature only when it is statistically indistinguishable from the
   reference, or preferred.
5. **Rap test corpus:** licensed dry vocals and beats (with permission),
   hand-labelled for pitch, lyrics, syllable onsets and beat grid. It plays
   the same role GuitarSet plays for our tab work, and drives the flow and
   transcription benchmarks.
6. **Reliability:** crash reporting, autosave, kill-the-process-mid-take
   recovery test, 8-hour soak test.

---

## 5. Revised roadmap (2–3 engineers: C++/DSP, React, Python/ML)

| Phase | Scope | Rough effort |
|---|---|---|
| **P0 Quality harness** | Golden renders, LUFS/TP/aliasing checks, latency rig, pluginval in CI, listening-test protocol, start the rap corpus | 3–4 weeks |
| **P1 VOX** (highest value for rappers) | Low-latency pitch correction (key from sidecar), de-esser, comp, EQ, oversampled saturation, throws, presets; VST3/AU/standalone; React UI | 3–4 months |
| **P2 BEAT** | MIDI-out drum/808 generator (port style rules from `compose`), sampler + licensed kit, drag-out MIDI/stems | 2–3 months |
| **P3 Booth + AI tools** | Standalone multitrack recording and comping (Tracktion Engine); sidecar: separation, graphical formant-preserving tuning, flow analysis, rhyme notebook | 3–4 months |
| **P4 MASTER + generation** | Loudness-targeted mastering; optional ACE-Step idea → stems; cloud sync | 2–3 months |

Plan on **about 9–14 months to a credible v1**. P1 alone is a shippable
product (a rap vocal-chain plugin), so revenue or user feedback can come
early.

---

## 6. Decisions needed from you

1. **Open source (AGPL) or closed commercial?** This drives the JUCE,
   Tracktion and GPL choices above.
2. **Plugin suite first (recommended), or a standalone studio first?**
3. **Target platforms:** macOS + Windows (recommended), or iOS too? Many
   rappers record on phones; AUv3 is possible from the same JUCE code.

---

## Sources

- Pitch correction: [x42-autotune](https://x42-plugins.com/x42/x42-autotune), [Zynthian AT1 docs](https://zynthian.org/engines/_engine-list/engine-x42-fat1), [Ardour forum on AT1 quality](https://discourse.ardour.org/t/pitch-correction/86293)
- Neural pitch / vocoder: [RMVPE](https://arxiv.org/abs/2306.15412v2), [FCPE](https://arxiv.org/html/2509.15140v1), [PC-NSF-HiFiGAN (openvpi)](https://github.com/openvpi/vocoders/releases), [pitch-shift restoration](https://arxiv.org/html/2601.10345v1)
- Time stretch: [HISE forum comparison](https://forum.hise.audio/post/67995), [Bungee comparison](https://bungee.parabolaresearch.com/compare), [Rubber Band R3](https://www.breakfastquay.com/news/20220707.html), [Tenacity issue](https://zff.dev/tenacityteam/tenacity/issues/377)
- AI generation quality: [ACE-Step 1.5 paper](https://arxiv.org/html/2602.00744v1), [ACE-Step 1.5 repo](https://github.com/ace-step/ACE-Step-1.5), [Z.Tools review](https://z.tools/blog/ace-step-v1-5-review)
- Latency: [Antares on low-latency monitoring](https://www.antarestech.com/blog/low-latency-monitoring-in-daws-what-it-is-and-why-it-matters), [latency calculator](https://bchillmix.com/pages/latency-calculator), [OpenStudio browser figures](https://gh.r-mj.com/braedonsaunders/openstudio)
- JUCE: [JUCE 8 EULA amendments](https://forum.juce.com/t/amendments-to-the-juce-end-user-licence-agreement-for-juce-8/61265), [revenue tiers](https://forum.juce.com/t/revenue-limits-for-juce-tiers/61058), [JUCE 8 features](https://audioxpress.com/news/juce-8-adds-important-new-features-and-enhancements-for-audio-application-and-plugin-development), [React in JUCE WebView write-up](https://zenn.dev/jun_murakami/articles/f9b3227fadfe7e)
- Tracktion Engine: [repo](https://github.com/Tracktion/tracktion_engine/), [licensing update](https://forum.juce.com/t/announcement-tracktion-engine-licensing-update/56897)
- Plugin hosting from Python (prototyping): [DawDreamer](https://GitHub.com/DBraun/DawDreamer), [pedalboard](https://github.com/spotify/pedalboard)
