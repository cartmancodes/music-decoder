# Research: a music production tool for rappers

Date: 2026-11-04 · Status: research / proposal (nothing built yet)

This note looks at what a rapper-focused production tool would need, which
open-source parts exist in late 2026, and how much of the current
`music-decoder` codebase we can reuse. Licences and legal points are flagged
where they affect whether the tool can be used commercially.

---

## 1. What rappers actually do (the workflow to support)

A typical session, start to finish:

1. **Get a beat.** Make one, buy or lease a "type beat", or flip a sample.
2. **Write.** Bars over the beat: rhyme schemes, syllable counts, flow.
3. **Record.** Many takes of each verse, plus punch-ins, doubles and ad-libs.
4. **Process vocals.** Pitch correction (subtle, or the hard-tune effect),
   EQ, compression, de-essing, reverb and delay throws.
5. **Mix and master.** Fit the vocal into the beat and hit streaming loudness.
6. **Export.** Stems, WAV/MP3, a snippet for socials.

Existing products split this up. Suno makes whole songs from a prompt.
BandLab is a free AI-assisted studio (Songstarter, Splitter, Mastering).
Voloco and Rapchat are phone apps for recording with effects. A local,
open-source tool covering steps 1–5, with good **analysis** (which is what
this repo already does well), is a real gap.

---

## 2. What we already have and can reuse

| Existing module | Reuse for rappers |
|---|---|
| `dsp/beats.py`, `dsp/beats_madmom.py` (madmom beat tracking) | Beat grid for the flow analyser, BPM for "type beat" metadata, snapping takes to the grid |
| `key/` (fusion estimator: CNN + Krumhansl-Kessler/Temperley + chord fit) | **Key for auto-tune scale snapping** (the key auto-tune needs is the beat's key), and key tags for beats |
| `chords/` | Chord roots drive the **808 bassline** (808s follow the root) |
| `separation/demucs.py` | Make acapellas/instrumentals, flip samples; add a RoFormer backend for vocals (§3.4) |
| `compose/` (MIDI arrangement, melody, voicings) | Extend with drum and 808 tracks and genre templates |
| `synth/fluidsynth_wrapper.py` + bundled soundfont | Preview MIDI beats now; swap in a drum-sample sampler later |
| `ingest/` (ffmpeg, yt-dlp) | Load reference tracks and beats |
| `ui/streamlit_app.py` | Fine for offline features; **can't record audio well**, so recording needs the React front end (§4) |

So the analysis side (BPM, key, chords, stems) is largely done. The new work
is **generation (beats), the writing tools, recording, and vocal DSP.**

---

## 3. Feature modules and recommended tech

### 3.1 Beat Lab: making beats

Three tiers, cheapest and most controllable first:

**(a) Rule-based, symbolic beats (recommended first; it fits `compose/`).**
Generate MIDI for drums plus an 808 from a style template and the chord
progression:
- Trap: hi-hats on a 1/16 grid with **1/32 and 1/24 (triplet) rolls**, a
  snare or clap on beat 3, syncopated kicks, and an 808 on the chord root
  with **pitch glides** (portamento) and kick/808 alignment. TrapGenerator
  used 1/32 resolution to capture hat rolls for the same reason.
- Boom-bap: swung 1/16 (54–62% swing), kick on 1 and the "and" of 2, snare
  on 2 and 4.
- Drill: about 140 BPM, half-time feel, sliding 808s, triplet hats.
- The output is deterministic MIDI (seeded, like the current `compose`), so
  it is easy to test and the user can edit it in any DAW.
- To render it, add a small sampler (one-shot WAVs mapped to MIDI notes)
  next to fluidsynth. A pitched 808 one-shot played across notes, with
  glides, is the key sound to get right.

**(b) Humanising / ML on MIDI.** Magenta's **GrooVAE** (open code, checkpoints
and Groove MIDI Dataset) adds human timing and velocity, and has a variant
that adds closed hi-hats around a given kick and snare. Its training data is
live drummers, not trap, so treat it as a "feel" knob, not a trap generator.

**(c) Generating beat audio.** The licence decides what is usable:

| Model | Weights licence (verify on the model card) | Fit |
|---|---|---|
| **ACE-Step** (v1 3.5B; v1.5 reported Feb 2026) | **Apache-2.0**, with a LoRA fine-tuning pipeline | **Best candidate** for a commercially usable local generator; fine-tune a LoRA on beats we have rights to |
| Stable Audio Open 1.0 | Stability AI Community License: commercial use allowed under $1M annual revenue | Good for loops and one-shots; trained on CC-licensed audio |
| MusicGen (Meta) | Weights reported as **CC-BY-NC** | Non-commercial only; avoid if beats might be sold |
| Magenta RealTime 2 (Jun 2026, 2.4B) | Open weights; **live** generation steered by MIDI, text or audio, ~200 ms control latency | A "jam with the AI" mode; runs on-device (JAX/MLX) |

Recommendation: ship (a) first. Add ACE-Step as an optional extra in the
same way Demucs is handled today (`fetch-models`, GPU optional), and
generate **stems or loops** rather than finished beats, so the producer
still arranges them.

**(d) Sample flipping.** Load a song, separate it, detect key and BPM (we
already can), chop at onsets (`librosa.onset`), time-stretch and pitch-shift
to the session, and lay the chops on pads. Sampling needs clearance; see §5.

### 3.2 The Pen: a lyric and rhyme workspace

- **Rhyme engine:** CMU Pronouncing Dictionary via `pronouncing`/NLTK, with
  a grapheme-to-phoneme fallback for slang and names (`g2p_en`). Detect
  **multisyllabic rhymes** by matching runs of *stressed vowel phonemes*
  across words and lines, not just word endings. RhymePad (PyPI, beta) does
  colour-coded end, internal, slant and multi rhymes on this exact stack,
  so it is worth reading or vendoring. Raplyzer computes a rhyme-density
  score.
- **Bar and flow grid:** count syllables per bar and lay them on a 16-cell
  grid per bar (one cell per 1/16) so writers can see their cadence before
  recording.
- **Suggestions:** an LLM (Claude via the API) can suggest rhymes and
  multis for a line, or point out weak bars. Keep it **assistive**, never
  writing whole verses by default; for most rappers authorship is the
  point.
- Storage: lyrics are versioned text, which fits Postgres well.

### 3.3 The Booth: recording

- **Browser recording** with an `AudioWorklet` (Baseline in all major
  browsers) writing PCM to OPFS or IndexedDB, then uploading. Textcast is an
  MIT example of AudioWorklet plus OPFS streaming. Browser round-trip
  latency is around 25–50 ms (self-reported by the OpenStudio project), so:
  - monitor the beat only, not the processed vocal, in the browser;
  - **measure latency with a loopback click** and shift takes by that
    amount automatically.
- Features: count-in, loop a section, **take lanes and comping**, punch-in,
  and dedicated double and ad-lib tracks.
- Local-only alternative: Python `sounddevice` capture from the CLI or a
  desktop wrapper gives lower latency but a worse UX.
- Playback and sequencing in the front end: **Tone.js** (transport, sampler,
  scheduling) and **wavesurfer.js** (waveforms and regions). There are
  several MIT React and Tone.js step sequencers (beatsequence, jsWave) to
  borrow patterns from.

### 3.4 Vocal processing

- **Pitch correction (auto-tune):**
  - Offline (MVP): pYIN (`librosa.pyin`) or CREPE for f0, snap to the
    **scale from our key detector**, then resynthesise with PSOLA (`psola`
    package). A retune-speed parameter controls smoothing; speed 0 gives
    the hard T-Pain/Future effect. Reference implementations exist
    (Wolfsound/HF Spaces `pitch_correction_utils.py`).
  - Better quality: a WORLD or neural vocoder keeps formants intact. Real
    time needs C++/WASM (or a commercial SDK such as Superpowered), so
    leave live auto-tune out of v1.
- **Effects chain:** Spotify **`pedalboard`** has HPF, EQ (shelf/peak),
  compressor, noise gate, limiter, reverb, delay, chorus and distortion, can
  host VST3/AU plugins, and has `AudioStream` for live use. It has **no
  de-esser**, so build one (band-pass sidechain to a compressor over
  5–9 kHz). Ship rap presets (clean, melodic, hard-tune, lo-fi, phone
  filter for ad-libs).
- **Stem separation for acapellas:** RoFormer models (BS-RoFormer,
  Mel-Band RoFormer) now beat htdemucs on vocals, at about 11.9 dB SDR on
  MVSEP's Multisong vocals. The `audio-separator` package (PyPI) runs them
  from a CLI or from Python. Add it as a second backend next to
  `separation/demucs.py`.

> **Licence warning:** `pedalboard` is **GPL-3.0**, its `PitchShift` uses
> Rubber Band (GPLv2+/commercial), and **Matchering is GPL-3.0**. This repo
> is MIT. Using them is fine for a tool you run yourself, but distributing
> a combined binary or app means GPL obligations. Either accept GPL for the
> production module, keep it as a separately installed extra, or pick
> permissive alternatives (e.g. `scipy.signal` filters plus our own
> compressor and limiter).

### 3.5 Flow analysis (the feature that sets us apart)

This builds on the repo's strength in analysis:

1. Separate the vocal, then transcribe with **WhisperX** (Whisper plus
   wav2vec2 forced alignment) to get word timestamps, then split words
   into syllable onsets (CMU phones, or vocal onset detection).
2. Overlay the syllables on our **madmom beat grid** and compute:
   - pocket: share of stressed syllables within ±30 ms of a 1/16 grid line;
   - **syllables per beat** per bar (density, double-time sections);
   - triplet-flow detection (onsets clustering on a 1/12 grid);
   - rhyme placement: where rhymes land in the bar (e.g. on beat 4).
3. Show it as a heatmap and compare takes, or compare against a reference
   verse.

Caveat: WhisperX was benchmarked on speech, not rap over beats. Fast
delivery, ad-libs and slang will hurt word error rate. Build a small
hand-labelled rap test set first, the same way we used GuitarSet for tabs,
and measure WER and timestamp error with and without separation.

### 3.6 Mix and master

- **Auto mix:** gain staging, vocal-to-beat balance via loudness, ducking
  the beat in 2–5 kHz under the vocal (dynamic EQ or sidechain), and
  bus-level reverb and delay sends.
- **Mastering:** **Matchering 2.0** matches RMS, frequency response, peak
  level and stereo width to a reference track the user picks, which is how
  rappers already think ("make it sound like X"). It is GPL-3 (see above)
  and offers little control, so pair it with `pyloudnorm` for
  integrated-LUFS targets (about −14 LUFS for streaming, true peak
  ≤ −1 dBTP) and a final limiter.

---

## 4. Architecture

Following the stack in RESEARCH.md (Python, React, Docker, Postgres):

```
React (Vite) ── Tone.js transport/sampler · wavesurfer.js · AudioWorklet recorder
   │  REST + WebSocket (job progress, already modelled by progress.py)
FastAPI ── projects, beats, takes, lyrics, presets
   │
   ├─ Postgres: projects, tracks, takes (+ latency offset), lyrics versions,
   │            analysis results (bpm, key, chords, flow metrics as JSONB)
   ├─ MinIO/S3 (or local disk): audio blobs, stems, renders
   └─ Worker queue (RQ/Celery or arq) with GPU optional:
        music_decoder.{analyze, separate, beatgen, autotune, flow, master}
docker compose: api · worker · postgres · minio · web
```

- Keep `music_decoder` as the **library core**, with new subpackages
  `beatgen/`, `vocals/` (tune, chain, de-ess), `lyrics/` (rhymes,
  syllables), `flow/` and `master/`, each with typed dataclasses in
  `types.py` and the same CLI/library/Streamlit layering as now.
- The CLI keeps working (`music-decoder beat --style trap --bpm 140
  --progression "Am F C G"`), and the web app is a front end over the same
  calls.
- Heavy models (ACE-Step, RoFormer, WhisperX) are optional extras fetched
  by `fetch-models`, like Demucs today.

---

## 5. Legal and ethical guardrails

- **AI-generated beats:** UMG and WMG settled with Udio (Oct and Nov 2025)
  and WMG with Suno (Nov 2025), moving to licensed models. UMG v. Suno and
  Sony v. both were still open as of mid-2026 (I have not checked their
  current status). Only use models with clear licences (ACE-Step Apache-2.0,
  Stable Audio Open within its terms), fine-tune only on data we have
  rights to, and **tag AI-generated content in exported metadata**. Some
  reports say EU AI Act disclosure rules for AI audio apply from Aug 2026.
- **Voice cloning:** don't ship "sound like <artist>" voice conversion (RVC
  and similar). Tennessee's **ELVIS Act** (in force Jul 2024) makes
  unauthorised voice simulation a violation and **reaches tool providers**.
  The federal NO FAKES Act of 2026 is pending. If voice conversion is ever
  added, make it consent-gated (the user's own voice only).
- **Sampling:** the sample-flip feature should warn that a released track
  needs clearance, and should not pull samples from YouTube automatically
  for anything beyond personal practice.

---

## 6. Phased roadmap

| Phase | Scope | Rough effort |
|---|---|---|
| **P1: offline MVP** (CLI + Streamlit) | Rule-based trap/boom-bap/drill beat generator (drums + 808 from progression, MIDI + WAV via a sampler); rhyme and syllable analyser for pasted lyrics; file-based auto-tune snapped to the detected key; pedalboard vocal presets; Matchering + LUFS master | 3–5 weeks |
| **P2: flow and separation** | RoFormer vocal backend via `audio-separator`; WhisperX transcription with flow metrics on the beat grid; build a rap test set and benchmark report (like the GuitarSet one) | 3–4 weeks |
| **P3: web studio** | FastAPI, Postgres and worker in Docker; React + Tone.js + wavesurfer; AudioWorklet recorder with latency calibration, take lanes and comping; projects | 6–8 weeks |
| **P4: generative** | ACE-Step loops and stems (optional GPU extra) with an optional LoRA on licensed beats; GrooVAE humanise; LLM rhyme suggestions; Magenta RT 2 jam mode | 4–6 weeks |

---

## 7. Open questions

1. **Local-only or hosted?** The current tool is local-first ("no
   uploads"). A hosted studio changes the design: auth, storage costs and
   GPU workers.
2. **Will users sell what they make?** This decides the model choice
   (rules out MusicGen) and whether GPL dependencies are acceptable.
3. **Who is the target user?** Bedroom rappers who need a beat and a booth,
   or producers who need analysis and generation? This sets whether P3 or
   P4 comes first.

---

## Sources

- Demucs, Basic Pitch and the existing pipeline: [RESEARCH.md](../../RESEARCH.md)
- ACE-Step licence and LoRA: [ComfyUI wiki](https://comfyui-wiki.com/en/models/ace-step), [Replicate readme](https://internal.replicate.com/lucataco/ace-step/readme), [vast.ai model page](https://vast.ai/model/ace-step-v1-3.5b)
- Stable Audio Open licence and training data: [Stability AI](https://stability.ai/news/stable-audio-open-research-paper), [Digital Music News](https://www.digitalmusicnews.com/?p=345258)
- MusicGen weights licence (forum report, verify on the model card): [HN thread](https://hn.matthewblode.com/item/36972347)
- Magenta RealTime / RT 2: [Magenta RT](https://magenta.tensorflow.org/magenta-realtime), [Magenta RT 2](https://magenta.tensorflow.org/magenta-realtime-2), [Live Music Models paper](https://arxiv.org/html/2508.04651v3), [Simon Willison](https://simonwillison.net/2025/Jun/20/magenta-realtime)
- GrooVAE: [Magenta](https://magenta.tensorflow.org/groovae), [paper](https://arxiv.org/pdf/1905.06118v2.pdf); trap hats: [TrapGenerator](https://github.com/mattiasu96/TrapGenerator)
- RoFormer separation: [Mel-Band RoFormer paper](https://arxiv.org/pdf/2310.01809), [MVSEP leaderboard](https://mirror.mvsep.com/algorithms), [audio-separator (PyPI)](https://pypi.org/project/audio-separator/)
- Rhyme analysis: [RhymePad](https://pypi.org/project/rhymepad/), [BarBreakDown](https://github.com/jakezz/BarBreakDown), [Raplyzer](https://github.com/ekQ/raplysaattori)
- Browser audio: [AudioWorklet baseline](https://web-platform-dx.github.io/web-features-explorer/features/audio-worklet.json), [OpenStudio](https://gh.r-mj.com/braedonsaunders/openstudio), [Textcast](https://awesome.ecosyste.ms/projects/github.com%2Fsirmews%2Ftextcast), [Superpowered](https://superpowered.com/riding-the-latest-waves-on-the-web), [beatsequence](https://awesome.ecosyste.ms/projects/github.com%2Fsteve-mackinnon%2Fbeatsequence), [jsWave](https://gh.r-mj.com/ruizterce/jsWave)
- Vocal DSP: [pedalboard](https://github.com/spotify/pedalboard), [PSOLA auto-tune reference](https://huggingface.co/spaces/nateraw/singing-songstarter/blob/main/pitch_correction_utils.py), [scale-snap variant](https://huggingface.co/spaces/kindahex/Hex-AutoTune/blob/main/autotune_script.py), [data-driven pitch corrector](https://github.com/jamesbrownjr/data_driven_pitch_corrector)
- Mastering: [Matchering](https://github.com/wokashi-rg/Matchering), [bonedo review](https://www.bonedo.de/artikel/matchering-20-erlaubt-kostenloses-mastering-auf-open-source-basis)
- Transcription and alignment: [WhisperX paper](https://arxiv.org/abs/2303.00747v1)
- Market: [BandLab / Suno comparison](https://vidmuse.ai/blog/best-ai-music-generator), [AI tools 2026](https://resources.rework.com/tools/ai-tools/best-ai-tools-for-music-production-2026), [Voloco stats](https://www.similarweb.com/it/app/google/com.jazarimusic.voloco/), [AI beat genres](https://getrhythmm.com/knowledge/what_is_the_best_ai_beat_generator_in_2026.php)
- Legal: [UMG–Udio](https://www.musicbusinessworldwide.com/universal-music-settles-udio-lawsuit-strikes-deal-for-licensed-ai-music-platform), [WMG–Suno](https://www.thestar.com.my/tech/tech-news/2025/11/26/warner-music-group-settles-copyright-case-with-suno-for-licensed-ai-music), [Suno/Udio 2026 status](https://blog.dubspot.com/suno-udio-ai-music-lawsuits-2026), [ELVIS Act](https://www.reedsmith.com/articles/tennessee-efforts-protect-rop-against-genai-imitations-deepfakes/), [NO FAKES Act 2026 text](https://www.govinfo.gov/content/pkg/BILLS-119s4591is/html/BILLS-119s4591is.htm)
