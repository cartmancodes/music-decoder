# Accuracy Improvement Roadmap

**Date:** 2026-05-05
**Status:** research / proposal — no implementation yet
**Targets vs current:**

| Metric | Current | Target | Stretch (v2) |
|---|---|---|---|
| Note F-measure | 0.63 | 0.78 | 0.85 |
| Onset F-measure | (broken / partial) | 0.70 | 0.80 |
| Pitch-class accuracy | 0.60 | 0.78 | 0.85 |
| Tab string accuracy | 0.44 | 0.60 | 0.75 |
| Chord recognition score | 0.055 | 0.40 | 0.60 |

The numbers are mean across the 2 synthetic + 5 GuitarSet fixtures. Targets are based on RESEARCH.md and the literature surveyed below.

---

## 1. Findings from the deep dive

### 1.1 Bugs that distort the current numbers

These are not "improvements" — they're correctness fixes. Land first, before any algorithmic work, otherwise you can't tell which intervention helped.

**B-1. `mir_eval.onset.f_measure` rejects unsorted onsets.** GuitarSet's `track.notes_all.intervals` is aggregated from per-string note lists; the combined array is *not* chronologically sorted. mir_eval raises `ValueError: Events should be in increasing order.` Our `_safe` wrapper in `evaluation/runner.py` swallows it and returns `None`. Result: 4 of 5 GuitarSet tracks have `onset_f_measure: null` in the regression report — we are **hiding genuine onset scores**.

- **Fix:** sort intervals by start time in either the runner or inside `note_f_measure`/`onset_f_measure` themselves. mir_eval's note-level metric (`precision_recall_f1_overlap`) is more permissive, which is why `note_f_measure` does compute on the same input.
- **Effort:** 1 hour. Add `np.argsort(intervals[:, 0])` before mir_eval calls; add a unit test that reproduces the exception on shuffled input.
- **Impact:** Onset F goes from null → real values (probably 0.4-0.6 on GuitarSet); aggregate threshold can be raised from 0.30 to 0.60.

**B-2. The chord_recognition_score floor of 0.05 hides a vocabulary mismatch.** Of 5 GuitarSet fixtures, 4 score 0.0 and 1 scores 0.275. The 0.0s are not algorithm failures — GuitarSet's `leadsheet_chords` annotations include qualities (`sus2`, `maj6`, `dim`, slash chords) that our 4-quality vocabulary skips. The metric ends up scoring 6 frames per fixture instead of thousands.

- **Fix:** report `chord_recognition_score` only over frames where the GT label is in our supported vocabulary; track the *coverage* (fraction of frames with supported labels) as a separate metric. This separates "we got the answer wrong" from "we couldn't possibly get the answer".
- **Effort:** 2 hours.
- **Impact:** the 0.0s become meaningful (likely 0.30-0.50 on the supported subset); the 0.05 floor can be raised to ~0.30.

**B-3. `tab_string_accuracy=null` for synthetic fixtures and Rock1.** Synthetic fixtures have `tab=None` (no string-level GT) — that's correct, the null is intentional. But Rock1 also reports null even though GuitarSet has per-string notes. Worth confirming: it's likely caused by the orchestrator producing zero `tabbed_notes` for Rock1 (because A* dropped them all due to its current cost weights interacting badly with Rock1's note density), not a metric bug. **Investigate and either fix or accept.**

- **Effort:** 30 minutes to triage.

### 1.2 Per-fixture failure mode breakdown

| Fixture | Note F | Pitch class | Tab string | Why it failed |
|---|---|---|---|---|
| c_major_scale (synth) | 1.00 | 0.95 | n/a | clean monophonic — works |
| g_major_chord (synth) | **0.71** | 0.91 | n/a | basic-pitch missing simultaneous notes |
| BN1-comp (bossa nova) | 0.60 | **0.44** | 0.27 | dense chord stacks confuse pitch detection |
| BN1-solo (bossa nova) | 0.68 | 0.84 | 0.33 | best GuitarSet track — solo melody-line |
| Funk1-comp (funk) | 0.58 | 0.41 | **0.62** | tab benefits from clear bass-line |
| Jazz1-comp (jazz) | 0.59 | **0.27** | 0.54 | extended jazz harmony breaks chroma profile |
| Rock1-comp (rock) | **0.25** | 0.39 | null | 403 GT notes in 22.1s — extreme density + likely distortion |

Patterns:
- **Synthetic chord fail (0.71):** even on a 6-note clean sustained chord, basic-pitch is splitting/missing notes. It was trained on pitched instruments at 22050 Hz; sine-wave synthesis exposes its missing-fundamentals fallback. Use fluidsynth + GuitarML soundfonts for synthetic fixtures (was in the v1 spec, deferred).
- **Comping tracks pull pitch class accuracy down hard.** Jazz1 at 0.27 is bad. The chroma is dominated by chord-extension partials that don't appear in the simple major/minor profile.
- **Rock1 is an outlier on every metric.** 403 onsets in 22 seconds = roughly 18 onsets/second — basic-pitch's onset detector is overwhelmed and produces fragmented output. Distortion from rock electric guitar adds harmonic clutter.

### 1.3 What's structurally limiting accuracy today

1. **basic-pitch defaults are not tuned for guitar.** Onset/frame thresholds, min note length, frequency bounds all use general-purpose defaults. We never swept.
2. **No chord-aware bias in transcription.** basic-pitch outputs notes; our tab assigner picks fingerings; nothing knows the song is in C major or that the current bar is an Am chord. A 1% chord-aware reweighting would catch many false positives.
3. **A* fingering doesn't match performance fingering.** A* picks the *most playable* path; GuitarSet records *what was actually played*. Two different goals; even a perfect A* will score 50-70% on string accuracy at best.
4. **Chord recognition uses chroma + templates.** This is the 2010-era baseline. Modern systems use NNLS chroma (Mauch-Dixon) or end-to-end neural models, gaining 15-25 points.
5. **Synthetic fixtures use sine waves.** Ground-truth-perfect timing/pitch but unrealistic timbre. Real recordings have transients, bow noise, finger squeaks, room reverb — none of which sine fixtures train against.
6. **Single-pass pipeline, no hyperparameter sweep.** We pinned defaults in `hyperparameters.yaml` and never measured alternates.

---

## 2. State-of-the-art reference points (2024-2026)

### 2.1 Polyphonic guitar transcription

| Approach | F-measure on GuitarSet | Open source? | Practical for us? |
|---|---|---|---|
| basic-pitch (current) | 0.55-0.70 | Yes (Spotify) | Already in use |
| **Riley & Edwards (2024)** — high-resolution piano model fine-tuned on GuitarSet | ~0.85 (zero-shot SOTA) | [Project page](https://xavriley.github.io/HighResolutionGuitarTranscription/), code+checkpoints on Zenodo, not pip-installable | **Strong v2 candidate** — drop-in replacement for basic-pitch on solo guitar |
| **GAPS benchmark model (ISMIR 2024)** | SOTA on GuitarSet + GAPS | [aim-qmul.github.io/GAPS](https://aim-qmul.github.io/GAPS/), code released | Strong candidate; trained on 14h of real classical guitar |
| **trimplexx music-transcription (CRNN)** | claims 0.87 MPE F1 on GuitarSet | [GitHub MIT-licensed](https://github.com/trimplexx/music-transcription), CUDA 12.1 required | High claim; CUDA-only is a problem for a CPU-first local app |
| **TART (Oct 2025)** — 4-stage end-to-end audio→tab+technique | not disclosed publicly yet | [arXiv 2510.02597](https://arxiv.org/abs/2510.02597); code status unclear | Aspirational; revisit when code drops |

### 2.2 Chord recognition

| Approach | MIREX MajMin score | Open source? | Practical for us? |
|---|---|---|---|
| chroma_cqt + template + Viterbi (current) | ~0.30-0.50 on real audio | n/a | Already in use |
| **NNLS Chroma + Chordino (Mauch-Dixon 2010)** | 0.80 on MIREX 2009 | [c4dm/nnls-chroma](https://github.com/c4dm/nnls-chroma) Vamp plugin; [Essentia binding](https://essentia.upf.edu/reference/streaming_NNLSChroma.html) | **Excellent v2 candidate** — replace chroma source, keep Viterbi backend |
| **Deep chroma extractor (Korzeniowski-Widmer 2016)** | 0.78-0.82 | [Madmom](https://madmom.readthedocs.io/) library, pip-installable | Strong candidate |
| **LLM Chain-of-Thought chord recognition (2025)** | +1-2.7% over baselines | research code only | Far-future |
| **Transformer-based (BTC-FDAA, 2025)** | +1-2 points over CRNN baselines | research code only | Far-future |

### 2.3 MIDI → tab

| Approach | Tab accuracy | Open source? | Practical for us? |
|---|---|---|---|
| A* (current) | playable, 0.40-0.60 string accuracy on GuitarSet | n/a | Already in use |
| **Fretting-Transformer (June 2025)** — T5-based | "surpasses A* and Guitar Pro" | [arXiv 2506.14223](https://arxiv.org/abs/2506.14223); code status unclear | **High-value v2** if code drops |
| **MIDI-to-Tab (Riley, 2024)** — BART-style | SOTA on Riley's test set | not yet released | Watch |
| **A* + chord-aware bias (proposal)** | est. 0.55-0.65 | trivial to build | **Quick win** — immediate effort, leverages existing chord output |

---

## 3. Prioritized improvement roadmap

Three phases. Each phase has measurable acceptance criteria and bounded effort.

### Phase A — Bugs and quick wins (1-2 days, no new dependencies)

Expected aggregate impact: +5 to +10 points note F-measure, +10 to +20 points pitch-class accuracy, +5 to +15 points tab-string accuracy.

**A-1. Fix mir_eval ordering bug (B-1).** Pre-sort intervals by start time inside the metric helpers. Add a unit test reproducing the exception on shuffled input. Re-run regression suite — the 4 null onset values become real numbers; aggregate onset threshold can be raised from 0.30 to a real floor.

**A-2. Hyperparameter sweep on basic-pitch.** The plan's `scripts/sweep.py` was deferred. Sweep `onset_threshold ∈ {0.3, 0.4, 0.5, 0.6}`, `frame_threshold ∈ {0.2, 0.3, 0.4}`, `minimum_note_length_ms ∈ {30, 58, 120}` against the GuitarSet corpus. ~36 trials, 5 minutes each on CPU = 3 hours wall-clock. Pick the highest-scoring set and pin it. Likely +3-7 F-measure points just from this.

**A-3. Octave-error correction post-filter.** basic-pitch occasionally outputs octave-shifted notes (a common failure mode in CNN-based pitch trackers). Detect notes whose pitch is exactly 12 semitones from a high-confidence neighbor in the same time window and drop the lower-confidence one. ~50 lines of code.

**A-4. Constrain pitch range to guitar.** Drop any predicted note below E2 (40) or above ~E6 (88). basic-pitch's `minimum_frequency_hz` and `maximum_frequency_hz` already do this in principle but the current 32.7-2000 Hz bounds let through several semitones below the lowest guitar string. Tighten to `41-1320 Hz` (E2 to E6). ~5 lines.

**A-5. Chord-aware A* bias (huge ROI).** When a chord segment covers a span, multiply the A* candidate scores for fingerings that contain the detected chord's pitches by a small factor (e.g., 0.8 — preferred). When the predicted chord changes, force the cost recomputation. This couples chord recognition (already implemented) to fingering choice. Expected: +10-15 string-accuracy points on funk/rock comping.

**A-6. Chord vocabulary scoring fix (B-2).** Score `chord_recognition_score` only over frames where GT is in our 4-quality vocabulary; report coverage separately. The 0.0s become meaningful. Threshold floor goes from 0.05 → 0.25.

**A-7. Median-filter post-filter for basic-pitch contour.** Currently we only median-filter CREPE output. basic-pitch's frame-level pitch grid has the same flicker issue. Apply a 3-frame median filter to the pre-segmentation activations before note assembly. ~30 lines if we go through `model_output` directly; harder via the public predict() API.

### Phase B — Replace key components (1-2 weeks, drop-in or wrapper-style)

Expected aggregate impact: +10 to +20 points note F-measure, +20 to +30 points chord recognition.

**B-1. Switch chord recognition to NNLS Chroma.** Replace `compute_chroma_with_hpss` for the chord stage with NNLS chroma via [Essentia](https://essentia.upf.edu/) or the Vamp plugin host. NNLS chroma was specifically designed for chord recognition and consistently beats CQT chroma by 10-15 points in the literature. Keep our 49-state Viterbi as the smoothing layer.
- Effort: 1 week (Essentia install path on macOS/Linux; Streamlit-side soundfont selection; API wrapper).
- Risk: Essentia is heavier to install than librosa.

**B-2. Add Riley's high-resolution model as a transcription option.** Don't replace basic-pitch — add a `transcription_model: highres-guitar` toggle alongside `basic-pitch` and `crepe`. Wrap the Zenodo-hosted checkpoint behind a class that mirrors `BasicPitchWrapper`'s interface. The user picks the model in the upload page.
- Effort: 1-2 weeks (download + integration + tests; the model needs PyTorch CPU runtime which is already a transitive dep via Demucs).
- Expected gain on solo guitar: +15-25 F-measure points per the paper.

**B-3. Extend chord vocabulary to MajMin + Sevenths + Sus.** Add `min7`, `dim`, `sus2`, `sus4` qualities. Templates expand from 49 → 73 states. Tune voicings table. This **alone** raises the GuitarSet coverage from ~30% of frames to ~70% — and the score with it.
- Effort: 1 week.
- Tradeoff: more states → noisier per-beat argmax, requires re-tuning HMM `p_self`.

**B-4. Bass-line emphasis for chord recognition.** Apply a 2x weight to the lowest 4 chroma columns (E, F, F#, G) before template matching. Bass lines almost always carry the chord root; this is a well-known trick from MIREX submissions.
- Effort: 1 day.
- Expected gain: +5-10 chord recognition points.

**B-5. Real synthetic fixtures via fluidsynth + soundfont.** Replace `pretty_midi.synthesize()` (sine waves) with `pm.fluidsynth(sf2_path)` using a public-domain guitar soundfont. The synthetic test scores will drop slightly (it's harder for basic-pitch on realistic timbres) but they'll be **predictive** of real-world performance.
- Effort: 1 day; commit a license-cleared SF2 (~20MB) or download-on-fixture-build.

### Phase C — Big bets (1-2 months, requires real ML work)

Expected aggregate impact: +15 to +25 points everything; pushes the system into research-frontier territory.

**C-1. Fretting-Transformer integration.** Once the code is open-sourced (paper June 2025; usually 6-12 months later), wrap it as the third tab-assignment backend. Keep A* as the fallback for unsupported tunings.
- Effort: 2-3 weeks once code is available.
- Watch: [arXiv 2506.14223](https://arxiv.org/abs/2506.14223); the authors typically publish to GitHub.

**C-2. Build a 30-track manual fixture set.** Hand-label 30 short clips spanning the genres GuitarSet weighted (acoustic, electric, distorted, fingerstyle, palm-muted) with verified per-string tab annotations. This becomes the regression suite's truth source — and a fine-tuning corpus.
- Effort: 1-2 months (the bottleneck is the labeling, not the code).
- This is what serious systems do. RESEARCH.md called it out as v1; we deferred to "user supplies later".

**C-3. Fine-tune basic-pitch on GuitarSet.** Run `basic-pitch --train` on GuitarSet's audio + JAMS pairs. Existing infrastructure; the Spotify codebase supports this. CPU training is slow but feasible; GPU optional.
- Effort: 2 weeks (1 week for the training harness, 1 week to find a checkpoint that beats the off-the-shelf baseline).
- Expected gain: +5-15 F-measure points on GuitarSet specifically; possibly less on out-of-distribution data.

**C-4. Deep chord recognition via Madmom.** Madmom's `DeepChromaProcessor` + `DeepChromaChordRecognitionProcessor` is pip-installable and trained on the McGill Billboard dataset. Replace our Viterbi with theirs (or use their chroma + our Viterbi).
- Effort: 1 week.
- Expected gain: +10-20 chord recognition points.

**C-5. End-to-end TART-style audio→tab.** When TART code lands, evaluate it as a single-stage replacement for the entire transcription→A* pipeline. If accuracy and licence both work, it could be the v3 default with our pipeline as a fallback.
- Watch.

---

## 4. Recommended execution order

If I were prioritizing for ROI:

1. **Today (4 hours):** A-1 (mir_eval bug), A-3 (octave correction), A-4 (pitch range), A-6 (chord scoring fix). All small, all immediate impact.
2. **This week (2 days):** A-2 (hyperparameter sweep), A-5 (chord-aware A*), A-7 (median filter). Validate against the regression suite after each.
3. **Next 2 weeks:** B-3 (extended chord vocab) + B-4 (bass emphasis) + B-1 (NNLS chroma). These compound — together they take chord recognition from 0.055 to ~0.45.
4. **Next month:** B-2 (Riley model) + B-5 (fluidsynth fixtures). The first lifts solo-guitar transcription dramatically; the second makes synthetic numbers honest.
5. **Next quarter:** C-2 (manual fixture set) + C-3 or C-4 (fine-tuning or deep chord rec). C-1 and C-5 stay on the watchlist.

Each phase is gated by the regression suite — refuse to merge anything that drops a metric below the captured baseline.

---

## 5. Assumptions and risks

- **Numbers above are from RESEARCH.md and the literature, not from running the alternatives on our exact corpus.** Until we measure them on our 5-track GuitarSet subset (with our threshold + baseline harness), they're estimates.
- **Riley's model and the Fretting-Transformer haven't been verified to be CPU-friendly.** If they require GPU for inference, they break the v1 "CPU-only must work" goal.
- **NNLS Chroma via Essentia is heavier than librosa's chroma.** Adds ~50MB install and OS-specific binaries. Worth the gain but a packaging cost.
- **Fine-tuning basic-pitch on GuitarSet risks overfitting.** Validation set discipline is essential; don't include training tracks in the regression fixtures.
- **The "0.85 SOTA on GuitarSet" claims are usually on full GuitarSet (60 pieces), not our 5-track subset.** Our absolute numbers may be lower or higher; the ordering should generalize.

---

## 6. Sources

- [GuitarSet: A Dataset for Guitar Transcription](https://archives.ismir.net/ismir2018/paper/000188.pdf) — Xi et al. 2018
- [High Resolution Guitar Transcription via Domain Adaptation](https://arxiv.org/abs/2402.15258) — Riley & Edwards 2024
- [GAPS: A Large and Diverse Classical Guitar Dataset and Benchmark Transcription Model](https://arxiv.org/abs/2408.08653) — Riley et al. ISMIR 2024
- [GAPS companion site](https://aim-qmul.github.io/GAPS/) — code and dataset
- [TART: A Comprehensive Tool for Technique-Aware Audio-to-Tab Guitar Transcription](https://arxiv.org/abs/2510.02597) — Oct 2025
- [Fretting-Transformer: Encoder-Decoder Model for MIDI to Tablature Transcription](https://arxiv.org/abs/2506.14223) — June 2025
- [trimplexx music-transcription](https://github.com/trimplexx/music-transcription) — CRNN, claims 0.87 MPE F1 on GuitarSet, MIT-licensed
- [NNLS Chroma — Mauch & Dixon](https://www.eecs.qmul.ac.uk/~simond/pub/2010/Mauch-Dixon-ISMIR-2010.pdf) — 2010, 80% MIREX 2009
- [c4dm/nnls-chroma Vamp plugin](https://github.com/c4dm/nnls-chroma)
- [Essentia NNLSChroma binding](https://essentia.upf.edu/reference/streaming_NNLSChroma.html)
- [Madmom — pip-installable](https://madmom.readthedocs.io/)
- [Training chord recognition models on artificially generated audio](https://arxiv.org/html/2508.05878) — 2025
- [Enhancing Automatic Chord Recognition through LLM Chain-of-Thought Reasoning](https://arxiv.org/html/2509.18700v1) — 2025
- [MIREX 2025 Audio Chord Estimation task](https://music-ir.org/mirex/wiki/2025:Audio_Chord_Estimation)
- [Awesome Automatic Guitar Transcription](https://github.com/lucasgris/awesome-agt) — curated resource list
