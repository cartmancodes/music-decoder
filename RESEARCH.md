Here's a thorough breakdown. Your goal — "scale + exact guitar notes from any audio" — is actually two distinct problems chained together, and they have very different difficulty levels and very different best-in-class solutions. Let me walk through each layer with what's actually working in 2026.

## The pipeline, end to end

The full chain looks like this:

**MP3 → waveform → (optional) source separation → pitch/note detection → key/scale estimation → MIDI → string-fret assignment → tablature**

Every stage has tradeoffs. The honest summary up front: solo guitar audio is *very* solvable today (~80–90% note F-measure with the right model). A full mixed song where you want to extract just the guitar part is solvable but messier (~50–70%, with artifacts). Key/scale detection on tonal music is excellent (~80–90%+).

## Stage 1 — Source separation (only if input is a full mix)

If you feed a full song with vocals, drums, bass, and guitar into a transcription model, you'll get garbage. You need to isolate guitar first.

The standard tool is Meta's **Demucs**. The v4 model "htdemucs_6s" adds piano and guitar as individual sources, though piano separation is noted as not working great. In practice, htdemucs_6s is the only mainstream open-source model that gives you a dedicated guitar stem from a mix. Quality on guitar is okay-not-great — it bleeds with "other" instruments, and clean electric leads survive better than acoustic strumming buried in a mix. Spleeter (Deezer) is older and only gives you 4 stems (vocals/drums/bass/other) so guitar lands in "other" mixed with keys/synths.

```bash
pip install demucs
demucs -n htdemucs_6s song.mp3
# outputs separated/htdemucs_6s/song/{guitar,bass,drums,vocals,piano,other}.wav
```

If your input is solo guitar already, skip this stage entirely — separation will only hurt fidelity.

## Stage 2 — Note transcription (audio → MIDI)

This is the hardest stage. The right tool depends on whether the audio is monophonic (one note at a time) or polyphonic (chords).

**Monophonic** (e.g. a guitar solo line): CREPE is a deep CNN that operates on raw time-domain waveforms and matches or outperforms pYIN. It hits roughly 95% raw pitch accuracy at the 50-cent threshold and around 91% at the strict 10-cent threshold on real music. For pure DSP without ML, librosa's `pyin` is the closest free baseline. A newer model, **SwiftF0** (2025), claims a 91.8% harmonic mean at 10 dB SNR, beats CREPE by over 12 points in noise, and runs roughly 42x faster on CPU — worth watching.

**Polyphonic** (chords, full guitar parts): the field has moved decisively to neural nets. Three options worth knowing:

1. **Spotify Basic Pitch** — pip-installable, lightweight, instrument-agnostic. It produces a MIDI file with pitch bends, supports polyphonic input, and is designed to handle one instrument at a time. It was trained on GuitarSet, MAESTRO, MedleyDB, Slakh, and iKala. This is the right starting point for almost any project — it's the lowest-effort path to working transcription.

2. **Domain-adapted piano models** — researchers found that fine-tuning high-resolution piano transcription models on guitar data beats purpose-built guitar models. Riley et al. used a high-resolution piano transcription model trained on commercial score-audio pairs to achieve state-of-the-art zero-shot results on GuitarSet. The GAPS dataset (14 hours of classical guitar audio-score pairs, 2024) is now the largest public training resource.

3. **Multi-instrument transformers** — Google's MT3 and the systems from the 2025 AMT Challenge handle multi-instrument transcription end to end. In the 2025 AMT Challenge, eight teams submitted solutions and two outperformed the MT3 baseline. These are heavier but generalize better across genres.

The frontier work — TART (UC Berkeley, Oct 2025) — is a four-stage pipeline doing audio-to-MIDI, expressive technique classification (slides, bends, percussive hits), string/fret assignment, and tablature generation, claimed as the first to produce detailed tablature with accurate fingerings and expressive labels from guitar audio. Not pip-installable yet, but the architecture is the blueprint.

## Stage 3 — Key/scale detection

This part is much more tractable. The classic approach is **Krumhansl-Schmuckler**: build a pitch-class profile from the audio, then correlate it against pre-defined key profiles for all 24 major and minor keys; the highest correlation wins.

The procedure:
1. Compute a chromagram (`librosa.feature.chroma_cqt` is the standard — CQT-based chroma is more accurate than STFT-based for music).
2. Average across time to get a 12-D pitch class distribution.
3. Correlate against the 24 Krumhansl-Kessler profiles.
4. Pick the argmax.

Two refinements that matter in practice. First, run harmonic-percussive source separation (`librosa.effects.hpss`) before computing chroma — drums and transients pollute the pitch-class estimate. Second, the **Temperley** revisions to the original profiles tend to outperform the originals on real music. Temperley proposed simpler matching, alternative profile values, segment-level pitch class presence rather than duration sums, and a modulation penalty between segments. For most pop/rock songs, K-S with HPSS gets you 80%+ accuracy; key changes mid-song are where it struggles.

For scale detection (beyond just major/minor), once you know the key you essentially know the diatonic scale. Modal detection (Dorian, Mixolydian, etc.) is harder and usually requires looking at characteristic note movements rather than just pitch-class statistics.

## Stage 4 — MIDI → guitar tablature (string/fret assignment)

Here's the subtlety most people miss: **a single pitch can be played on multiple string-fret combinations on a guitar**. E4 is the open high E, the 5th fret of the B string, the 9th fret of the G string, the 14th fret of the D string, and so on. So "the exact guitar notes" is genuinely ambiguous — what you actually want is the *most playable* fingering.

Three approaches, in increasing sophistication:

1. **Naive lowest-fret** — for each MIDI note, pick the string-fret combination with the lowest fret. Simple, often produces awkward fingerings. Used as the baseline in the Fretting-Transformer paper; it achieves 100% pitch accuracy by definition but fails on playability metrics.

2. **A\* path search with biomechanical cost** — model the fretboard as a graph, weight transitions by hand movement, span, and ergonomic cost. Burlet & Fujinaga's A\*-Guitar is the canonical version. Solid baseline, no training data needed.

3. **Transformer-based assignment** — the Fretting-Transformer uses a T5-based encoder-decoder conditioned on guitar-specific ergonomic principles, modeling physical playability constraints, and outperforms A\* and naive baselines on tab accuracy. Riley et al.'s MIDI-to-Tab uses a BART-style architecture with quintile auto-regressive inference and beam search.

For a practical project, start with A\* — it's interpretable, works well, and there are open-source implementations. Move to a learned model only if A\* outputs feel awkward in your test cases.

## Recommended architecture for your project

Given your stack (Python, React, Docker, Postgres), here's how I'd lay it out:

```
[React frontend] 
    → uploads MP3 / paste YouTube URL
    → [FastAPI backend in Docker]
        → ffmpeg: decode to 22050 Hz mono WAV
        → demucs (htdemucs_6s): isolate guitar stem (optional toggle)
        → basic-pitch: WAV → MIDI note events
        → librosa: chromagram → K-S key detection
        → music21 / custom: MIDI → tab via A* fret assignment
        → Postgres: store user uploads, MIDI events, predicted key, tablature
    → React renders: 
        - chromagram heatmap
        - waveform with note onsets
        - detected key/scale
        - interactive tab viewer (VexFlow or AlphaTab in React)
        - playable MIDI (Tone.js)
```

A minimal end-to-end Python prototype:

```python
import librosa, numpy as np
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH

# 1. Transcribe to MIDI
model_output, midi_data, note_events = predict(
    "song.mp3", model_or_model_path=ICASSP_2022_MODEL_PATH
)
midi_data.write("song.mid")

# 2. Detect key (Krumhansl-Schmuckler)
y, sr = librosa.load("song.mp3", sr=22050)
y_h, _ = librosa.effects.hpss(y)
chroma = librosa.feature.chroma_cqt(y=y_h, sr=sr).mean(axis=1)

major = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
minor = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
keys = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

scores = []
for i in range(12):
    scores.append((f"{keys[i]} major", np.corrcoef(np.roll(major,i), chroma)[0,1]))
    scores.append((f"{keys[i]} minor", np.corrcoef(np.roll(minor,i), chroma)[0,1]))
key, _ = max(scores, key=lambda x: x[1])
print("Key:", key)

# 3. Map MIDI notes → guitar fret/string (naive baseline)
TUNING = [40, 45, 50, 55, 59, 64]  # EADGBE in MIDI numbers
def to_tab(midi_note):
    options = [(i, midi_note - t) for i, t in enumerate(TUNING) 
               if 0 <= midi_note - t <= 24]
    if not options: return None
    return min(options, key=lambda x: x[1])  # lowest-fret heuristic

for note in midi_data.instruments[0].notes[:20]:
    print(note.start, note.pitch, to_tab(note.pitch))
```

Replace the naive `to_tab` with A\* search for production quality.

## Honest accuracy expectations

I want to set realistic expectations because the marketing around "AI music transcription" is hyperbolic:

- **Solo monophonic guitar (clean recording)**: 90%+ note accuracy, near-perfect pitch class. Production-ready.
- **Solo polyphonic guitar (chords, fingerstyle)**: 75–85% note F-measure with Basic Pitch or domain-adapted models. Usable but needs review.
- **Guitar in a full mix**: 50–70% after Demucs separation. Bass notes get confused with guitar low strings, harmonics from other instruments leak in. Often unusable without manual cleanup.
- **Distorted electric guitar**: harder than clean — distortion adds harmonics that confuse pitch detection. Models trained on EGDB do better than general models.
- **Key detection on tonal pop/rock**: 80–90%. Struggles on jazz, modal, or atonal music.
- **Tab fingering quality**: A\* gets you "valid but sometimes weird." Transformer-based methods get closer to human fingering choices.

## Where to push if you want state of the art

If you want to go beyond Basic Pitch:
- Train a fine-tuned model on **GuitarSet** (60 pieces with hexaphonic per-string ground truth) plus **GAPS** (14 hours classical) plus **EGDB** (240 electric guitar performances).
- Use **CQT** (constant-Q transform) as input rather than mel-spectrograms — it's geometrically aligned with musical pitch.
- Add a **technique classifier** (slides, bends, hammer-ons, palm mutes) as a second head on the model.
- Use **stereo input** if available — stereo CQT representation helps separate individual guitar parts within a mix using spatial cues from studio recordings.

This would be a serious research project — a few months of work for a small team — but the path is clear and the datasets exist.

Want me to actually scaffold the Docker + FastAPI + React project structure, or write a more complete reference implementation of the A\* fret-assignment algorithm?