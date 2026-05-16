# Music Decoder — Usage Guide

Other docs: [Technical](architecture-technical.md) · [Overview](architecture-overview.md) · **Usage (this file)**

A complete guide for users who want to run Music Decoder. Covers
installation, the three usage surfaces (CLI, library, Streamlit UI),
output layout, troubleshooting, and tips for getting good results.

---

## Prerequisites

- **Python 3.11** (3.12 also supported; 3.13 is not).
- **ffmpeg** — for decoding any audio file you load.
- **fluidsynth** — only required if you want non-sine MIDI playback for
  composition output, or for rendering the synthetic regression
  fixtures via a soundfont. The default `compose()` flow renders WAV
  through pretty-MIDI's built-in sine synthesizer and works without
  fluidsynth.
- **A GeneralMIDI soundfont** (a `.sf2` file) — only required if you
  want fluidsynth-rendered output. See [Tips: Soundfonts](#soundfonts)
  below.

### macOS

```bash
brew install ffmpeg fluidsynth
```

### Ubuntu / Debian

```bash
sudo apt update
sudo apt install ffmpeg libsndfile1 libfluidsynth3
```

### Windows

ffmpeg: download from <https://www.gyan.dev/ffmpeg/builds/> and add
`bin/` to your `PATH`. fluidsynth: `winget install --id FluidSynth.FluidSynth`,
then make sure the install dir is on `PATH`. Most usage is supported,
though the project's CI and integration tests are exercised on macOS
and Linux only.

---

## Installation

### Option 0: `setup.sh` (recommended for development clones)

The repo ships a `setup.sh` that does the whole prerequisites-and-install
dance for you: detects your OS, installs system binaries (ffmpeg,
fluidsynth, libsndfile, plus python@3.11 if missing), creates `.venv`,
installs the package with dev extras (`pip install -e ".[dev]"`),
downloads the synthesis soundfont, and runs `music-decoder doctor`.

```bash
git clone https://github.com/cartmancodes/music-decoder.git
cd music-decoder
source ./setup.sh
```

`setup.sh` is sourceable: `source ./setup.sh` runs the installer **and**
activates `.venv` in your current shell. Plain `./setup.sh` works too
but you must `source .venv/bin/activate` afterwards (a subprocess can't
modify the parent shell's environment).

Re-running is idempotent. Flags:

- `--no-system` — skip the brew/apt step (use existing system binaries).
- `--no-soundfont` — skip the ~6 MB SF2 download.
- `--recreate` — wipe and rebuild `.venv` from scratch.
- `--python /path/to/python3.11` — pick a specific interpreter.

### Option 1: pipx (recommended for end users)

[pipx](https://pipx.pypa.io/) installs the tool into an isolated
environment and exposes the `music-decoder` command on your `PATH`.

```bash
git clone https://github.com/cartmancodes/music-decoder.git
cd music-decoder
pipx install ./
```

### Option 2: pip in a virtualenv (development)

```bash
git clone https://github.com/cartmancodes/music-decoder.git
cd music-decoder
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The `dev` extras add `pytest`, `pytest-cov`, `pytest-xdist`, `ruff`,
`mypy`, and `types-PyYAML`.

---

## First run

After installing, run the doctor command to verify your dependencies
are wired up.

```bash
music-decoder doctor
```

Expected output (formatting may vary slightly):

```text
ffmpeg on PATH................. OK
fluidsynth..................... OK
soundfont present.............. OK
```

Each "FAIL" line includes a hint with the recommended remediation.
`doctor` exits with a non-zero status if anything fails, which makes
it easy to wire into shell scripts.

The first time you actually call `analyze()`, the underlying ML models
download to your local cache (typically under
`~/.cache/torch/hub` for Demucs, `~/.cache/basic-pitch` for basic-pitch,
and `~/.madmom/` for madmom). Combined size is roughly 400 MB. The
download happens once.

---

## CLI reference

The CLI is grouped under `music-decoder` with four subcommands:
`analyze`, `compose`, `ui`, and `doctor`.

### `analyze` — chord progression + tab

```text
music-decoder analyze SOURCE [--tuning ...] [--solo-guitar | --full-mix]
                              [--no-separation]
                              [--format pretty | json]
```

- `SOURCE` (positional) — a local audio file (`.mp3`, `.wav`, `.flac`)
  or a YouTube URL (`https://youtu.be/...` or
  `https://www.youtube.com/watch?v=...`).
- `--tuning` — guitar tuning preset. One of:
  - `EADGBE` (default; standard tuning).
  - `Drop D` or `Drop-D` (case-sensitive aliases for the same preset).
  - `Eb` (half-step down).
  - `D standard` (whole-step down).
  - `Drop C`.
  - `DADGAD`.
- `--solo-guitar / --full-mix` — declare what the input is. Default
  is `--full-mix`. With `--solo-guitar`, source separation is skipped
  even if `--no-separation` isn't set.
- `--no-separation` — skip the Demucs guitar-isolation step on full
  mixes (useful when you trust the original mix or want to save time).
- `--format pretty | json` — output format. `pretty` (default) prints
  a human-readable summary. `json` dumps the full `AnalysisResult`
  dataclass tree (paths converted to strings, tuples to lists).

`analyze` does not write artifacts to disk other than the YouTube cache;
there is no `--out` flag.

#### Examples

Analyze a YouTube link:

```bash
music-decoder analyze 'https://youtu.be/dQw4w9WgXcQ'
```

**Always quote YouTube URLs.** zsh and bash treat `&` (background) and
`?` (glob) as special characters — an unquoted URL with `&list=...` will
silently truncate the URL at the `&` and leave you wondering why nothing
happened. Single quotes are safest. The `&list=RD...` "radio mix"
playlist parameter is recognised and ignored: yt-dlp is invoked with
`noplaylist=True`, so you get the single video referenced by `?v=`,
even when the URL was copied from a YouTube auto-mix.

```bash
# All of these are equivalent — the playlist context is dropped:
music-decoder analyze 'https://www.youtube.com/watch?v=ilNt2bikxDI'
music-decoder analyze 'https://www.youtube.com/watch?v=ilNt2bikxDI&list=RDilNt2bikxDI'
music-decoder analyze 'https://youtu.be/ilNt2bikxDI'
```

Expected pretty output:

```text
Source:     https://youtu.be/dQw4w9WgXcQ
Duration:   213.45s @ 22050 Hz
Key:        A minor (corr=0.86)
Tempo:      113.0 BPM
Chords (24):
   0.00 -  3.21s  Am
   3.21 -  6.40s  F
   6.40 -  9.52s  C
   9.52 - 12.71s  G
   ...
Tab (312 notes; first 16s):
e|---15--------------------------------------------3--------------|
B|-----------------------------------------------------------3----|
G|---------7--5--5-5-----------------------------------------4----|
D|---10----------7------------------------------------------------|
A|----------------------------------------------------3-----------|
E|-----------------------1--------------1-------------------------|
```

The pretty output shows a **bounded tab excerpt** (first 16 s) — the
full transcription would overplot a fixed-width staff. For the complete
per-note tab use `--format json` and read the `tab` array. If the
transcriber found no notes (common on dense full mixes), you'll see
`Tab: (no notes transcribed — common on dense full mixes)` instead;
key and chords are still reported.

Analyze a local file as a solo guitar recording, skipping Demucs:

```bash
music-decoder analyze song.mp3 --solo-guitar --no-separation
```

Analyze a Drop D recording and emit JSON for piping into `jq`:

```bash
music-decoder analyze rock.wav --tuning "Drop D" --format json | jq '.chord_progression[0:4]'
```

### `compose` — generate an arrangement

```text
music-decoder compose --scale TONIC:MODE --progression "CHORDS..."
                       [--style arpeggio | strum | fingerstyle]
                       [--tempo BPM] [--bars N] [--seed N]
                       [--tuning ...] --out DIR
                       [--format pretty | json]
```

- `--scale` (required) — `TONIC:MODE` where `TONIC` is one of
  `C, C#, D, D#, E, F, F#, G, G#, A, A#, B` (flats also accepted:
  `Db`, `Eb`, `Gb`, `Ab`, `Bb`) and `MODE` is `major` or `minor`.
  Examples: `C:major`, `A:minor`, `Eb:major`.
- `--progression` (required) — space-separated chord symbols. Quality
  suffixes recognised: bare (major), `m`, `7`, `maj7`, `m7`, `dim`,
  `sus4`, `aug`, `min`, `min7`. Examples: `"Cmaj7 Am7 Dm7 G7"`,
  `"C G Am F"`, `"Dm7 G7 Cmaj7"`.
- `--style` — `fingerstyle` (default), `strum`, or `arpeggio`.
- `--tempo` — beats per minute, float. Default `100.0`.
- `--bars` — bars per chord. Default `1`.
- `--seed` — integer; same seed → same melody. Default: a fresh random
  seed each run.
- `--tuning` — same set as `analyze`. Default `EADGBE`.
- `--out` (required) — directory where the per-take folder is created.
- `--format pretty | json`.

#### Examples

A I-vi-ii-V in C major with a fixed seed:

```bash
music-decoder compose \
  --scale C:major \
  --progression "Cmaj7 Am7 Dm7 G7" \
  --style fingerstyle --tempo 100 --seed 42 \
  --out ./out
```

Expected pretty output (paths will differ):

```text
MIDI:    out/20260508T142510-7c4a3b21/composition.mid
WAV:     out/20260508T142510-7c4a3b21/composition.wav
ASCII tab:
e|---5---5-7---7---5-...
B|---5-------5-------|
G|---5-------5-------|
D|---5-------5-------|
A|-3-------3---------|
E|-------------------|
```

A minor blues feel with strummed chords:

```bash
music-decoder compose \
  --scale A:minor \
  --progression "Am Dm E7 Am" \
  --style strum --tempo 80 --bars 2 --seed 7 \
  --out ./out
```

A Drop D power-chord progression, fingerstyle:

```bash
music-decoder compose \
  --scale D:major \
  --progression "D G A D" \
  --tuning "Drop D" --style fingerstyle --seed 1 \
  --out ./out
```

JSON output for programmatic consumption:

```bash
music-decoder compose --scale C:major --progression "C G Am F" \
  --seed 1 --out ./out --format json | jq '.metadata'
```

### `ui` — launch the Streamlit app

```bash
music-decoder ui
```

This shells out to `streamlit run` against
[ui/streamlit_app.py](../src/music_decoder/ui/streamlit_app.py) with
`--server.headless false --browser.gatherUsageStats false`. Your
default browser opens at `http://localhost:8501`.

### `doctor` — verify dependencies

```bash
music-decoder doctor
```

Checks ffmpeg, the `fluidsynth` Python module, and the soundfont
configured in `config/runtime.yaml`. Returns non-zero on any failure.

---

## Library API

Everything the CLI does is also a normal Python function call.

### Imports

```python
from music_decoder import (
    analyze, compose,
    Scale, ChordSymbol, Tuning,
    STANDARD_EADGBE, DROP_D, EB_HALF_STEP_DOWN,
    D_STANDARD, DROP_C, DADGAD,
    AnalysisResult, Composition,
    KeyEstimate, ChordSegment, Note, TabbedNote,
    TabPosition, VoicedChord,
)
```

### `analyze()`

```python
def analyze(
    source: str | Path,
    *,
    declared_kind: Literal["solo_guitar", "full_mix"] = "full_mix",
    tuning: Tuning = STANDARD_EADGBE,
    use_separation: bool = True,
    progress: ProgressCallback | None = None,
) -> AnalysisResult: ...
```

Minimal example:

```python
from music_decoder import analyze

result = analyze("song.mp3", declared_kind="solo_guitar", use_separation=False)

print(f"Key: {result.key.tonic} {result.key.mode}")
print(f"Tempo: {result.tempo_bpm:.1f} BPM")
for seg in result.chord_progression[:8]:
    print(f"  {seg.start_s:6.2f}-{seg.end_s:6.2f}s  {seg.chord.to_label()}")
```

With a progress callback (e.g. for a CLI progress bar):

```python
from music_decoder import analyze

def progress(stage: str, fraction: float) -> None:
    print(f"[{stage}] {fraction * 100:.0f}%")

result = analyze("https://youtu.be/dQw4w9WgXcQ", progress=progress)
```

### `compose()`

```python
def compose(
    scale: Scale,
    progression: Sequence[ChordSymbol],
    *,
    bars_per_chord: int = 1,
    tempo_bpm: float = 100.0,
    style: Literal["arpeggio", "strum", "fingerstyle"] = "fingerstyle",
    tuning: Tuning = STANDARD_EADGBE,
    seed: int | None = None,
    out_dir: Path,           # required
) -> Composition: ...
```

`out_dir` is a required keyword argument; `compose()` writes a
`<timestamp>-<hash>/` subdirectory under it.

Minimal example:

```python
from pathlib import Path
from music_decoder import compose, Scale, ChordSymbol

result = compose(
    scale=Scale("C", "major"),
    progression=[ChordSymbol.parse(c) for c in ("Cmaj7", "Am7", "Dm7", "G7")],
    bars_per_chord=1,
    tempo_bpm=100.0,
    style="fingerstyle",
    seed=42,
    out_dir=Path("./out"),
)

print(result.midi_path)   # ./out/20260508T...-.../composition.mid
print(result.wav_path)    # ./out/20260508T...-.../composition.wav
print(result.ascii_tab)
```

### Helper constructors

```python
Scale.parse("C:major")          # → Scale(tonic="C", mode="major")
Scale.parse("Eb:minor")         # → Scale(tonic="D#", mode="minor")  (flats normalize to sharps)

ChordSymbol.parse("Cmaj7")      # → ChordSymbol(root="C", quality="maj7")
ChordSymbol.parse("F#m7")       # → ChordSymbol(root="F#", quality="min7")
ChordSymbol.parse("Bb")         # → ChordSymbol(root="A#", quality="maj")
ChordSymbol.parse("Asus4").to_label()  # → "Asus4"
```

### Tuning presets

```python
from music_decoder import (
    STANDARD_EADGBE, DROP_D, EB_HALF_STEP_DOWN,
    D_STANDARD, DROP_C, DADGAD,
)

print(STANDARD_EADGBE.open_pitches)   # (40, 45, 50, 55, 59, 64) — E2 A2 D3 G3 B3 E4
```

You can also build a custom tuning:

```python
from music_decoder import Tuning

OPEN_G = Tuning("Open G", (38, 43, 50, 55, 59, 62))   # D2 G2 D3 G3 B3 D4
```

---

## Streamlit UI walkthrough

Launch with:

```bash
music-decoder ui
```

The UI is a single page with three tabs:

### Analyze tab

- **Source** — radio: "File upload" or "YouTube URL".
- **Audio file** — drag-drop or browse for an `.mp3`, `.wav`, or `.flac`.
- **YouTube URL** — paste a full URL.
- **Tuning** — dropdown of the six presets.
- **Declared kind** — `full_mix` or `solo_guitar`.
- **Use Demucs separation** — checkbox (only effective on full mix).
- **Analyze** button — runs the pipeline; a progress bar shows each
  stage.

After completion you see:

- Detected key, tempo, duration as a one-line summary.
- The chord progression rendered as a horizontal timeline with chord
  labels at their time spans.
- The full ASCII tab in a code block.
- An HTML5 audio player for the original (or cached YouTube) audio.

If the pipeline raises a `MusicDecoderError` (corrupt audio, YouTube
unreachable, etc.), the tab shows a red `st.error` block with the
exception message.

### Compose tab

- **Tonic** — dropdown (12 notes).
- **Mode** — `major` or `minor`.
- **Progression** — free text, space-separated (default
  `"Cmaj7 Am7 Dm7 G7"`).
- **Style** — `fingerstyle`, `strum`, `arpeggio`.
- **Tempo** — slider, 40 to 220 BPM.
- **Bars per chord** — slider, 1 to 4.
- **Seed (optional)** — text box; leave empty for a fresh random seed.
- **Tuning** — dropdown.
- **Compose** button.

After completion you see the WAV in a player, the ASCII tab, and
download buttons for both the MIDI and WAV files.

### About tab

Shows the package version and the loaded hyperparameter-set ID
(currently `2026-05-08-v2-baseline`).

---

## Output layout

### YouTube cache

Path: `<data_dir>/yt_cache/<video_id>.wav`.

Where `<data_dir>` is `~/.music-decoder` by default (overridable via
`MUSIC_DECODER_DATA_DIR` or [`config/runtime.yaml`](../config/runtime.yaml)).
The cache is keyed by the 11-character YouTube video ID. A second
`analyze()` on the same URL skips the download and reads from the
cache directly. You can wipe the directory at any time; the next
analyze will re-download.

### Composition output

Per-take directory: `<out_dir>/<timestamp>-<hash>/`.

- `composition.mid` — the MIDI score (two tracks: melody, accompaniment).
- `composition.wav` — the rendered audio at 22050 Hz, 16-bit PCM.

`<timestamp>` is `YYYYmmddTHHMMSS` (UTC-naive local clock).
`<hash>` is the first 8 hex chars of `sha1(scale | progression |
tempo | style | seed)`. Same inputs → same hash, but the timestamp
differs each run, so re-running with the same inputs creates a fresh
folder rather than overwriting. `--out` from the CLI is the parent
`<out_dir>`. The Streamlit UI uses
`<data_dir>/compositions/` (default `~/.music-decoder/compositions/`).

---

## JSON output schema

`music-decoder analyze --format json` emits the full `AnalysisResult`
as JSON:

```json
{
  "source": "song.mp3",
  "audio_path": "song.mp3",
  "duration_s": 213.45,
  "sample_rate_hz": 22050,
  "key": {
    "tonic": "A",
    "mode": "minor",
    "profile": "krumhansl_kessler",
    "correlation": 0.86,
    "margin": 0.04
  },
  "chord_progression": [
    {
      "start_s": 0.0,
      "end_s": 3.21,
      "root": "A",
      "quality": "min",
      "confidence": 0.79
    },
    ...
  ],
  "tab": [
    {
      "note": {
        "start_s": 0.04,
        "end_s": 0.21,
        "pitch": 57,
        "velocity": 87,
        "confidence": 0.71
      },
      "position": {"string": 4, "fret": 0},
      "cost_breakdown": {"path_total": 12.7}
    },
    ...
  ],
  "tempo_bpm": 113.0,
  "beat_times_s": [0.0, 0.53, 1.06, ...],
  "metadata": {
    "hyperparameter_set": "2026-05-08-v2-baseline"
  }
}
```

Field reference:

- `source` — the original input string (path or URL).
- `audio_path` — the local WAV/MP3 actually decoded (the YouTube cache
  file when source was a URL).
- `duration_s` — duration of the loaded audio in seconds.
- `sample_rate_hz` — the audio's sample rate (22050 by default).
- `key.tonic` — one of `C, C#, D, D#, E, F, F#, G, G#, A, A#, B`
  (sharps only).
- `key.mode` — `major` or `minor`.
- `key.profile` — `krumhansl_kessler` or `temperley` — which textbook
  profile the consensus picked.
- `key.correlation` — Pearson correlation between the audio's
  pitch-class profile and the matched key profile, in `[-1, 1]`.
  Higher is better.
- `key.margin` — `correlation - second_best_correlation`. A high
  margin means the second-best key was much worse, which is a strong
  signal.
- `chord_progression[i].root` — chord root, or `"N"` for no-chord.
- `chord_progression[i].quality` — one of
  `maj | min | 7 | maj7 | min7 | dim | sus4 | aug` (or `""` when root
  is `"N"`).
- `chord_progression[i].confidence` — mean of the per-beat similarity
  scores during the segment, `[0, 1]`.
- `tab[i].note.pitch` — MIDI note number (60 = C4).
- `tab[i].note.velocity` — 0-127.
- `tab[i].note.confidence` — basic-pitch's per-note confidence,
  `[0, 1]`.
- `tab[i].position.string` — 0-indexed, low-to-high (0 = low E in
  EADGBE).
- `tab[i].position.fret` — 0 = open string.
- `tempo_bpm` — beat-tracker estimate.
- `beat_times_s` — flat list of beat onsets in seconds.
- `metadata.hyperparameter_set` — ID stamped from
  `config/hyperparameters.yaml`.

`music-decoder compose --format json` emits a `Composition` with the
same conventions:

```json
{
  "midi_path": "out/20260508T142510-7c4a3b21/composition.mid",
  "wav_path":  "out/20260508T142510-7c4a3b21/composition.wav",
  "ascii_tab": "e|---5...",
  "melody_notes": [{"start_s": 0.0, "end_s": 0.57, "pitch": 67, ...}, ...],
  "chord_voicings": [
    {
      "chord": {"root": "C", "quality": "maj7"},
      "positions": [
        {"string": 0, "fret": -1}, {"string": 1, "fret": 3},
        {"string": 2, "fret": 2}, {"string": 3, "fret": 0},
        {"string": 4, "fret": 0}, {"string": 5, "fret": 0}
      ]
    },
    ...
  ],
  "metadata": {
    "scale": "C:major",
    "progression": ["Cmaj7", "Am7", "Dm7", "G7"],
    "bars_per_chord": 1,
    "tempo_bpm": 100.0,
    "style": "fingerstyle",
    "seed": 42,
    "tuning": "EADGBE"
  }
}
```

In `chord_voicings[i].positions`, `fret = -1` means the string is
muted (not played).

---

## Configuration

### Runtime config

[`config/runtime.yaml`](../config/runtime.yaml):

```yaml
data_dir: ~/.music-decoder
log_level: INFO
ffmpeg_path: ffmpeg
fluidsynth_soundfont: GeneralUser-GS.sf2
sample_rate_hz: 22050
youtube_cache_dir: ${data_dir}/yt_cache
composition_out_dir: ${data_dir}/compositions
```

Override any key via environment variable:

```bash
export MUSIC_DECODER_DATA_DIR=/tmp/md-test
export MUSIC_DECODER_LOG_LEVEL=DEBUG
export MUSIC_DECODER_YOUTUBE_CACHE_DIR=/var/cache/md/yt
music-decoder analyze song.mp3
```

`${data_dir}` references in YAML resolve from sibling keys via a
4-pass interpolation pass.

### Hyperparameters

[`config/hyperparameters.yaml`](../config/hyperparameters.yaml) records
the canonical hyperparameter set for the current pipeline. Its `id`
is stamped into every `AnalysisResult.metadata["hyperparameter_set"]`
so you can correlate a result with its config.

Most adapter functions read their hyperparameters from this YAML at
call time — notably `chord_detection.backend` (used by
`recognize_chords`) and the `basic_pitch.minimum_frequency_hz` /
`maximum_frequency_hz` bounds (used by `transcribe`). Some sub-keys
remain hard-coded in the adapter modules: the `tab_assignment.weights`
block and `post_processing.median_filter_window` are not currently
overridable from the YAML. Those defaults are intentional for v2 and
are listed in the technical doc §4.x; we track the gap as future work
in the v2 design spec §14.

### Eval thresholds

[`config/eval_thresholds.yaml`](../config/eval_thresholds.yaml):

```yaml
chord_recognition_score: 0.60
key_mirex_score: 0.75
tab_string_accuracy: 0.55
```

Consumed by `pytest -m regression` only.

---

## Troubleshooting

### `ffmpeg binary not found on PATH`

Source: ingest layer can't find ffmpeg. Confirm with `which ffmpeg`.
On macOS: `brew install ffmpeg`. On Ubuntu: `sudo apt install ffmpeg`.
If you need a non-default location, point to it via
`MUSIC_DECODER_FFMPEG_PATH=/full/path/to/ffmpeg` or by editing
`config/runtime.yaml`.

### `fluidsynth: FAIL` from doctor

The Python `fluidsynth` module isn't importable. Install both the
system library and the Python binding:

- macOS: `brew install fluidsynth && pip install pyfluidsynth`.
- Ubuntu: `sudo apt install libfluidsynth3 && pip install pyfluidsynth`.

If you don't plan to use fluidsynth-rendered output (the default
`compose()` flow uses pretty-MIDI's sine synth), this is a soft
warning; analyze and compose will still work.

### `soundfont present: FAIL`

The path in `config/runtime.yaml` points at a `.sf2` file that
doesn't exist. Either:

- Run the bundled downloader:
  ```bash
  python scripts/download_soundfont.py
  ```
  This drops a public-domain `TimGM6mb.sf2` (~6 MB) into
  `tests/fixtures/synthetic/soundfont/`.
- Or supply your own:
  ```bash
  python scripts/download_soundfont.py \
    --output ~/.music-decoder/GeneralUser-GS.sf2
  ```
- Or edit `config/runtime.yaml` to point `fluidsynth_soundfont` at
  whatever `.sf2` you have.

### YouTube errors

Common shapes: "Video unavailable", "Private video",
"This video is not available in your country".

- Confirm the URL plays in a normal browser.
- If it's region-locked, you can't bypass it; pick a different source.
- If yt-dlp itself is out of date, update it:
  `pip install --upgrade yt-dlp`. YouTube periodically changes its
  player, and yt-dlp ships frequent fixes.
- The error is wrapped as `YouTubeError` and shown as a red banner in
  the UI / a clean error in the CLI.

### YouTube error mentions a video ID I didn't supply

If the error is `[youtube] <other-id>: This video is not available`
where `<other-id>` doesn't match the `?v=` in your URL, you almost
certainly hit the playlist-context trap. The fix shipped — yt-dlp is
invoked with `noplaylist=True` — but if you're running an older install
without it, upgrade with `pip install -e .` or re-run `./setup.sh`.

### Shell ate part of my URL / "command not found: list=…"

Your YouTube URL contained `&` or `?` and wasn't quoted. The shell
backgrounded the command at the `&` and tried to execute the rest as a
separate statement. **Single-quote URLs**:

```bash
music-decoder analyze 'https://www.youtube.com/watch?v=ilNt2bikxDI&list=RDilNt2bikxDI'
```

### Wrong `music-decoder` on PATH (stale global install)

If you previously ran `pip install` or `pipx install ./` outside a
virtualenv, you may have a `music-decoder` binary in
`/opt/homebrew/bin/` or `~/.local/bin/` that shadows the one in
`.venv/bin/`. Symptom: tracebacks reference site-packages paths instead
of your repo (`/opt/homebrew/lib/python3.11/site-packages/music_decoder/...`),
or the doctor reports config/runtime paths that don't exist.

Diagnose:

```bash
which music-decoder       # should be inside <repo>/.venv/bin
type music-decoder
```

Fix:

```bash
# Activate the venv shipped by setup.sh — its bin/ dir takes priority on PATH
source .venv/bin/activate

# Or, if you really want to remove the stale global install:
/opt/homebrew/opt/python@3.11/bin/python3.11 -m pip uninstall music-decoder
```

### `CorruptAudioError` on a file you know plays

ffmpeg failed to decode it cleanly. Common causes: file truncated, a
codec ffmpeg wasn't built with, or a sample rate mismatch. Try:

```bash
ffmpeg -i broken.mp3 fixed.wav
music-decoder analyze fixed.wav
```

Re-encoding to PCM WAV via ffmpeg manually is the fastest workaround.

### `SilentAudioError` / `ClipTooShortError`

The audio is below `RMS = 1e-4` (effectively silence) or shorter than
1.0 second. Double-check you're feeding the right file.

### Low-confidence results

If `result.key.correlation < 0.5` or many `result.chord_progression[i].confidence < 0.4`,
the audio is hard for the model. Mitigations:

- For full-mix audio, try with and without `--no-separation`. Demucs
  helps on dense mixes but can introduce artifacts on already-clean
  guitar.
- For solo guitar, force `--solo-guitar` so Demucs is skipped.
- Heavily distorted electric guitar fools the chord detector. If the
  song has a clean version (acoustic, demo), prefer that.
- Check the tempo: if the beat tracker latched to the wrong tempo
  (you'll see it in `result.tempo_bpm`), chord boundaries will be
  misaligned.

### `tempo_bpm` is double or half what I expected

This is expected behavior, not a bug. The beat tracker has an inherent
octave ambiguity — for a 76 BPM song it may report ~152 BPM (double-time)
or for a 140 BPM song ~70 BPM (half-time), because both are metrically
valid interpretations of the same pulse. In end-to-end testing this
showed up on ~2 of 3 real tracks. It does **not** affect key or chord-root
detection (those run on a beat-synchronous chroma whose grid is internally
consistent regardless of the octave). If you only need chords, ignore the
absolute BPM; if you need the true tempo, halve/double it to the range
that matches how the song feels.

### What accuracy should I expect?

On clean, well-mixed recordings with clear harmony, end-to-end testing
against songs with documented progressions gave:

- **Key:** correct tonic+mode on all sampled tracks (6/6; correlation
  typically 0.75–0.92).
- **Chord progression:** the detected chord *sequence* (not just the
  root set) reproduced the documented cycle on every sampled track —
  including correct minor-quality chords (e.g. the vi in a I–vi–IV–V).
  ~97–100% of segments within the known chord set; mean confidence ~0.85.
- **Tablature:** the A\* fret assignment is reliable when notes exist —
  positions stay in range and spread sensibly across all six strings.
  Note *transcription* is the weak link: on ~2 of 6 sampled tracks the
  separated-guitar stem yielded a very sparse tab (tens of notes for a
  whole song) even though chords were ~100% correct. The tab is a guide,
  not a faithful transcription, and is sparser than the chord output.

Accuracy drops on heavily distorted electric guitar, dense mixes without
separation, ambiguous modal tonality, and very short clips. Treat the
output as a strong first draft, not ground truth — spot-check against
your ear. Chord/key detection is consistently stronger than the
note-level tab.

### `No notes detected`

basic-pitch returned zero notes. Causes: very quiet audio, very dense
polyphony, or a recording that isn't actually pitched (drums-only).
The result still has a key + chord progression; only `tab` is empty.

### `analyze()` is slow on the first run

First-call model downloads (Demucs, basic-pitch, madmom) total ~400 MB.
The CLI shows download progress on stderr. Subsequent runs are fast
(typically 10-30 s on CPU for a 3-minute song).

### Streamlit UI hangs after clicking Analyze

Open the terminal where you launched `music-decoder ui`. Errors and
download progress print there. If a stage is taking a long time, the
progress bar will tick at the start and end of each stage but stays
flat in between (the underlying ML models don't expose mid-stage
progress).

### `InvalidProgressionError: No canonical voicing for ...`

You passed a chord whose root or quality isn't covered by the
voicing table. Confirm your spelling: roots must be one of
`C, C#, D, D#, E, F, F#, G, G#, A, A#, B` (flats `Db`, `Eb`, `Gb`,
`Ab`, `Bb` are accepted and normalized). Qualities must be empty
(major), `m`, `7`, `maj7`, `m7`, `dim`, `sus4`, or `aug`.

---

## Tips for best results

### Solo guitar vs full mix

Use `--solo-guitar` only when the recording is genuinely a single
guitar track. With it set:

- Source separation is skipped (faster).
- The chord detector still treats the audio as harmonically rich.

If you have a band recording but the guitar is mixed loud and clear
(modern pop with a single rhythm guitar, demos), `--full-mix --no-separation`
is often a good middle ground — Demucs adds latency and can sometimes
oversubtract.

### Tuning presets

Pick the tuning your guitar is actually in. The A* fret search uses
the open-string MIDI pitches, so passing the wrong tuning can produce
unplayable fingerings (very high frets or out-of-range notes).

Custom tunings: see [Library API: Tuning presets](#tuning-presets).

### Composition style choices

- **`fingerstyle`** — alternating bass + treble. Best for ballads,
  folk, slow tempos. Default for a reason; sounds reasonable on most
  progressions.
- **`strum`** — block chords on beats 1 and 3. Best for fast pop /
  rock progressions.
- **`arpeggio`** — chord tones rolled out one at a time. Best for
  delicate, classical-leaning pieces.

Use `--bars 2` or higher when the chord changes feel too fast at
default. Use `--tempo 60-80` for ballads, `--tempo 100-120` for pop,
`--tempo 140+` for upbeat material.

### Seed reproducibility

`--seed N` makes `compose()` fully deterministic. Same `(scale,
progression, bars, tempo, style, seed, tuning)` always produces the
same melody. Useful for:

- Iterating on a chord progression while keeping the melody constant.
- Sharing a seed in a bug report so we can reproduce.
- Generating multiple takes by sweeping seed: `for s in 1 2 3 4 5; do
  music-decoder compose --seed $s ...; done`.

Without `--seed`, the random number generator is seeded from
NumPy's default entropy source, so every run differs.

### Soundfonts

`compose()` auto-selects fluidsynth when both
`config/runtime.yaml` `fluidsynth_soundfont` resolves to an existing
`.sf2` file and the `fluidsynth` Python module is importable; otherwise
it falls back to `pretty_midi.PrettyMIDI.synthesize` (a built-in sine
synth) with a single warning log line. To get a richer timbre, install
fluidsynth and supply a soundfont. The simplest way:

```bash
python scripts/download_soundfont.py
```

This downloads `TimGM6mb.sf2` (~6 MB, public domain) into
`tests/fixtures/synthetic/soundfont/`. The `render_wav` adapter looks
in that directory automatically; you can also move the `.sf2` wherever
you like and point `config/runtime.yaml` `fluidsynth_soundfont` at it.
Callers who want to force a backend can pass `backend=SynthBackend.SINE`
or `backend=SynthBackend.FLUIDSYNTH` to `synth.render_wav` directly.

### Caching the YouTube downloads

Repeated `analyze()` on the same URL is fast — the WAV is cached at
`<data_dir>/yt_cache/<video_id>.wav`. To force a re-download, delete
that one file. To wipe the cache entirely, remove the directory.

### Working with the JSON output

The CLI's `--format json` is the easiest way to script around
analyze. Stdout carries **only** the JSON document — all pipeline
diagnostics (yt-dlp download progress, basic-pitch's `Predicting MIDI`
line, model logs) are redirected to stderr, so `| jq` works directly.
When saving to a file you'll still see that progress on your terminal
(it's on stderr); add `2>/dev/null` to silence it:

```bash
music-decoder analyze 'https://youtu.be/dQw4w9WgXcQ' --format json 2>/dev/null > out.json
```

A few useful one-liners with `jq`:

```bash
# Just the chord labels in order
music-decoder analyze song.mp3 --format json \
  | jq -r '.chord_progression[] | "\(.start_s) \(.root)\(.quality)"'

# Notes per (string, fret)
music-decoder analyze song.mp3 --format json \
  | jq '.tab | group_by(.position.string) | map({string: .[0].position.string, count: length})'

# Just the metadata (handy for hyperparameter set lookup)
music-decoder analyze song.mp3 --format json | jq '.metadata'
```

---

## Development

If you cloned the repo and want to run the test suite, lint, etc.:

```bash
make dev          # pip install -e ".[dev]"
make test-fast    # unit tests only
make test         # unit + integration
make regression   # accuracy regression suite (pytest -m regression)
make lint         # ruff
make typecheck    # mypy --strict
make run          # alias for `music-decoder ui`
```

The regression suite expects the synthetic fixtures under
`tests/fixtures/synthetic/`. The first run renders them to WAV (via
fluidsynth + soundfont if available, else sine fallback); the
rendered WAVs are cached on disk via a `.synth_marker` invalidation
file.

---

## Quick reference card

```bash
# Install (development clone)
source ./setup.sh                # full env + activate .venv in this shell
music-decoder doctor

# Or, end-user install
pipx install ./

# Analyze a song — ALWAYS single-quote YouTube URLs (& and ? are shell-special)
music-decoder analyze 'https://youtu.be/dQw4w9WgXcQ'
music-decoder analyze <FILE-OR-URL> [--solo-guitar] [--no-separation] [--tuning ...]
music-decoder analyze song.mp3 --format json | jq '.chord_progression'

# Compose a piece
music-decoder compose --scale C:major --progression "Cmaj7 Am7 Dm7 G7" \
                       --style fingerstyle --tempo 100 --seed 42 --out ./out

# Launch UI
music-decoder ui

# Library
from music_decoder import analyze, compose, Scale, ChordSymbol
analyze("song.mp3")
compose(scale=Scale("C","major"),
        progression=[ChordSymbol.parse(c) for c in ("Cmaj7","Am7","Dm7","G7")],
        seed=42, out_dir=Path("./out"))
```
