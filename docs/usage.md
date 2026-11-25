# Music Decoder — Usage Guide

Other docs: [Technical](architecture-technical.md) · [Overview](architecture-overview.md) · **Usage (this file)**

A complete guide for users who want to run Music Decoder. Covers
installation, the three usage surfaces (CLI, library, Streamlit UI),
output layout, troubleshooting, and tips for getting good results.

---

## Prerequisites

- **Python 3.11** (3.12 also supported; 3.13 is not).
- **ffmpeg** — for decoding any audio file you load.
- **fluidsynth** — renders composition output (and the synthetic
  regression fixtures) with real instrument sounds. Without it,
  `compose()` falls back to pretty-MIDI's built-in sine synthesizer.
  `productionize` needs the `fluidsynth` command-line program for its
  strings.
- A General MIDI soundfont is **not** a prerequisite: pretty-MIDI ships
  one (`TimGM6mb.sf2`), and `setup.sh` / `music-decoder fetch-models`
  install the higher-quality GeneralUser GS. See
  [Tips: Soundfonts](#soundfonts).

### macOS

```bash
brew install ffmpeg fluidsynth
```

### Ubuntu / Debian

```bash
sudo apt update
sudo apt install ffmpeg libsndfile1 libfluidsynth3 fluidsynth
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
pre-downloads the models with `music-decoder fetch-models` (Demucs
weights ~52 MB + GeneralUser GS soundfont ~30 MB) and the `productionize`
drum and percussion samples with `music-decoder fetch-samples` (~41 MB),
and runs `music-decoder doctor`. The only later downloads are the piano
samples a `productionize` song needs (or all ~700 MB up front with
`music-decoder fetch-samples --piano`).

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
- `--no-models` — skip the model and sample pre-downloads (Demucs then
  downloads on the first full-mix analysis, samples on the first
  `productionize`; composition uses the bundled soundfont).
- `--no-soundfont` — skip only the GeneralUser GS soundfont.
- `--recreate` — wipe and rebuild `.venv` from scratch.
- `--python /path/to/python3.11` — pick a specific interpreter.

### Option 1: pipx (recommended for end users)

[pipx](https://pipx.pypa.io/) installs the tool into an isolated
environment and exposes the `music-decoder` command on your `PATH`.

```bash
git clone https://github.com/cartmancodes/music-decoder.git
cd music-decoder
pipx install ./
music-decoder fetch-models   # optional: pre-download Demucs + soundfont
```

### Option 2: pip in a virtualenv (development)

```bash
git clone https://github.com/cartmancodes/music-decoder.git
cd music-decoder
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools hatchling "numpy<2" "cython<3"
pip install --no-build-isolation "madmom>=0.16,<0.17"
pip install -e ".[dev]"
music-decoder fetch-models   # optional: pre-download Demucs + soundfont
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
ffmpeg on PATH.................. OK
fluidsynth...................... OK
soundfont....................... OK
  using installed: ~/.music-decoder/GeneralUser-GS.sf2
demucs weights.................. OK
```

Each "FAIL" line includes a hint with the recommended remediation.
`doctor` exits with a non-zero status if anything fails, which makes
it easy to wire into shell scripts. `demucs weights ... NOT CACHED` is
informational, not a failure.

Models: basic-pitch and madmom ship their weights inside their pip
packages, so they never download. The only download is Demucs
(`htdemucs_6s`, ~52 MB, into `~/.cache/torch/hub`), needed only for
full-mix (non-solo) analysis. `setup.sh` fetches it up front; otherwise
run `music-decoder fetch-models` once, or it downloads automatically on
the first full-mix `analyze()`.

---

## CLI reference

The CLI is grouped under `music-decoder` with seven subcommands:
`analyze`, `compose`, `productionize`, `ui`, `doctor`, `fetch-models`,
and `fetch-samples`.

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
                       [--tuning ...] [--sound soundfont | plucked]
                       [--candidates N] [--rank-audio]
                       [--melody-model none | amt]
                       [--melody-engine tune | rules]
                       [--melody-pick bands | phrase] --out DIR
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
- `--bars` — bars per chord; must be ≥ 1 (`0` or a negative number is
  rejected with an error). Default `1`.
- `--seed` — integer; same seed → same music (for a given
  `metadata.composer_version`). Default: a fresh random seed each run,
  recorded as `metadata.seed_used` (see
  [Seed reproducibility](#seed-reproducibility)).
- `--tuning` — same set as `analyze`. Default `EADGBE`.
- `--sound soundfont | plucked` — how the WAV sounds. `soundfont`
  (default) is the General MIDI guitar from the soundfont; `plucked` is a
  built-in string model (an extended Karplus–Strong voice per note, placed
  on the string and fret the tab says) with no soundfont needed. Only the
  WAV changes: the MIDI and the tab are the same for both.
- `--candidates N` — `fingerstyle` only (the other styles ignore it). The
  composer writes N pieces and keeps the one with the best score from the
  tool's own measures of a good tune. Default `200` with the phrase pick
  (the default), `12` with `--melody-pick bands` or `--melody-engine rules`;
  `1` gives a single piece. Must be ≥ 1.
- `--rank-audio` — `fingerstyle` only. Instead of trusting the score
  alone, render the best 4 candidates and let Audiobox Aesthetics (a
  pretrained model that rates how enjoyable and how well produced a clip
  sounds) break the tie. An audio difference under 0.3 on the model's
  1–10 scale counts only in proportion, so a gap that small does not
  overturn the score's order by itself. Needs the `ml` extra, see
  [Optional models](#optional-models-the-ml-extra). With `strum` or
  `arpeggio` it is ignored, and the command says so on stderr.
- `--melody-model none | amt` — `fingerstyle` only. With `amt` a pretrained
  melody model, the Anticipatory Music Transformer, proposes a tune over
  the bass line, and the composer uses the proposal as a soft hint: its own
  rules still decide every note. Needs the `ml` extra. Default `none`.
  With `strum` or `arpeggio` it is ignored, with a note on stderr.
  The model works in seconds, not bars, and proposes for the first **40
  seconds** of the piece at most (about 16 bars at 100 bpm, 10 bars at 60 bpm),
  so its run time stops growing past that length. In a longer piece the
  bars after 40 s are composed without the hint; the command says so on
  stderr (`note: the melody model guides the first 40 s`) and
  `metadata.melody_model_seconds` records the seconds it covered.
- `--melody-engine tune | rules` — `fingerstyle` only. Which generator
  writes the melody. `tune` (the default, new in composer version 6) builds it
  from the statistics of real tunes: real bar rhythms, real note-to-note
  moves, real chord-tone and scale-degree habits (see
  [what the tune engine does](#what-the-tune-engine-does-composer-version-6)).
  `rules` is composer version 5's hand-written generator, note for note, so
  a seed from version 5 still gives its old melody. `strum` and `arpeggio` always use the rules
  engine and ignore the option without a message. It works together with
  `--melody-model amt`, `--candidates` and `--rank-audio`.
- `--melody-pick bands | phrase` — `fingerstyle` with the `tune` engine.
  How the best candidate is chosen. `phrase` (the default since composer
  version 8) keeps the one a small model trained on real pop melodies finds
  about as predictable as a real tune: neither more surprising nor more
  obvious (see [the phrase model](#the-phrase-model)). With `phrase` the
  composer writes 200 candidates unless you give `--candidates`. `bands`
  (version 7's rule) keeps the one whose melody sits most inside the ranges
  of real tunes, from 12 candidates. In two blind listening rounds
  (2026-11-25 and 26, 8 rated pieces per option) `phrase` averaged 6.5 out of
  10 against 5.4 for `bands`. With `--melody-engine rules` the default is
  `bands`, and asking for `phrase` is refused with an error; with `strum` or
  `arpeggio` the option is ignored, with a note on stderr if you gave
  `phrase`.
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

Expected pretty output (paths will differ; the notes are what composer
version 8 writes for this seed: the tune engine's best of 200 by the phrase
pick, on the shipped statistics and phrase model):

```text
MIDI:    out/20260508T142510-7c4a3b21/composition.mid
WAV:     out/20260508T142510-7c4a3b21/composition.wav
ASCII tab:
Fingerstyle
   Cmaj7             Am7               Dm7               G7
e|---------0-0-0---|---------0-0-0---|-0---------------|-3---------------|
B|-1---1---------0-|-1---1---------1-|-----------------|-3---------------|
G|-------0---------|-------0---------|-------5-------5-|-0---------------|
D|-----2-------2---|-----2-------2---|-----3-------3---|-3---------------|
A|-3-------3-------|-0-------0-------|-5-------5-------|-2---------------|
E|-----------------|-----------------|-----------------|-3---------------|
   1 & 2 & 3 & 4 &   1 & 2 & 3 & 4 &   1 & 2 & 3 & 4 &   1 & 2 & 3 & 4 &

Chords
   Cmaj7  Am7  Dm7  G7
e|-0------0----5----3--|
B|-0------1----6----3--|
G|-0------0----5----0--|
D|-2------2----3----3--|
A|-3------0----5----2--|
E|-x------x----x----3--|
```

With `--style fingerstyle` (the default) the tab is **one guitar part**:
every note of the melody and the accompaniment, on its string, in rhythm.
Each bar is eight columns (the count row `1 & 2 & 3 & 4 &` runs under every
line), bars are separated by `|`, four bars sit on a line, and the chord
name is written above its bar. A note that is still ringing is not drawn
again, and a fret of 10 or more widens its column for every string.
The **Chords** block under it shows each chord shape under its name
(`x` = muted string). In the example, the bass notes on the A and D strings
are the thumb, the notes on the top strings are the melody, and in the last
bar a rolled G7 sits under the final melody note, a held G3 on the G string.
`strum` and `arpeggio` still print the older two-part text: a **Melody**
block (one column per note, no rhythm) and the **Chords** block. The MIDI
file has the exact timing in every style.

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

The v5 and v6 options, on a fingerstyle piece (the second and third need the
`ml` extra, see [Optional models](#optional-models-the-ml-extra)):

```bash
# the plucked-string sound instead of the soundfont guitar
music-decoder compose --scale C:major --progression "C G Am F" --bars 2 \
  --seed 601 --sound plucked --out ./out

# keep the best of 200 (the default) by the symbolic score, then let the
# audio model choose among the best 4
music-decoder compose --scale A:minor --progression "Am Dm E7 Am" --bars 2 \
  --seed 602 --rank-audio --out ./out

# let the Anticipatory Music Transformer propose the tune
music-decoder compose --scale E:minor --progression "Em C G D" --bars 2 \
  --seed 604 --tuning DADGAD --melody-model amt --out ./out

# one single piece, no choice among candidates
music-decoder compose --scale C:major --progression "C G Am F" \
  --seed 1 --candidates 1 --out ./out

# composer version 5's hand-written melody instead of the tune engine
music-decoder compose --scale C:major --progression "C G Am F" --bars 2 \
  --seed 601 --melody-engine rules --out ./out

# composer version 7's choice: the best of 12 by the band score
music-decoder compose --scale E:major --progression "E B C#m G#m A E A B" \
  --bars 2 --tempo 110 --seed 26001 --melody-pick bands --out ./out
```

Measured on an Apple M4 (CPU), an 8-bar fingerstyle piece takes about 1–2
seconds with the defaults, 2–4 seconds with `--sound plucked`, 6–8 seconds
with `--rank-audio` (it renders and scores 4 candidates) and 24–29 seconds
with `--melody-model amt` (the model is loaded on every call).
Those times are for `--melody-pick bands`. The default phrase pick takes about
2–7 seconds for a 16-bar piece (writing 200 candidates takes about 3 of them
and the model about 2), plus about 1 second to load the model. Only the chosen candidate's MIDI, WAV and tab
are kept.

### `productionize` — master a composition for release

```text
music-decoder productionize SOURCE --out DIR [--name NAME]
                             [--title TITLE] [--key KEY]
                             [--progression "CHORDS..."]
                             [--percussion NAME[@SECTIONS]]...
                             [--verse held | guide-line | skeleton | build]
                             [--style boom-bap | pop-rap]
                             [--lead guitar | synth]
                             [--samples-dir DIR] [--no-qc]
                             [--format pretty | json]
```

Takes what `compose` wrote and renders it as a finished boom-bap
instrumental: an acoustic drum kit, a synthesized bass, a sampled grand
piano and strings, mixed, mastered to -10.5 LUFS under a -1.8 dBTP
ceiling, and checked.

- `SOURCE` — a take folder written by `compose`, or its `composition.mid`.
  A take folder's `composition.json` supplies the key, progression and
  tempo, and `--key` / `--progression` override it.
- `--out` (required) — where the files go (see
  [Production output](#production-output)).
- `--name` — file name of the master, default `production`.
- `--title`, `--key` — title and key tags of the MP3 (key like `F#m`).
  A key that parses (`Am`, `Bb`, `F#m`) is also the home chord and
  replaces the take's scale and its default title; one that does not
  (`"C major"`) is a tag only.
- `--progression` — the chords of the composition, e.g. `"Am F C G"`.
  The bass follows their roots. Without it, the bass follows a take
  folder's `composition.json` progression; with no manifest either, each
  bar's bass note is the lowest note of the accompaniment in that bar.
- `--percussion` — add `shaker`, `tambourine` or `congas`; repeat the
  option for several. `NAME@1,6` limits one to those sections (numbered
  from 0; default sections 2-6). Off by default.
- `--verse` — what the four verses play (default `build`):
  - `held` — the composer's chord shapes held under the whole hook melody.
  - `guide-line` — piano voicings (root and fifth low, third and seventh
    in the middle) that move as little as possible between chords, and one
    held note a bar from a stepwise line over the chord tones and the
    in-scale ninth. Sparse, leaves room for a vocal.
  - `skeleton` — the same voicings under the hook melody's notes on beats
    1 and 3 only: the tune at half its density.
  - `build` (default) — verses 1-2 as `guide-line`, verses 3-4 as
    `skeleton`, so the verse grows into the second hook.
- `--style` — `boom-bap` (default, described here) or `pop-rap`, an
  upbeat trap-drum instrumental (see [Pop-rap style](#pop-rap-style)).
- `--lead` — the pop-rap lead sound: `guitar` (default, an electric
  guitar) or `synth`. Ignored by boom-bap.
- `--samples-dir` — where samples are kept (default
  `<data_dir>/samples`, or `MUSIC_DECODER_SAMPLES_DIR`). Pop-rap uses no
  samples and ignores it.
- `--no-qc` — skip the acceptance checks.

The composition is one pass over its chord progression. The production
plays that pass eight times, as eight sections, then a final chord:

| # | Section | Piano | Drums | Bass | Strings |
|---|---------|-------|-------|------|---------|
| 0 | Intro | accompaniment only, soft, extra reverb | none | none | none |
| 1 | Hook 1 | melody + accompaniment as composed | light | yes | none |
| 2-5 | Verses 1-4 | held rolled chords + melody (see `--verse`) | full | yes | none |
| 6 | Hook 2 | melody + accompaniment as composed | full | yes | fade in |
| 7 | Outro | held rolled chords + melody, soft, extra reverb | one kick + crash | last note decays | yes |

The final chord is the song's home chord, even when the loop ends
elsewhere (a `Cmaj7 Am7 Dm7 G7` loop ends on Cmaj7, not G7). The key comes
from an explicit `--key` that parses (`Am`, `Bb`), else the take's
manifest or composition scale, else the loop's first chord;
the chord is voiced like the verses (root and fifth low, the third and
seventh in the middle), rolled, and the strings hold its top notes.

Every drum hit is on an 8th note; the groove plays through the last bar
of each section and a change is marked only by an open hi-hat and a
crash. Samples not already fetched by `setup.sh` / `music-decoder
fetch-samples` download on first use (drums about 40 MB, piano 1-2 MB per
note and loudness used). A render takes under a minute
and about 3 GB of memory, so run one at a time.

Example:

```bash
music-decoder compose --scale A:minor --progression "Am F C G" \
  --tempo 92 --seed 7 --out ./out
music-decoder productionize ./out/20261113T085802-79b50081 \
  --out ./release --name night_drive --title "Night Drive" --key Am \
  --progression "Am F C G"
```

Expected output (abridged):

```text
WAV:     release/night_drive.wav
MP3:     release/night_drive.mp3
Raw mix: release/pro_raw.wav (+ 8 stems)
Master:  -10.6 LUFS, -1.93 dBTP, limiter up to 2.0 dB
Credit:  Piano: Salamander Grand Piano V3 by Alexander Holm (CC BY 3.0)
Checks:
PASS  format              44100 Hz, PCM_24, 2 ch, 90.60 s  (expected ...)
PASS  loudness (WAV)      -10.57 LUFS  (expected -10.5 +-0.3 LU)
...
PASS  drum grid           308 of 308 hits on an 8th note  (expected all, within +-3 ms)
```

The checks cover format, loudness and true peak of the WAV and the MP3,
clipping, the lead-in and the tail, DC offset, stereo width and mono
compatibility, tempo and beat steadiness, level across section changes,
and the drum grid. If one fails, the files are still written, the report
is printed, and the command exits with status 1. The checks do not
replace listening.

`--hisaab` renders the reference song the sound was tuned on ("Hisaab",
F# minor, 80.1 BPM) and also compares section levels and octave bands
against its reference master. Pass its final note file as `SOURCE`;
without one the notes are rebuilt from the song's build spec.

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

Checks ffmpeg and the `fluidsynth` Python module, shows which soundfont
synthesis will use (the configured one, else the bundled TimGM6mb), and
whether the Demucs weights are cached. Returns non-zero on any failure
(an uncached Demucs model is reported but not a failure).

### `fetch-models` — pre-download models

```bash
music-decoder fetch-models                 # Demucs weights + GeneralUser GS soundfont
music-decoder fetch-models --no-soundfont  # Demucs only
music-decoder fetch-models --no-demucs     # soundfont only
music-decoder fetch-models --composer-models  # also Audiobox Aesthetics + AMT (~930 MB, needs the `ml` extra)
```

### Optional models: the `ml` extra

Two optional models help `compose` and are never needed for anything else.
They are not part of the default install, and no default code path or test
imports them:

| Model | Used by | Download | Licence |
|---|---|---|---|
| Audiobox Aesthetics (`facebook/audiobox-aesthetics`) | `compose --rank-audio` | ~415 MB | CC-BY-4.0 (Meta) |
| Anticipatory Music Transformer (`stanford-crfm/music-small-800k`) | `compose --melody-model amt` | ~512 MB (a pickled `pytorch_model.bin`, read with torch's weights-only loader) | Apache-2.0 (Stanford CRFM) |

Install the Python packages and fetch both weights (about 930 MB in the
Hugging Face cache, `~/.cache/huggingface/hub`):

```bash
pip install -e ".[ml]"                      # transformers, huggingface_hub, audiobox_aesthetics, anticipation
music-decoder fetch-models --composer-models
```

The `ml` extra pulls in `transformers`, `huggingface_hub`,
`audiobox_aesthetics`, `tqdm` and `anticipation` (not on PyPI, so it is
pinned to a git commit; this needs `git` on the PATH). Each download pins a
model revision and sends no token. Both models run on the CPU, so a seed
gives the same result every time. If you ask for `--rank-audio` or
`--melody-model amt` without the extra or the weights, the command stops
before doing any work with an error that names both fixes.

### `fetch-samples` — pre-download production samples

```bash
music-decoder fetch-samples            # drum kit + percussion (146 files)
music-decoder fetch-samples --piano    # also every piano sample (~700 MB)
music-decoder fetch-samples --list     # print URL and local path, fetch nothing
```

Optional: `productionize` fetches what a song needs on first use. Use
this to prepare a machine that will be offline.

Idempotent: anything already present is kept. `fetch-models` downloads the Demucs
`htdemucs_6s` weights (~52 MB) into the torch hub cache and GeneralUser
GS (~30 MB) to the `fluidsynth_soundfont` path in `config/runtime.yaml`
(`~/.music-decoder/GeneralUser-GS.sf2`). Exits non-zero if a download
fails; the app keeps working either way.

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

`compose()`'s signature is frozen. In composer version 8 it keeps the best of
200 fingerstyle candidates by the phrase pick, with the tune engine writing
each melody. To choose
the sound, the number of candidates, the melody engine or the optional models,
call `compose_with_options()` from
`music_decoder.compose.api` (it is not exported from the package):

```python
from music_decoder.compose.api import compose_with_options

result = compose_with_options(
    Scale("C", "major"),
    [ChordSymbol.parse(c) for c in ("C", "G", "Am", "F")],
    bars_per_chord=2, seed=601, out_dir=Path("./out"),
    sound="plucked",          # or "soundfont" (default)
    candidates=None,          # pieces to write (None: 200, or 12 with bands)
    rank_audio=False,         # True needs the `ml` extra and the weights
    melody_model=None,        # "amt" needs the `ml` extra and the weights
    melody_engine="tune",     # or "rules": composer v5's melody, note for note
    melody_pick=None,         # "phrase" (tune engine) or "bands"; None picks by engine
)
```

Asking for `rank_audio` or `melody_model="amt"` without the extra or the
weights raises `music_decoder.compose.ml.MLUnavailableError`. Any other
`melody_engine` raises `ValueError`; for `strum` and `arpeggio` it is ignored.
`candidates` defaults to `None`: 200 with the phrase pick, else 12.
`melody_pick="phrase"` with `melody_engine="rules"` raises `ValueError`; for
`strum` and `arpeggio` it is ignored. A missing or unreadable weights file
raises `music_decoder.compose.phrase_model.PhraseModelError`, naming the
file; without `melody_pick="phrase"` the file is never read and torch is
never imported.
`compose_with_options()` defaults to `"tune"`, but the lower-level
`arrange()` and `arrange_candidates()` default to `"rules"` (and to fewer
candidates), so code that calls them directly is unchanged.

### `productionize()`

```python
from music_decoder.productionize import productionize, ProductionConfig

def productionize(
    source: Song | Composition | Path | str,
    *,
    out_dir: Path,           # required
    name: str = "production",
    config: ProductionConfig | None = None,
    samples_dir: Path | None = None,
    progression: Sequence[ChordSymbol] | None = None,
    title: str | None = None,
    key: str | None = None,
    check: bool = True,
    progress: ProgressCallback | None = None,
) -> Production: ...
```

`source` is what `compose()` returned, a take folder, or a composer MIDI
file. It is imported from `music_decoder.productionize`, not from the
package root.

For a `Composition` passed from Python, `key` and `progression` only set
the MP3 tags, as before; the take's own metadata already supplies the
scale and the progression.

```python
release = productionize(result, out_dir=Path("./release"), name="night_drive")

print(release.wav_path, release.mp3_path)
print(release.master.integrated_lufs, release.master.true_peak_dbtp)
print(release.qc.passed)        # False if an acceptance check failed
print(release.qc.format())      # the report the CLI prints
```

Every level, EQ setting and time constant of the sound lives in
`ProductionConfig` (frozen dataclasses in
[productionize/config.py](../src/music_decoder/productionize/config.py)).
To change one, pass a modified copy:

```python
from dataclasses import replace
from music_decoder.productionize.config import PercussionLayer, with_percussion

cfg = ProductionConfig()
cfg = replace(cfg, mix=replace(cfg.mix, bass_dbfs=-17.5))      # quieter bass
cfg = with_percussion(cfg, (PercussionLayer("shaker"),))       # shaker in sections 2-6
release = productionize(result, out_dir=Path("./release"), config=cfg)
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
(currently `2026-10-08-v3-guitarset-tuned`).

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
- `composition.wav` — the rendered audio, 16-bit PCM: 22050 Hz with the
  default soundfont sound, 44100 Hz mono with `--sound plucked`.
- `composition.json` — the take's settings: scale, progression, tempo, bars
  per chord, seed and the rest of the metadata; `productionize` reads it.

`<timestamp>` is `YYYYmmddTHHMMSS` (UTC-naive local clock).
`<hash>` is the first 8 hex chars of `sha1(scale | progression |
tempo | style | seed)`. Same inputs → same hash, but the timestamp
differs each run, so re-running with the same inputs creates a fresh
folder rather than overwriting. `--out` from the CLI is the parent
`<out_dir>`. The Streamlit UI uses
`<data_dir>/compositions/` (default `~/.music-decoder/compositions/`).

### Pop-rap style

`--style pop-rap` renders the same composition as an upbeat pop-rap
instrumental instead: half-time trap drums, a sine 808 on each bar's root,
a soft saw pad, a plucked copy of the accompaniment, and your melody as the
lead. Nothing is sampled except the guitar, so it needs no download.
The hats and fx are rolled off above 16 kHz, so the MP3 stays under its
true-peak ceiling. Beat jitter is reported but not checked for pop-rap,
because the beat tracker is unreliable on half-time trap; the drum-grid
check covers timing, and it uses 16th notes for pop-rap.

After the MP3 is encoded, a guard measures its true peak. If it is too
close to the ceiling, the track is mastered again with a lower ceiling, at
most 3 times, and the number of re-masters is recorded as
`mp3_guard_passes` in the JSON metadata. The guard runs even with
`--no-qc`.

```bash
music-decoder productionize ./out/<take-folder> --style pop-rap \
  --lead synth --out ./release --name my_track
```

It lays the composition's one pass over this form. P is the whole number
of passes closest to 16 bars (at least one pass): 16 bars for a 4- or
8-bar pass, 12 for a 12-bar pass.

| Section | Bars | Lead | Drums and 808 | Pad and pluck | Other |
|---|---|---|---|---|---|
| Intro | 8 | none | hats from bar 5 | yes | riser in the last 2 bars |
| Hook 1 | P | melody | yes | yes | crash on bar 1 |
| Verse | P | none (room for a rap) | yes, with hat rolls; none in its last bar | yes | crash on bar 1 |
| Hook 2 | P | melody plus a quiet double | yes | yes | crash on bar 1 |
| Outro | 4 | melody's first 4 bars | none | yes | — |
| Final chord | 2 bars held | — | — | home chord on the pad | — |

The quiet double in hook 2 is the other lead sound.

- **Stems:** `stem_<name>.wav` for `drums`, `808`, `pad`, `pluck`, `fx`
  and `lead` (float, without the lead-in).
- **Tempo:** it suits pieces of about 100–130 bpm. The drums are half-time
  trap at the piece's own tempo, so a much slower or faster piece sounds
  off.
- **Boom-bap only:** `--percussion`, `--verse` and `--hisaab` are usage
  errors with `--style pop-rap`.
- **Lead sound:** the guitar is played from the installed soundfont by
  FluidSynth. Both leads need FluidSynth and the soundfont, because the
  synth lead's hook-2 double is the guitar. It checks for both before it
  writes anything. The synth is mixed louder than the guitar (-16.5 against -18
  dBFS RMS), so it sits more forward in the mix.
- **Source:** it needs composer notes (a melody and an accompaniment), so
  it takes a take folder or `composition.mid`, not a ready song.

In the library:

```python
from music_decoder.productionize import PopRapConfig, ProductionConfig

cfg = ProductionConfig(style="pop-rap", poprap=PopRapConfig(lead="synth"))
release = productionize(result, out_dir=Path("./release"), config=cfg)
```

### Production output

`productionize` writes straight into `--out` (re-running overwrites):

- `<name>.wav` — the master: 44.1 kHz, 24-bit PCM, stereo, with 0.12 s
  of silence at the start.
- `<name>.mp3` — 320 kbps, ID3v2.3 tags (title, BPM, key, genre).
- `<name>.json` — metadata, what the master did, and the check report.
- `pro_raw.wav` — the pre-master mix, 32-bit float.
- `stem_<element>.wav` — one float stem each for `drums`, `bass`,
  `piano`, `strings`, `kick`, `snare`, `hats`, `crash` (and `percussion`
  when on). About 35 MB per stem for a 1:40 track. The pop-rap style
  writes `drums`, `808`, `pad`, `pluck`, `fx` and `lead` instead.

Samples are cached in `<data_dir>/samples/` (`piano/`, `drums/`, `perc/`).

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
    "profile": "fusion",
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
    "hyperparameter_set": "2026-10-08-v3-guitarset-tuned"
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
- `key.profile` — which estimator produced the key: `fusion` (v3
  default: profiles + neural classifier + chord fit), `cnn`, or
  `krumhansl_kessler` / `temperley` when the profile-only backend is
  configured.
- `key.correlation` — for `fusion`, the winning key's average
  standardized score across the cues (higher is more confident; typically
  1–3); for the profile backends, the Pearson correlation in `[-1, 1]`;
  for `cnn`, the classifier's probability.
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
  "ascii_tab": "Fingerstyle\n   Cmaj7 ...",
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
    "seed_used": 42,
    "tuning": "EADGBE",
    "composer_version": 8,
    "sound": "soundfont",
    "candidates": 200,
    "chosen_candidate": 25,
    "score": 3.63,
    "final_fallback": false,
    "identity_misses": [],
    "rank_audio": false,
    "audio_scores": null,
    "melody_model": null,
    "melody_model_seconds": null,
    "melody_engine": "tune",
    "melody_pick": "phrase",
    "phrase_model": "9070f033c2b2"
  }
}
```

The metadata keys added in version 5: `seed_used` (the seed the piece was
made with: `seed`, or the one drawn when no `--seed` was given);
`sound` (`soundfont` or `plucked`);
`candidates` (how many fingerstyle pieces were written; `1` for strum and
arpeggio) and `chosen_candidate` (the index of the one kept); `score` (its
symbolic score, `null` for strum and arpeggio); `final_fallback` (true when
the final chord could not be re-voiced under the final note);
`identity_misses` (the 0-based bars whose accompaniment alone lacks an
essential tone; `null` for strum and arpeggio, which do not count them); `rank_audio` (was the audio re-rank on), `audio_scores`
(`{"CE": ..., "PQ": ...}` of the chosen candidate, or `null`),
`melody_model` (`"amt"` or `null`) and `melody_model_seconds` (how many
seconds from the start the melody model proposed for, at most 40; `null`
without it).

The metadata key added in version 6: `melody_engine` (`"tune"` or
`"rules"`: the generator that wrote the melody; always `"rules"` for strum
and arpeggio). With the tune engine `score` is on a smaller scale than in version 5
(a few points, not a dozen) and can drop below zero: it is the same kind of
sum, but the melody terms are replaced by a penalty for lying outside the
ranges of real tunes (see
[what the tune engine does](#what-the-tune-engine-does-composer-version-6)),
so compare scores only within one engine.

The metadata keys added in version 8: `melody_pick` (`"phrase"` or
`"bands"`; always `"bands"` for strum, arpeggio and the rules engine) and
`phrase_model` (the first 12 hex digits of the weights file's sha256 when
`melody_pick` is `"phrase"`, else `null`). With `"phrase"` the `score` uses
the phrase distance in place of the band distance, so compare it only with
other `"phrase"` scores. A `--melody-pick bands` run keeps version 7's take
folder name.

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

Without fluidsynth, `compose()` falls back to pretty-MIDI's sine synth;
analyze and compose still work.

### `the strings need the fluidsynth command-line tool`

`productionize` renders its strings with the `fluidsynth` program, not
the Python module `doctor` checks. Install it (`brew install fluidsynth`,
`apt install fluidsynth`); `setup.sh` does. It uses
`/usr/share/sounds/sf2/FluidR3_GM.sf2` when present (`apt install
fluid-soundfont-gm`), else the soundfont composition uses.

### `N samples are missing ...` / `could not download ...`

`productionize` could not fetch a sample from GitHub. Check the network,
then retry: finished downloads are kept. For an offline machine, run
`music-decoder fetch-samples --piano` once while online.

### `soundfont: FAIL`

No soundfont was found at all — neither the one configured in
`config/runtime.yaml` nor pretty-MIDI's bundled `TimGM6mb.sf2` (which
means the `pretty_midi` install is broken). Fix with either:

- `music-decoder fetch-models --no-demucs` (installs GeneralUser GS at the
  configured path), or
- edit `config/runtime.yaml` to point `fluidsynth_soundfont` at any
  `.sf2` you have.

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

If Demucs wasn't pre-fetched, the first full-mix analysis downloads its
weights (~52 MB; progress on stderr). Run `music-decoder fetch-models`
once to avoid that. Each process also loads the models on first use, so
the first analysis in a session is a few seconds slower than later ones.

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

- **`fingerstyle`** — one guitar part that one person can play from the
  tab. In each bar the fretting hand holds one shape of the chord and
  everything else is picked inside it (the shape may change from bar to
  bar, see below):
  - the thumb plays a **Travis bass**, a quarter note on every beat,
    alternating the root (beats 1 and 3) with another string of the shape
    (beats 2 and 4);
  - **fills** pluck a chord tone on a free string on the "and" of beats 2
    and 4, and are how the third or seventh is heard when the melody sits
    on the string that carries it (when a third or seventh still has no
    place, the alternate bass on beat 2 or 4 plays it, above the root);
  - the **melody** is chosen only from notes the held shape can reach
    (open strings, or frets near the shape) on the top strings, always
    the highest note when it starts and never on a string the thumb is
    using;
  - the **final bar** is one held melody note over one rolled chord. The
    roll may be voiced differently from the bar's shape so that every
    chord tone, the third and seventh included, sounds under the held note
    (a D ending on D4 is rolled as `x-5-4-2-(3)-x`, where the `(3)` is the
    held melody note, which puts F# under it).

  Every note has one string and fret, the MIDI file and the tab agree note
  for note, and the tab text is the combined tab shown above. Best for
  ballads, folk and slow tempos. It is the default because it sounds
  reasonable on most progressions.

  What composer version 5 adds to fingerstyle (the hook, arch and peak
  bullet below describes the version 5 melody, so it holds only with
  `--melody-engine rules`; the other bullets hold with both engines):

  - **Per-bar shapes.** Each bar may use a different shape of its chord
    (open, or a barre up to the 9th fret). The composer picks the
    sequence by dynamic programming: it prefers shapes that keep the third
    and seventh playable under the melody, shapes that reach more melody
    notes, open and library shapes, a melody range centred on G4, and a
    small hand movement between bars. A chord held for several bars
    usually keeps its shape. After the part is written, any bar whose
    accompaniment alone still lacks an essential tone is tried with the
    other shapes (the melody's pitches never change) and kept only if
    playability and identity both hold. In the evaluation matrices this
    removes the old 2-6% of seventh-chord bars that lost their seventh,
    and widens the melody by 2-3 semitones. The **Chords** block under the
    tab lists every shape the piece used, in order of first use.
  - **Hooks, an arch, peaks and form** (`--melody-engine rules` only; the
    default tune engine has no hook, arch or peak rule, see below). The first
    bar's first two to four notes are the **hook**: every opening phrase restates it, transposed to
    the chord, and the answering phrase restates it with exactly one
    interval changed (by a semitone; the rhythm is kept). Each four-bar phrase follows an
    **arch**, rising to **one** unique highest note (its peak, 55-75% of
    the way through) and falling to the cadence. Pieces of four phrases or
    more (13 bars or more) follow **A A' B A'**, where B has a contrasting
    rhythm and its arch sits two semitones higher; a piece of three phrases
    (9 to 12 bars) is A A' A', so it ends at home, and an 8-bar piece has
    two phrases, A A'. The hard melody rules and the reach sets still apply.
  - **Expressive performance.** The velocities in the MIDI are no longer
    flat: the melody is at least 8 louder than the accompaniment sounding
    with it, strong beats are louder, each phrase swells toward its peak
    and falls to the cadence, thumb notes on beats 1 and 3 are louder than
    on 2 and 4, fills are softest, and every velocity stays in 30-118.
    Accompaniment notes **let ring**: they last until their string is
    plucked again or the bar ends, and are cut short wherever ringing on
    would break playability or sit at or above a starting melody note. The
    tab still draws onsets only.
  - **Audio-only micro-timing.** The WAV is rendered from a temporary
    performance MIDI whose onsets move by at most +-6 ms (seeded, so the
    WAV repeats). The strokes of one chord stay together, the final roll
    keeps its stagger, and `composition.mid` and the tab stay exactly on
    the eighth-note grid.
  - **Best of 12.** The composer writes 12 pieces (`--candidates`) and
    keeps the one with the best score. The score is a weighted sum of the
    tool's own measures: accompaniment identity, melody above the
    accompaniment, hand movement and a final-chord fallback, plus the
    measures of the melody. With the version 5 melody (`--melody-engine
    rules`) those are chord tones on strong beats, stepwise motion, rhythm
    variety, hook, peak, arch and range, minus repeated notes. With the
    tune engine (the default) they are replaced by one penalty: how far
    the melody lies outside the ranges real tunes occupy. The weights are
    in `compose/scoring.py`. `--rank-audio` re-ranks the best 4 with
    Audiobox Aesthetics.
  - **Melody model.** `--melody-model amt` asks the Anticipatory Music
    Transformer for one tune per piece over the thumb pattern (the first
    40 seconds at most). Each melody note's target is the highest proposed
    note sounding at its onset, or else the nearest proposed note starting
    within an eighth after it; a note close to the target, in any octave,
    gets a soft bonus in the pitch scoring. The hard rules, the reach sets
    and the cadence still decide, so the result is as playable as without
    the model.

  Composer version 6 changes the melody only: see
  [What the tune engine does](#what-the-tune-engine-does-composer-version-6)
  below. Everything above works as in version 5 except the hook, arch and
  peak bullet, which is rules-engine only: the shapes, thumb, fills,
  performance, best of 12 and the models apply to both engines. The A A' B A'
  form still sets which bars copy an earlier rhythm, but B's contrasting
  rhythm and higher arch are rules-engine behaviour.

  Limits. The melody's pitches are never changed to fit the chord, and in
  rare cases an accompaniment bar still lacks an essential tone, which
  `identity_misses` lists. For a few endings, mostly diminished (and
  rarely augmented) chords, no roll can put every chord tone under the held
  note: the piece then ends on a thinned, still playable roll of the bar's
  shape, and `final_fallback` says so. The best-of-12 score picks against
  both.
- **`strum`** — a pop strum on the eighth-note grid (down, down-up,
  up-down-up), with strings staggered a few milliseconds like a real
  pick stroke. Best for pop and rock progressions.
- **`arpeggio`** — bass on the downbeat, then seven eighth notes rolling
  up and down the chord. Best for delicate, classical-leaning pieces.

In `strum` and `arpeggio` the accompaniment leaves out chord notes that
would sit on or just under the melody (never the bass, and never the notes
that make the chord's shape major, minor or a seventh), and the piece ends
on one ringing chord. In `fingerstyle` nothing is left out after the fact:
the melody is chosen to fit the held shape, and the piece ends on a held
melody note over one rolled chord. `strum` and `arpeggio` are unchanged by
composer version 5: no per-bar shapes, no dynamics, no candidates, and
`--sound plucked` is their only new option. The melody is written in four-bar phrases with a
repeated motif, resolves leading tones, follows altered chords (G# over
E7 in A minor), and ends on a long tonic (or on the last chord's root
if that chord does not contain the tonic).

Use `--bars 2` or higher when the chord changes feel too fast at
default. Use `--tempo 60-80` for ballads, `--tempo 100-120` for pop,
`--tempo 140+` for upbeat material.

### What the tune engine does (composer version 6)

Listeners found the version 5 melodies unlike real tunes, so version 6
measured real melodies and writes the fingerstyle melody to match. In plain
words, compared with the version 5 melody (`--melody-engine rules`):

- **Repeated notes are allowed.** A real tune repeats the same note in about
  one move out of five; version 5 almost never did.
- **More notes, and more kinds of bar.** The number of notes per bar follows
  the tempo (a slow piece gets more notes per bar), and a piece uses several
  different bar rhythms instead of one reused pattern. Bar rhythms are real
  ones, drawn according to how often each follows the bar before it.
- **Off-the-beat notes and late entries.** About two notes in five fall off
  the beat and some bars start after beat 1, which is how real tunes swing
  and lead into a phrase. The first note of the piece is still on beat 1, and the
  piece still ends on one held note over one rolled chord.
- **Figures come back.** A short figure (a few notes, as steps in the scale)
  is replayed in a later bar with the same rhythm, so a listener hears a
  tune return. The answering phrase of A A' (and A A' A') often copies the
  rhythm of the first phrase. The tune engine has no separate hook, arch or
  peak rules; returning figures do that work.
- **Chord tones are likely, not forced.** Strong beats favour chord tones at
  the rate real melodies use them (about three in four on beats 1 and 3),
  and the melody leans on the scale degrees real melodies lean on. Phrases
  end on a chord tone, and not on the tonic unless nothing else fits, so
  they sound open until the last note.
- **The highest note can come anywhere** in a phrase; version 5 forced it
  into the second half.
- **No move is larger than a fifth** between two consecutive melody notes
  (version 5 allowed up to an octave, rarely), and the hard rules of version
  3 still hold: no tritone, no seventh, and the move into a new chord and into
  the last note at most a fifth. The melody is still chosen from the notes
  the held chord shape can reach, so everything about playability is as
  before.

A repeated note keeps the share that real tunes give it, but a repeat that
fits the harmony and the register poorly is made less likely, never more.

Best of 12 uses the same measure: each candidate is scored by how far its
features lie outside the 25th to 75th percentile range of eight-bar windows
of real tunes in its mode (repeat, step, third and leap shares, notes per
second, rhythm variety, recurring bar rhythms and figures, off-beat and
late-entry shares, chord tones on strong beats, pentatonic share and range),
so the tool keeps the candidate closest to a real tune.

`--melody-engine tune` is for `fingerstyle` only. `strum` and `arpeggio`
always use the rules engine, as before.

**Where the statistics come from.** The engine reads one small file,
`src/music_decoder/compose/data/tune_stats.json`, shipped with the package.
It holds counts and percentiles, nothing else: how often each one-bar rhythm
occurs and follows another, how often each scale-step move follows another,
chord-tone and scale-degree shares, notes per second and the percentile
ranges above. No melody is stored or copied: the longest context in it is a
pair of consecutive moves.

- **POP909, cleaned by Dai et al.** (MIT licence): 909 pop songs with the
  melody, chords, key and human phrase labels
  (<https://github.com/Dsqvival/hierarchical-structure-analysis>, built on
  POP909 by Wang et al., ISMIR 2020). The dataset itself is not in this
  repository.
- **The project's own pilot recordings.** The pitch statistics (interval
  shares, pentatonic share, scale-degree shares, range) of the vocal lines of
  400 songs of the project's own YouTube pilot (the other 88 of its 488 songs
  could not be extracted). The shipped file includes them: the scale-degree
  shares are the mean of the two sources, and the pitch ranges are widened to
  cover both (for example the range band is 8.9 to 15 semitones, and the repeat
  band reaches 0.34). The rhythm and move tables are POP909's alone, because
  the pilot's timing is too noisy. Only these statistics are stored, never audio
  or notes. `meta.sources` in `tune_stats.json` names both sources.

To rebuild the file, fetch the dataset once and run the builder. It reads the
dataset from disk and takes a few seconds:

```bash
git clone --depth 1 https://github.com/Dsqvival/hierarchical-structure-analysis \
  ~/.music-decoder/datasets/pop909-hsa

python scripts/build_tune_stats.py \
  --pop909 ~/.music-decoder/datasets/pop909-hsa/POP909 \
  --out src/music_decoder/compose/data/tune_stats.json \
  [--pilot FILE...]
```

`--pop909` is the `POP909` folder inside the clone. `--pilot` takes one or
more JSON-lines files of vocal lines (one song per line,
`{"tonic_pc": 0, "mode": "major", "notes": [[eighth_slot, midi_pitch], ...]}`),
written by a throwaway extractor (`extract_vocal_lines.py` in
`spikes/preference-pilot/`) that is not part of the repository, and neither
are its output files (`vocal_lines_*.jsonl`). The shipped file was built with
all 400 lines, and rebuilding with the same `--pilot` files reproduces it
exactly, apart from `meta.built` (the date of the build). Without `--pilot`
the result is POP909 only, which is a different file from the shipped one (no
`pilot` entry, other degree shares and bands), so the gates are not
guaranteed to pass on it: use it for a comparison, not as a replacement. The
script prints the counts and the bands it measured.

A rebuilt file that differs in content changes the tune the engine writes for
a given seed. After a rebuild, run `pytest tests/unit/test_compose_tune_gates.py`:
it checks the engine against the new bands (see
[the technical doc](architecture-technical.md#533-composer-v6-the-tune-engine)).

#### The phrase model

`--melody-pick phrase` uses a small network (a causal transformer of 1.28
million weights) trained on the melodies of POP909 (the same MIT-licensed,
cleaned release as above) to read whole phrases. It sees each melody as one
slot per sixteenth note, each slot either "no new note" or a new note at a
pitch relative to the key. Alongside, it sees the chord, the next chord,
where the bar sits in its phrase, whether the phrase restates an earlier one,
and the tempo. Reading up to 32 bars back, it needs 24% fewer bits to
predict unseen real songs than the same network reading one bar at a time.

For a candidate it measures the bits per bar the model needs to predict the
melody, averaged over 8-bar windows (the final whole note left out). Real
tunes need about 14–15 bits per bar (median 14.9 in major and 14.2 in minor,
on songs the model never saw). The tune engine's melodies usually need more,
about 17–23. The pick prefers the candidate nearest the real median, by
`2.0` points of score per interquartile range away from it. A melody with a
note outside the model's range (from an octave below to an eleventh above
the key's tonic just under the melody's middle pitch) counts as 10 ranges away.

The weights ship as `src/music_decoder/compose/data/phrase_model.npz`
(3.05 MB; numpy arrays in half precision, read without pickle, beside a
record of its sources, licence, data split, settings and measured numbers).
They need `torch`, which is installed with the package. To retrain them
(about 13 minutes on an Apple-silicon GPU, longer on the CPU):

```bash
python scripts/train_phrase_model.py ~/.music-decoder/datasets/pop909-hsa/POP909 \
  [--out src/music_decoder/compose/data/phrase_model.npz] [--steps 1500]
```

Songs are split 80/10/10 into training, validation and held-out with a
fixed seed (909); every "real tune" number above comes from the held-out
songs. On the same machine, retraining reproduces the shipped numbers (held-out
14.92 bits per bar). A retrained file has a new hash, so it can change which
candidate is picked for a seed.

### Seed reproducibility

`--seed N` makes `compose()` fully deterministic. Same `(scale,
progression, bars, tempo, style, seed, tuning)` always produces the
same melody (with the same numpy version, since numpy does not
guarantee its random streams across releases). Useful for:

- Iterating on a chord progression while keeping the melody constant.
- Sharing a seed in a bug report so we can reproduce.
- Generating multiple takes by sweeping seed: `for s in 1 2 3 4 5; do
  music-decoder compose --seed $s ...; done`.
- Keeping a take you made without `--seed`: every run records the seed
  it used as `metadata.seed_used` (`--format json`). Run the same command
  again with `--seed <seed_used>` to rebuild the same piece: the same
  MIDI, tab and WAV.

Seeds are reproducible within one composer version, which is recorded
in `metadata.composer_version` (currently `8`). Version 2 (November
2026) replaced the uniform quarter-note generator with phrase-planned
melodies, so a seed saved from an older version now gives a different
(more musical) take. Version 3 (November 2026) plays a faithful,
root-position chord shape that needs at most four fingers in every
tuning, and its melodies never leap by a tritone or a seventh (a leap
wider than a fifth is rare, arpeggiates one chord and steps back), so
a seed saved from version 2 also gives a different take. Version 4
(November 2026) makes `fingerstyle` one playable guitar part, so a
seed gives a different `fingerstyle` take than in version 3; `strum`
and `arpeggio` output is unchanged from version 3. Version 5
(November 2026) again changes only `fingerstyle`: it picks a shape per
bar, writes a hook and a phrase arch, performs the part with dynamics and
let-ring, and keeps the best of 12 candidates. A seed saved from version 4
therefore gives a different `fingerstyle` piece, and the seed now
names the best of 12 pieces, not one (the chosen index is
`metadata.chosen_candidate`). Version 6 (November 2026) changes only the
`fingerstyle` melody: the default tune engine writes it from the statistics
of real tunes, so a seed saved from version 5 gives a different melody. To
get the version 5 melody back for a seed, pass `--melody-engine rules`.
Version 7 (November 2026) changes only `fingerstyle`, and only the tune
engine's scoring and the audio re-rank: the tune score now counts the last
bars of a pass that is not a whole number of 8-bar windows, and the
`--rank-audio` re-rank uses fixed scales, so a candidate's total no longer
depends on which others made the shortlist. A seed saved from version 6 can
therefore pick a different candidate. Version 8 (November 2026) changes only
the `fingerstyle` choice of candidate with the tune engine: it writes 200
and keeps the one the phrase model finds closest to real tunes, so a seed
saved from version 7 gives a different piece. To get the version 7 piece
back for a seed, pass `--melody-pick bands`. `strum` and `arpeggio` are still
unchanged from version 3.
`--candidates`, `--rank-audio`, `--melody-model`, `--melody-engine` and
`--melody-pick` change which piece a seed gives; `--sound` and the audio
micro-timing change only the WAV. The same seed with the same options is
reproducible with or without the models. Compose
output does not depend on `config/hyperparameters.yaml` or
`MUSIC_DECODER_HYPERPARAMETERS`.

To measure the composer, run `python scripts/compose_musicality_report.py`.
It prints every musicality metric over the fixed evaluation matrices;
`tests/fixtures/compose/musicality_baseline.json` holds the
pre-version-2 numbers for comparison (measured at commit `69c2fa7`), and
`tests/fixtures/compose/musicality_v2.json` holds the version-2 numbers
(measured at commit `b41e74f`, with `--matrices all`). The default run
builds the development matrices only; `--matrices acceptance` builds the
held-out acceptance matrix.

Without `--seed`, the random number generator is seeded from
NumPy's default entropy source, so every run differs.

### Soundfonts

`compose()` renders with fluidsynth whenever the `fluidsynth` Python
module is importable, using the first soundfont it finds:

1. `config/runtime.yaml` `fluidsynth_soundfont` (default
   `GeneralUser-GS.sf2`, relative to `data_dir` = `~/.music-decoder`) —
   installed by `setup.sh` / `music-decoder fetch-models`;
2. pretty-MIDI's bundled `TimGM6mb.sf2` (6 MB, always available).

Only if fluidsynth itself is missing does it fall back to
`pretty_midi.PrettyMIDI.synthesize` (a sine synth), with one warning log
line. To use another soundfont, point `fluidsynth_soundfont` at it or run
`python scripts/download_soundfont.py --url <url> --dest <path>`.
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
make test-production  # full-song productionize renders (downloads ~180 MB samples)
make lint         # ruff
make typecheck    # mypy --strict
make run          # alias for `music-decoder ui`
```

The regression suite expects the synthetic fixtures under
`tests/fixtures/synthetic/`. The first run renders them to WAV with
fluidsynth (a `.sf2` under `tests/fixtures/synthetic/soundfont/` if
present, else pretty-MIDI's bundled TimGM6mb; sine only without any
soundfont); the
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

# ...with the plucked-string sound, or the optional models (`pip install -e ".[ml]"`
# and `music-decoder fetch-models --composer-models` first)
music-decoder compose --scale C:major --progression "C G Am F" --bars 2 --seed 601 \
                       --sound plucked --rank-audio --melody-model amt --out ./out

# ...with composer version 5's melody instead of the tune engine
music-decoder compose --scale C:major --progression "C G Am F" --bars 2 --seed 601 \
                       --melody-engine rules --out ./out

# Master it for release (WAV + MP3 + stems + checks)
music-decoder productionize ./out/<take-folder> --out ./release --name my_track

# Launch UI
music-decoder ui

# Library
from music_decoder import analyze, compose, Scale, ChordSymbol
analyze("song.mp3")
compose(scale=Scale("C","major"),
        progression=[ChordSymbol.parse(c) for c in ("Cmaj7","Am7","Dm7","G7")],
        seed=42, out_dir=Path("./out"))
```
