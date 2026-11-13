# music-decoder

Local audio analysis tool. Four things it does:

1. **Identify chord progression** for a YouTube URL or local MP3/WAV/FLAC.
2. **Generate guitar tablature** for the same.
3. **Suggest a music composition** given a scale and a chord progression — outputs MIDI, WAV, and ASCII tab.
4. **Productionize a composition** — render it as a boom-bap instrumental (real drum kit, bass, sampled grand piano, strings) mastered to a 24-bit WAV and a 320 kbps MP3.

## Install

Prerequisites: Python 3.11 or 3.12, `ffmpeg`, `fluidsynth` (for synthesis).

The fastest way is the bundled `setup.sh`. It installs system binaries
(ffmpeg, fluidsynth, libsndfile), creates `.venv`, installs the package
with dev extras, pre-downloads the models (`music-decoder fetch-models`:
Demucs weights + GeneralUser GS soundfont), and runs `doctor` — so nothing
downloads on first use:

```bash
source ./setup.sh   # also activates .venv in your current shell
# or:
./setup.sh          # runs setup; you must `source .venv/bin/activate` after
```

Re-running is idempotent. Flags: `--no-system`, `--no-models`, `--no-soundfont`,
`--recreate`, `--python <path>`. See [docs/usage.md](docs/usage.md#installation) for the manual install paths
(brew/apt, pipx).

## Usage

```bash
# Analyze a YouTube link (always single-quote URLs — & and ? are shell-special)
music-decoder analyze 'https://youtu.be/dQw4w9WgXcQ'

# Playlist context (&list=...) is ignored automatically; just quote the URL
music-decoder analyze 'https://www.youtube.com/watch?v=ilNt2bikxDI&list=RDilNt2bikxDI'

# Analyze a local file as solo guitar (skip Demucs)
music-decoder analyze song.mp3 --solo-guitar --no-separation

# Compose over a I–vi–ii–V in C major
music-decoder compose \
    --scale C:major \
    --progression "Cmaj7 Am7 Dm7 G7" \
    --style fingerstyle --tempo 100 --seed 42 \
    --out ./out

# Turn that composition into a mastered instrumental (WAV + MP3 + stems)
music-decoder productionize ./out/<take-folder> --out ./release --name my_track

# Launch the Streamlit UI
music-decoder ui

# Verify dependencies (also shows soundfont + Demucs cache status)
music-decoder doctor

# Pre-download models (done by setup.sh; run once after a pip/pipx install)
music-decoder fetch-models

# Pre-download the drum and percussion samples for productionize (optional)
music-decoder fetch-samples
```

## Library

```python
from pathlib import Path

from music_decoder import analyze, compose, Scale, ChordSymbol

result = analyze("https://youtu.be/dQw4w9WgXcQ")
print(result.key, result.chord_progression[:4])

comp = compose(
    scale=Scale("C", "major"),
    progression=[ChordSymbol.parse(c) for c in ("Cmaj7", "Am7", "Dm7", "G7")],
    out_dir=Path("./out"), seed=42,
)
print(comp.midi_path, comp.wav_path)

from music_decoder.productionize import productionize

release = productionize(comp, out_dir=Path("./release"), name="my_track")
print(release.wav_path, release.mp3_path, release.qc.passed)
```

## Development

```bash
make dev          # install with dev extras (pip install -e ".[dev]")
make test-fast    # unit tests only
make test         # unit + integration
make regression   # accuracy regression suite (pytest -m regression)
make lint         # ruff
make typecheck    # mypy --strict
make run          # launch the Streamlit UI (alias for `music-decoder ui`)

# accuracy benchmark on GuitarSet (data in tests/fixtures/guitarset/, git-ignored)
python scripts/benchmark_guitarset.py --split dev --by-style
```

## Documentation

- [docs/usage.md](docs/usage.md) — full usage guide: install paths, CLI
  reference, library API, JSON schema, troubleshooting, tips.
- [docs/architecture-overview.md](docs/architecture-overview.md) —
  high-level pipeline and module map.
- [docs/architecture-technical.md](docs/architecture-technical.md) —
  detailed technical reference (DSP, key/chord/tab algorithms, config).
- [docs/reports/2026-10-08-guitarset-benchmark.md](docs/reports/2026-10-08-guitarset-benchmark.md) —
  measured accuracy on GuitarSet and the decisions behind the v3 defaults.

## License

MIT.

`productionize` plays third-party samples, downloaded on first use:
Salamander Grand Piano V3 by Alexander Holm (CC BY 3.0 — **credit it on
release**: "Piano: Salamander Grand Piano V3 by Alexander Holm (CC BY 3.0)")
and Virtuosity Drums (CC0).
