# music-decoder

Local audio analysis tool. Three things it does:

1. **Identify chord progression** for a YouTube URL or local MP3/WAV/FLAC.
2. **Generate guitar tablature** for the same.
3. **Suggest a music composition** given a scale and a chord progression — outputs MIDI, WAV, and ASCII tab.

## Install

Prerequisites: Python 3.11, `ffmpeg`, `fluidsynth` (for synthesis).

- macOS: `brew install ffmpeg fluidsynth`
- Ubuntu: `apt install ffmpeg libfluidsynth3`

```bash
pipx install ./        # or: pip install -e .   (development)
```

## Usage

```bash
# Analyze a YouTube link
music-decoder analyze https://youtu.be/dQw4w9WgXcQ

# Analyze a local file as solo guitar (skip Demucs)
music-decoder analyze song.mp3 --solo-guitar --no-separation

# Compose over a I–vi–ii–V in C major
music-decoder compose \
    --scale C:major \
    --progression "Cmaj7 Am7 Dm7 G7" \
    --style fingerstyle --tempo 100 --seed 42 \
    --out ./out

# Launch the Streamlit UI
music-decoder ui

# Verify dependencies
music-decoder doctor
```

## Library

```python
from music_decoder import analyze, compose, Scale, ChordSymbol

result = analyze("https://youtu.be/dQw4w9WgXcQ")
print(result.key, result.chord_progression[:4])

comp = compose(
    scale=Scale("C", "major"),
    progression=[ChordSymbol.parse(c) for c in ("Cmaj7","Am7","Dm7","G7")],
    out_dir="./out", seed=42,
)
print(comp.midi_path, comp.wav_path)
```

## Development

```bash
make dev          # install with dev extras
make test-fast    # unit tests
make test         # unit + integration
make regression   # accuracy regression suite
make lint         # ruff
make typecheck    # mypy --strict
```

## Architecture

See [docs/superpowers/specs/2026-05-08-refactor-design.md](docs/superpowers/specs/2026-05-08-refactor-design.md).

## Docker

```bash
docker compose up
```

## License

MIT.
