# music-decoder

Local audio analysis tool: takes any MP3/WAV/FLAC and produces a key/scale
estimate plus a guitar tablature with explicit string + fret positions.
Accuracy is the primary success metric; mir_eval-driven regression tests
run against committed fixtures every time.

## Install (recommended)

Prerequisites: Python 3.11, `ffmpeg` on PATH, `chromaprint` (`fpcalc`) for song
fingerprinting (optional). On macOS: `brew install ffmpeg chromaprint fluidsynth`.
On Ubuntu: `apt install ffmpeg libchromaprint1 libfluidsynth3`.

    pipx install ./

(or `pip install -e .` for development.)

Run:

    music-decoder

Opens the Streamlit UI at http://localhost:8501 and starts a worker daemon in the
background. The first run downloads model weights (~500 MB) into your user cache.

## Subcommands

- `music-decoder ui` — UI only (assumes a worker is already running).
- `music-decoder worker` — worker daemon only.
- `music-decoder doctor` — verify ffmpeg, data dirs, and configuration.
- `music-decoder download-models` — pre-fetch basic-pitch, CREPE, Demucs weights.

## Development

```bash
make dev          # install with dev extras
make test-fast    # fast unit tests
make test         # full unit + slow + regression
make lint         # ruff
make typecheck    # mypy
make eval         # regression accuracy harness
make fixtures     # download GuitarSet via mirdata
```

## Architecture

See [docs/superpowers/specs/2026-05-04-music-decoder-design.md](docs/superpowers/specs/2026-05-04-music-decoder-design.md)
for the full design and [docs/superpowers/plans/2026-05-04-music-decoder.md](docs/superpowers/plans/2026-05-04-music-decoder.md)
for the implementation plan.

## Docker (optional)

```bash
docker compose up
```
Mounts a named volume for the SQLite database and artifacts.

## Configuration

- `config/runtime.yaml` — paths, log level, ffmpeg location. Override any field via
  `MUSIC_DECODER_<UPPERCASE_KEY>` env vars.
- `config/hyperparameters.yaml` — pinned hyperparameter set (id is recorded in
  every job so results are reproducible).
- `config/eval_thresholds.yaml` — accuracy gate values used by the regression test.

## License

MIT.
