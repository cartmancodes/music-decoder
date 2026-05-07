"""Click-based CLI for music-decoder.

Subcommands:
- analyze <source>: chord progression + tab from file or YouTube URL
- compose: generate an arrangement from a scale + progression
- ui: launch the Streamlit UI
- doctor: verify ffmpeg, fluidsynth, models
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

import click

from music_decoder.api import analyze, compose
from music_decoder.errors import MusicDecoderError
from music_decoder.tabs.tuning import (
    DADGAD,
    DROP_C,
    DROP_D,
    D_STANDARD,
    EB_HALF_STEP_DOWN,
    STANDARD_EADGBE,
)
from music_decoder.types import ChordSymbol, Scale

_TUNINGS = {
    "EADGBE": STANDARD_EADGBE,
    "Drop D": DROP_D,
    "Drop-D": DROP_D,
    "Eb": EB_HALF_STEP_DOWN,
    "D standard": D_STANDARD,
    "Drop C": DROP_C,
    "DADGAD": DADGAD,
}


def _serialize(obj: object) -> object:
    if is_dataclass(obj):
        return {k: _serialize(v) for k, v in asdict(obj).items()}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (tuple, list)):
        return [_serialize(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    return obj


@click.group()
@click.version_option()
def main() -> None:
    """Music Decoder — chords, tabs, and composition suggestions."""


@main.command("analyze")
@click.argument("source")
@click.option("--tuning", type=click.Choice(list(_TUNINGS)), default="EADGBE")
@click.option("--solo-guitar/--full-mix", default=False)
@click.option("--no-separation", is_flag=True, default=False)
@click.option("--out", type=click.Path(file_okay=False), default=None)
@click.option("--format", "fmt", type=click.Choice(["pretty", "json"]),
              default="pretty")
def cli_analyze(
    source: str, tuning: str, solo_guitar: bool, no_separation: bool,
    out: str | None, fmt: str,
) -> None:
    """Identify chord progression + guitar tab from a file or YouTube URL."""
    try:
        result = analyze(
            source,
            declared_kind="solo_guitar" if solo_guitar else "full_mix",
            tuning=_TUNINGS[tuning],
            use_separation=not no_separation,
            out_dir=Path(out) if out else None,
        )
    except MusicDecoderError as e:
        raise click.ClickException(str(e)) from e

    if fmt == "json":
        click.echo(json.dumps(_serialize(result), indent=2, default=str))
    else:
        click.echo(f"Source:     {result.source}")
        click.echo(f"Duration:   {result.duration_s:.2f}s @ {result.sample_rate_hz} Hz")
        click.echo(f"Key:        {result.key.tonic} {result.key.mode}"
                   f" (corr={result.key.correlation:.2f})")
        click.echo(f"Tempo:      {result.tempo_bpm:.1f} BPM")
        click.echo(f"Chords ({len(result.chord_progression)}):")
        for seg in result.chord_progression:
            click.echo(f"  {seg.start_s:6.2f}–{seg.end_s:6.2f}s  {seg.chord.to_label()}")


@main.command("compose")
@click.option("--scale", required=True, help="e.g. 'C:major' or 'A:minor'")
@click.option("--progression", required=True,
              help="space-separated chord symbols, e.g. 'Cmaj7 Am7 Dm7 G7'")
@click.option("--style", type=click.Choice(["arpeggio", "strum", "fingerstyle"]),
              default="fingerstyle")
@click.option("--tempo", type=float, default=100.0)
@click.option("--bars", type=int, default=1)
@click.option("--seed", type=int, default=None)
@click.option("--tuning", type=click.Choice(list(_TUNINGS)), default="EADGBE")
@click.option("--out", type=click.Path(file_okay=False), required=True)
@click.option("--format", "fmt", type=click.Choice(["pretty", "json"]),
              default="pretty")
def cli_compose(
    scale: str, progression: str, style: str, tempo: float, bars: int,
    seed: int | None, tuning: str, out: str, fmt: str,
) -> None:
    """Generate a simple arrangement from a scale + chord progression."""
    try:
        s = Scale.parse(scale)
        chords = [ChordSymbol.parse(c) for c in progression.split()]
        result = compose(
            scale=s, progression=chords,
            bars_per_chord=bars, tempo_bpm=tempo,
            style=style, tuning=_TUNINGS[tuning], seed=seed,
            out_dir=Path(out),
        )
    except MusicDecoderError as e:
        raise click.ClickException(str(e)) from e
    except ValueError as e:
        raise click.ClickException(str(e)) from e

    if fmt == "json":
        click.echo(json.dumps(_serialize(result), indent=2, default=str))
    else:
        click.echo(f"MIDI:    {result.midi_path}")
        click.echo(f"WAV:     {result.wav_path}")
        click.echo("ASCII tab:")
        click.echo(result.ascii_tab)


@main.command("ui")
def cli_ui() -> None:
    """Launch the Streamlit UI."""
    entry = Path(__file__).parents[1] / "ui" / "streamlit_app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(entry),
           "--server.headless", "false",
           "--browser.gatherUsageStats", "false"]
    raise SystemExit(subprocess.call(cmd, env=os.environ.copy()))


def _doctor_checks() -> int:
    rc = 0

    def check(name: str, ok: bool, hint: str = "") -> None:
        nonlocal rc
        click.echo(f"{name:.<32} {'OK' if ok else 'FAIL'}")
        if not ok:
            if hint:
                click.echo(f"  hint: {hint}")
            rc = 1

    check("ffmpeg on PATH", shutil.which("ffmpeg") is not None,
          "Install ffmpeg: brew install ffmpeg / apt install ffmpeg")

    try:
        import fluidsynth  # noqa: F401
        check("fluidsynth", True)
    except Exception:
        check("fluidsynth", False, "pip install pyfluidsynth + brew install fluidsynth")

    try:
        from music_decoder.config.runtime import load_runtime_config
        cfg = load_runtime_config()
        sf = Path(cfg.fluidsynth_soundfont).expanduser()
        check("soundfont present", sf.exists() if sf.is_absolute() else True,
              f"expected at {sf}")
    except Exception as e:
        check("soundfont present", False, str(e))

    return rc


@main.command("doctor")
def cli_doctor() -> None:
    """Verify external dependencies and model availability."""
    rc = _doctor_checks()
    sys.exit(rc)


if __name__ == "__main__":
    main()
