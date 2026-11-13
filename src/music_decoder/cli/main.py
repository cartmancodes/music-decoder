"""Click-based CLI for music-decoder.

Subcommands:
- analyze <source>: chord progression + tab from file or YouTube URL
- compose: generate an arrangement from a scale + progression
- ui: launch the Streamlit UI
- productionize <source>: render composer output as a mastered instrumental
- doctor: verify ffmpeg, fluidsynth, models
- fetch-models: pre-download Demucs weights and the GeneralUser GS soundfont
- fetch-samples: pre-download the drum, percussion and piano samples
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path
from typing import Literal, cast

import click

from music_decoder.api import analyze, compose
from music_decoder.errors import MusicDecoderError
from music_decoder.productionize import ProductionConfig, Song, hisaab_song, productionize
from music_decoder.productionize.config import (
    PERCUSSION_OPTIONS,
    VERSE_STYLES,
    PercussionLayer,
    with_percussion,
)
from music_decoder.productionize.samples import (
    SampleStore,
    drum_refs,
    percussion_refs,
    piano_refs,
)
from music_decoder.tabs.render import render_ascii_tab
from music_decoder.tabs.tuning import (
    D_STANDARD,
    DADGAD,
    DROP_C,
    DROP_D,
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


@contextlib.contextmanager
def _quiet_stdout() -> Iterator[None]:
    """Redirect OS-level fd 1 to fd 2 for the duration of the block.

    Third-party libraries in the pipeline write progress/diagnostics
    straight to stdout — yt-dlp's ``[download] …`` bars, basic-pitch's
    ``Predicting MIDI for …`` line, and TensorFlow C-level logs. That
    corrupts ``--format json`` output and the documented
    ``analyze … --format json | jq`` workflow. Redirecting at the file-
    descriptor level (not just ``sys.stdout``) catches C-extension and
    subprocess writes too. The result is printed afterward to the
    restored, pristine stdout; the noise lands on stderr where it
    belongs.
    """
    sys.stdout.flush()
    saved_fd = os.dup(1)
    try:
        os.dup2(2, 1)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved_fd, 1)
        os.close(saved_fd)


def _serialize(obj: object) -> object:
    if is_dataclass(obj) and not isinstance(obj, type):
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
@click.option("--format", "fmt", type=click.Choice(["pretty", "json"]), default="pretty")
def cli_analyze(
    source: str,
    tuning: str,
    solo_guitar: bool,
    no_separation: bool,
    fmt: str,
) -> None:
    """Identify chord progression + guitar tab from a file or YouTube URL."""
    try:
        with _quiet_stdout():
            result = analyze(
                source,
                declared_kind="solo_guitar" if solo_guitar else "full_mix",
                tuning=_TUNINGS[tuning],
                use_separation=not no_separation,
            )
    except MusicDecoderError as e:
        raise click.ClickException(str(e)) from e

    if fmt == "json":
        click.echo(json.dumps(_serialize(result), indent=2, default=str))
    else:
        click.echo(f"Source:     {result.source}")
        click.echo(f"Duration:   {result.duration_s:.2f}s @ {result.sample_rate_hz} Hz")
        click.echo(
            f"Key:        {result.key.tonic} {result.key.mode} (corr={result.key.correlation:.2f})"
        )
        click.echo(f"Tempo:      {result.tempo_bpm:.1f} BPM")
        click.echo(f"Chords ({len(result.chord_progression)}):")
        for seg in result.chord_progression:
            click.echo(f"  {seg.start_s:6.2f}-{seg.end_s:6.2f}s  {seg.chord.to_label()}")
        # Tablature. The full song would overplot a fixed-width tab, so show
        # a bounded excerpt (first 16s). Empty tab is a documented outcome
        # for dense mixes — say so explicitly rather than print blank staves.
        if result.tab:
            window = [t for t in result.tab if t.note.start_s <= 16.0]
            click.echo(f"Tab ({len(result.tab)} notes; first 16s):")
            click.echo(render_ascii_tab(window or list(result.tab)[:48]))
        else:
            click.echo("Tab: (no notes transcribed — common on dense full mixes)")


@main.command("compose")
@click.option("--scale", required=True, help="e.g. 'C:major' or 'A:minor'")
@click.option(
    "--progression", required=True, help="space-separated chord symbols, e.g. 'Cmaj7 Am7 Dm7 G7'"
)
@click.option(
    "--style", type=click.Choice(["arpeggio", "strum", "fingerstyle"]), default="fingerstyle"
)
@click.option("--tempo", type=float, default=100.0)
@click.option("--bars", type=int, default=1)
@click.option("--seed", type=int, default=None)
@click.option("--tuning", type=click.Choice(list(_TUNINGS)), default="EADGBE")
@click.option("--out", type=click.Path(file_okay=False), required=True)
@click.option("--format", "fmt", type=click.Choice(["pretty", "json"]), default="pretty")
def cli_compose(
    scale: str,
    progression: str,
    style: str,
    tempo: float,
    bars: int,
    seed: int | None,
    tuning: str,
    out: str,
    fmt: str,
) -> None:
    """Generate a simple arrangement from a scale + chord progression."""
    try:
        s = Scale.parse(scale)
        chords = [ChordSymbol.parse(c) for c in progression.split()]
        with _quiet_stdout():
            result = compose(
                scale=s,
                progression=chords,
                bars_per_chord=bars,
                tempo_bpm=tempo,
                style=cast(Literal["arpeggio", "strum", "fingerstyle"], style),
                tuning=_TUNINGS[tuning],
                seed=seed,
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


def _parse_percussion(specs: tuple[str, ...]) -> tuple[PercussionLayer, ...]:
    """``NAME`` or ``NAME@1,6`` (0-based section numbers) per --percussion."""
    layers = []
    for spec in specs:
        name, _, sections = spec.partition("@")
        if name not in PERCUSSION_OPTIONS:
            raise click.BadParameter(
                f"unknown percussion {name!r}; choose from {', '.join(sorted(PERCUSSION_OPTIONS))}"
            )
        if not sections:
            layers.append(PercussionLayer(name))
            continue
        try:
            layers.append(PercussionLayer(name, tuple(int(s) for s in sections.split(","))))
        except ValueError as e:
            raise click.BadParameter(
                f"sections of {spec!r} must be numbers, e.g. {name}@1,6"
            ) from e
    return tuple(layers)


@main.command("productionize")
@click.argument("source", required=False)
@click.option("--out", type=click.Path(file_okay=False), required=True)
@click.option("--name", default="production", help="file name of the master (without extension)")
@click.option("--title", default=None, help="title tag of the MP3")
@click.option("--key", default=None, help="key tag of the MP3, e.g. 'F#m'")
@click.option(
    "--progression", default=None, help="chords of SOURCE, e.g. 'Am Dm E7 Am' (default: inferred)"
)
@click.option(
    "--percussion",
    multiple=True,
    help="add shaker, tambourine or congas; NAME or NAME@SECTIONS (e.g. tambourine@1,6)",
)
@click.option(
    "--hisaab",
    is_flag=True,
    default=False,
    help="render the reference song; SOURCE is then its note file (optional)",
)
@click.option(
    "--verse",
    type=click.Choice(sorted(VERSE_STYLES)),
    default="held",
    help="verse style for composer songs (see the usage guide)",
)
@click.option("--samples-dir", type=click.Path(file_okay=False), default=None)
@click.option("--no-qc", is_flag=True, default=False, help="skip the acceptance checks")
@click.option("--format", "fmt", type=click.Choice(["pretty", "json"]), default="pretty")
def cli_productionize(
    source: str | None,
    out: str,
    name: str,
    title: str | None,
    key: str | None,
    progression: str | None,
    percussion: tuple[str, ...],
    hisaab: bool,
    verse: str,
    samples_dir: str | None,
    no_qc: bool,
    fmt: str,
) -> None:
    """Render composer output as a mastered instrumental (WAV + MP3 + stems).

    SOURCE is a folder written by `compose` or its composition.mid.
    """
    if source is None and not hisaab:
        raise click.UsageError("give a composition folder or MIDI file (or --hisaab)")
    config = with_percussion(ProductionConfig(), _parse_percussion(percussion))
    config = replace(config, arrange=replace(config.arrange, verse_style=verse))

    def progress(stage: str, fraction: float) -> None:
        click.echo(f"  {stage} ...", err=True)

    try:
        song: Song | str
        if hisaab:
            song = hisaab_song(Path(source) if source else None)
        else:
            song = cast(str, source)
        chords = [ChordSymbol.parse(c) for c in progression.split()] if progression else None
        with _quiet_stdout():
            result = productionize(
                song,
                out_dir=Path(out),
                name=name,
                config=config,
                samples_dir=Path(samples_dir) if samples_dir else None,
                progression=chords,
                title=title,
                key=key,
                check=not no_qc,
                progress=progress,
            )
    except MusicDecoderError as e:
        raise click.ClickException(str(e)) from e
    except ValueError as e:
        raise click.ClickException(str(e)) from e

    qc = result.qc
    if fmt == "json":
        payload = cast(dict[str, object], _serialize(result))
        if qc is not None:
            payload["qc"] = qc.to_dict()
        click.echo(json.dumps(payload, indent=2, default=str))
    else:
        click.echo(f"WAV:     {result.wav_path}")
        click.echo(f"MP3:     {result.mp3_path}")
        click.echo(f"Raw mix: {result.raw_mix_path} (+ {len(result.stem_paths)} stems)")
        click.echo(
            f"Master:  {result.master.integrated_lufs:.1f} LUFS, "
            f"{result.master.true_peak_dbtp:.2f} dBTP, "
            f"limiter up to {result.master.limiter_max_reduction_db:.1f} dB"
        )
        for credit in result.metadata.get("credits", []):
            click.echo(f"Credit:  {credit}")
        if qc is not None:
            click.echo("Checks:")
            click.echo(qc.format())
    if qc is not None and not qc.passed:
        raise click.ClickException(
            f"{len(qc.failures)} acceptance check(s) failed: "
            + ", ".join(c.name for c in qc.failures)
        )


def _suppress_streamlit_email_prompt() -> None:
    """Pre-seed ~/.streamlit/credentials.toml so Streamlit's first-run
    interactive 'Email:' prompt never blocks on stdin.

    Streamlit shows that prompt only when the credentials file is absent
    and the server is not headless. We keep headless=false (so the
    browser still opens) and just write an empty email once, which
    permanently dismisses the prompt. Never overwrites an existing file.
    """
    cred = Path.home() / ".streamlit" / "credentials.toml"
    if cred.exists():
        return
    try:
        cred.parent.mkdir(parents=True, exist_ok=True)
        cred.write_text('[general]\nemail = ""\n')
    except OSError:
        # Non-fatal: worst case the user sees the prompt once.
        pass


@main.command("ui")
def cli_ui() -> None:
    """Launch the Streamlit UI."""
    _suppress_streamlit_email_prompt()
    entry = Path(__file__).parents[1] / "ui" / "streamlit_app.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(entry),
        "--server.headless",
        "false",
        "--browser.gatherUsageStats",
        "false",
    ]
    env = os.environ.copy()
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    raise SystemExit(subprocess.call(cmd, env=env))


def _doctor_checks() -> int:
    rc = 0

    def check(name: str, ok: bool, hint: str = "") -> None:
        nonlocal rc
        click.echo(f"{name:.<32} {'OK' if ok else 'FAIL'}")
        if not ok:
            if hint:
                click.echo(f"  hint: {hint}")
            rc = 1

    check(
        "ffmpeg on PATH",
        shutil.which("ffmpeg") is not None,
        "Install ffmpeg: brew install ffmpeg / apt install ffmpeg",
    )

    try:
        import fluidsynth  # noqa: F401

        check("fluidsynth", True)
    except Exception:
        check("fluidsynth", False, "pip install pyfluidsynth + brew install fluidsynth")

    from music_decoder import assets
    from music_decoder.synth import _resolve_soundfont_path

    sf = _resolve_soundfont_path()
    check(
        "soundfont",
        sf is not None,
        "no soundfont found; run: music-decoder fetch-models",
    )
    if sf is not None:
        source = "bundled TimGM6mb" if sf == assets.bundled_soundfont() else "installed"
        click.echo(f"  using {source}: {sf}")

    # Optional: only full-mix analysis needs Demucs, and it downloads lazily,
    # so a cold cache is reported but doesn't fail the health check.
    try:
        cached = assets.demucs_weights_cached()
    except Exception:
        cached = False
    click.echo(f"{'demucs weights':.<32} {'OK' if cached else 'NOT CACHED'}")
    if not cached:
        click.echo(
            "  hint: run `music-decoder fetch-models` to avoid a ~52 MB download on first use"
        )

    return rc


@main.command("doctor")
def cli_doctor() -> None:
    """Verify external dependencies and model availability."""
    rc = _doctor_checks()
    sys.exit(rc)


@main.command("fetch-models")
@click.option("--no-demucs", is_flag=True, help="Skip the Demucs separation weights (~52 MB).")
@click.option("--no-soundfont", is_flag=True, help="Skip the GeneralUser GS soundfont (~30 MB).")
def cli_fetch_models(no_demucs: bool, no_soundfont: bool) -> None:
    """Pre-download models so first runs don't stall (idempotent).

    Fetches the Demucs htdemucs_6s weights into the torch cache and the
    GeneralUser GS soundfont to the path in config/runtime.yaml. Composition
    works without the soundfont (pretty_midi's bundled one is used).
    """
    from music_decoder import assets

    rc = 0
    if not no_demucs:
        click.echo("Demucs htdemucs_6s weights ...")
        try:
            assets.fetch_demucs()
            click.echo("  OK (cached)")
        except Exception as e:
            click.echo(f"  FAILED: {e}")
            rc = 1
    if not no_soundfont:
        click.echo("GeneralUser GS soundfont ...")
        try:
            path = assets.fetch_soundfont()
            click.echo(f"  OK: {path}")
        except Exception as e:
            click.echo(f"  FAILED: {e} (the bundled TimGM6mb soundfont will be used)")
            rc = 1
    sys.exit(rc)


@main.command("fetch-samples")
@click.option("--samples-dir", type=click.Path(file_okay=False), default=None)
@click.option("--piano", is_flag=True, help="Also fetch every piano sample (~700 MB).")
@click.option("--list", "list_only", is_flag=True, help="Print URL and local path; fetch nothing.")
def cli_fetch_samples(samples_dir: str | None, piano: bool, list_only: bool) -> None:
    """Pre-download the samples `productionize` plays (idempotent).

    Fetches the drum kit and the optional percussion. Piano samples are
    fetched as songs need them unless --piano is given.
    """
    config = ProductionConfig()
    refs = [*drum_refs(config.drums), *percussion_refs(PERCUSSION_OPTIONS)]
    if piano:
        refs += piano_refs(config.piano)
    if list_only:
        for ref in refs:
            click.echo(f"{ref.url}\t{ref.path}")
        return
    store = SampleStore(Path(samples_dir) if samples_dir else None)
    try:
        fetched = store.ensure(refs)
    except MusicDecoderError as e:
        raise click.ClickException(str(e)) from e
    click.echo(f"{len(refs)} samples in {store.root} ({fetched} downloaded)")


if __name__ == "__main__":
    main()
