"""Public entry point: composer output (or a song) to a mastered release."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from music_decoder.productionize.config import ProductionConfig
from music_decoder.productionize.master import (
    MasterReport,
    encode_mp3,
    master,
    with_lead_in,
    write_wav,
)
from music_decoder.productionize.mix import render_mix
from music_decoder.productionize.qc import QcReport, run_qc
from music_decoder.productionize.samples import PIANO_CREDIT, SampleStore
from music_decoder.productionize.song import (
    ProductionError,
    Song,
    song_from_composition,
    song_from_midi,
)
from music_decoder.progress import NullProgress, ProgressCallback
from music_decoder.types import ChordSymbol, Composition

COMPOSITION_MIDI = "composition.mid"
RAW_MIX = "pro_raw.wav"


@dataclass(frozen=True)
class Production:
    """Everything :func:`productionize` wrote."""

    wav_path: Path
    mp3_path: Path
    raw_mix_path: Path
    stem_paths: dict[str, Path]
    master: MasterReport
    qc: QcReport | None
    metadata: dict[str, Any] = field(default_factory=dict)


def resolve_song(
    source: Song | Composition | Path | str,
    *,
    progression: Sequence[ChordSymbol] | None = None,
    title: str | None = None,
    key: str | None = None,
) -> Song:
    """The song for any accepted source: a ``Song``, what ``compose()``
    returned, a composition folder, or a composer MIDI file."""
    if isinstance(source, Song):
        song = source
    elif isinstance(source, Composition):
        song = song_from_composition(source)
    else:
        path = Path(source).expanduser()
        if path.is_dir():
            path = path / COMPOSITION_MIDI
        if not path.is_file():
            raise ProductionError(f"no composer MIDI file at {path}")
        song = song_from_midi(path, progression=progression)
    if title is not None:
        song = replace(song, title=title)
    if key is not None:
        song = replace(song, key=key)
    return song


def productionize(
    source: Song | Composition | Path | str,
    *,
    out_dir: Path,
    name: str = "production",
    config: ProductionConfig | None = None,
    samples_dir: Path | None = None,
    progression: Sequence[ChordSymbol] | None = None,
    title: str | None = None,
    key: str | None = None,
    check: bool = True,
    progress: ProgressCallback | None = None,
) -> Production:
    """Render ``source`` with real drums, bass, sampled piano and strings,
    master it, and write ``<name>.wav`` (24-bit), ``<name>.mp3`` (320 kbps),
    the float pre-master mix and one float stem per element into ``out_dir``.

    ``source`` is what ``compose()`` returned, a composition folder, a
    composer MIDI file, or a ready :class:`Song`. ``progression`` names the
    chords of a MIDI source (otherwise each bar's lowest accompaniment note
    is its bass root). ``title`` and ``key`` (e.g. ``"F#m"``) override the MP3
    tags. With ``check`` the acceptance checks run on the
    result; a failed check is reported in ``Production.qc``, not raised.
    ``progress`` is called with each stage's name as it starts.
    """
    cb: ProgressCallback = progress if progress is not None else NullProgress()
    cfg = config or ProductionConfig()
    song = resolve_song(source, progression=progression, title=title, key=key)
    sr = song.grid.sample_rate
    out_dir.mkdir(parents=True, exist_ok=True)
    store = SampleStore(samples_dir)

    with tempfile.TemporaryDirectory(prefix="productionize-") as tmp:
        mix = render_mix(song, cfg, store, Path(tmp), cb)
    raw_mix_path = write_wav(out_dir / RAW_MIX, mix.mix, sr, subtype="FLOAT")
    stem_paths = {
        stem: write_wav(out_dir / f"stem_{stem}.wav", audio, sr, subtype="FLOAT")
        for stem, audio in mix.stems.items()
    }

    cb("master", 0.0)
    mastered, report = master(mix.mix, cfg.master, sr)
    cb("export", 0.0)
    wav_path = write_wav(
        out_dir / f"{name}.wav", with_lead_in(mastered, cfg.master, sr), sr, subtype="PCM_24"
    )
    tags = {"title": song.title, "TBPM": str(round(song.grid.tempo_bpm)), "genre": cfg.master.genre}
    if song.key:
        tags["TKEY"] = song.key
    mp3_path = encode_mp3(wav_path, out_dir / f"{name}.mp3", tags, cfg.master)

    qc = None
    if check:
        cb("checks", 0.0)
        qc = run_qc(
            song,
            wav_path,
            mp3_path,
            mix.drum_onsets_s,
            cfg.qc,
            cfg.master,
            tail_s=cfg.mix.tail_s,
        )
    metadata: dict[str, Any] = {
        "title": song.title,
        "key": song.key,
        "tempo_bpm": song.grid.tempo_bpm,
        "duration_s": song.body_s + cfg.mix.tail_s + cfg.master.lead_in_s,
        "sections": [s.name for s in song.sections],
        "notes": song.source,
        "percussion": [layer.option for layer in cfg.percussion.layers],
        "credits": [PIANO_CREDIT],
    }
    (out_dir / f"{name}.json").write_text(
        json.dumps(
            {
                "metadata": metadata,
                "master": asdict(report),
                "qc": qc.to_dict() if qc is not None else None,
            },
            indent=2,
        )
    )
    return Production(
        wav_path=wav_path,
        mp3_path=mp3_path,
        raw_mix_path=raw_mix_path,
        stem_paths=stem_paths,
        master=report,
        qc=qc,
        metadata=metadata,
    )
