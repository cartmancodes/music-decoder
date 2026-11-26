"""Public entry point: composer output (or a song) to a mastered release."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from music_decoder.productionize import dsp
from music_decoder.productionize.config import (
    ComposerArrangeConfig,
    MasterConfig,
    PopRapConfig,
    ProductionConfig,
)
from music_decoder.productionize.instruments import synths
from music_decoder.productionize.instruments.piano import piano_note_refs
from music_decoder.productionize.master import (
    MasterReport,
    encode_mp3,
    master,
    with_lead_in,
    write_wav,
)
from music_decoder.productionize.mix import MixResult, render_mix
from music_decoder.productionize.poprap import PopRapSong, lead_notes, poprap_song
from music_decoder.productionize.poprap_mix import render_poprap_mix
from music_decoder.productionize.qc import QcReport, decode_mp3, run_qc
from music_decoder.productionize.samples import PIANO_CREDIT, SampleStore
from music_decoder.productionize.song import (
    ProductionError,
    Song,
    composition_kwargs,
    manifest_kwargs,
    parse_key_tag,
    read_manifest,
    scale_title,
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


def _path_kwargs(
    path: Path, progression: Sequence[ChordSymbol] | None, key: str | None
) -> dict[str, Any]:
    """What a composer MIDI file's take manifest says, if it has one, with an
    explicit ``progression`` replacing the manifest's. An explicit ``key``
    that parses (``"Am"``, ``"Bb"``) replaces the manifest's scale and its
    title; one that does not (``"C major"``) is a tag only and leaves the
    manifest's scale and title alone. With no manifest this is just those
    arguments."""
    manifest = read_manifest(path)
    kw = manifest_kwargs(manifest) if manifest else {}
    if progression is not None:
        kw["progression"] = progression
    if key is not None:
        kw["key"] = key
        named = parse_key_tag(key)
        if named is not None:
            kw["scale"] = named
            kw["title"] = scale_title(named)
    kw.setdefault("key", "")
    return kw


def resolve_song(
    source: Song | Composition | Path | str,
    *,
    progression: Sequence[ChordSymbol] | None = None,
    title: str | None = None,
    key: str | None = None,
    arrange: ComposerArrangeConfig | None = None,
) -> Song:
    """The song for any accepted source: a ``Song``, what ``compose()``
    returned, a composition folder, or a composer MIDI file. ``arrange``
    lays out composer sources (verse style and the rest)."""
    if isinstance(source, Song):
        song = source
    elif isinstance(source, Composition):
        song = song_from_composition(source, arrange=arrange)
    else:
        path = Path(source).expanduser()
        if path.is_dir():
            path = path / COMPOSITION_MIDI
        if not path.is_file():
            raise ProductionError(f"no composer MIDI file at {path}")
        # The take's scale (or an explicit key tag) tells the arrangement which
        # chord is home.
        song = song_from_midi(path, **_path_kwargs(path, progression, key), arrange=arrange)
    if title is not None:
        song = replace(song, title=title)
    if key is not None:
        song = replace(song, key=key)
    return song


def resolve_poprap_song(
    source: Song | Composition | Path | str,
    *,
    cfg: PopRapConfig,
    progression: Sequence[ChordSymbol] | None = None,
    title: str | None = None,
    key: str | None = None,
) -> PopRapSong:
    """The pop-rap song for a composition, a composition folder or a composer
    MIDI file. A ready ``Song`` is a boom-bap arrangement and is refused."""
    if isinstance(source, Song):
        raise ProductionError("pop-rap needs a composer file or composition, not a ready Song")
    if isinstance(source, Composition):
        song = poprap_song(source.midi_path, cfg=cfg, **composition_kwargs(source))
    else:
        path = Path(source).expanduser()
        if path.is_dir():
            path = path / COMPOSITION_MIDI
        if not path.is_file():
            raise ProductionError(f"no composer MIDI file at {path}")
        song = poprap_song(path, cfg=cfg, **_path_kwargs(path, progression, key))
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
    """Render ``source``, master it, and write ``<name>.wav`` (24-bit),
    ``<name>.mp3`` (320 kbps), the float pre-master mix and one float stem per
    element into ``out_dir``.

    ``config.style`` picks the sound. Boom-bap (the default) renders real
    drums, bass, sampled piano and strings from a composer source or a ready
    :class:`Song`. Pop-rap renders trap drums, an 808, a pad, a pluck and a
    lead from a composer source only; it needs ``samples_dir`` only for the
    piano lead.

    ``source`` is what ``compose()`` returned, a composition folder, a
    composer MIDI file, or (boom-bap only) a ready :class:`Song`.
    ``progression`` names the chords of a MIDI source (otherwise the take
    manifest's progression, if it has one, else each bar's lowest
    accompaniment note is its bass root). ``title`` and ``key`` (e.g.
    ``"F#m"``) override the MP3 tags. With ``check`` the acceptance checks run
    on the result; a failed check is reported in ``Production.qc``, not
    raised. ``progress`` is called with each stage's name as it starts.
    """
    cb: ProgressCallback = progress if progress is not None else NullProgress()
    cfg = config or ProductionConfig()
    if cfg.style == "pop-rap":
        return _productionize_poprap(
            source,
            out_dir=out_dir,
            name=name,
            cfg=cfg,
            progression=progression,
            title=title,
            key=key,
            check=check,
            cb=cb,
            samples_dir=samples_dir,
        )

    song = resolve_song(source, progression=progression, title=title, key=key, arrange=cfg.arrange)
    out_dir.mkdir(parents=True, exist_ok=True)
    store = SampleStore(samples_dir)

    with tempfile.TemporaryDirectory(prefix="productionize-") as tmp:
        mix = render_mix(song, cfg, store, Path(tmp), cb)
    metadata: dict[str, Any] = {
        "title": song.title,
        "verse_style": cfg.arrange.verse_style,
        "key": song.key,
        "tempo_bpm": song.grid.tempo_bpm,
        "duration_s": song.body_s + cfg.mix.tail_s + cfg.master.lead_in_s,
        "sections": [s.name for s in song.sections],
        "notes": song.source,
        "percussion": [layer.option for layer in cfg.percussion.layers],
        "credits": [PIANO_CREDIT],
        "style": "boom-bap",
        "lead": None,
    }
    return _master_and_report(
        song,
        mix,
        cfg,
        out_dir=out_dir,
        name=name,
        tail_s=cfg.mix.tail_s,
        drum_grid="8th",
        metadata=metadata,
        check=check,
        cb=cb,
    )


def _productionize_poprap(
    source: Song | Composition | Path | str,
    *,
    out_dir: Path,
    name: str,
    cfg: ProductionConfig,
    progression: Sequence[ChordSymbol] | None,
    title: str | None,
    key: str | None,
    check: bool,
    cb: ProgressCallback,
    samples_dir: Path | None = None,
) -> Production:
    song = resolve_poprap_song(
        source, cfg=cfg.poprap, progression=progression, title=title, key=key
    )
    # The guitar and synth leads need the guitar (the lead, or the synth's hook-2
    # double), so look before writing. The piano lead plays alone.
    if cfg.poprap.lead != "piano":
        synths.require_guitar(cfg.poprap.guitar_lead)
    store = None
    if cfg.poprap.lead == "piano":
        # Fetch (or, offline, fail on) the piano samples before writing anything.
        store = SampleStore(samples_dir)
        main, _ = lead_notes(song)
        store.ensure(
            piano_note_refs(
                main, bar_s=song.grid.bar_s, tail_start_s=song.body_s, cfg=cfg.poprap.piano_lead
            )
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="productionize-") as tmp:
        mix = render_poprap_mix(song, cfg.poprap, Path(tmp), cb, store=store)
    tail_s = cfg.poprap.form.tail_s
    metadata: dict[str, Any] = {
        "title": song.title,
        "verse_style": None,
        "key": song.key,
        "tempo_bpm": song.grid.tempo_bpm,
        "duration_s": song.body_s + tail_s + cfg.master.lead_in_s,
        "sections": [s.name for s in song.sections],
        "notes": song.source,
        "percussion": [],
        "credits": [PIANO_CREDIT] if cfg.poprap.lead == "piano" else [],
        "style": "pop-rap",
        "lead": cfg.poprap.lead,
        "mp3_guard_passes": 0,
    }
    return _master_and_report(
        song,
        mix,
        cfg,
        out_dir=out_dir,
        name=name,
        tail_s=tail_s,
        drum_grid="16th",
        metadata=metadata,
        check=check,
        cb=cb,
        mp3_guard=cfg.poprap,
        check_beat_jitter=False,
    )


def mp3_true_peak_dbtp(mp3: Path, master_cfg: MasterConfig) -> float:
    """The MP3's true peak as ``run_qc`` measures it (decoded to float)."""
    audio, _ = decode_mp3(mp3)
    return dsp.gain_to_db(dsp.true_peak(audio, master_cfg.oversample))


def _master_and_report(
    song: Song | PopRapSong,
    mix: MixResult,
    cfg: ProductionConfig,
    *,
    out_dir: Path,
    name: str,
    tail_s: float,
    drum_grid: Literal["8th", "16th"],
    metadata: dict[str, Any],
    check: bool,
    cb: ProgressCallback,
    mp3_guard: PopRapConfig | None = None,
    check_beat_jitter: bool = True,
) -> Production:
    """Write the raw mix and stems, master, export, check and report. The part
    of :func:`productionize` the two styles share. ``mp3_guard`` (pop-rap only)
    re-masters while the MP3's true peak is too close to the delivery ceiling;
    ``check_beat_jitter`` off (pop-rap) reports the jitter without checking it."""
    sr = song.grid.sample_rate
    raw_mix_path = write_wav(out_dir / RAW_MIX, mix.mix, sr, subtype="FLOAT")
    stem_paths = {
        stem: write_wav(out_dir / f"stem_{stem}.wav", audio, sr, subtype="FLOAT")
        for stem, audio in mix.stems.items()
    }

    tags = {"title": song.title, "TBPM": str(round(song.grid.tempo_bpm)), "genre": cfg.master.genre}
    if song.key:
        tags["TKEY"] = song.key

    def master_and_export(master_cfg: MasterConfig) -> tuple[MasterReport, Path, Path]:
        cb("master", 0.0)
        mastered, rep = master(mix.mix, master_cfg, sr)
        cb("export", 0.0)
        wav = write_wav(
            out_dir / f"{name}.wav", with_lead_in(mastered, master_cfg, sr), sr, subtype="PCM_24"
        )
        return rep, wav, encode_mp3(wav, out_dir / f"{name}.mp3", tags, master_cfg)

    report, wav_path, mp3_path = master_and_export(cfg.master)
    if mp3_guard is not None:
        master_cfg = cfg.master
        threshold = cfg.qc.mp3_true_peak_max_dbtp - mp3_guard.mp3_guard_margin_db
        passes = 0
        while passes < mp3_guard.mp3_guard_attempts:
            excess = mp3_true_peak_dbtp(mp3_path, master_cfg) - threshold
            if excess <= 0.0:
                break
            master_cfg = replace(
                master_cfg,
                ceiling_dbtp=master_cfg.ceiling_dbtp - (excess + mp3_guard.mp3_guard_margin_db),
            )
            report, wav_path, mp3_path = master_and_export(master_cfg)
            passes += 1
        metadata["mp3_guard_passes"] = passes

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
            tail_s=tail_s,
            drum_grid=drum_grid,
            check_beat_jitter=check_beat_jitter,
        )
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
