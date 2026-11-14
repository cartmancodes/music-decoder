"""Sample sources, download, loading, resampling and round robin.

Sources (pinned to a commit so a render is reproducible):

- Piano: Salamander Grand Piano V3 by Alexander Holm (CC BY 3.0).
- Drums and percussion: Virtuosity Drums (CC0).

Samples are fetched on demand into a local directory with the layout
``piano/Fs4v8.flac``, ``drums/oh_kick_snoff_vl3_rr1.flac`` and
``perc/Tamb1_Shake_rr1_Close.wav`` (``#`` is stored as ``s``).
"""

from __future__ import annotations

import os
import urllib.parse
import urllib.request
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import numpy.typing as npt
import soundfile as sf
from scipy import signal

from music_decoder.errors import MusicDecoderError
from music_decoder.productionize.config import (
    PERCUSSION_OPTIONS,
    DrumPiece,
    DrumsConfig,
    PercussionVoice,
    PianoConfig,
)

SALAMANDER_COMMIT = "3382bf9496bba2486f5ab0de55a264d1dfc38404"
VIRTUOSITY_COMMIT = "9f04cf9a734527edfbb0a4eee1f674e45bbf71bc"
PIANO_BASE_URL = (
    "https://raw.githubusercontent.com/sfzinstruments/SalamanderGrandPiano/"
    f"{SALAMANDER_COMMIT}/Samples/"
)
DRUMS_BASE_URL = f"https://raw.githubusercontent.com/sfzinstruments/virtuosity_drums/{VIRTUOSITY_COMMIT}/Samples/"
PIANO_CREDIT = "Piano: Salamander Grand Piano V3 by Alexander Holm (CC BY 3.0)"

_NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_MAGIC = {".flac": b"fLaC", ".wav": b"RIFF"}
_CHUNK = 1 << 20
DOWNLOAD_WORKERS = 8


class SampleError(MusicDecoderError):
    """A sample could not be downloaded or read."""


@dataclass(frozen=True)
class SampleRef:
    """Where a sample comes from and where it is kept (relative path)."""

    url: str
    path: str


def note_name(midi_pitch: int) -> str:
    """Scientific pitch name, e.g. 66 -> ``F#4``."""
    return f"{_NOTE_NAMES[midi_pitch % 12]}{midi_pitch // 12 - 1}"


def piano_ref(midi_pitch: int, layer: int) -> SampleRef:
    stem = f"{note_name(midi_pitch)}v{layer}.flac"
    return SampleRef(
        url=PIANO_BASE_URL + urllib.parse.quote(stem),
        path=f"piano/{stem.replace('#', 's')}",
    )


def drum_ref(piece: DrumPiece, mic: str, layer: int, round_robin: int | None) -> SampleRef:
    rr = "" if round_robin is None else f"_rr{round_robin}"
    stem = f"{mic}_{piece.articulation}_vl{layer}{rr}.flac"
    return SampleRef(url=f"{DRUMS_BASE_URL}{mic}/{piece.folder}/{stem}", path=f"drums/{stem}")


def percussion_ref(voice: PercussionVoice, round_robin: int) -> SampleRef:
    stem = f"{voice.name}_rr{round_robin}_Close.wav"
    return SampleRef(url=f"{DRUMS_BASE_URL}perc/close/{voice.folder}/{stem}", path=f"perc/{stem}")


def drum_takes(piece: DrumPiece, layers: Sequence[int]) -> list[tuple[int, int | None]]:
    """Every (layer, round robin) take of ``piece`` over ``layers``."""
    rrs: Sequence[int | None] = piece.round_robins or (None,)
    return [(layer, rr) for layer in layers for rr in rrs]


def drum_refs(cfg: DrumsConfig) -> list[SampleRef]:
    """Every drum sample the pattern can play."""
    plan: list[tuple[DrumPiece, Sequence[int]]] = [
        (cfg.kick, (cfg.kick_main_layer, cfg.kick_other_layer)),
        (cfg.snare, (*cfg.snare_layers, *cfg.snare_light_layers)),
        (cfg.hat_closed, (cfg.hat_layer, cfg.hat_light_layer)),
        (cfg.hat_open, (cfg.hat_open_layer,)),
        (cfg.crash, (cfg.crash_hook_layer, cfg.crash_other_layer)),
    ]
    refs: dict[str, SampleRef] = {}
    for piece, layers in plan:
        for layer, rr in drum_takes(piece, layers):
            for mic, _ in piece.mic_weights:
                ref = drum_ref(piece, mic, layer, rr)
                refs[ref.path] = ref
    return list(refs.values())


def percussion_refs(options: Iterable[str]) -> list[SampleRef]:
    refs: list[SampleRef] = []
    for name in options:
        for voice in PERCUSSION_OPTIONS[name].voices:
            refs.extend(percussion_ref(voice, rr) for rr in range(1, voice.round_robins + 1))
    return refs


def piano_refs(cfg: PianoConfig, pitches: Iterable[int] | None = None) -> list[SampleRef]:
    """All velocity layers of the given sampled pitches (default: every one)."""
    chosen = cfg.sample_pitches if pitches is None else sorted(set(pitches))
    return [piano_ref(p, layer) for p in chosen for layer in range(1, len(cfg.layer_bounds) + 1)]


def default_samples_dir() -> Path:
    """``MUSIC_DECODER_SAMPLES_DIR`` or ``<data_dir>/samples``."""
    override = os.environ.get("MUSIC_DECODER_SAMPLES_DIR")
    if override:
        return Path(override).expanduser()
    from music_decoder.config.runtime import load_runtime_config

    return Path(load_runtime_config().data_dir).expanduser() / "samples"


def _has_magic(path: Path, suffix: str) -> bool:
    """True if ``path`` starts with the header of a ``suffix`` audio file."""
    try:
        with open(path, "rb") as f:
            return f.read(4) == _MAGIC.get(suffix, b"")
    except OSError:
        return False


def _download(url: str, target: Path) -> None:
    """Download to a ``.part`` file, check the header, then rename into place.

    An interrupted or bogus download (an HTML error page, say) never leaves
    a broken sample behind.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    try:
        with urllib.request.urlopen(url, timeout=120) as resp, open(part, "wb") as out:
            while chunk := resp.read(_CHUNK):
                out.write(chunk)
        if not _has_magic(part, target.suffix):
            raise SampleError(f"download from {url} is not a {target.suffix} file")
        os.replace(part, target)
    except OSError as e:
        raise SampleError(f"could not download {url}: {e}") from e
    finally:
        part.unlink(missing_ok=True)


class SampleStore:
    """A local sample directory that fills itself from the pinned sources."""

    def __init__(self, root: Path | None = None, *, download: bool = True) -> None:
        self.root = root if root is not None else default_samples_dir()
        self.download = download

    def path(self, ref: SampleRef) -> Path:
        return self.root / ref.path

    def missing(self, refs: Iterable[SampleRef]) -> list[SampleRef]:
        unique = {r.path: r for r in refs}
        return [r for r in unique.values() if not self.path(r).exists()]

    def ensure(self, refs: Iterable[SampleRef]) -> int:
        """Make every sample available locally; returns how many were fetched."""
        todo = self.missing(refs)
        if not todo:
            return 0
        if not self.download:
            names = ", ".join(r.path for r in todo[:5])
            raise SampleError(
                f"{len(todo)} samples are missing under {self.root} (e.g. {names}); "
                "run `music-decoder fetch-samples`"
            )
        with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
            list(pool.map(lambda r: _download(r.url, self.path(r)), todo))
        return len(todo)

    def load(
        self, ref: SampleRef, *, max_seconds: float | None = None
    ) -> tuple[npt.NDArray[np.float64], int]:
        """The sample as float64 ``(frames, channels)`` and its sample rate."""
        self.ensure([ref])
        try:
            with sf.SoundFile(str(self.path(ref))) as f:
                frames = -1 if max_seconds is None else int(max_seconds * f.samplerate)
                data = f.read(frames, dtype="float64", always_2d=True)
                return np.asarray(data, dtype=np.float64), int(f.samplerate)
        except (sf.LibsndfileError, OSError) as e:
            raise SampleError(f"could not read sample {self.path(ref)}: {e}") from e


def resample(
    x: npt.NDArray[np.float64],
    sr_from: int,
    sr_to: int,
    *,
    semitones: float = 0.0,
    max_denominator: int = 400,
) -> npt.NDArray[np.float64]:
    """Polyphase resample along axis 0, optionally shifting pitch up by ``semitones``."""
    ratio = Fraction((sr_to / sr_from) / 2.0 ** (semitones / 12.0)).limit_denominator(
        max_denominator
    )
    if ratio == 1:
        return x
    return np.asarray(
        signal.resample_poly(x, ratio.numerator, ratio.denominator, axis=0), dtype=np.float64
    )


class RoundRobin:
    """Cycles through ``n`` takes without playing the same one twice in a row."""

    def __init__(self, n: int, rng: np.random.Generator) -> None:
        if n < 1:
            raise ValueError("a round robin needs at least one take")
        self.n = n
        self._rng = rng
        self._index = int(rng.integers(n))

    def next(self) -> int:
        if self.n > 1:
            self._index = (self._index + int(self._rng.integers(1, self.n))) % self.n
        return self._index
