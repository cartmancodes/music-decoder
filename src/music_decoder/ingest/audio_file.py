# src/music_decoder/ingest/audio_file.py
from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile

from music_decoder.types import AudioSource, LoadedAudio

from .ffmpeg import decode_to_wav

SR_STANDARD = 22050
SR_HIGH = 44100
MIN_DURATION_S = 1.0
SILENCE_RMS_THRESHOLD = 1e-4


class AudioIoError(Exception):
    """Base."""


class CorruptAudioError(AudioIoError):
    pass


class SilentAudioError(AudioIoError):
    pass


class ClipTooShortError(AudioIoError):
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate(samples: np.ndarray[object, np.dtype[np.float32]], sr: int) -> None:
    if not np.isfinite(samples).all():
        raise CorruptAudioError("samples contain NaN or Inf")
    duration_s = samples.size / sr
    if duration_s < MIN_DURATION_S:
        raise ClipTooShortError(f"duration {duration_s:.2f}s < {MIN_DURATION_S}s")
    rms = float(np.sqrt(np.mean(samples**2)))
    if rms < SILENCE_RMS_THRESHOLD:
        raise SilentAudioError(f"rms {rms:.6f} below silence threshold")


def load_audio(source: AudioSource) -> LoadedAudio:
    target_sr = SR_HIGH if source.requested_quality == "high" else SR_STANDARD
    if not source.path.exists():
        raise CorruptAudioError(f"file does not exist: {source.path}")
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "decoded.wav"
        try:
            decode_to_wav(source.path, wav, sr=target_sr)
        except Exception as e:
            raise CorruptAudioError(str(e)) from e
        sr, raw = wavfile.read(str(wav))
        if sr != target_sr:
            raise CorruptAudioError(f"unexpected sample rate {sr}, wanted {target_sr}")
        if raw.dtype == np.int16:
            samples: np.ndarray[object, np.dtype[np.float32]] = (
                raw.astype(np.float32) / 32768.0
            )
        else:
            samples = raw.astype(np.float32)
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        _validate(samples, sr)
    return LoadedAudio(
        samples=samples, sr=sr, duration_s=samples.size / sr,
        sha256=_sha256(source.path), source=source,
    )


# v2 alias: prefer `load_audio_file` going forward.
load_audio_file = load_audio
