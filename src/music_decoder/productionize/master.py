"""Master: float mix to a loudness target under a true-peak ceiling.

Subsonic filter, glue compressor, oversampled soft clipper, look-ahead
true-peak limiter, then a loudness loop; export as 24-bit WAV and MP3.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pyloudnorm
import soundfile as sf
from scipy.ndimage import minimum_filter1d, uniform_filter1d

from music_decoder.ingest.ffmpeg import find_ffmpeg
from music_decoder.productionize import dsp
from music_decoder.productionize.config import MasterConfig
from music_decoder.productionize.dsp import Audio
from music_decoder.productionize.song import ProductionError


@dataclass(frozen=True)
class MasterReport:
    integrated_lufs: float
    true_peak_dbtp: float
    gain_db: float
    limiter_max_reduction_db: float
    limiter_p95_reduction_db: float
    iterations: int


def loudness(x: Audio, sr: int) -> float:
    """Integrated loudness in LUFS (ITU-R BS.1770)."""
    return float(pyloudnorm.Meter(sr).integrated_loudness(x))


def ceiling_gain(cfg: MasterConfig) -> float:
    """The linear true-peak ceiling the limiter holds."""
    return dsp.db_to_gain(cfg.ceiling_dbtp) * cfg.ceiling_margin


def soft_clip(x: Audio, cfg: MasterConfig) -> Audio:
    """Oversampled clipper: linear below the knee, tanh above it."""
    loud = np.abs(x)[np.abs(x) > cfg.silence_floor]
    if loud.size == 0:
        return x
    threshold = cfg.clip_ratio * float(np.percentile(loud, cfg.clip_percentile))
    u = dsp.oversample(x, cfg.oversample) / threshold
    knee = cfg.clip_knee
    mag = np.abs(u)
    over = knee + (1.0 - knee) * np.tanh((mag - knee) / (1.0 - knee))
    clipped = np.where(mag <= knee, u, np.sign(u) * over)
    return dsp.downsample(clipped * threshold, cfg.oversample)[: len(x)]


def limiter_gain(x: Audio, ceiling: float, cfg: MasterConfig, sr: int) -> npt.NDArray[np.float64]:
    """Per-sample gain that keeps the oversampled peak under ``ceiling``."""
    n = len(x)
    up = np.abs(dsp.oversample(x, cfg.oversample)).max(axis=1)
    peak = up[: n * cfg.oversample].reshape(n, cfg.oversample).max(axis=1)
    need = np.minimum(1.0, ceiling / np.maximum(peak, 1e-12))
    ahead = max(int(cfg.lookahead_s * sr), 1)
    held = minimum_filter1d(need, size=2 * ahead + 1, mode="nearest")
    target = np.minimum(held, uniform_filter1d(held, size=ahead, mode="nearest"))

    block = cfg.limiter_block
    n_blocks = -(-n // block)
    padded = np.ones(n_blocks * block)
    padded[:n] = target
    floors = padded.reshape(n_blocks, block).min(axis=1)
    coef = float(np.exp(-block / (sr * cfg.release_s)))
    smoothed = np.empty(n_blocks)
    gain = 1.0
    for i, floor in enumerate(floors.tolist()):
        gain = floor if floor < gain else coef * gain + (1.0 - coef) * floor
        smoothed[i] = gain
    centres = np.arange(n_blocks) * block + block / 2.0
    released = np.interp(np.arange(n), centres, smoothed)
    return np.asarray(np.minimum(released, target), dtype=np.float64)


def master(mix: Audio, cfg: MasterConfig, sr: int) -> tuple[Audio, MasterReport]:
    """The mastered audio (no lead-in yet) and what the chain did."""
    if not np.any(np.abs(mix) > cfg.silence_floor):
        raise ProductionError("the mix is silent; nothing to master")
    x = dsp.butter(mix, "highpass", cfg.subsonic_hz, sr, order=cfg.subsonic_order)
    x = dsp.compress(x, cfg.glue.rms_multiple * dsp.rms(x), cfg.glue, sr)
    x = soft_clip(x, cfg)

    ceiling = ceiling_gain(cfg)
    gain_db = cfg.target_lufs - loudness(x, sr)
    iterations = 0
    while True:
        iterations += 1
        driven = x * dsp.db_to_gain(gain_db)
        reduction = limiter_gain(driven, ceiling, cfg, sr)
        out = driven * reduction[:, None]
        error = cfg.target_lufs - loudness(out, sr)
        if abs(error) <= cfg.loudness_tolerance_lu or iterations >= cfg.loudness_iterations:
            break
        gain_db += error
    peak = dsp.true_peak(out, cfg.oversample)
    if peak > ceiling:
        out = out * (ceiling / peak)
        peak = ceiling
    active = reduction[reduction < 1.0]
    report = MasterReport(
        integrated_lufs=loudness(out, sr),
        true_peak_dbtp=dsp.gain_to_db(peak),
        gain_db=gain_db,
        limiter_max_reduction_db=-dsp.gain_to_db(float(reduction.min())),
        limiter_p95_reduction_db=(
            -dsp.gain_to_db(float(np.percentile(active, 5))) if active.size else 0.0
        ),
        iterations=iterations,
    )
    return out, report


def with_lead_in(audio: Audio, cfg: MasterConfig, sr: int) -> Audio:
    return np.concatenate([np.zeros((round(cfg.lead_in_s * sr), audio.shape[1])), audio])


def write_wav(path: Path, audio: Audio, sr: int, *, subtype: str) -> Path:
    """``subtype`` is ``"PCM_24"`` for the master, ``"FLOAT"`` for mix and stems."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr, subtype=subtype)
    return path


def mp3_command(
    ffmpeg: Path, wav: Path, mp3: Path, tags: dict[str, str], cfg: MasterConfig
) -> list[str]:
    cmd = [str(ffmpeg), "-y", "-loglevel", "error", "-i", str(wav)]
    cmd += ["-codec:a", "libmp3lame", "-b:a", cfg.mp3_bitrate, "-id3v2_version", "3"]
    for key, value in tags.items():
        cmd += ["-metadata", f"{key}={value}"]
    return [*cmd, str(mp3)]


def encode_mp3(
    wav: Path, mp3: Path, tags: dict[str, str], cfg: MasterConfig, ffmpeg: Path | None = None
) -> Path:
    try:
        binary = find_ffmpeg(ffmpeg)
    except FileNotFoundError as e:
        raise ProductionError(f"MP3 export needs ffmpeg: {e}") from e
    result = subprocess.run(
        mp3_command(binary, wav, mp3, tags, cfg), capture_output=True, text=True
    )
    if result.returncode != 0:
        raise ProductionError(f"ffmpeg could not encode {mp3.name}: {result.stderr.strip()}")
    return mp3
