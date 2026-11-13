"""Acceptance checks on a mastered WAV and its MP3.

Each measurement is a small function on audio arrays; :func:`run_qc` turns
them into pass/fail :class:`Check` rows. Checks that compare against a
reference master (section levels, octave bands) run only for songs that
have one; without it they are reported as measurements.

Numbers are not the whole test: the result still has to be judged by ear.
"""

from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import soundfile as sf
from scipy import signal

from music_decoder.ingest.ffmpeg import find_ffmpeg
from music_decoder.productionize import dsp
from music_decoder.productionize.config import MasterConfig, QcConfig
from music_decoder.productionize.dsp import Audio
from music_decoder.productionize.master import ceiling_gain, loudness
from music_decoder.productionize.song import ProductionError, Song

_TP_SLACK_DB = 0.01  # measurement noise on a peak that sits on the ceiling
_FULL_SCALE = 1.0 - 2.0**-15


@dataclass(frozen=True)
class Check:
    """One row of the report. ``passed`` is ``None`` for a measurement that
    has nothing to be compared against."""

    name: str
    passed: bool | None
    value: str
    expected: str


@dataclass(frozen=True)
class QcReport:
    checks: tuple[Check, ...]

    @property
    def passed(self) -> bool:
        return all(c.passed is not False for c in self.checks)

    @property
    def failures(self) -> tuple[Check, ...]:
        return tuple(c for c in self.checks if c.passed is False)

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "checks": [asdict(c) for c in self.checks]}

    def format(self) -> str:
        mark = {True: "PASS", False: "FAIL", None: "info"}
        width = max(len(c.name) for c in self.checks)
        return "\n".join(
            f"{mark[c.passed]}  {c.name:<{width}}  {c.value}  (expected {c.expected})"
            for c in self.checks
        )


# ---- Measurements ----------------------------------------------------------


def first_sound_s(x: Audio, sr: int, threshold_dbfs: float) -> float | None:
    """Time of the first sample above ``threshold_dbfs``."""
    above = np.nonzero(np.max(np.abs(x), axis=1) > dsp.db_to_gain(threshold_dbfs))[0]
    return float(above[0]) / sr if len(above) else None


def tail_peak_db(x: Audio, sr: int, window_s: float) -> float:
    return dsp.gain_to_db(float(np.max(np.abs(x[-int(window_s * sr) :]))))


def stereo_correlation(x: Audio) -> float:
    if np.std(x[:, 0]) == 0.0 or np.std(x[:, 1]) == 0.0:
        return 1.0
    return float(np.corrcoef(x[:, 0], x[:, 1])[0, 1])


def mono_fold_down_db(x: Audio) -> float:
    """Level of the mono sum relative to the stereo signal."""
    return dsp.gain_to_db(dsp.rms(x.mean(axis=1, keepdims=True)) / max(dsp.rms(x), 1e-12))


def low_band_correlation(x: Audio, sr: int, cutoff_hz: float) -> float:
    return stereo_correlation(dsp.butter(x, "lowpass", cutoff_hz, sr, order=4))


def full_scale_samples(x: Audio) -> int:
    return int(np.count_nonzero(np.abs(x) >= _FULL_SCALE))


def beat_grid(x: Audio, sr: int, expected_bpm: float) -> tuple[float, float]:
    """(tempo in BPM, beat jitter in seconds) from librosa's beat tracker.

    The tracker starts from ``expected_bpm`` (its default prior of 120 BPM
    reads a slow backbeat with 8th-note hats as double time). Tempo is the
    slope of a straight line through the tracked beats; jitter is the
    spread of the beats around that line.
    """
    import librosa

    mono = librosa.resample(x.mean(axis=1).astype(np.float32), orig_sr=sr, target_sr=22_050)
    _, beats = librosa.beat.beat_track(
        y=mono, sr=22_050, hop_length=256, start_bpm=expected_bpm, units="time"
    )
    beats = np.asarray(beats, dtype=np.float64)
    if len(beats) < 4:
        return 0.0, float("inf")
    index = np.arange(len(beats))
    slope, intercept = np.polyfit(index, beats, 1)
    return 60.0 / float(slope), float(np.std(beats - (slope * index + intercept)))


def octave_band_levels(
    x: Audio, sr: int, centres_hz: Sequence[float], reference_lufs: float, n_fft: int
) -> npt.NDArray[np.float64]:
    """Level (dB) of each octave band of the mono sum, after matching the
    mono sum's loudness to ``reference_lufs``.

    Band power is summed from a Hann short-time spectrum, the way a
    spectrum-based check sees it: a tone near a band edge leaks into the
    neighbouring band.
    """
    mono = x.mean(axis=1)
    mono = mono * dsp.db_to_gain(reference_lufs - loudness(mono[:, None], sr))
    freqs, _, spectrum = signal.stft(
        mono, fs=sr, nperseg=n_fft, noverlap=n_fft - n_fft // 4, window="hann", scaling="psd"
    )
    power = (np.abs(spectrum) ** 2).mean(axis=1)
    step = freqs[1] - freqs[0]
    levels = []
    for centre in centres_hz:
        band = (freqs >= centre / np.sqrt(2.0)) & (freqs < centre * np.sqrt(2.0))
        levels.append(10.0 * np.log10(max(float(power[band].sum() * step), 1e-20)))
    return np.asarray(levels)


def off_grid(onsets_s: Sequence[float], grid_s: float, tolerance_s: float) -> list[float]:
    """Onsets farther than ``tolerance_s`` from a multiple of ``grid_s``."""
    onsets = np.asarray(onsets_s, dtype=np.float64)
    error = np.abs(onsets - np.round(onsets / grid_s) * grid_s)
    return [float(t) for t in onsets[error > tolerance_s]]


def decode_mp3(mp3: Path, ffmpeg: Path | None = None) -> tuple[Audio, int]:
    """The MP3 decoded to float (so overs are visible, not clipped)."""
    try:
        binary = find_ffmpeg(ffmpeg)
    except FileNotFoundError as e:
        raise ProductionError(f"checking the MP3 needs ffmpeg: {e}") from e
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "decoded.wav"
        cmd = [
            str(binary),
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(mp3),
            "-c:a",
            "pcm_f32le",
            str(wav),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise ProductionError(f"ffmpeg could not decode {mp3.name}: {result.stderr.strip()}")
        audio, sr = sf.read(str(wav), dtype="float64", always_2d=True)
    return audio, int(sr)


# ---- The report ------------------------------------------------------------


def _within(value: float, target: float, tolerance: float) -> bool:
    return abs(value - target) <= tolerance


def _between(value: float, bounds: tuple[float, float]) -> bool:
    return bounds[0] <= value <= bounds[1]


def run_qc(
    song: Song,
    wav_path: Path,
    mp3_path: Path | None,
    drum_onsets_s: Sequence[float],
    cfg: QcConfig,
    master_cfg: MasterConfig,
    *,
    tail_s: float,
    ffmpeg: Path | None = None,
) -> QcReport:
    """Every acceptance check for a finished master."""
    checks: list[Check] = []

    def add(name: str, passed: bool | None, value: str, expected: str) -> None:
        checks.append(Check(name, passed, value, expected))

    info = sf.info(str(wav_path))
    x, sr = sf.read(str(wav_path), dtype="float64", always_2d=True)
    g = song.grid
    lead = master_cfg.lead_in_s
    expected_s = song.body_s + tail_s + lead
    add(
        "format",
        info.samplerate == g.sample_rate
        and info.subtype == "PCM_24"
        and info.channels == 2
        and abs(info.duration - expected_s) < 0.005,
        f"{info.samplerate} Hz, {info.subtype}, {info.channels} ch, {info.duration:.2f} s",
        f"{g.sample_rate} Hz, PCM_24, 2 ch, {expected_s:.2f} s",
    )

    target = master_cfg.target_lufs
    wav_lufs = loudness(x, sr)
    add(
        "loudness (WAV)",
        _within(wav_lufs, target, cfg.loudness_tolerance_lu),
        f"{wav_lufs:.2f} LUFS",
        f"{target:.1f} +-{cfg.loudness_tolerance_lu} LU",
    )
    ceiling_db = dsp.gain_to_db(ceiling_gain(master_cfg))
    wav_peak = dsp.gain_to_db(dsp.true_peak(x, master_cfg.oversample))
    add(
        "true peak (WAV)",
        wav_peak <= ceiling_db + _TP_SLACK_DB,
        f"{wav_peak:.2f} dBTP",
        f"at or below {ceiling_db:.2f}",
    )
    if mp3_path is not None:
        mp3, mp3_sr = decode_mp3(mp3_path, ffmpeg)
        mp3_lufs = loudness(mp3, mp3_sr)
        add(
            "loudness (MP3)",
            _within(mp3_lufs, target, cfg.loudness_tolerance_lu),
            f"{mp3_lufs:.2f} LUFS",
            f"{target:.1f} +-{cfg.loudness_tolerance_lu} LU",
        )
        mp3_peak = dsp.gain_to_db(dsp.true_peak(mp3, master_cfg.oversample))
        mp3_ceiling = cfg.mp3_true_peak_max_dbtp
        if song.reference is not None and song.reference.mp3_true_peak_dbtp is not None:
            mp3_ceiling = song.reference.mp3_true_peak_dbtp
        add(
            "true peak (MP3)",
            mp3_peak <= mp3_ceiling,
            f"{mp3_peak:.2f} dBTP",
            f"at or below {mp3_ceiling:.2f}",
        )
        clipped = full_scale_samples(mp3)
        add("samples at full scale (MP3)", clipped == 0, str(clipped), "0")

    first = first_sound_s(x, sr, cfg.first_sound_dbfs)
    add(
        f"first sound above {cfg.first_sound_dbfs:.0f} dBFS",
        first is not None and _within(first, lead, cfg.first_sound_tolerance_s),
        "none" if first is None else f"{first * 1000:.0f} ms",
        f"{lead * 1000:.0f} +-{cfg.first_sound_tolerance_s * 1000:.0f} ms",
    )
    tail = tail_peak_db(x, sr, cfg.tail_window_s)
    add(
        f"peak of the last {cfg.tail_window_s * 1000:.0f} ms",
        tail < cfg.tail_peak_dbfs,
        f"{tail:.0f} dBFS",
        f"below {cfg.tail_peak_dbfs:.0f}",
    )
    dc = float(np.max(np.abs(x.mean(axis=0))))
    add("DC offset", dc < cfg.dc_offset_max, f"{dc:.5f}", f"under {cfg.dc_offset_max}")

    offset = round(lead * sr)
    spans = [(a + offset, b + offset) for a, b in map(song.section_span, range(len(song.sections)))]
    levels = [dsp.rms_db(x[a:b]) for a, b in spans]
    shown = ", ".join(f"{v:.1f}" for v in levels)
    ref = song.reference
    if ref is not None and len(ref.section_rms_dbfs) == len(levels):
        add(
            "section RMS (dBFS)",
            all(
                _within(v, t, ref.section_rms_tolerance_db)
                for v, t in zip(levels, ref.section_rms_dbfs, strict=True)
            ),
            shown,
            ", ".join(f"{t:.1f}" for t in ref.section_rms_dbfs)
            + f" +-{ref.section_rms_tolerance_db:.0f} dB",
        )
    else:
        add("section RMS (dBFS)", None, shown, "no reference master")

    correlation = stereo_correlation(x)
    add(
        "stereo correlation",
        correlation >= cfg.stereo_correlation_min,
        f"{correlation:.2f}",
        f"at least {cfg.stereo_correlation_min}",
    )
    fold = mono_fold_down_db(x)
    add(
        "mono fold-down",
        fold >= cfg.mono_fold_down_min_db,
        f"{fold:.1f} dB",
        f"within {cfg.mono_fold_down_min_db:.0f} dB",
    )
    low = low_band_correlation(x, sr, cfg.low_band_hz)
    add(
        f"L/R correlation below {cfg.low_band_hz:.0f} Hz",
        low >= cfg.low_correlation_min,
        f"{low:.3f}",
        f"{cfg.low_correlation_min} or more",
    )

    wanted = g.tempo_bpm
    tempo, jitter = beat_grid(x, sr, wanted)
    # Without a reference master a beat tracker's octave error is not a fault.
    octaves = (1.0,) if ref is not None else (0.5, 1.0, 2.0)
    add(
        "tempo from audio",
        any(_within(tempo, wanted * k, cfg.tempo_tolerance_bpm * k) for k in octaves),
        f"{tempo:.1f} BPM",
        f"{wanted:.1f} +-{cfg.tempo_tolerance_bpm} BPM",
    )
    add(
        "beat jitter",
        jitter < cfg.beat_jitter_max_s,
        f"{jitter * 1000:.1f} ms",
        f"under {cfg.beat_jitter_max_s * 1000:.0f} ms",
    )

    beat = round(g.beat_s * sr)
    bar = round(g.bar_s * sr)
    last_ok, first_ok = True, True
    last_shown, first_shown = [], []
    grooves = ("light", "full")
    for i in range(len(song.sections) - 1):
        _, end = spans[i]
        reference_bar = dsp.rms(x[end - bar : end])
        if reference_bar <= 0.0:
            continue
        last = dsp.rms(x[end - beat : end]) / reference_bar
        nxt = dsp.rms(x[end : end + beat]) / reference_bar
        last_shown.append(f"{last:.2f}")
        first_shown.append(f"{nxt:.2f}")
        # Only where the groove carries on must the level carry on too.
        if song.sections[i].drums in grooves and song.sections[i + 1].drums in grooves:
            last_ok &= _between(last, cfg.last_beat_ratio)
            first_ok &= _between(nxt, cfg.first_beat_ratio)
    add(
        "last beat of a section vs its last bar",
        last_ok,
        ", ".join(last_shown),
        f"{cfg.last_beat_ratio[0]}-{cfg.last_beat_ratio[1]} where the groove continues",
    )
    add(
        "next section's first beat vs that bar",
        first_ok,
        ", ".join(first_shown),
        f"{cfg.first_beat_ratio[0]}-{cfg.first_beat_ratio[1]} where the groove continues",
    )

    verse = np.concatenate([x[spans[i][0] : spans[i][1]] for i in song.verse_sections])
    bands = octave_band_levels(
        verse, sr, cfg.octave_centres_hz, cfg.octave_reference_lufs, cfg.octave_fft
    )
    shown = ", ".join(f"{v:.1f}" for v in bands)
    if ref is not None and len(ref.octave_band_db) == len(bands):
        add(
            "octave bands, verses (dB)",
            all(
                _within(float(v), t, ref.octave_band_tolerance_db)
                for v, t in zip(bands, ref.octave_band_db, strict=True)
            ),
            shown,
            ", ".join(f"{t:.1f}" for t in ref.octave_band_db)
            + f" +-{ref.octave_band_tolerance_db:.0f} dB",
        )
    else:
        add("octave bands, verses (dB)", None, shown, "no reference master")

    eighth = g.beat_s / 2.0
    stray = off_grid(drum_onsets_s, eighth, cfg.drum_grid_tolerance_s)
    add(
        "drum grid",
        not stray,
        f"{len(drum_onsets_s) - len(stray)} of {len(drum_onsets_s)} hits on an 8th note",
        f"all, within +-{cfg.drum_grid_tolerance_s * 1000:.0f} ms",
    )
    return QcReport(tuple(checks))
