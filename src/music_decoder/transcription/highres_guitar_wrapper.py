"""High-resolution transcription backend (Phase B-2).

Wraps Kong et al.'s ``piano_transcription_inference`` package — the same
architecture Riley & Edwards 2024 fine-tuned for guitar. We integrate the
architecture; the v1 stock checkpoint is piano-trained and performs WORSE
than basic-pitch on guitar audio (verified A/B: 0.34 vs 0.60 F-measure on
Bossa nova). The integration is forward-looking scaffolding for when a
fine-tuned guitar checkpoint becomes available — the user supplies it via
the ``checkpoint_path`` parameter, and the rest of the pipeline doesn't
care.

The roadmap [docs/superpowers/research/2026-05-05-accuracy-improvement-roadmap.md]
calls this out: at the time of writing, Riley's fine-tuned weights aren't
publicly hosted; the Zenodo record (10984521) is the access-restricted
Leduc dataset, not the model. Watch the project page for updates.
"""
from __future__ import annotations

import statistics
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pretty_midi
import scipy.io.wavfile as wavfile

from music_decoder.logging_setup import get_logger
from music_decoder.pipeline.contracts import (
    LoadedAudio,
    TranscribedNote,
    TranscriptionResult,
)

_log = get_logger("transcription.highres_guitar")
_HIGHRES_SR = 16000  # piano_transcription_inference operates at 16kHz mono.


def transcribe_highres_guitar(
    audio: LoadedAudio,
    *,
    output_dir: Path,
    checkpoint_path: Path | None = None,
    device: str = "cpu",
) -> TranscriptionResult:
    """Run Kong et al.'s high-resolution architecture on the audio.

    ``checkpoint_path`` is None to use the package's auto-downloaded stock
    piano checkpoint. To use a fine-tuned guitar checkpoint, pass an
    explicit path.
    """
    try:
        # Lazy-import so an unavailable piano_transcription_inference doesn't
        # crash the rest of the pipeline at module-import time.
        from piano_transcription_inference import (
            PianoTranscription,
        )
        from piano_transcription_inference import (
            sample_rate as PT_SAMPLE_RATE,
        )
    except ImportError as e:
        raise ImportError(
            "piano_transcription_inference is not installed. Run "
            "`pip install piano_transcription_inference` (one-time, ~14KB) "
            "and ensure the checkpoint is downloadable; see "
            f"docs/training.md for guitar-fine-tune workflow. Original error: {e}"
        ) from e

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_midi_path = output_dir / "raw_highres_guitar.mid"
    post_midi_path = output_dir / "post_highres_guitar.mid"

    # Resample to the model's expected rate (16 kHz mono).
    samples = audio.samples
    if audio.sr != PT_SAMPLE_RATE:
        from librosa import resample
        samples = resample(samples, orig_sr=audio.sr, target_sr=PT_SAMPLE_RATE)
    samples = samples.astype(np.float32)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)

    transcriptor = PianoTranscription(
        device=device,
        checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
    )

    # piano_transcription_inference writes a MIDI file; we then parse it.
    with tempfile.TemporaryDirectory() as tmp:
        tmp_midi = Path(tmp) / "transcribed.mid"
        transcriptor.transcribe(samples, str(tmp_midi))
        # Copy to the persistent paths
        raw_midi_path.write_bytes(tmp_midi.read_bytes())
        post_midi_path.write_bytes(tmp_midi.read_bytes())

    pm = pretty_midi.PrettyMIDI(str(raw_midi_path))
    notes: list[TranscribedNote] = []
    for inst in pm.instruments:
        for n in inst.notes:
            notes.append(
                TranscribedNote(
                    start_s=float(n.start),
                    end_s=float(n.end),
                    pitch=int(n.pitch),
                    velocity=int(n.velocity),
                    confidence=min(1.0, n.velocity / 127.0),
                )
            )
    notes.sort(key=lambda n: (n.start_s, n.pitch))
    median_conf = (
        statistics.median([n.confidence for n in notes]) if notes else 0.0
    )

    return TranscriptionResult(
        notes=notes,
        model="highres-guitar",
        raw_midi_path=raw_midi_path,
        post_midi_path=post_midi_path,
        hyperparameters={
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else "stock-piano",
            "device": device,
            "sample_rate_hz": PT_SAMPLE_RATE,
        },
        median_confidence=float(median_conf),
    )


def _smoke_test_audio_pipeline(
    samples: np.ndarray[Any, np.dtype[Any]], sr: int, output_path: Path,
) -> None:
    """Helper used by tests: write a temporary WAV and check it round-trips."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(output_path), sr, (samples * 32767).astype(np.int16))
