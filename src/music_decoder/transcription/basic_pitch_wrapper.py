from __future__ import annotations

import statistics
import tempfile
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile

from music_decoder.config.hyperparameters import BasicPitchParams
from music_decoder.pipeline.contracts import (
    LoadedAudio,
    TranscribedNote,
    TranscriptionResult,
)


def transcribe_basic_pitch(
    audio: LoadedAudio, params: BasicPitchParams, *, output_dir: Path,
) -> TranscriptionResult:
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import predict

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_midi_path = output_dir / "raw_basic_pitch.mid"
    post_midi_path = output_dir / "post_basic_pitch.mid"   # post-processing fills it later

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "input.wav"
        wavfile.write(str(wav_path), audio.sr, (audio.samples * 32767).astype(np.int16))
        _, midi_data, note_events = predict(
            str(wav_path),
            model_or_model_path=ICASSP_2022_MODEL_PATH,
            onset_threshold=params.onset_threshold,
            frame_threshold=params.frame_threshold,
            minimum_note_length=params.minimum_note_length_ms,
            minimum_frequency=params.minimum_frequency_hz,
            maximum_frequency=params.maximum_frequency_hz,
        )
    midi_data.write(str(raw_midi_path))
    midi_data.write(str(post_midi_path))   # placeholder until post-processing runs

    notes: list[TranscribedNote] = []
    for start, end, pitch, amplitude, _pitch_bend in note_events:
        notes.append(TranscribedNote(
            start_s=float(start), end_s=float(end),
            pitch=int(pitch), velocity=round(min(127, amplitude * 127)),
            confidence=float(min(1.0, max(0.0, amplitude))),
        ))
    median_conf = statistics.median([n.confidence for n in notes]) if notes else 0.0

    return TranscriptionResult(
        notes=notes, model="basic-pitch",
        raw_midi_path=raw_midi_path, post_midi_path=post_midi_path,
        hyperparameters={
            "onset_threshold": params.onset_threshold,
            "frame_threshold": params.frame_threshold,
            "minimum_note_length_ms": params.minimum_note_length_ms,
            "minimum_frequency_hz": params.minimum_frequency_hz,
            "maximum_frequency_hz": params.maximum_frequency_hz,
        },
        median_confidence=float(median_conf),
    )
