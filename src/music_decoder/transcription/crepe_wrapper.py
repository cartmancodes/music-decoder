from __future__ import annotations

import statistics
from pathlib import Path

import numpy as np
import pretty_midi

from music_decoder.config.hyperparameters import CrepeParams
from music_decoder.pipeline.contracts import (
    LoadedAudio,
    TranscribedNote,
    TranscriptionResult,
)


def _hz_to_midi(freq: float) -> int:
    if freq <= 0:
        return -1
    return int(np.round(69 + 12 * np.log2(freq / 440.0)))


def _segment_runs(
    midi_seq: np.ndarray[object, np.dtype[np.int_]],
    conf_seq: np.ndarray[object, np.dtype[np.float64]],
    step_s: float,
) -> list[tuple[float, float, int, float]]:
    """Group consecutive frames with the same pitch into note events."""
    notes: list[tuple[float, float, int, float]] = []
    i = 0
    while i < len(midi_seq):
        if midi_seq[i] < 0:
            i += 1
            continue
        j = i
        while j + 1 < len(midi_seq) and midi_seq[j + 1] == midi_seq[i]:
            j += 1
        start_s = i * step_s
        end_s = (j + 1) * step_s
        avg_conf = float(np.mean(conf_seq[i : j + 1]))
        notes.append((start_s, end_s, int(midi_seq[i]), avg_conf))
        i = j + 1
    return notes


def transcribe_crepe(
    audio: LoadedAudio,
    params: CrepeParams,
    *,
    output_dir: Path,
    median_filter_window: int = 5,
) -> TranscriptionResult:
    import crepe

    output_dir.mkdir(parents=True, exist_ok=True)
    samples = audio.samples
    if audio.sr != 16000:
        from librosa import resample
        samples = resample(samples, orig_sr=audio.sr, target_sr=16000)
    _time, frequency, confidence, _ = crepe.predict(
        samples, sr=16000,
        model_capacity=params.model_capacity,
        step_size=params.step_size_ms,
        viterbi=params.viterbi,
        verbose=0,
    )
    from music_decoder.transcription.post_processing import median_filter_pitch_contour
    smoothed_frequency: np.ndarray[object, np.dtype[np.float64]] = median_filter_pitch_contour(
        np.asarray(frequency, dtype=float), window=median_filter_window
    )
    midi_seq = np.array(
        [_hz_to_midi(f) if c >= 0.5 else -1
         for f, c in zip(smoothed_frequency, confidence, strict=True)]
    )
    step_s = params.step_size_ms / 1000.0
    runs = _segment_runs(midi_seq, confidence, step_s)
    notes = [
        TranscribedNote(start_s=s, end_s=e, pitch=p, velocity=80, confidence=c)
        for s, e, p, c in runs
    ]
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=24)
    for n in notes:
        inst.notes.append(pretty_midi.Note(
            velocity=n.velocity, pitch=n.pitch, start=n.start_s, end=n.end_s,
        ))
    pm.instruments.append(inst)
    raw = output_dir / "raw_crepe.mid"
    post = output_dir / "post_crepe.mid"
    pm.write(str(raw))
    pm.write(str(post))
    median_conf = statistics.median([n.confidence for n in notes]) if notes else 0.0
    return TranscriptionResult(
        notes=notes, model="crepe",
        raw_midi_path=raw, post_midi_path=post,
        hyperparameters={
            "model_capacity": params.model_capacity,
            "step_size_ms": params.step_size_ms,
            "viterbi": params.viterbi,
        },
        median_confidence=float(median_conf),
    )
