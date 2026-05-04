from __future__ import annotations

import json
import shutil
import traceback
from datetime import UTC, datetime

import matplotlib.pyplot as plt
import numpy as np
from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, selectinload

from music_decoder.artifacts.base import ArtifactStore
from music_decoder.audio_io.load import load_audio
from music_decoder.beat_tracking.beats import track_beats
from music_decoder.beat_tracking.time_signature import infer_time_signature
from music_decoder.config.hyperparameters import HyperparameterSet
from music_decoder.key_detection.api import detect_key
from music_decoder.key_detection.chroma import compute_chroma_with_hpss
from music_decoder.logging_setup import get_logger
from music_decoder.midi_synth.fluidsynth_wrapper import SynthBackend, synthesize_midi_to_wav
from music_decoder.persistence.models import Job
from music_decoder.persistence.repositories import (
    JobProgressRepo,
    JobRepo,
    KeyEstimateRepo,
    NoteRepo,
    TempoEstimateRepo,
)
from music_decoder.pipeline.contracts import AudioSource, LoadedAudio, SeparationResult
from music_decoder.pipeline.events import StageEventEmitter
from music_decoder.separation.demucs import isolate_guitar
from music_decoder.tab_assignment.assigner import assign_tab
from music_decoder.tab_assignment.tuning import get_preset
from music_decoder.transcription.basic_pitch_wrapper import transcribe_basic_pitch
from music_decoder.transcription.crepe_wrapper import transcribe_crepe
from music_decoder.transcription.post_processing import apply_post_processing
from music_decoder.ui.components.chromagram import render_chromagram_figure
from music_decoder.ui.components.waveform import render_waveform_figure

_log = get_logger("orchestrator")


def process_audio(
    *, job_id: int, engine: Engine, artifacts: ArtifactStore,
    hyperparameters: HyperparameterSet,
) -> None:
    with Session(engine) as s:
        job_repo = JobRepo(s)
        progress = JobProgressRepo(s)
        note_repo = NoteRepo(s)
        key_repo = KeyEstimateRepo(s)
        tempo_repo = TempoEstimateRepo(s)

        # Load job with its upload relationship eagerly to avoid detached issues
        job = s.execute(
            select(Job).where(Job.id == job_id).options(selectinload(Job.upload))
        ).scalar_one()

        if job.status not in ("queued", "running"):
            return
        job.status = "running"
        job.started_at = datetime.now(UTC)
        s.commit()

        # Re-load after commit to get fresh state with upload
        job = s.execute(
            select(Job).where(Job.id == job_id).options(selectinload(Job.upload))
        ).scalar_one()

        emit = StageEventEmitter(progress, job_id=job_id)
        try:
            with emit("audio_io") as summary:
                source = AudioSource(
                    path=artifacts.path_for(job.upload.artifact_path),
                    declared_kind=job.upload.declared_kind,  # type: ignore[arg-type]
                    requested_quality=job.requested_quality,  # type: ignore[arg-type]
                    requested_tuning=get_preset(job.requested_tuning),
                )
                audio = load_audio(source)
                summary["duration_s"] = audio.duration_s
                summary["sr"] = audio.sr
            s.commit()

            with emit("separation") as summary:
                # C4: respect use_demucs toggle; skip separation when user opts out
                if not job.use_demucs or audio.source.declared_kind == "solo_guitar":
                    separation = SeparationResult(
                        guitar_samples=None, sr=audio.sr,
                        skipped_reason=(
                            "solo_guitar declared"
                            if audio.source.declared_kind == "solo_guitar"
                            else "user_disabled_demucs"
                        ),
                        bleed_estimate_db=None,
                    )
                else:
                    separation = isolate_guitar(audio)
                samples_for_pitch = (
                    separation.guitar_samples if separation.guitar_samples is not None
                    else audio.samples
                )
                summary["skipped_reason"] = separation.skipped_reason
            s.commit()

            output_dir = artifacts.path_for(f"derived/{job_id}")
            with emit("transcription") as summary:
                if job.transcription_model == "basic-pitch":
                    audio_for_t = LoadedAudio(
                        samples=samples_for_pitch, sr=audio.sr,
                        duration_s=samples_for_pitch.size / audio.sr,
                        sha256=audio.sha256, source=source,
                    )
                    raw_t = transcribe_basic_pitch(
                        audio_for_t, hyperparameters.basic_pitch,
                        output_dir=output_dir,
                    )
                else:
                    audio_for_t = LoadedAudio(
                        samples=samples_for_pitch, sr=audio.sr,
                        duration_s=samples_for_pitch.size / audio.sr,
                        sha256=audio.sha256, source=source,
                    )
                    raw_t = transcribe_crepe(
                        audio_for_t, hyperparameters.crepe, output_dir=output_dir,
                        median_filter_window=hyperparameters.post_processing.median_filter_window,
                    )
                summary["raw_note_count"] = len(raw_t.notes)
            s.commit()

            with emit("key_detection") as summary:
                key_result = detect_key(
                    samples_for_pitch, sr=audio.sr,
                    hpss_margin=hyperparameters.key_detection.hpss_margin,
                    segment_length_s=hyperparameters.key_detection.windowed_segment_length_s,
                    hop_s=hyperparameters.key_detection.windowed_hop_s,
                )
                key_rows: list[dict] = []  # type: ignore[type-arg]
                for profile, top3 in key_result.global_top3_per_profile.items():
                    for rank, est in enumerate(top3, start=1):
                        key_rows.append({
                            "scope": "global", "window_start_s": None,
                            "window_end_s": None, "profile": profile,
                            "rank": rank, "tonic": est.tonic, "mode": est.mode,
                            "correlation": est.correlation, "margin": est.margin,
                        })
                for ws, we, est in key_result.windowed_segments:
                    key_rows.append({
                        "scope": "window", "window_start_s": float(ws),
                        "window_end_s": float(we), "profile": est.profile,
                        "rank": 1, "tonic": est.tonic, "mode": est.mode,
                        "correlation": est.correlation, "margin": est.margin,
                    })
                key_repo.bulk_insert(job_id, key_rows)
                summary["consensus"] = (
                    f"{key_result.consensus_key.tonic} {key_result.consensus_key.mode}"
                    if key_result.consensus_key else None
                )
            s.commit()

            with emit("beat_tracking") as summary:
                grid = track_beats(
                    samples_for_pitch, sr=audio.sr,
                    start_bpm=hyperparameters.beat_tracking.start_bpm,
                    tightness=hyperparameters.beat_tracking.tightness,
                )
                # Time signature inference using simple beat-strength proxy:
                if grid.beat_times_s.size > 4:
                    diffs = np.diff(grid.beat_times_s)
                    strengths = 1.0 / (diffs + 1e-6)
                else:
                    strengths = np.zeros(0)
                ts = infer_time_signature(
                    strengths,
                    min_confidence=hyperparameters.beat_tracking.ts_min_confidence,
                )
                tempo_repo.upsert(
                    job_id=job_id, tempo_bpm=grid.tempo_bpm,
                    beat_times_s_json=json.dumps(grid.beat_times_s.tolist()),
                    downbeat_times_s_json=json.dumps(grid.downbeat_times_s.tolist()),
                    ts_numerator=ts.numerator, ts_denominator=ts.denominator,
                    ts_confidence=ts.confidence, ts_assumed=ts.assumed,
                )
                summary["tempo_bpm"] = grid.tempo_bpm
                summary["ts"] = f"{ts.numerator}/{ts.denominator}"
            s.commit()

            with emit("visualizations") as summary:
                # C3: render chromagram + waveform PNGs into derived/{job_id}
                # TODO: cache chroma if measured to matter
                chroma = compute_chroma_with_hpss(
                    samples_for_pitch, sr=audio.sr,
                    hpss_margin=hyperparameters.key_detection.hpss_margin,
                )
                chroma_path = artifacts.path_for(f"derived/{job_id}/chromagram.png")
                chroma_path.parent.mkdir(parents=True, exist_ok=True)
                fig = render_chromagram_figure(
                    chroma.astype(np.float64), sr=audio.sr, hop_length=512,
                )
                fig.savefig(chroma_path, dpi=120)
                plt.close(fig)

                onsets_s = np.array([n.start_s for n in raw_t.notes])
                wave_path = artifacts.path_for(f"derived/{job_id}/waveform.png")
                fig2 = render_waveform_figure(
                    samples_for_pitch.astype(np.float64), sr=audio.sr, onsets_s=onsets_s,
                )
                fig2.savefig(wave_path, dpi=120)
                plt.close(fig2)
                summary["chromagram_png"] = str(chroma_path)
                summary["waveform_png"] = str(wave_path)
            s.commit()

            with emit("post_processing") as summary:
                cleaned = apply_post_processing(
                    raw_t.notes, params=hyperparameters.post_processing,
                    beats=grid.beat_times_s,
                )
                summary["cleaned_note_count"] = len(cleaned)
            s.commit()

            with emit("tab_assignment") as summary:
                tab_result = assign_tab(
                    cleaned, tuning=get_preset(job.requested_tuning),
                    weights=hyperparameters.tab_assignment.weights,
                    max_fret=hyperparameters.tab_assignment.max_fret,
                )
                note_rows: list[dict] = []  # type: ignore[type-arg]
                for tn in tab_result.tabbed_notes:
                    note_rows.append({
                        "start_s": tn.note.start_s, "end_s": tn.note.end_s,
                        "pitch": tn.note.pitch, "velocity": tn.note.velocity,
                        "confidence": tn.note.confidence,
                        "string": tn.position.string, "fret": tn.position.fret,
                        "cost_breakdown_json": json.dumps(tn.cost_breakdown),
                        "dropped_reason": None,
                    })
                for dropped_note, reason in tab_result.notes_dropped:
                    note_rows.append({
                        "start_s": dropped_note.start_s, "end_s": dropped_note.end_s,
                        "pitch": dropped_note.pitch, "velocity": dropped_note.velocity,
                        "confidence": dropped_note.confidence,
                        "string": None, "fret": None, "cost_breakdown_json": None,
                        "dropped_reason": reason,
                    })
                note_repo.bulk_insert(job_id, note_rows)
                summary["assigned"] = len(tab_result.tabbed_notes)
                summary["dropped"] = len(tab_result.notes_dropped)
            s.commit()

            with emit("midi_synth") as summary:
                # C2: produce synthesized.wav + source.wav under derived/{job_id}
                synth_wav = artifacts.path_for(f"derived/{job_id}/synthesized.wav")
                synthesize_midi_to_wav(
                    raw_t.post_midi_path, synth_wav, sr=audio.sr,
                    backend=SynthBackend.SINE,
                )
                src_copy = artifacts.path_for(f"derived/{job_id}/source.wav")
                src_copy.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(audio.source.path, src_copy)
                summary["synthesized_wav"] = str(synth_wav)
                summary["source_wav"] = str(src_copy)
            s.commit()

            job.status = "succeeded"
            job.finished_at = datetime.now(UTC)
            s.commit()
        except Exception as e:
            _log.exception("pipeline_failed", extra={"job_id": job_id})
            job_repo.mark_failed(
                job_id, type(e).__name__, str(e), traceback.format_exc(),
            )
            s.commit()
            raise
