from __future__ import annotations

import hashlib
import json as _json
from typing import Any, Literal

from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from music_decoder.artifacts.base import ArtifactStore
from music_decoder.persistence.models import (
    AccuracyReport,
    Job,
    JobProgress,
    KeyEstimate,
    Note,
    TabReference,
    TempoEstimate,
)
from music_decoder.persistence.repositories import JobRepo, TabReferenceRepo, UploadRepo


def enqueue_upload(
    *,
    engine: Engine,
    artifacts: ArtifactStore,
    original_filename: str,
    mime_type: str,
    content: bytes,
    declared_kind: Literal["solo_guitar", "full_mix"],
    transcription_model: Literal["basic-pitch", "crepe"],
    requested_tuning: str,
    requested_quality: Literal["standard", "high"],
    use_demucs: bool,
    hyperparameter_set: str,
) -> int:
    sha = hashlib.sha256(content).hexdigest()
    with Session(engine) as s:
        existing = UploadRepo(s).find_by_sha256(sha)
        if existing is not None:
            upload_id = existing.id
        else:
            upload = UploadRepo(s).create(
                sha256=sha, original_filename=original_filename, mime_type=mime_type,
                duration_s=None, sample_rate_hz=None, declared_kind=declared_kind,
                artifact_path="placeholder",
            )
            s.flush()
            ext = original_filename.rsplit(".", 1)[-1].lower() or "wav"
            key = f"uploads/{upload.id}/source.{ext}"
            artifacts.put(key, content)
            upload.artifact_path = key
            upload_id = upload.id

        job = JobRepo(s).enqueue(
            upload_id=upload_id, transcription_model=transcription_model,
            requested_tuning=requested_tuning, requested_quality=requested_quality,
            use_demucs=use_demucs, hyperparameter_set=hyperparameter_set,
        )
        s.commit()
        return int(job.id)


def record_tab_reference(
    engine: Engine,
    job_id: int,
    raw_input: str,
    predicted_tabs: list[dict[str, Any]],
) -> tuple[Any, float]:
    """Fetch/parse a user-pasted tab, compute similarity, persist a TabReference row.

    Returns (FetchedTab, similarity_score).
    """
    from music_decoder.pipeline.contracts import TabbedNote, TabPosition, TranscribedNote
    from music_decoder.tab_reference.alignment import similarity_to_prediction
    from music_decoder.tab_reference.base import RawTabInput
    from music_decoder.tab_reference.parser import parse_ascii_tab
    from music_decoder.tab_reference.user_paste import UserPasteProvider

    fetched = UserPasteProvider().fetch(RawTabInput(text=raw_input))
    ref_positions = parse_ascii_tab(fetched.raw_text)

    tabbed: list[TabbedNote] = []
    for r in predicted_tabs:
        if r.get("string") is None or r.get("fret") is None:
            continue
        tabbed.append(TabbedNote(
            note=TranscribedNote(
                start_s=float(r["start_s"]),
                end_s=float(r["end_s"]),
                pitch=int(r["pitch"]),
                velocity=int(r["velocity"]),
                confidence=float(r["confidence"]),
            ),
            position=TabPosition(
                string=int(r["string"]),
                fret=int(r["fret"]),
            ),
            cost_breakdown={},
        ))

    sim = similarity_to_prediction(tabbed, ref_positions)

    with Session(engine) as s:
        TabReferenceRepo(s).create(
            job_id=job_id,
            source=fetched.source,
            song_acoustid=None,
            raw_text=fetched.raw_text,
            similarity_to_prediction=sim,
            disagreement_spans_json=None,
        )
        s.commit()

    return fetched, sim


def job_status(engine: Engine, job_id: int) -> dict[str, object] | None:
    with Session(engine) as s:
        job = s.get(Job, job_id)
        if job is None:
            return None
        progress = list(s.execute(
            select(JobProgress).where(JobProgress.job_id == job_id)
            .order_by(JobProgress.started_at)
        ).scalars())
        return {
            "id": job.id,
            "status": job.status,
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "finished_at": job.finished_at.isoformat() if job.finished_at else None,
            "error_class": job.error_class,
            "error_message": job.error_message,
            "progress": [
                {
                    "stage": p.stage,
                    "started_at": p.started_at.isoformat() if p.started_at else None,
                    "ended_at": p.ended_at.isoformat() if p.ended_at else None,
                    "success": p.success, "error": p.error,
                    "summary": p.summary_json,
                }
                for p in progress
            ],
        }


def load_results(engine: Engine, job_id: int) -> dict[str, object] | None:
    with Session(engine) as s:
        job = s.get(Job, job_id)
        if job is None:
            return None
        keys = list(s.execute(select(KeyEstimate).where(KeyEstimate.job_id == job_id)
                              .order_by(KeyEstimate.scope, KeyEstimate.profile,
                                        KeyEstimate.rank)).scalars())
        tempo = s.get(TempoEstimate, job_id)
        notes = list(s.execute(select(Note).where(Note.job_id == job_id)
                                .order_by(Note.start_s)).scalars())
        refs = list(s.execute(select(TabReference).where(TabReference.job_id == job_id)
                                .order_by(TabReference.created_at)).scalars())
        report = s.get(AccuracyReport, job_id)
        from music_decoder.persistence.models import ChordSegment as ChordRow
        chord_rows = list(s.execute(
            select(ChordRow).where(ChordRow.job_id == job_id).order_by(ChordRow.start_s)
        ).scalars())
        return {
            "job": {
                "id": job.id, "status": job.status,
                "transcription_model": job.transcription_model,
                "requested_tuning": job.requested_tuning,
                "requested_quality": job.requested_quality,
                "hyperparameter_set": job.hyperparameter_set,
            },
            "keys": [
                {"scope": k.scope, "profile": k.profile, "rank": k.rank,
                 "tonic": k.tonic, "mode": k.mode, "correlation": float(k.correlation),
                 "margin": float(k.margin),
                 "window_start_s": k.window_start_s, "window_end_s": k.window_end_s}
                for k in keys
            ],
            "tempo": ({
                "tempo_bpm": float(tempo.tempo_bpm),
                "beat_times_s": _json.loads(tempo.beat_times_s_json),
                "downbeat_times_s": _json.loads(tempo.downbeat_times_s_json),
                "ts_numerator": tempo.ts_numerator,
                "ts_denominator": tempo.ts_denominator,
                "ts_confidence": float(tempo.ts_confidence),
                "ts_assumed": bool(tempo.ts_assumed),
            } if tempo else None),
            "notes": [
                {"start_s": float(n.start_s), "end_s": float(n.end_s),
                 "pitch": int(n.pitch), "velocity": int(n.velocity),
                 "confidence": float(n.confidence),
                 "string": n.string, "fret": n.fret,
                 "dropped_reason": n.dropped_reason}
                for n in notes
            ],
            "tab_references": [
                {"source": r.source, "raw_text": r.raw_text,
                 "similarity_to_prediction": (
                    float(r.similarity_to_prediction)
                    if r.similarity_to_prediction is not None else None),
                 "disagreement_spans": (
                    _json.loads(r.disagreement_spans_json)
                    if r.disagreement_spans_json else [])}
                for r in refs
            ],
            "accuracy_report": ({
                "fixture_name": report.fixture_name,
                "note_f_measure": report.note_f_measure,
                "key_mirex_score": report.key_mirex_score,
                "tab_string_accuracy": report.tab_string_accuracy,
                "full_metrics": _json.loads(report.full_metrics_json),
            } if report else None),
            "chord_segments": [
                {"start_s": float(c.start_s), "end_s": float(c.end_s),
                 "root": c.root, "quality": c.quality,
                 "confidence": float(c.confidence)}
                for c in chord_rows
            ],
        }
