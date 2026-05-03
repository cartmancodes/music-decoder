from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import (
    AccuracyReport,
    Job,
    JobProgress,
    KeyEstimate,
    Note,
    TabReference,
    TempoEstimate,
    Upload,
)


def _now() -> datetime:
    return datetime.now(UTC)


class UploadRepo:
    def __init__(self, session: Session) -> None:
        self.s = session

    def create(
        self,
        *,
        sha256: str,
        original_filename: str,
        mime_type: str,
        duration_s: float | None,
        sample_rate_hz: int | None,
        declared_kind: str,
        artifact_path: str,
    ) -> Upload:
        u = Upload(
            sha256=sha256,
            original_filename=original_filename,
            mime_type=mime_type,
            duration_s=duration_s,
            sample_rate_hz=sample_rate_hz,
            declared_kind=declared_kind,
            artifact_path=artifact_path,
            created_at=_now(),
        )
        self.s.add(u)
        self.s.flush()
        return u

    def find_by_sha256(self, sha256: str) -> Upload | None:
        return self.s.execute(
            select(Upload).where(Upload.sha256 == sha256)
        ).scalar_one_or_none()


class JobRepo:
    def __init__(self, session: Session) -> None:
        self.s = session

    def enqueue(
        self,
        upload_id: int,
        transcription_model: str,
        requested_tuning: str,
        requested_quality: str,
        use_demucs: bool,
        hyperparameter_set: str,
    ) -> Job:
        j = Job(
            upload_id=upload_id,
            status="queued",
            transcription_model=transcription_model,
            requested_tuning=requested_tuning,
            requested_quality=requested_quality,
            use_demucs=use_demucs,
            hyperparameter_set=hyperparameter_set,
            created_at=_now(),
        )
        self.s.add(j)
        self.s.flush()
        return j

    def get(self, job_id: int) -> Job:
        return self.s.execute(select(Job).where(Job.id == job_id)).scalar_one()

    def pop_next_queued(self) -> Job | None:
        j = self.s.execute(
            select(Job).where(Job.status == "queued").order_by(Job.id.asc()).limit(1)
        ).scalar_one_or_none()
        if j is None:
            return None
        j.status = "running"
        j.started_at = _now()
        self.s.flush()
        return j

    def mark_running(self, job_id: int) -> None:
        j = self.get(job_id)
        j.status = "running"
        j.started_at = j.started_at or _now()

    def mark_succeeded(self, job_id: int) -> None:
        j = self.get(job_id)
        j.status = "succeeded"
        j.finished_at = _now()

    def mark_failed(
        self,
        job_id: int,
        error_class: str,
        error_message: str,
        traceback_text: str | None,
    ) -> None:
        j = self.get(job_id)
        j.status = "failed"
        j.finished_at = _now()
        j.error_class = error_class
        j.error_message = error_message
        j.error_traceback = traceback_text


class JobProgressRepo:
    def __init__(self, session: Session) -> None:
        self.s = session

    def record(
        self,
        *,
        job_id: int,
        stage: str,
        started_at: datetime,
        ended_at: datetime | None,
        success: bool | None,
        error: str | None,
        summary_json: str = "{}",
    ) -> JobProgress:
        p = JobProgress(
            job_id=job_id,
            stage=stage,
            started_at=started_at,
            ended_at=ended_at,
            success=success,
            error=error,
            summary_json=summary_json,
        )
        self.s.add(p)
        self.s.flush()
        return p


class NoteRepo:
    def __init__(self, session: Session) -> None:
        self.s = session

    def bulk_insert(self, job_id: int, notes: Iterable[dict]) -> None:  # type: ignore[type-arg]
        self.s.add_all([Note(job_id=job_id, **n) for n in notes])
        self.s.flush()

    def list_for_job(self, job_id: int) -> list[Note]:
        return list(
            self.s.execute(
                select(Note).where(Note.job_id == job_id).order_by(Note.start_s)
            ).scalars()
        )


class KeyEstimateRepo:
    def __init__(self, session: Session) -> None:
        self.s = session

    def bulk_insert(self, job_id: int, rows: Iterable[dict]) -> None:  # type: ignore[type-arg]
        self.s.add_all([KeyEstimate(job_id=job_id, **r) for r in rows])
        self.s.flush()


class TempoEstimateRepo:
    def __init__(self, session: Session) -> None:
        self.s = session

    def upsert(
        self,
        *,
        job_id: int,
        tempo_bpm: float,
        beat_times_s_json: str,
        downbeat_times_s_json: str,
        ts_numerator: int,
        ts_denominator: int,
        ts_confidence: float,
        ts_assumed: bool,
    ) -> TempoEstimate:
        existing = self.s.get(TempoEstimate, job_id)
        if existing:
            existing.tempo_bpm = tempo_bpm
            existing.beat_times_s_json = beat_times_s_json
            existing.downbeat_times_s_json = downbeat_times_s_json
            existing.ts_numerator = ts_numerator
            existing.ts_denominator = ts_denominator
            existing.ts_confidence = ts_confidence
            existing.ts_assumed = ts_assumed
            return existing
        t = TempoEstimate(
            job_id=job_id,
            tempo_bpm=tempo_bpm,
            beat_times_s_json=beat_times_s_json,
            downbeat_times_s_json=downbeat_times_s_json,
            ts_numerator=ts_numerator,
            ts_denominator=ts_denominator,
            ts_confidence=ts_confidence,
            ts_assumed=ts_assumed,
        )
        self.s.add(t)
        self.s.flush()
        return t


class TabReferenceRepo:
    def __init__(self, session: Session) -> None:
        self.s = session

    def create(
        self,
        *,
        job_id: int,
        source: str,
        song_acoustid: str | None,
        raw_text: str,
        similarity_to_prediction: float | None,
        disagreement_spans_json: str | None,
    ) -> TabReference:
        r = TabReference(
            job_id=job_id,
            source=source,
            song_acoustid=song_acoustid,
            raw_text=raw_text,
            similarity_to_prediction=similarity_to_prediction,
            disagreement_spans_json=disagreement_spans_json,
            created_at=_now(),
        )
        self.s.add(r)
        self.s.flush()
        return r


class AccuracyReportRepo:
    def __init__(self, session: Session) -> None:
        self.s = session

    def upsert(
        self,
        *,
        job_id: int,
        fixture_name: str | None,
        note_f_measure: float | None,
        onset_f_measure: float | None,
        pitch_class_accuracy: float | None,
        key_mirex_score: float | None,
        tab_string_accuracy: float | None,
        self_confidence_summary_json: str,
        full_metrics_json: str,
    ) -> AccuracyReport:
        existing = self.s.get(AccuracyReport, job_id)
        if existing:
            existing.fixture_name = fixture_name
            existing.note_f_measure = note_f_measure
            existing.onset_f_measure = onset_f_measure
            existing.pitch_class_accuracy = pitch_class_accuracy
            existing.key_mirex_score = key_mirex_score
            existing.tab_string_accuracy = tab_string_accuracy
            existing.self_confidence_summary_json = self_confidence_summary_json
            existing.full_metrics_json = full_metrics_json
            return existing
        r = AccuracyReport(
            job_id=job_id,
            fixture_name=fixture_name,
            note_f_measure=note_f_measure,
            onset_f_measure=onset_f_measure,
            pitch_class_accuracy=pitch_class_accuracy,
            key_mirex_score=key_mirex_score,
            tab_string_accuracy=tab_string_accuracy,
            self_confidence_summary_json=self_confidence_summary_json,
            full_metrics_json=full_metrics_json,
        )
        self.s.add(r)
        self.s.flush()
        return r
