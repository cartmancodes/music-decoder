from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Upload(Base):
    __tablename__ = "uploads"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    sha256: Mapped[str] = mapped_column(String, nullable=False, index=True)
    original_filename: Mapped[str] = mapped_column(String, nullable=False)
    mime_type: Mapped[str] = mapped_column(String, nullable=False)
    duration_s: Mapped[float | None] = mapped_column(Float)
    sample_rate_hz: Mapped[int | None] = mapped_column(Integer)
    declared_kind: Mapped[str] = mapped_column(
        String, CheckConstraint("declared_kind IN ('solo_guitar','full_mix')"),
        nullable=False,
    )
    artifact_path: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    jobs: Mapped[list[Job]] = relationship(
        back_populates="upload", cascade="all, delete-orphan", passive_deletes=True,
    )


class Job(Base):
    __tablename__ = "jobs"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    upload_id: Mapped[int] = mapped_column(
        ForeignKey("uploads.id", ondelete="CASCADE"), nullable=False, index=True,
    )
    status: Mapped[str] = mapped_column(
        String, CheckConstraint(
            "status IN ('queued','running','succeeded','failed','cancelled')"
        ), nullable=False,
    )
    transcription_model: Mapped[str] = mapped_column(
        String, CheckConstraint("transcription_model IN ('basic-pitch','crepe')"),
        nullable=False,
    )
    requested_tuning: Mapped[str] = mapped_column(String, nullable=False)
    requested_quality: Mapped[str] = mapped_column(
        String, CheckConstraint("requested_quality IN ('standard','high')"),
        nullable=False,
    )
    use_demucs: Mapped[bool] = mapped_column(Boolean, nullable=False)
    hyperparameter_set: Mapped[str] = mapped_column(String, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime)
    error_class: Mapped[str | None] = mapped_column(String)
    error_message: Mapped[str | None] = mapped_column(Text)
    error_traceback: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    upload: Mapped[Upload] = relationship(back_populates="jobs")
    progress: Mapped[list[JobProgress]] = relationship(
        back_populates="job", cascade="all, delete-orphan", passive_deletes=True,
    )
    notes: Mapped[list[Note]] = relationship(
        back_populates="job", cascade="all, delete-orphan", passive_deletes=True,
    )
    chord_segments: Mapped[list[ChordSegment]] = relationship(
        back_populates="job", cascade="all, delete-orphan", passive_deletes=True,
    )

    __table_args__ = (Index("jobs_queued_idx", "status"),)


class JobProgress(Base):
    __tablename__ = "job_progress"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    job_id: Mapped[int] = mapped_column(
        ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False,
    )
    stage: Mapped[str] = mapped_column(String, nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime)
    success: Mapped[bool | None] = mapped_column(Boolean)
    error: Mapped[str | None] = mapped_column(Text)
    summary_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")

    job: Mapped[Job] = relationship(back_populates="progress")
    __table_args__ = (Index("job_progress_job_idx", "job_id", "started_at"),)


class Note(Base):
    __tablename__ = "notes"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    job_id: Mapped[int] = mapped_column(
        ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False,
    )
    start_s: Mapped[float] = mapped_column(Float, nullable=False)
    end_s: Mapped[float] = mapped_column(Float, nullable=False)
    pitch: Mapped[int] = mapped_column(Integer, nullable=False)
    velocity: Mapped[int] = mapped_column(Integer, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    string: Mapped[int | None] = mapped_column(Integer)
    fret: Mapped[int | None] = mapped_column(Integer)
    cost_breakdown_json: Mapped[str | None] = mapped_column(Text)
    dropped_reason: Mapped[str | None] = mapped_column(String)

    job: Mapped[Job] = relationship(back_populates="notes")
    __table_args__ = (Index("notes_job_idx", "job_id", "start_s"),)


class KeyEstimate(Base):
    __tablename__ = "key_estimates"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    job_id: Mapped[int] = mapped_column(
        ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False,
    )
    scope: Mapped[str] = mapped_column(
        String, CheckConstraint("scope IN ('global','window')"), nullable=False,
    )
    window_start_s: Mapped[float | None] = mapped_column(Float)
    window_end_s: Mapped[float | None] = mapped_column(Float)
    profile: Mapped[str] = mapped_column(
        String, CheckConstraint("profile IN ('krumhansl_kessler','temperley')"),
        nullable=False,
    )
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    tonic: Mapped[str] = mapped_column(String, nullable=False)
    mode: Mapped[str] = mapped_column(
        String, CheckConstraint("mode IN ('major','minor')"), nullable=False,
    )
    correlation: Mapped[float] = mapped_column(Float, nullable=False)
    margin: Mapped[float] = mapped_column(Float, nullable=False)

    __table_args__ = (Index("key_estimates_job_idx", "job_id", "scope"),)


class TempoEstimate(Base):
    __tablename__ = "tempo_estimates"
    job_id: Mapped[int] = mapped_column(
        ForeignKey("jobs.id", ondelete="CASCADE"), primary_key=True,
    )
    tempo_bpm: Mapped[float] = mapped_column(Float, nullable=False)
    beat_times_s_json: Mapped[str] = mapped_column(Text, nullable=False)
    downbeat_times_s_json: Mapped[str] = mapped_column(Text, nullable=False)
    ts_numerator: Mapped[int] = mapped_column(Integer, nullable=False)
    ts_denominator: Mapped[int] = mapped_column(Integer, nullable=False)
    ts_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    ts_assumed: Mapped[bool] = mapped_column(Boolean, nullable=False)


class TabReference(Base):
    __tablename__ = "tab_references"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    job_id: Mapped[int] = mapped_column(
        ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False,
    )
    source: Mapped[str] = mapped_column(String, nullable=False)
    song_acoustid: Mapped[str | None] = mapped_column(String)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    similarity_to_prediction: Mapped[float | None] = mapped_column(Float)
    disagreement_spans_json: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class AccuracyReport(Base):
    __tablename__ = "accuracy_reports"
    job_id: Mapped[int] = mapped_column(
        ForeignKey("jobs.id", ondelete="CASCADE"), primary_key=True,
    )
    fixture_name: Mapped[str | None] = mapped_column(String)
    note_f_measure: Mapped[float | None] = mapped_column(Float)
    onset_f_measure: Mapped[float | None] = mapped_column(Float)
    pitch_class_accuracy: Mapped[float | None] = mapped_column(Float)
    key_mirex_score: Mapped[float | None] = mapped_column(Float)
    tab_string_accuracy: Mapped[float | None] = mapped_column(Float)
    self_confidence_summary_json: Mapped[str] = mapped_column(Text, nullable=False)
    full_metrics_json: Mapped[str] = mapped_column(Text, nullable=False)


class ChordSegment(Base):
    __tablename__ = "chord_segments"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    job_id: Mapped[int] = mapped_column(
        ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False,
    )
    start_s: Mapped[float] = mapped_column(Float, nullable=False)
    end_s: Mapped[float] = mapped_column(Float, nullable=False)
    root: Mapped[str] = mapped_column(String, nullable=False)
    quality: Mapped[str] = mapped_column(String, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)

    job: Mapped[Job] = relationship(back_populates="chord_segments")
    __table_args__ = (Index("chord_segments_job_idx", "job_id", "start_s"),)
