"""init schema

Revision ID: 0001_init
Revises:
Create Date: 2026-05-04

"""
from alembic import op
import sqlalchemy as sa


revision = "0001_init"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "uploads",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("sha256", sa.String(), nullable=False),
        sa.Column("original_filename", sa.String(), nullable=False),
        sa.Column("mime_type", sa.String(), nullable=False),
        sa.Column("duration_s", sa.Float(), nullable=True),
        sa.Column("sample_rate_hz", sa.Integer(), nullable=True),
        sa.Column("declared_kind", sa.String(), nullable=False),
        sa.Column("artifact_path", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.CheckConstraint("declared_kind IN ('solo_guitar','full_mix')"),
    )
    op.create_index("ix_uploads_sha256", "uploads", ["sha256"])

    op.create_table(
        "jobs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("upload_id", sa.Integer(),
                  sa.ForeignKey("uploads.id", ondelete="CASCADE"), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("transcription_model", sa.String(), nullable=False),
        sa.Column("requested_tuning", sa.String(), nullable=False),
        sa.Column("requested_quality", sa.String(), nullable=False),
        sa.Column("use_demucs", sa.Boolean(), nullable=False),
        sa.Column("hyperparameter_set", sa.String(), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.Column("error_class", sa.String(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("error_traceback", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.CheckConstraint("status IN ('queued','running','succeeded','failed','cancelled')"),
        sa.CheckConstraint("transcription_model IN ('basic-pitch','crepe')"),
        sa.CheckConstraint("requested_quality IN ('standard','high')"),
    )
    op.create_index("ix_jobs_upload_id", "jobs", ["upload_id"])
    op.create_index("jobs_queued_idx", "jobs", ["status"])
    # Partial index for hot lookups of pending jobs:
    op.execute(
        "CREATE INDEX IF NOT EXISTS jobs_queued_partial_idx "
        "ON jobs(status) WHERE status IN ('queued','running')"
    )

    op.create_table(
        "job_progress",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("job_id", sa.Integer(),
                  sa.ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("stage", sa.String(), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=False),
        sa.Column("ended_at", sa.DateTime(), nullable=True),
        sa.Column("success", sa.Boolean(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("summary_json", sa.Text(), nullable=False, server_default="{}"),
    )
    op.create_index("job_progress_job_idx", "job_progress", ["job_id", "started_at"])

    op.create_table(
        "notes",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("job_id", sa.Integer(),
                  sa.ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("start_s", sa.Float(), nullable=False),
        sa.Column("end_s", sa.Float(), nullable=False),
        sa.Column("pitch", sa.Integer(), nullable=False),
        sa.Column("velocity", sa.Integer(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("string", sa.Integer(), nullable=True),
        sa.Column("fret", sa.Integer(), nullable=True),
        sa.Column("cost_breakdown_json", sa.Text(), nullable=True),
        sa.Column("dropped_reason", sa.String(), nullable=True),
    )
    op.create_index("notes_job_idx", "notes", ["job_id", "start_s"])

    op.create_table(
        "key_estimates",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("job_id", sa.Integer(),
                  sa.ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("scope", sa.String(), nullable=False),
        sa.Column("window_start_s", sa.Float(), nullable=True),
        sa.Column("window_end_s", sa.Float(), nullable=True),
        sa.Column("profile", sa.String(), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("tonic", sa.String(), nullable=False),
        sa.Column("mode", sa.String(), nullable=False),
        sa.Column("correlation", sa.Float(), nullable=False),
        sa.Column("margin", sa.Float(), nullable=False),
        sa.CheckConstraint("scope IN ('global','window')"),
        sa.CheckConstraint("profile IN ('krumhansl_kessler','temperley')"),
        sa.CheckConstraint("mode IN ('major','minor')"),
    )
    op.create_index("key_estimates_job_idx", "key_estimates", ["job_id", "scope"])

    op.create_table(
        "tempo_estimates",
        sa.Column("job_id", sa.Integer(),
                  sa.ForeignKey("jobs.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("tempo_bpm", sa.Float(), nullable=False),
        sa.Column("beat_times_s_json", sa.Text(), nullable=False),
        sa.Column("downbeat_times_s_json", sa.Text(), nullable=False),
        sa.Column("ts_numerator", sa.Integer(), nullable=False),
        sa.Column("ts_denominator", sa.Integer(), nullable=False),
        sa.Column("ts_confidence", sa.Float(), nullable=False),
        sa.Column("ts_assumed", sa.Boolean(), nullable=False),
    )

    op.create_table(
        "tab_references",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("job_id", sa.Integer(),
                  sa.ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("source", sa.String(), nullable=False),
        sa.Column("song_acoustid", sa.String(), nullable=True),
        sa.Column("raw_text", sa.Text(), nullable=False),
        sa.Column("similarity_to_prediction", sa.Float(), nullable=True),
        sa.Column("disagreement_spans_json", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )

    op.create_table(
        "accuracy_reports",
        sa.Column("job_id", sa.Integer(),
                  sa.ForeignKey("jobs.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("fixture_name", sa.String(), nullable=True),
        sa.Column("note_f_measure", sa.Float(), nullable=True),
        sa.Column("onset_f_measure", sa.Float(), nullable=True),
        sa.Column("pitch_class_accuracy", sa.Float(), nullable=True),
        sa.Column("key_mirex_score", sa.Float(), nullable=True),
        sa.Column("tab_string_accuracy", sa.Float(), nullable=True),
        sa.Column("self_confidence_summary_json", sa.Text(), nullable=False),
        sa.Column("full_metrics_json", sa.Text(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("accuracy_reports")
    op.drop_table("tab_references")
    op.drop_table("tempo_estimates")
    op.drop_index("key_estimates_job_idx", table_name="key_estimates")
    op.drop_table("key_estimates")
    op.drop_index("notes_job_idx", table_name="notes")
    op.drop_table("notes")
    op.drop_index("job_progress_job_idx", table_name="job_progress")
    op.drop_table("job_progress")
    op.execute("DROP INDEX IF EXISTS jobs_queued_partial_idx")
    op.drop_index("jobs_queued_idx", table_name="jobs")
    op.drop_index("ix_jobs_upload_id", table_name="jobs")
    op.drop_table("jobs")
    op.drop_index("ix_uploads_sha256", table_name="uploads")
    op.drop_table("uploads")
