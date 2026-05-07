"""widen transcription_model check constraint to include highres-guitar

Revision ID: 0003_highres_guitar_transcription_model
Revises: 0002_chord_segments
Create Date: 2026-05-07

The original check constraint in 0001_init was unnamed; SQLite check constraints
can only be modified by recreating the table.  We use batch_alter_table with
``recreate='always'`` which generates the right copy-rename-drop sequence.
"""
import sqlalchemy as sa
from alembic import op


revision = "0003_highres_guitar_transcription_model"
down_revision = "0002_chord_segments"
branch_labels = None
depends_on = None


def _job_columns() -> list[sa.Column]:
    """Job columns matching ``persistence.models.Job`` after Phase B-2."""
    return [
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
    ]


def _new_constraints() -> list[sa.CheckConstraint]:
    return [
        sa.CheckConstraint(
            "status IN ('queued','running','succeeded','failed','cancelled')"
        ),
        sa.CheckConstraint(
            "transcription_model IN ('basic-pitch','crepe','highres-guitar')"
        ),
        sa.CheckConstraint("requested_quality IN ('standard','high')"),
    ]


def _old_constraints() -> list[sa.CheckConstraint]:
    return [
        sa.CheckConstraint(
            "status IN ('queued','running','succeeded','failed','cancelled')"
        ),
        sa.CheckConstraint(
            "transcription_model IN ('basic-pitch','crepe')"
        ),
        sa.CheckConstraint("requested_quality IN ('standard','high')"),
    ]


def upgrade() -> None:
    with op.batch_alter_table(
        "jobs",
        recreate="always",
        copy_from=sa.Table(
            "jobs",
            sa.MetaData(),
            *_job_columns(),
            *_old_constraints(),
        ),
    ) as batch_op:
        # Replace the table-level check constraints with the widened set.
        for c in _new_constraints():
            batch_op.create_check_constraint(c.name or "_anon", c.sqltext)


def downgrade() -> None:
    with op.batch_alter_table(
        "jobs",
        recreate="always",
        copy_from=sa.Table(
            "jobs",
            sa.MetaData(),
            *_job_columns(),
            *_new_constraints(),
        ),
    ) as batch_op:
        for c in _old_constraints():
            batch_op.create_check_constraint(c.name or "_anon", c.sqltext)
