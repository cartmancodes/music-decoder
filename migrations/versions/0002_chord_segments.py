"""chord_segments table

Revision ID: 0002_chord_segments
Revises: 0001_init
Create Date: 2026-05-05

"""
from alembic import op
import sqlalchemy as sa


revision = "0002_chord_segments"
down_revision = "0001_init"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "chord_segments",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("job_id", sa.Integer(),
                  sa.ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("start_s", sa.Float(), nullable=False),
        sa.Column("end_s", sa.Float(), nullable=False),
        sa.Column("root", sa.String(), nullable=False),
        sa.Column("quality", sa.String(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
    )
    op.create_index("chord_segments_job_idx", "chord_segments", ["job_id", "start_s"])


def downgrade() -> None:
    op.drop_index("chord_segments_job_idx", table_name="chord_segments")
    op.drop_table("chord_segments")
