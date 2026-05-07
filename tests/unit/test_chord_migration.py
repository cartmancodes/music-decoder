from pathlib import Path

from sqlalchemy import inspect

from music_decoder.persistence.session import build_engine, run_migrations


def test_migration_creates_chord_segments_table(tmp_path: Path) -> None:
    db = tmp_path / "x.sqlite3"
    engine = build_engine(db)
    run_migrations(engine)
    insp = inspect(engine)
    assert "chord_segments" in insp.get_table_names()
    cols = {c["name"] for c in insp.get_columns("chord_segments")}
    assert {"id", "job_id", "start_s", "end_s", "root", "quality", "confidence"}.issubset(cols)
    indexes = [idx["name"] for idx in insp.get_indexes("chord_segments")]
    assert "chord_segments_job_idx" in indexes
