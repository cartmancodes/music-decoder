from pathlib import Path

from sqlalchemy import inspect

from music_decoder.persistence.session import build_engine, run_migrations


def test_run_migrations_creates_all_tables(tmp_path: Path):
    db_path = tmp_path / "app.sqlite3"
    engine = build_engine(db_path)
    run_migrations(engine)
    insp = inspect(engine)
    expected = {
        "uploads", "jobs", "job_progress", "notes",
        "key_estimates", "tempo_estimates", "tab_references", "accuracy_reports",
        "alembic_version",
    }
    assert expected.issubset(set(insp.get_table_names()))


def test_engine_enables_wal_and_foreign_keys(tmp_path: Path):
    db_path = tmp_path / "app.sqlite3"
    engine = build_engine(db_path)
    with engine.connect() as conn:
        from sqlalchemy import text

        assert conn.execute(text("PRAGMA journal_mode")).scalar_one().lower() == "wal"
        assert conn.execute(text("PRAGMA foreign_keys")).scalar_one() == 1
