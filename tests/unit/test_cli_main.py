

from music_decoder.cli.main import build_parser, run_worker_only


def test_build_parser_has_expected_subcommands():
    p = build_parser()
    assert "worker" in p.format_help()
    assert "ui" in p.format_help()


def test_run_worker_only_starts_and_exits(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSIC_DECODER_DB_PATH", str(tmp_path / "app.sqlite3"))
    monkeypatch.setenv("MUSIC_DECODER_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("MUSIC_DECODER_LOG_LEVEL", "WARNING")
    # We test the assembly path without actually entering the polling loop.
    from music_decoder.cli import main
    called = {"runs": 0}

    class FakeDaemon:
        def __init__(self, *a, **kw): pass
        def install_signal_handlers(self): pass
        def run(self): called["runs"] += 1

    monkeypatch.setattr(main, "WorkerDaemon", FakeDaemon)
    run_worker_only(runtime_yaml="config/runtime.yaml",
                    hp_yaml="config/hyperparameters.yaml")
    assert called["runs"] == 1
