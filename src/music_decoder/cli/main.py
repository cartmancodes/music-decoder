from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from sqlalchemy.engine import Engine

from music_decoder.artifacts.base import ArtifactStore
from music_decoder.artifacts.filesystem import FilesystemArtifactStore
from music_decoder.cli.health import check_ffmpeg, ensure_data_dirs
from music_decoder.cli.models_download import warm_models
from music_decoder.config.hyperparameters import HyperparameterSet, load_hyperparameters
from music_decoder.config.runtime import RuntimeConfig, load_runtime_config
from music_decoder.logging_setup import configure_logging, get_logger
from music_decoder.persistence.session import build_engine, run_migrations
from music_decoder.worker.daemon import WorkerDaemon

_log = get_logger("cli")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="music-decoder")
    sub = p.add_subparsers(dest="cmd", required=False)

    sub.add_parser("ui", help="Run the Streamlit UI only.")
    sub.add_parser("worker", help="Run the worker daemon only.")
    sub.add_parser("doctor", help="Verify ffmpeg, data dirs, and models.")
    sub.add_parser("download-models",
                   help="Download all model weights into the user cache.")

    fi = sub.add_parser(
        "fixture-init",
        help="Bootstrap a manual evaluation fixture from an audio file.",
    )
    fi.add_argument("audio", help="path to the audio file (.wav, .mp3, .flac)")
    fi.add_argument("--name", required=True,
                    help="fixture name (no extension); becomes <name>.json")
    fi.add_argument("--tuning", default="EADGBE",
                    help="tuning preset (default: EADGBE)")
    fi.add_argument("--output-dir", default="tests/fixtures/manual",
                    help="directory to write the JSON")
    fi.add_argument("--force", action="store_true",
                    help="overwrite if the JSON exists")

    fv = sub.add_parser(
        "fixture-validate",
        help="Validate a manual evaluation fixture against the schema.",
    )
    fv.add_argument("fixture", help="path to the fixture JSON")

    p.add_argument("--runtime-yaml", default="config/runtime.yaml")
    p.add_argument("--hp-yaml", default="config/hyperparameters.yaml")
    return p


def _bootstrap(
    runtime_yaml: str, hp_yaml: str
) -> tuple[RuntimeConfig, HyperparameterSet, Engine, ArtifactStore]:
    runtime = load_runtime_config(Path(runtime_yaml))
    hp = load_hyperparameters(Path(hp_yaml))
    configure_logging(runtime.log_level)
    db_path = Path(os.path.expanduser(str(runtime.db_path)))
    artifact_dir = Path(os.path.expanduser(str(runtime.artifact_dir)))
    model_cache = Path(os.path.expanduser(str(
        runtime.model_cache_dir or "~/Library/Caches/music-decoder/models"
    )))
    ensure_data_dirs(db_path=db_path, artifact_dir=artifact_dir, model_cache=model_cache)
    engine = build_engine(db_path)
    run_migrations(engine)
    artifacts = FilesystemArtifactStore(root=artifact_dir)
    return runtime, hp, engine, artifacts


def run_worker_only(runtime_yaml: str, hp_yaml: str) -> None:
    _runtime, hp, engine, artifacts = _bootstrap(runtime_yaml, hp_yaml)
    daemon = WorkerDaemon(engine=engine, artifacts=artifacts, hyperparameters=hp)
    daemon.install_signal_handlers()
    daemon.run()


def run_ui_only(runtime_yaml: str, hp_yaml: str) -> int:
    """Launch Streamlit by re-execing through `streamlit run`."""
    entry = Path(__file__).parents[1] / "ui" / "streamlit_app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(entry),
           "--server.headless", "false",
           "--browser.gatherUsageStats", "false"]
    env = os.environ.copy()
    env["MUSIC_DECODER_RUNTIME_YAML"] = runtime_yaml
    env["MUSIC_DECODER_HP_YAML"] = hp_yaml
    return subprocess.call(cmd, env=env)


def run_doctor(runtime_yaml: str, hp_yaml: str) -> None:
    check_ffmpeg()
    runtime, hp, _engine, _artifacts = _bootstrap(runtime_yaml, hp_yaml)
    print("ffmpeg ........ ok")
    print(f"db ............ {runtime.db_path}")
    print(f"artifacts ..... {runtime.artifact_dir}")
    print(f"hyperparameter set: {hp.id}")


def run_default(runtime_yaml: str, hp_yaml: str) -> int:
    """Default command: spawn worker subprocess + Streamlit in this process."""
    check_ffmpeg()
    worker_proc = subprocess.Popen(
        [sys.executable, "-m", "music_decoder.cli.main", "worker",
         "--runtime-yaml", runtime_yaml, "--hp-yaml", hp_yaml],
    )
    try:
        return run_ui_only(runtime_yaml, hp_yaml)
    finally:
        worker_proc.terminate()
        try:
            worker_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            worker_proc.kill()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cmd = args.cmd or "default"
    runtime_yaml = args.runtime_yaml
    hp_yaml = args.hp_yaml
    if cmd == "worker":
        run_worker_only(runtime_yaml, hp_yaml)
        return 0
    if cmd == "ui":
        return run_ui_only(runtime_yaml, hp_yaml)
    if cmd == "doctor":
        run_doctor(runtime_yaml, hp_yaml)
        return 0
    if cmd == "download-models":
        warm_models()
        return 0
    if cmd == "fixture-init":
        from music_decoder.cli.fixture import cli_init
        return cli_init(
            audio=args.audio, name=args.name, tuning=args.tuning,
            force=args.force, output_dir=args.output_dir,
        )
    if cmd == "fixture-validate":
        from music_decoder.cli.fixture import cli_validate
        return cli_validate(fixture=args.fixture)
    return run_default(runtime_yaml, hp_yaml)


if __name__ == "__main__":
    sys.exit(main())
