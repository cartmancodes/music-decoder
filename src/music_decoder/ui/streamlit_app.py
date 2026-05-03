from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from music_decoder.artifacts.filesystem import FilesystemArtifactStore
from music_decoder.config.hyperparameters import load_hyperparameters
from music_decoder.config.runtime import load_runtime_config
from music_decoder.persistence.session import build_engine, run_migrations


@st.cache_resource
def app_context() -> dict[str, object]:
    runtime_path = Path(os.environ.get("MUSIC_DECODER_RUNTIME_YAML", "config/runtime.yaml"))
    hp_path = Path(os.environ.get("MUSIC_DECODER_HP_YAML", "config/hyperparameters.yaml"))
    runtime = load_runtime_config(runtime_path)
    hp = load_hyperparameters(hp_path)
    engine = build_engine(Path(os.path.expanduser(str(runtime.db_path))))
    run_migrations(engine)
    artifacts = FilesystemArtifactStore(
        root=Path(os.path.expanduser(str(runtime.artifact_dir))),
    )
    return {"runtime": runtime, "hp": hp, "engine": engine, "artifacts": artifacts}


def main() -> None:
    st.set_page_config(page_title="Music Decoder", layout="wide")
    st.title("Music Decoder")
    st.markdown(
        "Local audio analysis tool. Upload an audio file to extract key, tempo, "
        "and a guitar tablature with explicit string + fret positions."
    )
    st.markdown("Use the sidebar to navigate to **Upload**, **Job**, or **Results**.")


if __name__ == "__main__":
    main()
