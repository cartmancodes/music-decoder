from __future__ import annotations

from typing import Literal, cast

import streamlit as st

from music_decoder.artifacts.base import ArtifactStore
from music_decoder.config.hyperparameters import HyperparameterSet
from music_decoder.tab_assignment.tuning import PRESETS
from music_decoder.ui.services import enqueue_upload
from music_decoder.ui.streamlit_app import app_context


def render() -> None:
    st.header("Upload audio")
    raw_ctx = app_context()
    from sqlalchemy.engine import Engine
    engine = cast(Engine, raw_ctx["engine"])
    artifacts = cast(ArtifactStore, raw_ctx["artifacts"])
    hp = cast(HyperparameterSet, raw_ctx["hp"])

    uploaded = st.file_uploader(
        "Choose an audio file (MP3, WAV, FLAC)",
        type=["mp3", "wav", "flac", "m4a", "ogg"],
    )
    declared_kind = cast(
        Literal["solo_guitar", "full_mix"],
        st.radio("Audio source", options=["solo_guitar", "full_mix"], horizontal=True),
    )
    transcription_model = cast(
        Literal["basic-pitch", "crepe"],
        st.radio(
            "Transcription model",
            options=["basic-pitch", "crepe"], horizontal=True,
            help="basic-pitch handles polyphonic; CREPE is monophonic only.",
        ),
    )
    requested_tuning = st.selectbox("Tuning", options=list(PRESETS.keys()))
    requested_quality = cast(
        Literal["standard", "high"],
        st.radio(
            "Quality", options=["standard", "high"], horizontal=True,
            help="High = 44.1 kHz, slower. Standard = 22.05 kHz, default.",
        ),
    )
    use_demucs = st.checkbox(
        "Run Demucs source separation (full mix only)",
        value=(declared_kind == "full_mix"),
    )
    submitted = st.button("Submit", type="primary", disabled=uploaded is None)
    if submitted and uploaded is not None:
        job_id = enqueue_upload(
            engine=engine, artifacts=artifacts,
            original_filename=uploaded.name,
            mime_type=uploaded.type or "application/octet-stream",
            content=uploaded.getvalue(),
            declared_kind=declared_kind,
            transcription_model=transcription_model,
            requested_tuning=requested_tuning or "EADGBE",
            requested_quality=requested_quality,
            use_demucs=use_demucs,
            hyperparameter_set=hp.id,
        )
        st.success(f"Job #{job_id} queued. Visit the Job page to watch progress.")
        st.markdown(f"[Open job](?id={job_id})")


render()
