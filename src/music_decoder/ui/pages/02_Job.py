from __future__ import annotations

import time
from typing import Any, cast

import streamlit as st

from music_decoder.ui.services import job_status
from music_decoder.ui.streamlit_app import app_context

_TERMINAL = ("succeeded", "failed", "cancelled")


def render() -> None:
    st.header("Job status")
    raw_ctx = app_context()
    from sqlalchemy.engine import Engine
    engine = cast(Engine, raw_ctx["engine"])
    qp = st.query_params
    raw_id = qp.get("id")
    if not raw_id:
        st.info("Pass `?id=N` to view a job.")
        return
    try:
        job_id = int(raw_id)
    except (TypeError, ValueError):
        st.error("Invalid job id.")
        return
    status = job_status(engine, job_id)
    if status is None:
        st.error(f"Job #{job_id} not found.")
        return
    st.subheader(f"Job #{job_id} — {status['status']}")
    st.write(f"started_at: {status['started_at']}")
    st.write(f"finished_at: {status['finished_at']}")
    if status["error_class"]:
        st.error(f"{status['error_class']}: {status['error_message']}")
    progress = cast(list[dict[str, Any]], status["progress"])
    if progress:
        st.markdown("**Stage progress**")
        for p in progress:
            ok = "ok" if p["success"] else ("running" if p["success"] is None else "fail")
            st.write(f"- `{p['stage']}` [{ok}] @ {p['started_at']}")
    if status["status"] == "succeeded":
        st.markdown(f"[View results](?id={job_id})")
    if status["status"] not in _TERMINAL:
        time.sleep(1.0)
        st.rerun()


render()
