from __future__ import annotations

from typing import Any, cast

import streamlit as st

from music_decoder.artifacts.base import ArtifactStore
from music_decoder.config.hyperparameters import HyperparameterSet
from music_decoder.pipeline.contracts import (
    TabbedNote,
    TabPosition,
    TranscribedNote,
)
from music_decoder.tab_reference.alignment import similarity_to_prediction
from music_decoder.tab_reference.base import RawTabInput
from music_decoder.tab_reference.parser import parse_ascii_tab
from music_decoder.tab_reference.user_paste import UserPasteProvider
from music_decoder.ui.components.confidence import confidence_color
from music_decoder.ui.components.tablature import (
    render_ascii_tab,
    render_svg_fretboard,
)
from music_decoder.ui.services import load_results
from music_decoder.ui.streamlit_app import app_context


def _to_tabbed_notes(rows: list[dict[str, Any]]) -> list[TabbedNote]:
    out = []
    for r in rows:
        if r["string"] is None or r["fret"] is None:
            continue
        out.append(TabbedNote(
            note=TranscribedNote(
                start_s=r["start_s"],
                end_s=r["end_s"],
                pitch=r["pitch"],
                velocity=r["velocity"],
                confidence=r["confidence"],
            ),
            position=TabPosition(
                string=r["string"],
                fret=r["fret"],
            ),
            cost_breakdown={},
        ))
    return out


def render() -> None:
    st.header("Results")
    raw_ctx = app_context()
    engine = raw_ctx["engine"]
    artifacts = cast(ArtifactStore, raw_ctx["artifacts"])
    hp = cast(HyperparameterSet, raw_ctx["hp"])
    qp = st.query_params
    raw_id = qp.get("id")
    if not raw_id:
        st.info("Pass `?id=N` to view results.")
        return
    job_id = int(raw_id)
    from sqlalchemy.engine import Engine
    payload = load_results(cast(Engine, engine), job_id)
    if payload is None:
        st.error("Job not found.")
        return
    job_info = cast(dict[str, Any], payload["job"])
    if job_info["status"] != "succeeded":
        st.warning(f"Job is in state {job_info['status']!r}; results not ready.")
        return
    thresholds: dict[str, float] = hp.ui.confidence_thresholds
    tabs = st.tabs([
        "Summary", "Visualization", "Tablature", "Playback", "Reference (UG)", "Diagnostics",
    ])
    keys_list = cast(list[dict[str, Any]], payload["keys"])
    tempo_info = cast(dict[str, Any] | None, payload["tempo"])
    notes_list = cast(list[dict[str, Any]], payload["notes"])
    refs_list = cast(list[dict[str, Any]], payload["tab_references"])

    with tabs[0]:
        st.subheader("Detected key")
        global_keys = [k for k in keys_list if k["scope"] == "global"]
        for k in global_keys:
            st.write(
                f"**{k['profile']}** rank {k['rank']}: "
                f"{k['tonic']} {k['mode']} (corr={k['correlation']:.3f}, "
                f"margin={k['margin']:.3f})"
            )
        if tempo_info:
            tag = " (assumed)" if tempo_info["ts_assumed"] else ""
            st.subheader("Tempo & meter")
            st.write(
                f"{tempo_info['tempo_bpm']:.1f} BPM, "
                f"{tempo_info['ts_numerator']}/{tempo_info['ts_denominator']}{tag}"
            )
        confs = [
            float(n["confidence"])
            for n in notes_list
            if n["dropped_reason"] is None
        ]
        if confs:
            from statistics import median
            st.metric("Median note confidence", f"{median(confs):.2f}")
        dropped = [n for n in notes_list if n["dropped_reason"]]
        if dropped:
            st.warning(f"{len(dropped)} notes were dropped during tab assignment.")

    with tabs[1]:
        st.info(
            "Visualization figures depend on the source audio; the worker emits "
            "chromagram and waveform PNGs into the artifact store, which this "
            "page loads when present."
        )
        for fn, caption in (("chromagram.png", "Chromagram"),
                            ("waveform.png", "Waveform with onsets")):
            key = f"derived/{job_id}/{fn}"
            if artifacts.exists(key):
                st.image(str(artifacts.path_for(key)), caption=caption)

    with tabs[2]:
        st.subheader("ASCII tablature")
        tabbed = _to_tabbed_notes(notes_list)
        st.code(render_ascii_tab(tabbed, n_strings=6, columns=64))
        st.subheader("Fretboard")
        st.markdown(render_svg_fretboard(tabbed, n_strings=6, max_fret=12),
                    unsafe_allow_html=True)
        st.subheader("Per-note confidence")
        for r in notes_list:
            color = confidence_color(
                float(r["confidence"]),
                high=thresholds["high"],
                medium=thresholds["medium"],
            )
            pos = (
                f"s{r['string']}f{r['fret']}"
                if r["string"] is not None
                else "DROPPED"
            )
            st.write(
                f"- t={r['start_s']:.2f}s pitch={r['pitch']} "
                f"({color}, conf={r['confidence']:.2f}) -> {pos}"
            )

    with tabs[3]:
        st.subheader("Playback")
        for fn, caption in (("synthesized.wav", "Predicted MIDI (synthesized)"),
                            ("source.wav", "Original audio")):
            key = f"derived/{job_id}/{fn}"
            if artifacts.exists(key):
                st.write(caption)
                st.audio(str(artifacts.path_for(key)))

    with tabs[4]:
        st.subheader("Reference tab (Ultimate Guitar paste)")
        st.write(
            "Paste a UG URL or the tab text. We never override our prediction; "
            "we only display the side-by-side disagreement."
        )
        text = st.text_area("URL or tab text", height=200)
        if st.button("Compare"):
            fetched = UserPasteProvider().fetch(RawTabInput(text=text))
            ref_positions = parse_ascii_tab(fetched.raw_text)
            tabbed = _to_tabbed_notes(notes_list)
            sim = similarity_to_prediction(tabbed, ref_positions)
            st.write(f"Similarity to prediction: {sim:.2f}")
            st.code(fetched.raw_text)
        if refs_list:
            st.markdown("**Previously pasted references**")
            for r in refs_list:
                st.write(
                    f"- source={r['source']}, "
                    f"similarity={r['similarity_to_prediction']}"
                )

    with tabs[5]:
        st.subheader("Diagnostics")
        st.json(job_info)
        accuracy = payload["accuracy_report"]
        if accuracy:
            st.subheader("Accuracy report")
            st.json(accuracy)


render()
