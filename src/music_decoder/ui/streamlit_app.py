"""Single-page Streamlit UI with Analyze / Compose / About tabs."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from music_decoder import analyze, compose
from music_decoder.config.runtime import load_runtime_config
from music_decoder.errors import MusicDecoderError
from music_decoder.tabs.tuning import (
    D_STANDARD,
    DADGAD,
    DROP_C,
    DROP_D,
    EB_HALF_STEP_DOWN,
    STANDARD_EADGBE,
)
from music_decoder.types import ChordSymbol, Scale
from music_decoder.ui.components.chord_progression import render_chord_timeline

_TUNINGS = {
    "EADGBE": STANDARD_EADGBE,
    "Drop D": DROP_D,
    "Eb": EB_HALF_STEP_DOWN,
    "D standard": D_STANDARD,
    "Drop C": DROP_C,
    "DADGAD": DADGAD,
}


def render() -> None:
    st.set_page_config(page_title="Music Decoder", layout="wide")
    st.title("Music Decoder")
    cfg = load_runtime_config()
    out_dir = Path(cfg.composition_out_dir).expanduser()

    tab_analyze, tab_compose, tab_about = st.tabs(["Analyze", "Compose", "About"])

    with tab_analyze:
        _render_analyze(out_dir)
    with tab_compose:
        _render_compose(out_dir)
    with tab_about:
        _render_about()


def _render_analyze(out_dir: Path) -> None:
    st.header("Analyze a song")
    src_kind = st.radio("Source", ("File upload", "YouTube URL"), horizontal=True)
    source: str | Path | None = None
    if src_kind == "File upload":
        f = st.file_uploader("Audio file", type=["mp3", "wav", "flac"])
        if f:
            tmp = out_dir / "uploads" / f.name
            tmp.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_bytes(f.read())
            source = tmp
    else:
        url = st.text_input("YouTube URL")
        if url:
            source = url

    tuning_name = st.selectbox("Tuning", list(_TUNINGS.keys()))
    declared = st.radio("Declared kind", ("full_mix", "solo_guitar"), horizontal=True)
    use_sep = st.checkbox("Use Demucs separation", value=True)

    if st.button("Analyze") and source is not None:
        bar = st.progress(0.0, text="starting…")

        def cb(stage: str, frac: float) -> None:
            bar.progress(min(max(frac, 0.0), 1.0), text=f"{stage} {int(frac * 100)}%")

        try:
            res = analyze(
                source,
                declared_kind=declared,  # type: ignore[arg-type]
                tuning=_TUNINGS[tuning_name],
                use_separation=use_sep,
                progress=cb,
            )
        except MusicDecoderError as e:
            st.error(str(e))
            return
        st.success("Done.")
        st.write(
            f"**Key:** {res.key.tonic} {res.key.mode}  "
            f"**Tempo:** {res.tempo_bpm:.1f} BPM  "
            f"**Duration:** {res.duration_s:.1f}s"
        )
        st.subheader("Chord progression")
        render_chord_timeline(res.chord_progression)
        st.subheader("Tablature")
        from music_decoder.tabs.render import render_ascii_tab

        st.code(render_ascii_tab(list(res.tab), num_strings=6))
        st.subheader("Original audio")
        st.audio(str(res.audio_path))


def _render_compose(out_dir: Path) -> None:
    st.header("Compose")
    tonic = st.selectbox(
        "Tonic",
        [
            "C",
            "C#",
            "D",
            "D#",
            "E",
            "F",
            "F#",
            "G",
            "G#",
            "A",
            "A#",
            "B",
        ],
    )
    mode = st.selectbox("Mode", ["major", "minor"])
    progression = st.text_input("Progression (space-separated)", value="Cmaj7 Am7 Dm7 G7")
    style = st.selectbox("Style", ["fingerstyle", "strum", "arpeggio"])
    tempo = st.slider("Tempo (BPM)", 40, 220, 100)
    bars = st.slider("Bars per chord", 1, 4, 1)
    seed_input = st.text_input("Seed (optional)", value="")
    seed = int(seed_input) if seed_input.strip() else None
    tuning_name = st.selectbox("Tuning", list(_TUNINGS.keys()), key="compose_tuning")

    if st.button("Compose"):
        try:
            chords = [ChordSymbol.parse(c) for c in progression.split()]
            res = compose(
                scale=Scale(tonic=tonic, mode=mode),  # type: ignore[arg-type]
                progression=chords,
                bars_per_chord=bars,
                tempo_bpm=float(tempo),
                style=style,
                tuning=_TUNINGS[tuning_name],
                seed=seed,
                out_dir=out_dir,
            )
        except (MusicDecoderError, ValueError) as e:
            st.error(str(e))
            return
        st.success("Done.")
        st.audio(str(res.wav_path))
        st.code(res.ascii_tab)
        st.download_button(
            "Download MIDI", data=res.midi_path.read_bytes(), file_name=res.midi_path.name
        )
        st.download_button(
            "Download WAV", data=res.wav_path.read_bytes(), file_name=res.wav_path.name
        )


def _render_about() -> None:
    import music_decoder
    from music_decoder.config.hyperparameters import load_hyperparameters

    st.write(f"Version: `{music_decoder.__version__}`")
    st.write(f"Hyperparameter set: `{load_hyperparameters().id}`")


if __name__ == "__main__":
    render()
