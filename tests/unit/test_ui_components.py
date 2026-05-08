import numpy as np

from music_decoder.tabs.render import (
    render_ascii_tab,
    render_svg_fretboard,
)
from music_decoder.types import (
    TabbedNote,
    TabPosition,
    TranscribedNote,
)
from music_decoder.ui.components.chromagram import render_chromagram_figure
from music_decoder.ui.components.confidence import confidence_color
from music_decoder.ui.components.waveform import render_waveform_figure


def test_confidence_color_thresholds():
    assert confidence_color(0.95, high=0.8, medium=0.5) == "green"
    assert confidence_color(0.65, high=0.8, medium=0.5) == "yellow"
    assert confidence_color(0.30, high=0.8, medium=0.5) == "red"


def _tn(pitch=60, start=0.0, end=0.5, string=4, fret=1, conf=0.9):
    return TabbedNote(
        note=TranscribedNote(start_s=start, end_s=end, pitch=pitch, velocity=80, confidence=conf),
        position=TabPosition(string=string, fret=fret),
        cost_breakdown={},
    )


def test_render_ascii_tab_six_rows_for_eadgbe():
    notes = [_tn(60, 0.0, 0.5, 4, 1), _tn(64, 0.5, 1.0, 5, 0)]
    text = render_ascii_tab(notes, n_strings=6, columns=8)
    rows = text.strip().split("\n")
    assert len(rows) == 6
    assert rows[0].startswith("e|")  # high E
    assert rows[5].startswith("E|")  # low E


def test_render_svg_fretboard_returns_svg():
    notes = [_tn(60, 0.0, 0.5, 4, 1, conf=0.95)]
    svg = render_svg_fretboard(notes, n_strings=6, max_fret=5)
    assert svg.startswith("<svg")
    assert "</svg>" in svg


def test_render_chromagram_figure_returns_figure():
    chroma = np.zeros((12, 50))
    chroma[0] = 1.0
    fig = render_chromagram_figure(chroma, sr=22050, hop_length=512)
    assert hasattr(fig, "axes")


def test_render_waveform_figure_returns_figure():
    samples = np.sin(2 * np.pi * 440 * np.arange(22050) / 22050)
    onsets = np.array([0.1, 0.5, 0.9])
    fig = render_waveform_figure(samples, sr=22050, onsets_s=onsets)
    assert hasattr(fig, "axes")
