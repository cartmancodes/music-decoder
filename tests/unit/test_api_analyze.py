"""Tests for the public ``analyze()`` entry point."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np

from music_decoder.api import analyze
from music_decoder.tabs.tuning import STANDARD_EADGBE
from music_decoder.types import (
    AnalysisResult,
    AudioSource,
    BeatGrid,
    ChordSegment,
    ChordSymbol,
    KeyEstimate,
    LoadedAudio,
    Note,
    TabbedNote,
    TabPosition,
)


def _mock_loaded_audio(tmp_path: Path) -> LoadedAudio:
    audio_path = tmp_path / "x.wav"
    source = AudioSource(
        path=audio_path,
        declared_kind="full_mix",
        requested_quality="standard",
        requested_tuning=STANDARD_EADGBE,
    )
    return LoadedAudio(
        samples=np.zeros(22050, dtype=np.float32),
        sr=22050,
        duration_s=1.0,
        sha256="abc",
        source=source,
    )


def test_analyze_returns_analysis_result(tmp_path: Path) -> None:
    audio = tmp_path / "x.wav"
    audio.write_bytes(b"WAV")
    with (
        mock.patch(
            "music_decoder.api.ingest_load",
            return_value=_mock_loaded_audio(tmp_path),
        ) as ingest,
        mock.patch("music_decoder.api.run_separation") as sep,
        mock.patch("music_decoder.api.track_beats") as beats,
        mock.patch("music_decoder.api.compute_chroma") as chroma,
        mock.patch("music_decoder.api.estimate_key") as key,
        mock.patch("music_decoder.api.recognize_chords") as chords,
        mock.patch("music_decoder.api.transcribe") as trans,
        mock.patch("music_decoder.api.assign_tabs") as tabs,
    ):
        sep.return_value = (np.zeros(22050, dtype=np.float32), 22050)
        beats.return_value = BeatGrid(
            tempo_bpm=120.0,
            beat_times_s=np.array([0.0, 0.5]),
            downbeat_times_s=np.array([0.0]),
            ts_numerator=4,
            ts_denominator=4,
            ts_confidence=0.5,
            ts_assumed=True,
        )
        chroma.return_value = np.zeros((12, 10))
        key.return_value = KeyEstimate(
            tonic="C",
            mode="major",
            profile="krumhansl_kessler",
            correlation=0.9,
            margin=0.2,
        )
        chords.return_value = (
            ChordSegment(
                start_s=0.0,
                end_s=1.0,
                root="C",
                quality="maj",
                confidence=0.8,
            ),
        )
        trans.return_value = (Note(start_s=0.0, end_s=0.5, pitch=60, velocity=80, confidence=0.9),)
        tabs.return_value = (
            TabbedNote(
                note=Note(
                    start_s=0.0,
                    end_s=0.5,
                    pitch=60,
                    velocity=80,
                    confidence=0.9,
                ),
                position=TabPosition(string=4, fret=3),
            ),
        )
        result = analyze(str(audio))

    assert isinstance(result, AnalysisResult)
    assert result.key.tonic == "C"
    assert len(result.chord_progression) == 1
    assert isinstance(result.chord_progression[0].chord, ChordSymbol)
    assert len(result.tab) == 1
    assert result.tempo_bpm == 120.0
    assert result.sample_rate_hz == 22050
    ingest.assert_called_once()


def test_analyze_skips_separation_for_solo_guitar(tmp_path: Path) -> None:
    audio = tmp_path / "x.wav"
    audio.write_bytes(b"WAV")
    loaded = _mock_loaded_audio(tmp_path)
    with (
        mock.patch(
            "music_decoder.api.ingest_load",
            return_value=loaded,
        ),
        mock.patch("music_decoder.api.run_separation") as sep,
        mock.patch("music_decoder.api.track_beats") as beats,
        mock.patch(
            "music_decoder.api.compute_chroma",
            return_value=np.zeros((12, 10)),
        ),
        mock.patch(
            "music_decoder.api.estimate_key",
            return_value=KeyEstimate(
                tonic="C",
                mode="major",
                profile="krumhansl_kessler",
                correlation=0.9,
                margin=0.2,
            ),
        ),
        mock.patch(
            "music_decoder.api.recognize_chords",
            return_value=(),
        ),
        mock.patch(
            "music_decoder.api.transcribe",
            return_value=(),
        ),
        mock.patch(
            "music_decoder.api.assign_tabs",
            return_value=(),
        ),
    ):
        beats.return_value = BeatGrid(
            tempo_bpm=100.0,
            beat_times_s=np.array([0.0]),
            downbeat_times_s=np.array([0.0]),
            ts_numerator=4,
            ts_denominator=4,
            ts_confidence=0.5,
            ts_assumed=True,
        )
        analyze(str(audio), declared_kind="solo_guitar")

    sep.assert_not_called()


def test_analyze_passes_chords_and_audio_to_key_estimation(tmp_path: Path) -> None:
    """Key fusion needs the recognized chords, so chords run before key."""
    audio = tmp_path / "x.wav"
    audio.write_bytes(b"WAV")
    segs = (ChordSegment(start_s=0.0, end_s=1.0, root="G", quality="maj", confidence=0.8),)
    key_est = KeyEstimate(tonic="G", mode="major", profile="fusion", correlation=1.0, margin=0.5)
    grid = BeatGrid(
        tempo_bpm=120.0,
        beat_times_s=np.array([0.0, 0.5]),
        downbeat_times_s=np.array([0.0]),
        ts_numerator=4,
        ts_denominator=4,
        ts_confidence=0.5,
        ts_assumed=True,
    )
    with (
        mock.patch("music_decoder.api.ingest_load", return_value=_mock_loaded_audio(tmp_path)),
        mock.patch("music_decoder.api.track_beats", return_value=grid),
        mock.patch("music_decoder.api.compute_chroma", return_value=np.zeros((12, 10))),
        mock.patch("music_decoder.api.estimate_key", return_value=key_est) as key,
        mock.patch("music_decoder.api.recognize_chords", return_value=segs),
        mock.patch("music_decoder.api.transcribe", return_value=()),
        mock.patch("music_decoder.api.assign_tabs", return_value=()),
    ):
        result = analyze(str(audio), declared_kind="solo_guitar")
    assert key.call_args.kwargs["chords"] == segs
    assert key.call_args.kwargs["sr"] == 22050
    assert result.key.profile == "fusion"
