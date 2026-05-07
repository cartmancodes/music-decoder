"""
Regression test: the real pipeline runs against synthetic fixtures.
Metrics are measured and checked against acceptance thresholds and a
stored baseline so future runs can detect regressions.
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from music_decoder.audio_io.load import load_audio
from music_decoder.beat_tracking.beats import track_beats
from music_decoder.chords.api import detect_chords
from music_decoder.config.hyperparameters import load_hyperparameters
from music_decoder.evaluation.fixtures.base import Fixture, GroundTruth
from music_decoder.evaluation.fixtures.guitarset import GuitarSetFixtures
from music_decoder.evaluation.fixtures.synthetic import SyntheticFixtures
from music_decoder.evaluation.regression import (
    check_against_baseline,
    check_against_thresholds,
    load_baseline,
    load_thresholds,
)
from music_decoder.evaluation.runner import run_evaluation
from music_decoder.key_detection.api import detect_key
from music_decoder.key_detection.chroma import compute_chroma_with_hpss
from music_decoder.types import AudioSource
from music_decoder.tab_assignment.assigner import assign_tab
from music_decoder.tab_assignment.tuning import get_preset
from music_decoder.transcription.basic_pitch_wrapper import transcribe_basic_pitch
from music_decoder.transcription.post_processing import apply_post_processing


def _real_pipeline(audio_path: Path, fixture: Fixture) -> dict[str, object]:
    hp = load_hyperparameters(Path("config/hyperparameters.yaml"))
    src = AudioSource(
        path=audio_path, declared_kind="solo_guitar",
        requested_quality="standard", requested_tuning=get_preset("EADGBE"),
    )
    audio = load_audio(src)
    transcription = transcribe_basic_pitch(
        audio, hp.basic_pitch, output_dir=Path("/tmp/md_eval_tmp"),
    )
    grid = track_beats(audio.samples, sr=audio.sr,
                       start_bpm=hp.beat_tracking.start_bpm,
                       tightness=hp.beat_tracking.tightness)
    cleaned = apply_post_processing(
        transcription.notes, params=hp.post_processing,
        beats=grid.beat_times_s,
    )
    key = detect_key(audio.samples, sr=audio.sr,
                     hpss_margin=hp.key_detection.hpss_margin,
                     segment_length_s=hp.key_detection.windowed_segment_length_s,
                     hop_s=hp.key_detection.windowed_hop_s)
    tab_result = assign_tab(
        cleaned, tuning=get_preset("EADGBE"),
        weights=hp.tab_assignment.weights, max_fret=hp.tab_assignment.max_fret,
    )
    chroma = compute_chroma_with_hpss(
        audio.samples, sr=audio.sr, hpss_margin=hp.key_detection.hpss_margin,
    )
    chord_result = detect_chords(
        chroma=chroma, sr=audio.sr, hop_length=512,
        beat_grid=grid, params=hp.chord_detection,
        audio_path=audio_path,
    )
    intervals = np.array([(n.start_s, n.end_s) for n in cleaned], dtype=float)
    pitches = np.array([n.pitch for n in cleaned], dtype=float)
    tab_intervals = np.array(
        [(t.note.start_s, t.note.end_s) for t in tab_result.tabbed_notes],
        dtype=float,
    ) if tab_result.tabbed_notes else np.zeros((0, 2))
    consensus = key.consensus_key
    return {
        "intervals": intervals, "pitches_midi": pitches,
        "key": (consensus.tonic, consensus.mode) if consensus else None,
        "tab": [(t.note.pitch, t.position.string, t.position.fret)
                for t in tab_result.tabbed_notes],
        "tab_intervals": tab_intervals,
        "chord_segments": chord_result.segments,
    }


def _synthetic_fixtures() -> list[Fixture]:
    return list(SyntheticFixtures(root=Path("tests/fixtures/synthetic")).load())


# A small selection of GuitarSet excerpts spanning genres and playing styles.
# These IDs are taken from the canonical GuitarSet release.
_GUITARSET_TRACK_IDS = [
    "00_BN1-129-Eb_comp",   # bossa nova comping
    "00_BN1-129-Eb_solo",   # bossa nova solo
    "00_Funk1-114-Ab_comp",
    "00_Jazz1-200-B_comp",
    "00_Rock1-130-A_comp",
]


def _guitarset_fixtures() -> list[Fixture]:
    cache = Path("tests/fixtures/guitarset")
    loader = GuitarSetFixtures(cache_dir=cache, track_ids=_GUITARSET_TRACK_IDS)
    if not loader.is_available():
        return []
    return list(loader.load())


def _all_fixtures() -> list[Fixture]:
    return _synthetic_fixtures() + _guitarset_fixtures()


@pytest.fixture
def trivial_fixture(tmp_path: Path) -> Fixture:
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")
    return Fixture(
        name="trivial", source="manual", audio_path=audio,
        ground_truth=GroundTruth(
            intervals=np.array([[0.0, 0.5]]), pitches_midi=np.array([60]),
            key=("C", "major"), tempo_bpm=120.0, tab=[(60, 4, 1)],
        ),
    )


@pytest.mark.regression
def test_thresholds_loadable() -> None:
    thresholds = load_thresholds(Path("config/eval_thresholds.yaml"))
    assert "note_f_measure" in thresholds
    assert "regression_tolerance" in thresholds


@pytest.mark.regression
@pytest.mark.slow
def test_real_pipeline_passes_thresholds() -> None:
    fixtures = _all_fixtures()
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    report_path = Path("evaluation_reports") / f"{ts}_regression.json"
    report = run_evaluation(_real_pipeline, fixtures, report_path=report_path)
    thresholds = load_thresholds(Path("config/eval_thresholds.yaml"))
    failures = check_against_thresholds(report, thresholds)
    assert failures == [], f"Threshold failures: {failures}"


@pytest.mark.regression
@pytest.mark.slow
def test_baseline_comparison_no_regression() -> None:
    fixtures = _all_fixtures()
    report = run_evaluation(_real_pipeline, fixtures)
    baseline = load_baseline(Path("evaluation_reports/baseline.json"))
    failures = check_against_baseline(report, baseline, tolerance=0.02)
    assert failures == [], f"Baseline regressions: {failures}"


@pytest.mark.regression
@pytest.mark.slow
def test_guitarset_fixtures_loadable_when_cached() -> None:
    """Skips cleanly when GuitarSet cache absent; surfaces a clear error
    when cache is present but the loader yields nothing."""
    cache = Path("tests/fixtures/guitarset")
    loader = GuitarSetFixtures(cache_dir=cache, track_ids=_GUITARSET_TRACK_IDS)
    if not loader.is_available():
        pytest.skip("GuitarSet cache not present; run `make fixtures` to download.")
    fixtures = list(loader.load())
    assert len(fixtures) >= 1, (
        f"GuitarSet cache present at {cache} but loader yielded 0 fixtures "
        f"for track ids {_GUITARSET_TRACK_IDS}. Possible mirdata API drift."
    )
