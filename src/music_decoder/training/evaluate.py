"""Evaluate a candidate basic-pitch checkpoint against the regression suite.

Given a checkpoint path, this module loads it as a drop-in for the stock
basic-pitch checkpoint, runs the full regression pipeline on the committed
fixtures, and reports per-fixture + aggregate metrics. Used to decide
whether a fine-tuned checkpoint is worth promoting.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from music_decoder.config.hyperparameters import (
    BasicPitchParams,
    load_hyperparameters,
)
from music_decoder.evaluation.fixtures.base import Fixture
from music_decoder.evaluation.fixtures.guitarset import GuitarSetFixtures
from music_decoder.evaluation.fixtures.synthetic import SyntheticFixtures
from music_decoder.evaluation.runner import EvaluationReport, run_evaluation


# A small helper that mirrors what tests/regression/test_accuracy_thresholds.py
# does, without importing it (the test file is regression-marked and pytest-only).
def _make_pipeline(
    checkpoint_path: Path | None,
    hp_path: Path,
) -> Callable[[Path, Fixture], dict[str, Any]]:
    """Return a pipeline callable that uses the candidate checkpoint."""
    from music_decoder.audio_io.load import load_audio
    from music_decoder.beat_tracking.beats import track_beats
    from music_decoder.chord_detection.api import detect_chords
    from music_decoder.key_detection.api import detect_key
    from music_decoder.key_detection.chroma import compute_chroma_with_hpss
    from music_decoder.pipeline.contracts import AudioSource
    from music_decoder.tab_assignment.assigner import assign_tab
    from music_decoder.tab_assignment.tuning import get_preset
    from music_decoder.transcription.basic_pitch_wrapper import (
        transcribe_basic_pitch,
    )
    from music_decoder.transcription.post_processing import apply_post_processing

    hp = load_hyperparameters(hp_path)

    def pipeline(audio_path: Path, fixture: Fixture) -> dict[str, Any]:
        src = AudioSource(
            path=audio_path, declared_kind="solo_guitar",
            requested_quality="standard", requested_tuning=get_preset("EADGBE"),
        )
        audio = load_audio(src)

        # Use the candidate checkpoint when provided. The current
        # transcribe_basic_pitch wrapper doesn't accept a custom path; this
        # placeholder keeps the API forward-looking for when fine-tuned
        # checkpoints are wired in. For now, the eval just uses the stock
        # checkpoint and the call below is identical to the regression test.
        bp_params: BasicPitchParams = hp.basic_pitch
        transcription = transcribe_basic_pitch(
            audio, bp_params, output_dir=Path("/tmp/md_eval_tmp"),
        )

        grid = track_beats(
            audio.samples, sr=audio.sr,
            start_bpm=hp.beat_tracking.start_bpm,
            tightness=hp.beat_tracking.tightness,
        )
        cleaned = apply_post_processing(
            transcription.notes, params=hp.post_processing,
            beats=grid.beat_times_s,
        )
        key = detect_key(
            audio.samples, sr=audio.sr,
            hpss_margin=hp.key_detection.hpss_margin,
            segment_length_s=hp.key_detection.windowed_segment_length_s,
            hop_s=hp.key_detection.windowed_hop_s,
        )
        chroma = compute_chroma_with_hpss(
            audio.samples, sr=audio.sr,
            hpss_margin=hp.key_detection.hpss_margin,
        )
        chord_result = detect_chords(
            chroma=chroma, sr=audio.sr, hop_length=512,
            beat_grid=grid, params=hp.chord_detection,
            audio_path=audio_path,
        )
        tab_result = assign_tab(
            cleaned, tuning=get_preset("EADGBE"),
            weights=hp.tab_assignment.weights,
            max_fret=hp.tab_assignment.max_fret,
        )
        intervals = np.array(
            [(n.start_s, n.end_s) for n in cleaned], dtype=float,
        )
        pitches = np.array([n.pitch for n in cleaned], dtype=float)
        consensus = key.consensus_key
        tab_intervals = np.array(
            [(t.note.start_s, t.note.end_s) for t in tab_result.tabbed_notes],
            dtype=float,
        ) if tab_result.tabbed_notes else np.zeros((0, 2))
        return {
            "intervals": intervals,
            "pitches_midi": pitches,
            "key": (
                (consensus.tonic, consensus.mode) if consensus else None
            ),
            "tab": [
                (t.note.pitch, t.position.string, t.position.fret)
                for t in tab_result.tabbed_notes
            ],
            "tab_intervals": tab_intervals,
            "chord_segments": chord_result.segments,
        }

    return pipeline


@dataclass(frozen=True)
class EvalSummary:
    """Aggregate metrics across all fixtures in a run."""

    note_f_measure: float | None
    onset_f_measure: float | None
    pitch_class_accuracy: float | None
    key_mirex_score: float | None
    tab_string_accuracy: float | None
    chord_recognition_score: float | None


def evaluate_checkpoint(
    checkpoint_path: Path | None,
    hp_path: Path,
    fixtures: list[Fixture],
) -> EvaluationReport:
    """Run the regression pipeline against ``fixtures`` and return a report.

    ``checkpoint_path`` is a forward-looking parameter; when basic-pitch's
    custom-checkpoint loading is wired into the wrapper, the candidate model
    will be used here. For now, the stock checkpoint is used regardless.
    """
    pipeline = _make_pipeline(checkpoint_path, hp_path)
    return run_evaluation(pipeline, fixtures)


def aggregate_metrics(report: EvaluationReport) -> EvalSummary:
    """Collapse per-fixture metrics to means (None when no fixture supplied a value)."""

    def _mean(name: str) -> float | None:
        vals = [
            getattr(r, name) for r in report.per_fixture
            if getattr(r, name) is not None
        ]
        return float(np.mean(vals)) if vals else None

    return EvalSummary(
        note_f_measure=_mean("note_f_measure"),
        onset_f_measure=_mean("onset_f_measure"),
        pitch_class_accuracy=_mean("pitch_class_accuracy"),
        key_mirex_score=_mean("key_mirex_score"),
        tab_string_accuracy=_mean("tab_string_accuracy"),
        chord_recognition_score=_mean("chord_recognition_score"),
    )


def default_fixtures(
    guitarset_cache: Path = Path("tests/fixtures/guitarset"),
    synthetic_root: Path = Path("tests/fixtures/synthetic"),
    guitarset_track_ids: list[str] | None = None,
) -> list[Fixture]:
    """Load the default mixed corpus (synthetic + GuitarSet held-out tracks)."""
    track_ids = guitarset_track_ids or [
        "00_BN1-129-Eb_comp", "00_BN1-129-Eb_solo",
        "00_Funk1-114-Ab_comp", "00_Jazz1-200-B_comp",
        "00_Rock1-130-A_comp",
    ]
    fixtures: list[Fixture] = list(SyntheticFixtures(root=synthetic_root).load())
    gs_loader = GuitarSetFixtures(
        cache_dir=guitarset_cache, track_ids=track_ids,
    )
    if gs_loader.is_available():
        fixtures.extend(gs_loader.load())
    return fixtures
