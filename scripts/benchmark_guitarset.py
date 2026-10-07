#!/usr/bin/env python
"""GuitarSet accuracy benchmark for the ``analyze()`` pipeline stages.

Runs the *public stage adapters* exactly as ``analyze(declared_kind="solo_guitar")``
does and scores them with standard ``mir_eval`` metrics:

- key    : MIREX weighted key score
- beats  : beat F-measure (±70 ms, first 5 s trimmed)
- chords : MIREX ``majmin`` duration-weighted recall vs the lead-sheet chords
- notes  : onset-only note P / R / F (50 ms, ±50 cents)
- tab_gt : string accuracy of ``assign_tabs`` fed the *ground-truth* notes
           (isolates the fingering algorithm from transcription errors)
- tab_e2e: string accuracy of ``assign_tabs`` on the transcribed notes

Dev split = players 00-04 (tune here); test split = player 05 (report only).
basic-pitch's raw network output is cached under ``out/bench-cache/bp`` so
decoding-parameter sweeps don't re-run the network.

    python scripts/benchmark_guitarset.py --split dev --stages key,chords
    python scripts/benchmark_guitarset.py --split test --json out/bench/v3-test.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import librosa
import numpy as np

from music_decoder.chords import recognize_chords
from music_decoder.dsp import compute_chroma, track_beats
from music_decoder.dsp.tempwav import temp_wav
from music_decoder.evaluation.guitarset import (
    DEV_PLAYERS,
    TEST_PLAYERS,
    GuitarSetTrack,
    iter_tracks,
)
from music_decoder.evaluation.metrics import (
    beat_f_measure,
    chord_majmin_score,
    key_mirex_score,
    note_onset_prf,
    tab_string_accuracy,
)
from music_decoder.key import estimate_key
from music_decoder.tabs import assign_tabs
from music_decoder.tabs.tuning import STANDARD_EADGBE
from music_decoder.types import TabbedNote, TranscribedNote

SR = 22050
CACHE = Path("out/bench-cache")
ALL_STAGES = ("key", "beats", "chords", "notes", "tab_gt", "tab_e2e")


def load_audio(track: GuitarSetTrack) -> np.ndarray[Any, np.dtype[np.float32]]:
    y, _ = librosa.load(str(track.audio_path), sr=SR, mono=True)
    return y.astype(np.float32)


def model_output(track: GuitarSetTrack, samples: np.ndarray[Any, Any]) -> dict[str, Any]:
    """Cached basic-pitch network output for *track*."""
    from music_decoder.transcription import basic_pitch_wrapper

    path = CACHE / "bp" / f"{track.track_id}.npz"
    if path.exists():
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    with temp_wav(samples, SR) as wav:
        out = basic_pitch_wrapper.run_model(wav)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    return out


def gt_note_arrays(track: GuitarSetTrack) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    iv = np.array([(n.start_s, n.end_s) for n in track.notes], dtype=float).reshape(-1, 2)
    pitches = np.array([n.pitch for n in track.notes], dtype=float)
    return iv, pitches


def score_tabs(track: GuitarSetTrack, tabbed: tuple[TabbedNote, ...] | list[TabbedNote]) -> float:
    truth = [
        (n.pitch, n.string, n.pitch - STANDARD_EADGBE.open_pitches[n.string])
        for n in track.notes
    ]
    gt_iv, _ = gt_note_arrays(track)
    pred_iv = np.array([(t.note.start_s, t.note.end_s) for t in tabbed], dtype=float).reshape(-1, 2)
    return tab_string_accuracy(list(tabbed), truth, pred_intervals=pred_iv, gt_intervals=gt_iv)


def notes_for(track: GuitarSetTrack, samples: np.ndarray[Any, Any]) -> tuple[TranscribedNote, ...]:
    from music_decoder.transcription import decode_model_output

    return decode_model_output(model_output(track, samples))


def evaluate_track(track: GuitarSetTrack, stages: tuple[str, ...]) -> dict[str, float]:
    samples = load_audio(track)
    row: dict[str, float] = {}
    if "key" in stages:
        key = estimate_key(compute_chroma(samples, SR), samples=samples, sr=SR)
        row["key"] = key_mirex_score(key, track.key)  # type: ignore[arg-type]
    grid = None
    if "beats" in stages or "chords" in stages:
        grid = track_beats(samples, SR)
    if "beats" in stages and grid is not None:
        row["beats"] = beat_f_measure(grid.beat_times_s, track.beats)
    if "chords" in stages and grid is not None:
        segs = list(recognize_chords(samples, SR, beat_grid=grid))
        row["chords"] = chord_majmin_score(segs, list(track.chords))
    notes: tuple[TranscribedNote, ...] | None = None
    if "notes" in stages or "tab_e2e" in stages:
        notes = notes_for(track, samples)
    if "notes" in stages and notes is not None:
        gt_iv, gt_p = gt_note_arrays(track)
        pred_iv = np.array([(n.start_s, n.end_s) for n in notes], dtype=float).reshape(-1, 2)
        pred_p = np.array([n.pitch for n in notes], dtype=float)
        p, r, f = note_onset_prf(pred_iv, pred_p, gt_iv, gt_p)
        row.update({"note_p": p, "note_r": r, "notes": f})
    if "tab_gt" in stages:
        gt_notes = [
            TranscribedNote(n.start_s, n.end_s, n.pitch, 80, 1.0) for n in track.notes
        ]
        row["tab_gt"] = score_tabs(track, assign_tabs(gt_notes))
    if "tab_e2e" in stages and notes is not None:
        row["tab_e2e"] = score_tabs(track, assign_tabs(notes))
    return row


def summarize(rows: dict[str, dict[str, float]]) -> dict[str, float]:
    keys = sorted({k for r in rows.values() for k in r})
    return {k: statistics.fmean(r[k] for r in rows.values() if k in r) for k in keys}


def by_style(rows: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    groups: dict[str, dict[str, dict[str, float]]] = {}
    for tid, r in rows.items():
        style = "".join(c for c in tid.split("_")[1].split("-")[0] if c.isalpha())
        kind = tid.rsplit("_", 1)[1]
        groups.setdefault(f"{style}-{kind}", {})[tid] = r
    return {g: summarize(rs) for g, rs in sorted(groups.items())}


def print_table(title: str, summary: dict[str, dict[str, float]]) -> None:
    cols = sorted({k for s in summary.values() for k in s})
    print(f"\n### {title}\n")
    print("| group | " + " | ".join(cols) + " |")
    print("|---|" + "---|" * len(cols))
    for g, s in summary.items():
        print(f"| {g} | " + " | ".join(f"{s.get(c, float('nan')):.3f}" for c in cols) + " |")


def tracks_for(split: str, limit: int | None) -> list[GuitarSetTrack]:
    players = DEV_PLAYERS if split == "dev" else TEST_PLAYERS
    tracks = list(iter_tracks(players=players))
    if limit is not None and limit < len(tracks):
        # Even stride keeps every player / style / comp-solo represented.
        idx = np.linspace(0, len(tracks) - 1, limit).round().astype(int)
        tracks = [tracks[i] for i in sorted(set(idx.tolist()))]
    return tracks


def run(
    split: str,
    stages: tuple[str, ...],
    limit: int | None,
    *,
    evaluate: Callable[[GuitarSetTrack, tuple[str, ...]], dict[str, float]] = evaluate_track,
) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    tracks = tracks_for(split, limit)
    t0 = time.time()
    for i, track in enumerate(tracks, 1):
        rows[track.track_id] = evaluate(track, stages)
        print(f"[{i}/{len(tracks)}] {track.track_id} {rows[track.track_id]}", file=sys.stderr)
    print(f"elapsed {time.time() - t0:.0f}s", file=sys.stderr)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--split", choices=("dev", "test"), default="dev")
    ap.add_argument("--stages", default=",".join(ALL_STAGES))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--by-style", action="store_true")
    args = ap.parse_args()
    stages = tuple(s for s in args.stages.split(",") if s)
    unknown = set(stages) - set(ALL_STAGES)
    if unknown:
        ap.error(f"unknown stages: {sorted(unknown)}")

    rows = run(args.split, stages, args.limit)
    summary = {"mean": summarize(rows)}
    if args.by_style:
        summary.update(by_style(rows))
    print_table(f"GuitarSet {args.split} (n={len(rows)})", summary)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"summary": summary, "tracks": rows}, indent=1))


if __name__ == "__main__":
    main()
