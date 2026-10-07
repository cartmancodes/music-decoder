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
        (n.pitch, n.string, n.pitch - STANDARD_EADGBE.open_pitches[n.string]) for n in track.notes
    ]
    gt_iv, _ = gt_note_arrays(track)
    pred_iv = np.array([(t.note.start_s, t.note.end_s) for t in tabbed], dtype=float).reshape(-1, 2)
    return tab_string_accuracy(list(tabbed), truth, pred_intervals=pred_iv, gt_intervals=gt_iv)


def notes_for(track: GuitarSetTrack, samples: np.ndarray[Any, Any]) -> tuple[TranscribedNote, ...]:
    from music_decoder.transcription import decode_model_output

    return decode_model_output(model_output(track, samples))


def evaluate_track(track: GuitarSetTrack, stages: tuple[str, ...]) -> dict[str, float]:
    needs_audio = bool(set(stages) - {"tab_gt"})
    samples = load_audio(track) if needs_audio else np.zeros(0, np.float32)
    row: dict[str, float] = {}
    # Mirror analyze(): beats -> chords -> key (key fusion uses the chords).
    grid = None
    segs = None
    if {"beats", "chords", "key"} & set(stages):
        grid = track_beats(samples, SR)
    if "beats" in stages and grid is not None:
        row["beats"] = beat_f_measure(grid.beat_times_s, track.beats)
    if ({"chords", "key"} & set(stages)) and grid is not None:
        segs = list(recognize_chords(samples, SR, beat_grid=grid))
    if "chords" in stages and segs is not None:
        row["chords"] = chord_majmin_score(segs, list(track.chords))
    if "key" in stages:
        chroma = compute_chroma(samples, SR)
        key = estimate_key(chroma, samples=samples, sr=SR, chords=segs)
        row["key"] = key_mirex_score(key, track.key)  # type: ignore[arg-type]
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
        gt_notes = [TranscribedNote(n.start_s, n.end_s, n.pitch, 80, 1.0) for n in track.notes]
        row["tab_gt"] = score_tabs(track, assign_tabs(gt_notes))
    if "tab_e2e" in stages and notes is not None:
        row["tab_e2e"] = score_tabs(track, assign_tabs(notes))
    return row


# ---- decoding sweep (basic-pitch thresholds + post-processing) --------------

Combo = tuple[float, float, int, bool, float, float]
# (onset, frame, min_note_ms, melodia, merge_gap_s, min_dur_s); gap/min_dur
# of 0 means "post-processing off".


def _decode_scores(track: GuitarSetTrack, combos: list[Combo]) -> list[tuple[float, float, float]]:
    from basic_pitch import note_creation
    from basic_pitch.constants import AUDIO_SAMPLE_RATE, FFT_HOP

    from music_decoder.transcription import resolve_params
    from music_decoder.transcription.basic_pitch_wrapper import notes_from_events
    from music_decoder.transcription.post_processing import drop_short_notes, merge_same_pitch

    out = model_output(track, load_audio(track) if not _cached(track) else np.zeros(0))
    bounds = resolve_params()
    gt_iv, gt_p = gt_note_arrays(track)
    scores = []
    for onset, frame, min_ms, melodia, gap, min_dur in combos:
        _m, events = note_creation.model_output_to_notes(
            out,
            onset_thresh=onset,
            frame_thresh=frame,
            min_note_len=round(min_ms / 1000 * (AUDIO_SAMPLE_RATE / FFT_HOP)),
            min_freq=bounds.minimum_frequency_hz,
            max_freq=bounds.maximum_frequency_hz,
            melodia_trick=melodia,
        )
        notes = notes_from_events(events)
        if gap > 0:
            notes = merge_same_pitch(notes, gap_s=gap)
        if min_dur > 0:
            notes = drop_short_notes(notes, min_duration_s=min_dur)
        iv = np.array([(n.start_s, n.end_s) for n in notes], dtype=float).reshape(-1, 2)
        pp = np.array([n.pitch for n in notes], dtype=float)
        scores.append(note_onset_prf(iv, pp, gt_iv, gt_p))
    return scores


def _cached(track: GuitarSetTrack) -> bool:
    return (CACHE / "bp" / f"{track.track_id}.npz").exists()


def sweep(
    tracks: list[GuitarSetTrack], combos: list[Combo], workers: int
) -> list[tuple[Combo, float, float, float]]:
    """Mean onset P/R/F per combo, best F first."""
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=workers) as pool:
        per_track = list(pool.map(_decode_scores, tracks, [combos] * len(tracks)))
    results = []
    for i, combo in enumerate(combos):
        ps, rs, fs = zip(*(t[i] for t in per_track), strict=True)
        results.append((combo, statistics.fmean(ps), statistics.fmean(rs), statistics.fmean(fs)))
    return sorted(results, key=lambda r: r[3], reverse=True)


def _print_sweep(
    title: str, results: list[tuple[Combo, float, float, float]], top: int = 8
) -> None:
    print(f"\n### {title}\n")
    print("| onset | frame | min_ms | melodia | merge_gap | min_dur | P | R | F |")
    print("|---|---|---|---|---|---|---|---|---|")
    for (o, fr, ms, mel, gap, md), p, r, f in results[:top]:
        print(f"| {o} | {fr} | {ms} | {mel} | {gap} | {md} | {p:.3f} | {r:.3f} | {f:.3f} |")


def sweep_notes(tracks: list[GuitarSetTrack], workers: int) -> Combo:
    """Coarse-to-fine search (threshold grid → length/melodia → refine → post-processing)."""
    import itertools as it

    # Make sure every network output is cached before fanning out.
    for t in tracks:
        if not _cached(t):
            model_output(t, load_audio(t))

    stage1 = [
        (o, f, 58, True, 0.0, 0.0)
        for o, f in it.product((0.3, 0.4, 0.5, 0.6, 0.7, 0.8), (0.2, 0.3, 0.4, 0.5, 0.6))
    ]
    r1 = sweep(tracks, stage1, workers)
    _print_sweep("stage 1: onset x frame", r1)
    o, f = r1[0][0][0], r1[0][0][1]

    stage2 = [(o, f, ms, mel, 0.0, 0.0) for ms, mel in it.product((35, 58, 90, 128), (True, False))]
    r2 = sweep(tracks, stage2, workers)
    _print_sweep("stage 2: min note length x melodia", r2)
    ms, mel = r2[0][0][2], r2[0][0][3]

    stage3 = [
        (round(o + do, 3), round(f + df, 3), ms, mel, 0.0, 0.0)
        for do, df in it.product((-0.05, 0.0, 0.05), (-0.05, 0.0, 0.05))
        if 0 < f + df < 1 and 0 < o + do < 1
    ]
    r3 = sweep(tracks, stage3, workers)
    _print_sweep("stage 3: refine thresholds", r3)
    o, f = r3[0][0][0], r3[0][0][1]

    stage4 = [(o, f, ms, mel, 0.0, 0.0)] + [
        (o, f, ms, mel, gap, md) for gap, md in it.product((0.02, 0.05), (0.0, 0.03, 0.05, 0.08))
    ]
    r4 = sweep(tracks, stage4, workers)
    _print_sweep("stage 4: post-processing", r4)
    return r4[0][0]


# ---- tab weight sweep (coordinate descent on ground-truth notes) -----------


def _tab_gt_score(track: GuitarSetTrack, weights: dict[str, float]) -> float:
    notes = [TranscribedNote(n.start_s, n.end_s, n.pitch, 80, 1.0) for n in track.notes]
    return score_tabs(track, assign_tabs(notes, weights=weights))


def _mean_tab(tracks: list[GuitarSetTrack], weights: dict[str, float], pool: Any) -> float:
    return statistics.fmean(pool.map(_tab_gt_score, tracks, [weights] * len(tracks)))


def sweep_tabs(tracks: list[GuitarSetTrack], workers: int, passes: int = 2) -> dict[str, float]:
    """Coordinate descent over A* weights, maximizing mean ``tab_gt``."""
    from concurrent.futures import ProcessPoolExecutor

    from music_decoder.tabs import _yaml_tab_params

    weights, _ = _yaml_tab_params()
    factors = (0.0, 0.25, 0.5, 2.0, 4.0)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        best = _mean_tab(tracks, weights, pool)
        print(f"start {best:.4f} {weights}", file=sys.stderr)
        for p in range(passes):
            for name in sorted(weights):
                base = weights[name] if weights[name] > 0 else 0.1
                for factor in factors:
                    trial = {**weights, name: round(base * factor, 4)}
                    score = _mean_tab(tracks, trial, pool)
                    if score > best + 1e-4:
                        best, weights = score, trial
                        print(f"pass {p} {name}={trial[name]} -> {best:.4f}", file=sys.stderr)
    print(f"\n### tab weight sweep (n={len(tracks)})\n\nbest tab_gt {best:.4f}: {weights}")
    return weights


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
    workers: int = 1,
    evaluate: Callable[[GuitarSetTrack, tuple[str, ...]], dict[str, float]] = evaluate_track,
) -> dict[str, dict[str, float]]:
    tracks = tracks_for(split, limit)
    t0 = time.time()
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(evaluate, tracks, [stages] * len(tracks)))
    else:
        results = []
        for i, track in enumerate(tracks, 1):
            results.append(evaluate(track, stages))
            print(f"[{i}/{len(tracks)}] {track.track_id} {results[-1]}", file=sys.stderr)
    print(f"elapsed {time.time() - t0:.0f}s", file=sys.stderr)
    return {t.track_id: r for t, r in zip(tracks, results, strict=True)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--split", choices=("dev", "test"), default="dev")
    ap.add_argument("--stages", default=",".join(ALL_STAGES))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--by-style", action="store_true")
    ap.add_argument("--sweep-notes", action="store_true", help="tune basic-pitch decoding")
    ap.add_argument("--sweep-tabs", action="store_true", help="tune A* tab weights")
    ap.add_argument("--workers", type=int, default=1, help="parallel processes")
    args = ap.parse_args()
    if args.sweep_tabs:
        sweep_tabs(tracks_for(args.split, args.limit), max(args.workers, 2))
        return
    if args.sweep_notes:
        best = sweep_notes(tracks_for(args.split, args.limit), max(args.workers, 2))
        print(f"\nbest: {best}")
        return
    stages = tuple(s for s in args.stages.split(",") if s)
    unknown = set(stages) - set(ALL_STAGES)
    if unknown:
        ap.error(f"unknown stages: {sorted(unknown)}")

    rows = run(args.split, stages, args.limit, workers=args.workers)
    summary = {"mean": summarize(rows)}
    if args.by_style:
        summary.update(by_style(rows))
    print_table(f"GuitarSet {args.split} (n={len(rows)})", summary)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"summary": summary, "tracks": rows}, indent=1))


if __name__ == "__main__":
    main()
