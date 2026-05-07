#!/usr/bin/env python3
"""Convert GuitarSet (mirdata cache) to basic-pitch's TFRecord training format.

basic-pitch's stock pipeline uses Apache Beam — fragile to install and even
more fragile to run when multiple Python interpreters live on the same
system (Beam's cloudpickle can't find ``basic_pitch`` from a worker process).
This script does the same conversion DIRECTLY, single-process, no Beam.

Output (matching basic_pitch.data.tf_example_deserialization's expected layout):
    <destination>/guitarset/splits/train/track_id.tfrecord
    <destination>/guitarset/splits/validation/track_id.tfrecord
    <destination>/guitarset/splits/test/track_id.tfrecord

Each TFRecord contains exactly one tf.train.Example per file, in the format
basic_pitch.data.tf_example_serialization.to_transcription_tfexample defines
(audio_wav bytes + sparse note/onset/contour indices/values + shapes).

Usage:
    python scripts/convert_guitarset_to_tfrecord.py \\
        --source tests/fixtures/guitarset \\
        --destination /tmp/bp_tfrecords/guitarset \\
        --train-percent 0.85 --validation-percent 0.10
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
import tempfile
from pathlib import Path
from typing import Iterable

import mirdata
import sox  # noqa: F401  required by basic-pitch's serializer
import tensorflow as tf

from basic_pitch.constants import (
    ANNOTATION_HOP,
    AUDIO_N_CHANNELS,
    AUDIO_SAMPLE_RATE,
    FREQ_BINS_CONTOURS,
    FREQ_BINS_NOTES,
    N_FREQ_BINS_CONTOURS,
    N_FREQ_BINS_NOTES,
)
from basic_pitch.data import tf_example_serialization


_log = logging.getLogger("convert_guitarset")


def _split_assignment(
    track_ids: list[str], train_pct: float, val_pct: float, seed: int | None,
) -> dict[str, str]:
    if seed is not None:
        random.seed(seed)
    shuffled = list(track_ids)
    random.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(round(n * train_pct))
    n_val = int(round(n * val_pct))
    splits: dict[str, str] = {}
    for i, track_id in enumerate(shuffled):
        if i < n_train:
            splits[track_id] = "train"
        elif i < n_train + n_val:
            splits[track_id] = "validation"
        else:
            splits[track_id] = "test"
    return splits


def _process_track(
    track_id: str,
    track: mirdata.core.Track,
    out_path: Path,
) -> bool:
    """Convert one GuitarSet track to a single-Example TFRecord. Returns True on success."""
    import numpy as np

    audio_path = Path(track.audio_mic_path)
    if not audio_path.exists():
        _log.warning("audio missing for %s: %s", track_id, audio_path)
        return False

    # Resample/mix to 22050 Hz stereo via sox (basic-pitch asserts both).
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        local_wav = Path(f.name)
    try:
        import sox as sox_mod
        tfm = sox_mod.Transformer()
        tfm.rate(AUDIO_SAMPLE_RATE)
        tfm.channels(AUDIO_N_CHANNELS)
        tfm.build(str(audio_path), str(local_wav))

        duration = sox_mod.file_info.duration(str(local_wav))
        if duration is None or duration < 1.0:
            _log.warning("track %s too short", track_id)
            return False
        time_scale = np.arange(0, duration + ANNOTATION_HOP, ANNOTATION_HOP)
        n_time_frames = len(time_scale)

        notes_obj = getattr(track, "notes_all", None) or getattr(track, "notes", None)
        if notes_obj is None or not hasattr(notes_obj, "to_sparse_index"):
            _log.warning("track %s has no notes annotation", track_id)
            return False

        note_indices, note_values = notes_obj.to_sparse_index(
            time_scale, "s", FREQ_BINS_NOTES, "hz",
        )
        onset_indices, onset_values = notes_obj.to_sparse_index(
            time_scale, "s", FREQ_BINS_NOTES, "hz", onsets_only=True,
        )
        contour_indices: list = []
        contour_values: list = []
        contour_obj = getattr(track, "multif0", None)
        if contour_obj is not None and hasattr(contour_obj, "to_sparse_index"):
            try:
                contour_indices, contour_values = contour_obj.to_sparse_index(
                    time_scale, "s", FREQ_BINS_CONTOURS, "hz",
                )
            except Exception as e:
                _log.warning("contour extraction failed for %s: %s", track_id, e)

        example = tf_example_serialization.to_transcription_tfexample(
            track_id,
            "guitarset",
            str(local_wav),
            note_indices,
            note_values,
            onset_indices,
            onset_values,
            contour_indices,
            contour_values,
            (n_time_frames, N_FREQ_BINS_NOTES),
            (n_time_frames, N_FREQ_BINS_CONTOURS),
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with tf.io.TFRecordWriter(str(out_path)) as writer:
            writer.write(example.SerializeToString())
    finally:
        if local_wav.exists():
            local_wav.unlink()
    return True


def convert(
    source: Path, destination: Path,
    *, train_pct: float, val_pct: float, seed: int | None,
    track_ids: Iterable[str] | None = None,
) -> dict[str, int]:
    """Convert GuitarSet to TFRecord. Returns counts per split."""
    ds = mirdata.initialize("guitarset", data_home=str(source))
    all_track_ids = list(track_ids) if track_ids is not None else list(ds.track_ids)
    if not all_track_ids:
        raise RuntimeError(f"no GuitarSet tracks found at {source}")

    assignment = _split_assignment(all_track_ids, train_pct, val_pct, seed)
    counts = {"train": 0, "validation": 0, "test": 0}

    for i, track_id in enumerate(all_track_ids, start=1):
        split = assignment[track_id]
        out_path = destination / "guitarset" / "splits" / split / f"{track_id}.tfrecord"
        if out_path.exists():
            _log.info("skip existing %s/%s.tfrecord (%d/%d)",
                      split, track_id, i, len(all_track_ids))
            counts[split] += 1
            continue
        track = ds.track(track_id)
        ok = _process_track(track_id, track, out_path)
        if ok:
            counts[split] += 1
            _log.info("wrote %s/%s.tfrecord (%d/%d)",
                      split, track_id, i, len(all_track_ids))
    return counts


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", default="tests/fixtures/guitarset",
        help="mirdata data_home for GuitarSet (must already be downloaded).",
    )
    parser.add_argument(
        "--destination", default="/tmp/bp_tfrecords/guitarset",
        help="Output directory. TFRecord splits go in train/, validation/, test/.",
    )
    parser.add_argument(
        "--train-percent", type=float, default=0.85,
        help="Fraction of tracks for training (default 0.85).",
    )
    parser.add_argument(
        "--validation-percent", type=float, default=0.10,
        help="Fraction for validation (default 0.10). Test = 1 - train - val.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for the train/val/test split.",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="If > 0, convert only the first N tracks (useful for smoke-testing).",
    )
    args = parser.parse_args()

    source = Path(args.source)
    destination = Path(args.destination)
    if not source.exists():
        print(f"GuitarSet cache missing: {source}", file=sys.stderr)
        return 2

    if args.limit > 0:
        ds = mirdata.initialize("guitarset", data_home=str(source))
        chosen = list(ds.track_ids)[: args.limit]
        counts = convert(
            source, destination,
            train_pct=args.train_percent, val_pct=args.validation_percent,
            seed=args.seed, track_ids=chosen,
        )
    else:
        counts = convert(
            source, destination,
            train_pct=args.train_percent, val_pct=args.validation_percent,
            seed=args.seed,
        )

    total = sum(counts.values())
    print(f"\nWrote {total} TFRecords to {destination}")
    for split, n in counts.items():
        print(f"  {split:11s} {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
