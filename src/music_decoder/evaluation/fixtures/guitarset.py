from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from .base import Fixture, GroundTruth

# Standard guitar tuning: MIDI pitches for open strings E A D G B e
_STANDARD_TUNING_MIDI = [40, 45, 50, 55, 59, 64]
_STRING_ORDER = ["E", "A", "D", "G", "B", "e"]


class GuitarSetFixtures:
    def __init__(self, cache_dir: Path, track_ids: list[str]) -> None:
        self.cache_dir = cache_dir
        self.track_ids = track_ids

    def is_available(self) -> bool:
        return self.cache_dir.exists() and any(self.cache_dir.iterdir())

    def _load_track(self, dataset, track_id: str) -> Fixture | None:  # type: ignore[no-untyped-def]
        if track_id not in dataset.track_ids:
            return None
        track = dataset.track(track_id)

        # Try notes_by_string first (older mirdata API), fall back to per-string notes dict
        tab: list[tuple[int, int, int]] | None = None

        if hasattr(track, "notes_by_string"):
            # Older mirdata API: notes_by_string is a list of NoteData per string
            notes_all_obj = getattr(track, "notes", None)
            if notes_all_obj is None:
                return None
            intervals_all = np.asarray(notes_all_obj.intervals, dtype=float)
            pitches_all = np.asarray(notes_all_obj.pitches, dtype=float)
            tab_rows: list[tuple[int, int, int]] = []
            for s_idx, sn in enumerate(track.notes_by_string or []):
                if sn is None:
                    continue
                open_midi = _STANDARD_TUNING_MIDI[s_idx] if s_idx < 6 else 40
                for p in np.asarray(sn.pitches, dtype=float):
                    fret = round(float(p)) - open_midi
                    tab_rows.append((round(float(p)), s_idx, fret))
            tab = tab_rows if tab_rows else None
        elif hasattr(track, "notes_all") and hasattr(track, "notes"):
            # Current mirdata API: notes is a dict keyed by string name
            notes_all_obj = getattr(track, "notes_all", None)
            if notes_all_obj is None:
                return None
            intervals_all = np.asarray(notes_all_obj.intervals, dtype=float)
            pitches_all = np.asarray(notes_all_obj.pitches, dtype=float)
            notes_dict = getattr(track, "notes", {})
            if isinstance(notes_dict, dict):
                tab_rows = []
                for s_idx, string_name in enumerate(_STRING_ORDER):
                    sn = notes_dict.get(string_name)
                    if sn is None:
                        continue
                    open_midi = _STANDARD_TUNING_MIDI[s_idx]
                    pitches = np.asarray(sn.pitches, dtype=float)
                    for p in pitches:
                        fret = round(float(p)) - open_midi
                        tab_rows.append((round(float(p)), s_idx, fret))
                tab = tab_rows if tab_rows else None
        else:
            # Minimal fallback: use whatever notes attr is available, no tab
            notes_obj = getattr(track, "notes", None)
            if notes_obj is None:
                return None
            if isinstance(notes_obj, dict):
                # Combine all per-string notes
                all_intervals: list[np.ndarray[object, np.dtype[np.float64]]] = []
                all_pitches: list[np.ndarray[object, np.dtype[np.float64]]] = []
                for sn in notes_obj.values():
                    if sn is not None:
                        all_intervals.append(np.asarray(sn.intervals, dtype=float))
                        all_pitches.append(np.asarray(sn.pitches, dtype=float))
                if not all_intervals:
                    return None
                intervals_all = np.concatenate(all_intervals)
                pitches_all = np.concatenate(all_pitches)
            else:
                intervals_all = np.asarray(notes_obj.intervals, dtype=float)
                pitches_all = np.asarray(notes_obj.pitches, dtype=float)

        gt = GroundTruth(
            intervals=intervals_all,
            pitches_midi=pitches_all,
            key=None,
            tempo_bpm=track.tempo if hasattr(track, "tempo") else None,
            tab=tab,
        )
        return Fixture(
            name=track_id,
            source="guitarset",
            audio_path=Path(track.audio_mic_path),
            ground_truth=gt,
        )

    def load(self) -> Iterator[Fixture]:
        if not self.is_available():
            return
        import mirdata

        ds = mirdata.initialize("guitarset", data_home=str(self.cache_dir))
        for tid in self.track_ids:
            fx = self._load_track(ds, tid)
            if fx is not None:
                yield fx
