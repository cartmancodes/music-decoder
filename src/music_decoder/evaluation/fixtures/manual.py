from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from .base import Fixture, GroundTruth


class ManualFixtures:
    def __init__(self, root: Path) -> None:
        self.root = root

    def load(self) -> Iterator[Fixture]:
        if not self.root.exists():
            return
        for json_path in sorted(self.root.glob("*.json")):
            data = json.loads(json_path.read_text())
            audio_rel = data.get("audio")
            if not audio_rel:
                continue
            audio_path = self.root / audio_rel
            if not audio_path.exists():
                continue
            tab_rows = data.get("tab", []) or []
            intervals = (
                np.array(
                    [(row["start_s"], row["end_s"]) for row in tab_rows],
                    dtype=float,
                )
                if tab_rows
                else np.zeros((0, 2))
            )
            pitches = (
                np.array([row["pitch"] for row in tab_rows], dtype=float)
                if tab_rows
                else np.zeros(0)
            )
            tab: list[tuple[int, int, int]] | None = None
            if tab_rows and all("string" in r and "fret" in r for r in tab_rows):
                tab = [(int(r["pitch"]), int(r["string"]), int(r["fret"])) for r in tab_rows]
            key_field = data.get("key")
            key = (key_field["tonic"], key_field["mode"]) if key_field else None
            gt = GroundTruth(
                intervals=intervals,
                pitches_midi=pitches,
                key=key,
                tempo_bpm=float(data["tempo_bpm"]) if "tempo_bpm" in data else None,
                tab=tab,
            )
            yield Fixture(
                name=json_path.stem,
                source="manual",
                audio_path=audio_path,
                ground_truth=gt,
            )
