"""Digests of the composer's strum and arpeggio output, frozen at composer v3.

``tests/fixtures/compose/v3_strum_arpeggio.json`` holds one SHA-256 per case.
``tests/unit/test_compose_frozen.py`` compares the current composer against it,
so changes aimed at the fingerstyle part cannot alter the other two styles.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator

from music_decoder.compose.api import accompany, arrange
from music_decoder.compose.tabtext import render_composition_tab
from music_decoder.evaluation.compose_musicality import (
    TEMPO_BPM,
    Inputs,
    edge_inputs,
    tunings_inputs,
    validation_inputs,
)
from music_decoder.tabs.tuning import PRESETS, STANDARD_EADGBE, Tuning

_STYLES = ("strum", "arpeggio")


def _cases() -> Iterator[tuple[Tuning, Inputs]]:
    for inputs in edge_inputs():
        yield STANDARD_EADGBE, inputs
    for tuning in PRESETS.values():
        for inputs in tunings_inputs():
            yield tuning, inputs
    for inputs in validation_inputs()[::10]:
        yield STANDARD_EADGBE, inputs


def strum_arpeggio_digests() -> dict[str, str]:
    """Map each case key to the SHA-256 hex of its strum or arpeggio output."""
    out: dict[str, str] = {}
    for tuning, (scale, progression, bars, seed) in _cases():
        arr = arrange(
            scale,
            progression,
            bars_per_chord=bars,
            tempo_bpm=TEMPO_BPM,
            style="strum",
            tuning=tuning,
            seed=seed,
        )
        tab_text = render_composition_tab(
            arr.melody_tab,
            arr.voicings,
            bars_per_chord=bars,
            tempo_bpm=TEMPO_BPM,
            tuning=tuning,
        )
        labels = " ".join(c.to_label() for c in progression)
        for style in _STYLES:
            accompaniment = (
                arr.accompaniment
                if style == "strum"
                else accompany(
                    arr.voicings,
                    arr.melody,
                    bars_per_chord=bars,
                    tempo_bpm=TEMPO_BPM,
                    style="arpeggio",
                    tuning=tuning,
                )
            )
            # Hash the fields, never repr(arr): new Arrangement fields must not move it.
            payload = repr((arr.melody, arr.voicings, accompaniment, arr.melody_tab, tab_text))
            key = f"{tuning.name}|{scale.tonic}:{scale.mode}|{labels}|{bars}|{seed}|{style}"
            out[key] = hashlib.sha256(payload.encode()).hexdigest()
    return out
