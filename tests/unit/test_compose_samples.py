"""The three human-flagged `compose` commands, run literally through the CLI.

Each command produced a faulty chord voicing and an over-wide melodic leap
(docs/superpowers/specs/2026-11-07-compose-voicings-leaps-design.md section 2.1).
The tests assert musical facts only; the strict xfail markers are removed by
the voicing and melody fixes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
from click.testing import CliRunner

from music_decoder.cli.main import _TUNINGS, main
from music_decoder.compose import metrics
from music_decoder.compose.voicings import fingers_needed, shape_faults
from music_decoder.types import ChordSymbol, Note, Tuning

SAMPLES = {
    "minor-10-B": [
        "--scale",
        "B:minor",
        "--progression",
        "Bm G D A Em7 C#dim F#7 Bm",
        "--style",
        "strum",
        "--tempo",
        "124",
        "--seed",
        "210",
        "--tuning",
        "EADGBE",
    ],
    "major-08-Ab": [
        "--scale",
        "Ab:major",
        "--progression",
        "Ab C7 Fm Db Eb7 Ab",
        "--style",
        "strum",
        "--tempo",
        "110",
        "--seed",
        "108",
        "--tuning",
        "Eb",
    ],
    "major-09-Eb": [
        "--scale",
        "Eb:major",
        "--progression",
        "Eb Cm Fm7 Bbsus4 Bb7 Eb",
        "--style",
        "fingerstyle",
        "--tempo",
        "72",
        "--seed",
        "109",
        "--tuning",
        "D standard",
    ],
}


def _run(args: list[str], tmp_path: Path) -> dict[str, Any]:
    with mock.patch("music_decoder.compose.api.render_wav"):
        result = CliRunner().invoke(
            main, ["compose", *args, "--out", str(tmp_path), "--format", "json"]
        )
    assert result.exit_code == 0, result.output
    out: dict[str, Any] = json.loads(result.output)
    return out


def _arg(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]


def _tuning(args: list[str]) -> Tuning:
    return _TUNINGS[_arg(args, "--tuning")]


def _shapes(out: dict[str, Any]) -> list[tuple[ChordSymbol, list[int]]]:
    return [
        (ChordSymbol(**v["chord"]), [p["fret"] for p in v["positions"]])
        for v in out["chord_voicings"]
    ]


@pytest.mark.parametrize("name", list(SAMPLES))
def test_sample_voicings_are_faithful(name: str, tmp_path: Path) -> None:
    args = SAMPLES[name]
    out = _run(args, tmp_path)
    tuning = _tuning(args)
    shapes = _shapes(out)
    sounding = {
        chord.to_label(): sorted(tuning.open_pitches[s] + f for s, f in enumerate(frets) if f >= 0)
        for chord, frets in shapes
    }
    # 1. The reported defect first (review r2 F3), so it is what fails on v2.
    if name == "minor-10-B":
        assert {p % 12 for p in sounding["C#dim"]} == {1, 4, 7}, "named: C#dim"
    if name == "major-08-Ab":
        assert len(sounding["D#7"]) >= 4 and {7, 1} <= {p % 12 for p in sounding["D#7"]}, (
            "named: D#7"
        )
    if name == "major-09-Eb":
        assert sounding["A#7"][0] % 12 == 10, "named: A#7 bass"
    # 2. Then every shape: faithful and playable (on v2, Eb-tuned C7 x43421 needs 5 fingers).
    for chord, frets in shapes:
        assert shape_faults(frets, tuning, chord) == (), f"audit faults: {chord.to_label()}"
        assert fingers_needed(frets) <= 4, f"audit fingers: {chord.to_label()}"


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(k, marks=pytest.mark.xfail(strict=True, reason="melodic leaps (Task 6)"))
        for k in SAMPLES
    ],
)
def test_sample_melodies_are_singable(name: str, tmp_path: Path) -> None:
    # Musical assertions only, in a fixed order (spec section 5.6). No version
    # check here: it would mask these assertions while the composer is still v2.
    args = SAMPLES[name]
    out = _run(args, tmp_path)
    mel = [Note(**d) for d in out["melody_notes"]]
    prog = [chord for chord, _ in _shapes(out)]
    tempo = float(_arg(args, "--tempo"))
    assert metrics.forbidden_interval_count(mel) == 0, "L1"
    assert metrics.unjustified_large_leap_count(mel, prog, 1, tempo) == 0, "L3"
    leaps, recovered = metrics.large_leap_recovery_counts(mel)
    assert recovered == leaps, "L2"
    assert max(metrics.chord_change_moves(mel, prog, 1, tempo)) <= 7, "L4max"
    final = abs(mel[-1].pitch - mel[-2].pitch)
    assert final <= 7 and final != 6, "L5"


@pytest.mark.parametrize("name", list(SAMPLES))
def test_sample_commands_are_deterministic(name: str, tmp_path: Path) -> None:
    a = _run(SAMPLES[name], tmp_path / "a")
    b = _run(SAMPLES[name], tmp_path / "b")
    assert a["melody_notes"] == b["melody_notes"] and a["chord_voicings"] == b["chord_voicings"]
