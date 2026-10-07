from pathlib import Path
from typing import Any

import numpy as np

from music_decoder.evaluation.guitarset import GTNote, parse_jams


def _jams() -> dict[str, Any]:
    def ann(ns: str, data: list[dict[str, Any]], src: str = "") -> dict[str, Any]:
        return {"namespace": ns, "annotation_metadata": {"data_source": src}, "data": data}

    return {
        "file_metadata": {"duration": 4.0},
        "annotations": [
            ann("note_midi", [{"time": 1.0, "duration": 0.5, "value": 44.2}], "0"),
            ann("note_midi", [{"time": 0.5, "duration": 0.5, "value": 64.9}], "5"),
            ann(
                "beat_position",
                [
                    {"time": 0.0, "value": {"position": 1}},
                    {"time": 0.5, "value": {"position": 2}},
                    {"time": 1.0, "value": {"position": 1}},
                ],
            ),
            ann(
                "chord",
                [
                    {"time": 0.0, "duration": 2.0, "value": "D#:maj"},
                    {"time": 2.0, "duration": 2.0, "value": "G#:maj6(*5)/1"},
                ],
            ),
            ann("chord", [{"time": 0.0, "duration": 4.0, "value": "X"}], "performed"),
            ann("key_mode", [{"time": 0.0, "duration": 4.0, "value": "Eb:major"}]),
        ],
    }


def test_parse_jams_notes_sorted_with_string_and_rounded_pitch() -> None:
    t = parse_jams(_jams(), track_id="00_BN1-129-Eb_comp", audio_path=Path("a.wav"))
    assert t.player == "00"
    assert t.notes == (GTNote(0.5, 1.0, 65, 5), GTNote(1.0, 1.5, 44, 0))


def test_parse_jams_uses_first_chord_annotation_and_keeps_raw_labels() -> None:
    t = parse_jams(_jams(), track_id="00_x", audio_path=Path("a.wav"))
    assert t.chords == ((0.0, 2.0, "D#:maj"), (2.0, 4.0, "G#:maj6(*5)/1"))


def test_parse_jams_key_normalised_to_sharps_and_beats() -> None:
    t = parse_jams(_jams(), track_id="00_x", audio_path=Path("a.wav"))
    assert t.key == ("D#", "major")
    np.testing.assert_allclose(t.beats, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(t.downbeats, [0.0, 1.0])
