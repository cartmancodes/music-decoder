"""CLI subcommands for authoring + validating manual evaluation fixtures.

Two subcommands:

- ``music-decoder fixture-init <audio.wav> --name <name>`` — creates a stub
  JSON in ``tests/fixtures/manual/<name>.json`` pre-populated with audio
  metadata, default tuning, and example tab/chord arrays. The user fills
  in the actual ground truth.
- ``music-decoder fixture-validate <fixture.json>`` — walks the schema and
  prints any rule violations with file:line references.

The validator runs through a list of ``(predicate, error_template)`` pairs;
adding a new rule is one line.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa

from music_decoder.tab_assignment.tuning import PRESETS

_MIDI_MIN = 0
_MIDI_MAX = 127
_VALID_ROOTS = {
    "C", "C#", "Db", "D", "D#", "Eb", "E", "F",
    "F#", "Gb", "G", "G#", "Ab", "A", "A#", "Bb", "B", "N",
}
_VALID_QUALITIES = {"maj", "min", "7", "maj7", ""}


@dataclass(frozen=True)
class ValidationError:
    """One validation rule violation."""

    field: str
    message: str

    def __str__(self) -> str:
        return f"{self.field}: {self.message}"


def init_fixture(
    audio_path: Path,
    *,
    name: str,
    output_dir: Path,
    tuning: str = "EADGBE",
    force: bool = False,
) -> Path:
    """Create a manual-fixture JSON stub from a real audio file.

    Returns the path to the written JSON.
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"audio file not found: {audio_path}")
    if tuning not in PRESETS:
        raise ValueError(
            f"unknown tuning preset {tuning!r}; one of {sorted(PRESETS)}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{name}.json"
    if out_path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite {out_path} (use --force)")

    duration_s = float(librosa.get_duration(path=str(audio_path)))

    stub = {
        "_comment": (
            "Manual evaluation fixture — fill in the tab + chords arrays with"
            " the actual ground truth. See "
            "tests/fixtures/manual/clip01.json (if present) for an example."
        ),
        "audio": audio_path.name,
        "tuning": tuning,
        "tempo_bpm": None,
        "key": {"tonic": "C", "mode": "major"},
        "tab": [
            {
                "_example": "delete this entry; add real notes here",
                "start_s": 0.0,
                "end_s": 0.5,
                "pitch": 60,
                "string": 4,
                "fret": 1,
            }
        ],
        "chords": [
            {
                "_example": "delete this entry; add real chords here",
                "start_s": 0.0,
                "end_s": min(duration_s, 4.0),
                "root": "C",
                "quality": "maj",
            }
        ],
        "_audio_metadata": {
            "duration_s": round(duration_s, 3),
            "filename": audio_path.name,
        },
    }
    out_path.write_text(json.dumps(stub, indent=2))
    return out_path


def validate_fixture(fixture_path: Path) -> list[ValidationError]:
    """Walk the fixture's schema and return any rule violations."""
    if not fixture_path.exists():
        return [ValidationError("file", f"fixture not found: {fixture_path}")]
    try:
        data: dict[str, Any] = json.loads(fixture_path.read_text())
    except json.JSONDecodeError as e:
        return [ValidationError("json", f"invalid JSON: {e}")]

    errors: list[ValidationError] = []

    # Required top-level fields
    audio_field = data.get("audio")
    if not audio_field:
        errors.append(ValidationError("audio", "required field missing"))
    else:
        audio_path = fixture_path.parent / str(audio_field)
        if not audio_path.exists():
            errors.append(
                ValidationError(
                    "audio",
                    f"file does not exist alongside fixture: {audio_path}",
                )
            )

    tuning = data.get("tuning")
    if tuning is None:
        errors.append(ValidationError("tuning", "required field missing"))
    elif tuning not in PRESETS:
        errors.append(
            ValidationError(
                "tuning",
                f"{tuning!r} is not a known preset; one of {sorted(PRESETS)}",
            )
        )

    # Validate the tab array if present
    tab = data.get("tab", [])
    if not isinstance(tab, list):
        errors.append(ValidationError("tab", "must be an array"))
    else:
        errors.extend(_validate_tab_rows(tab, tuning_name=tuning))

    # Validate chords array if present
    chords = data.get("chords", [])
    if not isinstance(chords, list):
        errors.append(ValidationError("chords", "must be an array"))
    else:
        errors.extend(_validate_chord_rows(chords))

    # Validate key if present
    key = data.get("key")
    if key is not None:
        if not isinstance(key, dict) or "tonic" not in key or "mode" not in key:
            errors.append(
                ValidationError("key", "must have 'tonic' and 'mode' keys")
            )

    return errors


def _validate_tab_rows(
    rows: list[Any], *, tuning_name: str | None,
) -> list[ValidationError]:
    out: list[ValidationError] = []
    open_pitches = (
        PRESETS[tuning_name].open_pitches if tuning_name in PRESETS else None
    )
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            out.append(ValidationError(f"tab[{i}]", "must be an object"))
            continue
        # Skip stub example rows
        if row.get("_example"):
            continue
        for required in ("start_s", "end_s", "pitch"):
            if required not in row:
                out.append(
                    ValidationError(f"tab[{i}].{required}", "required")
                )
        start_s = row.get("start_s")
        end_s = row.get("end_s")
        pitch = row.get("pitch")
        if isinstance(start_s, (int, float)) and isinstance(end_s, (int, float)):
            if end_s <= start_s:
                out.append(
                    ValidationError(
                        f"tab[{i}]",
                        f"end_s ({end_s}) must be > start_s ({start_s})",
                    )
                )
        if isinstance(pitch, (int, float)):
            if pitch < _MIDI_MIN or pitch > _MIDI_MAX:
                out.append(
                    ValidationError(
                        f"tab[{i}].pitch",
                        f"{pitch} out of MIDI range [{_MIDI_MIN}, {_MIDI_MAX}]",
                    )
                )
        # If string + fret are both present and tuning is known, sanity-check.
        string = row.get("string")
        fret = row.get("fret")
        if (open_pitches is not None
                and isinstance(string, int)
                and isinstance(fret, int)
                and isinstance(pitch, (int, float))):
            if 0 <= string < len(open_pitches) and fret >= 0:
                expected = open_pitches[string] + fret
                if expected != round(float(pitch)):
                    out.append(
                        ValidationError(
                            f"tab[{i}]",
                            f"string={string} fret={fret} on {tuning_name} "
                            f"yields MIDI {expected}, but pitch={pitch}",
                        )
                    )
            elif string < 0 or string >= len(open_pitches):
                out.append(
                    ValidationError(
                        f"tab[{i}].string",
                        f"{string} out of range for {tuning_name} "
                        f"(0..{len(open_pitches) - 1})",
                    )
                )
    return out


def _validate_chord_rows(rows: list[Any]) -> list[ValidationError]:
    out: list[ValidationError] = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            out.append(ValidationError(f"chords[{i}]", "must be an object"))
            continue
        if row.get("_example"):
            continue
        for required in ("start_s", "end_s", "root", "quality"):
            if required not in row:
                out.append(
                    ValidationError(f"chords[{i}].{required}", "required")
                )
        root = row.get("root")
        quality = row.get("quality")
        if isinstance(root, str) and root not in _VALID_ROOTS:
            out.append(
                ValidationError(
                    f"chords[{i}].root",
                    f"{root!r} is not a valid pitch class",
                )
            )
        if isinstance(quality, str) and quality not in _VALID_QUALITIES:
            out.append(
                ValidationError(
                    f"chords[{i}].quality",
                    f"{quality!r} is not in supported set "
                    f"{sorted(_VALID_QUALITIES)}",
                )
            )
        start_s = row.get("start_s")
        end_s = row.get("end_s")
        if isinstance(start_s, (int, float)) and isinstance(end_s, (int, float)):
            if end_s <= start_s:
                out.append(
                    ValidationError(
                        f"chords[{i}]",
                        f"end_s ({end_s}) must be > start_s ({start_s})",
                    )
                )
    return out


def _format_errors(errors: list[ValidationError]) -> str:
    if not errors:
        return ""
    lines = [f"  - {e}" for e in errors]
    return "\n".join(lines)


def cli_init(
    audio: str, name: str, tuning: str, force: bool, output_dir: str,
    print_fn: Callable[[str], None] = print,
) -> int:
    """Wired entry point for ``music-decoder fixture-init``."""
    audio_path = Path(audio)
    out_dir = Path(output_dir)
    try:
        out_path = init_fixture(
            audio_path, name=name, output_dir=out_dir,
            tuning=tuning, force=force,
        )
    except FileNotFoundError as e:
        print_fn(f"error: {e}")
        return 2
    except FileExistsError as e:
        print_fn(f"error: {e}")
        return 2
    except ValueError as e:
        print_fn(f"error: {e}")
        return 2
    print_fn(f"wrote {out_path}")
    print_fn("Edit the JSON to fill in your tab + chord ground truth, then run:")
    print_fn(f"  music-decoder fixture-validate {out_path}")
    return 0


def cli_validate(
    fixture: str, print_fn: Callable[[str], None] = print,
) -> int:
    """Wired entry point for ``music-decoder fixture-validate``."""
    errors = validate_fixture(Path(fixture))
    if errors:
        print_fn(f"validation failed: {len(errors)} error(s)")
        print_fn(_format_errors(errors))
        return 1
    print_fn(f"OK: {fixture}")
    return 0
