from __future__ import annotations

# Re-export the canonical Tuning dataclass from music_decoder.types so the
# CLI / compose / tabs modules all share one nominal type. (Historically
# tabs/tuning.py defined its own structurally-identical Tuning, which made
# mypy --strict flag every cross-module call site.)
from music_decoder.types import Tuning

__all__ = [
    "DADGAD",
    "DROP_C",
    "DROP_D",
    "D_STANDARD",
    "EB_HALF_STEP_DOWN",
    "PRESETS",
    "STANDARD_EADGBE",
    "Tuning",
    "get_preset",
]


PRESETS: dict[str, Tuning] = {
    "EADGBE": Tuning("EADGBE", (40, 45, 50, 55, 59, 64)),  # E2 A2 D3 G3 B3 E4
    "Drop_D": Tuning("Drop_D", (38, 45, 50, 55, 59, 64)),  # D2 A2 D3 G3 B3 E4
    "Eb": Tuning("Eb", (39, 44, 49, 54, 58, 63)),  # half-step down
    "D_standard": Tuning("D_standard", (38, 43, 48, 53, 57, 62)),  # full step down
    "Drop_C": Tuning("Drop_C", (36, 43, 48, 53, 57, 62)),
    "DADGAD": Tuning("DADGAD", (38, 45, 50, 55, 57, 62)),
}

# Named module-level constants for ergonomic imports.
STANDARD_EADGBE = PRESETS["EADGBE"]
DROP_D = PRESETS["Drop_D"]
EB_HALF_STEP_DOWN = PRESETS["Eb"]
D_STANDARD = PRESETS["D_standard"]
DROP_C = PRESETS["Drop_C"]
DADGAD = PRESETS["DADGAD"]


def get_preset(name: str) -> Tuning:
    if name not in PRESETS:
        raise KeyError(f"Unknown tuning preset: {name!r}")
    return PRESETS[name]
