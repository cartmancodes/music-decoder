"""Tab assignment adapters for the public ``analyze()`` pipeline."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from music_decoder.tabs.assigner import assign_tab as _assign_tab_impl
from music_decoder.tabs.tuning import STANDARD_EADGBE, Tuning
from music_decoder.types import TabbedNote, TranscribedNote

# Fallback only (YAML ``tab_assignment.weights`` is authoritative); mirrors the
# values tuned on GuitarSet dev by ``scripts/benchmark_guitarset.py --sweep-tabs``.
_DEFAULT_WEIGHTS: dict[str, float] = {
    "w_move": 0.5,
    "w_string": 0.15,
    "w_span": 0.5,
    "w_high": 1.6,
    "w_open": 0.05,
    "w_chord_intra": 0.3,
}
_DEFAULT_MAX_FRET = 22


def _yaml_tab_params() -> tuple[dict[str, float], int]:
    """``tab_assignment`` weights / max_fret from YAML; code defaults if unreadable."""
    try:
        from music_decoder.config.hyperparameters import load_hyperparameters

        tab = load_hyperparameters().tab_assignment
        return {**_DEFAULT_WEIGHTS, **tab.weights}, int(tab.max_fret)
    except Exception:  # pragma: no cover - defensive
        return dict(_DEFAULT_WEIGHTS), _DEFAULT_MAX_FRET


def assign_tabs(
    notes: Iterable[TranscribedNote],
    *,
    tuning: Tuning = STANDARD_EADGBE,
    max_fret: int | None = None,
    weights: dict[str, float] | None = None,
) -> Sequence[TabbedNote]:
    """Assign tab positions for a stream of notes.

    Weights and the fret cap come from ``config/hyperparameters.yaml``
    (``tab_assignment``) unless passed explicitly.
    """
    yaml_weights, yaml_max_fret = _yaml_tab_params()
    result = _assign_tab_impl(
        notes,
        tuning=tuning,
        weights=weights if weights is not None else yaml_weights,
        max_fret=max_fret if max_fret is not None else yaml_max_fret,
    )
    return tuple(result.tabbed_notes)


__all__ = ["assign_tabs"]
