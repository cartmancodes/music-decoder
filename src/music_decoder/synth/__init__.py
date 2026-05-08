"""Synth package: MIDI -> WAV rendering.

Exposes :func:`render_wav` as a thin alias over the existing
:func:`fluidsynth_wrapper.synthesize_midi_to_wav` so callers (notably
``compose.api``) can import a stable name regardless of the underlying
backend implementation.
"""

from __future__ import annotations

from pathlib import Path

from music_decoder.logging_setup import get_logger
from music_decoder.synth.fluidsynth_wrapper import (
    SynthBackend,
    synthesize_midi_to_wav,
)

_log = get_logger("synth.adapter")


def _resolve_soundfont_path() -> Path | None:
    """Return an absolute path to the configured soundfont, or None.

    Resolves ``RuntimeConfig.fluidsynth_soundfont`` against the runtime
    config; treats relative paths as relative to the runtime ``data_dir``.
    Returns None when the field is missing, the file doesn't exist, or the
    runtime config can't be loaded.
    """
    try:
        from music_decoder.config.runtime import load_runtime_config

        cfg = load_runtime_config()
    except Exception:
        return None
    sf_field = getattr(cfg, "fluidsynth_soundfont", None)
    if not sf_field:
        return None
    sf_path = Path(str(sf_field)).expanduser()
    if not sf_path.is_absolute():
        # Treat relative paths as data_dir-relative; fall back to project
        # tests/fixtures soundfont dir if that miss too.
        candidates = [Path(cfg.data_dir).expanduser() / sf_path]
        # Project soundfont fixture (used by `scripts/download_soundfont.py`).
        candidates.append(
            Path(__file__).resolve().parents[3]
            / "tests"
            / "fixtures"
            / "synthetic"
            / "soundfont"
            / sf_path
        )
        for c in candidates:
            if c.exists():
                return c
        return None
    return sf_path if sf_path.exists() else None


def _fluidsynth_importable() -> bool:
    try:
        import fluidsynth  # noqa: F401
    except Exception:
        return False
    return True


def render_wav(
    midi_path: Path,
    out_path: Path,
    *,
    backend: SynthBackend | None = None,
) -> Path:
    """Render `midi_path` to `out_path` as WAV.

    With ``backend=None`` (the default) auto-selects fluidsynth when both
    a soundfont is configured and the ``fluidsynth`` Python module is
    importable; otherwise falls back to the built-in pretty-MIDI sine
    synthesizer with a single-line warning.
    """
    soundfont = _resolve_soundfont_path() if backend is None else None
    chosen: SynthBackend
    if backend is not None:
        chosen = backend
        return synthesize_midi_to_wav(
            midi_path,
            out_path,
            backend=chosen,
            soundfont_path=soundfont,
        )
    if soundfont is not None and _fluidsynth_importable():
        chosen = SynthBackend.FLUIDSYNTH
    else:
        if soundfont is None:
            _log.warning("synth_fallback_to_sine", extra={"reason": "no_soundfont"})
        else:
            _log.warning(
                "synth_fallback_to_sine", extra={"reason": "fluidsynth_unimportable"}
            )
        chosen = SynthBackend.SINE
    return synthesize_midi_to_wav(
        midi_path,
        out_path,
        backend=chosen,
        soundfont_path=soundfont if chosen == SynthBackend.FLUIDSYNTH else None,
    )


__all__ = ["SynthBackend", "render_wav", "synthesize_midi_to_wav"]
