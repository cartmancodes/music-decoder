"""Synth package: MIDI -> WAV rendering.

Exposes :func:`render_wav` as a thin alias over the existing
:func:`fluidsynth_wrapper.synthesize_midi_to_wav` so callers (notably
``compose.api``) can import a stable name regardless of the underlying
backend implementation.
"""

from __future__ import annotations

from pathlib import Path

from music_decoder.logging_setup import get_logger
from music_decoder.paths import project_root
from music_decoder.synth.fluidsynth_wrapper import (
    SynthBackend,
    synthesize_midi_to_wav,
)

_log = get_logger("synth.adapter")


def _configured_soundfont_path() -> Path | None:
    """The soundfont named in the runtime config, if it exists on disk.

    Relative paths resolve against the runtime ``data_dir`` first, then the
    project's synthetic-fixture soundfont dir.
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
    if sf_path.is_absolute():
        return sf_path if sf_path.exists() else None
    candidates = [
        Path(cfg.data_dir).expanduser() / sf_path,
        project_root() / "tests" / "fixtures" / "synthetic" / "soundfont" / sf_path,
    ]
    return next((c for c in candidates if c.exists()), None)


def _resolve_soundfont_path() -> Path | None:
    """Absolute path of the soundfont to synthesize with, or None.

    Prefers the configured soundfont (e.g. GeneralUser GS installed by
    ``music-decoder fetch-models``); otherwise falls back to pretty_midi's
    bundled ``TimGM6mb.sf2`` so synthesis works with no download.
    """
    from music_decoder.assets import bundled_soundfont

    return _configured_soundfont_path() or bundled_soundfont()


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
