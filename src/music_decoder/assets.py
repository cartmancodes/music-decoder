"""Provisioning of the models and assets the pipeline needs outside the wheel.

Most models ship inside their pip packages (basic-pitch, madmom). Two things
don't:

- **Demucs ``htdemucs_6s`` weights** (~52 MB) — fetched by
  ``demucs.pretrained.get_model`` into the torch hub cache on first use.
  :func:`fetch_demucs` triggers that ahead of time.
- **A General MIDI soundfont** for composition synthesis. pretty_midi bundles
  ``TimGM6mb.sf2`` (6 MB), which :func:`bundled_soundfont` returns so synthesis
  works with no download at all; :func:`fetch_soundfont` installs the
  higher-fidelity GeneralUser GS at the path ``config/runtime.yaml`` names.

``music-decoder fetch-models`` (and ``setup.sh``) call these.
"""

from __future__ import annotations

import os
from pathlib import Path

DEMUCS_MODEL = "htdemucs_6s"
# GeneralUser GS by S. Christian Collins — official repository.
GENERALUSER_GS_URL = "https://github.com/mrbumpy409/GeneralUser-GS/raw/main/GeneralUser-GS.sf2"

_CHUNK = 1 << 20


def bundled_soundfont() -> Path | None:
    """pretty_midi's bundled ``TimGM6mb.sf2``, or ``None`` if unavailable."""
    try:
        import pretty_midi
    except ImportError:  # pragma: no cover - pretty_midi is a hard dependency
        return None
    path = Path(pretty_midi.__file__).parent / "TimGM6mb.sf2"
    return path if path.exists() else None


def is_soundfont(path: Path) -> bool:
    """True if *path* starts with a RIFF ``sfbk`` header (SF2)."""
    try:
        with open(path, "rb") as f:
            head = f.read(12)
    except OSError:
        return False
    return len(head) == 12 and head[:4] == b"RIFF" and head[8:12] == b"sfbk"


def soundfont_target() -> Path:
    """Where the configured soundfont lives (``data_dir``-relative if not absolute)."""
    from music_decoder.config.runtime import load_runtime_config

    cfg = load_runtime_config()
    sf = Path(cfg.fluidsynth_soundfont).expanduser()
    return sf if sf.is_absolute() else Path(cfg.data_dir).expanduser() / sf


def fetch_soundfont(
    dest: Path | None = None,
    url: str = GENERALUSER_GS_URL,
    *,
    force: bool = False,
) -> Path:
    """Download a soundfont to *dest* (default: :func:`soundfont_target`).

    Idempotent: an existing valid soundfont is kept unless *force*. Downloads
    to a ``.part`` file, validates the SF2 header, then renames into place, so
    an interrupted or bogus download (e.g. an HTML error page) never leaves a
    broken file behind. Raises ``RuntimeError`` on an invalid download.
    """
    import urllib.request

    target = dest if dest is not None else soundfont_target()
    if not force and is_soundfont(target):
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp, open(part, "wb") as out:
            while chunk := resp.read(_CHUNK):
                out.write(chunk)
        if not is_soundfont(part):
            raise RuntimeError(f"download from {url} is not a SoundFont (SF2) file")
        os.replace(part, target)
    finally:
        part.unlink(missing_ok=True)
    return target


def demucs_signatures(model: str = DEMUCS_MODEL) -> list[str]:
    """Checkpoint signatures a Demucs model needs (from demucs' remote manifest)."""
    import demucs
    import yaml

    manifest = Path(demucs.__file__).parent / "remote" / f"{model}.yaml"
    data = yaml.safe_load(manifest.read_text()) or {}
    return [str(sig) for sig in data.get("models", [])]


def demucs_weights_cached(model: str = DEMUCS_MODEL) -> bool:
    """True if every checkpoint *model* needs is already in the torch hub cache."""
    import torch

    checkpoints = Path(torch.hub.get_dir()) / "checkpoints"
    sigs = demucs_signatures(model)
    return bool(sigs) and all(any(checkpoints.glob(f"{sig}-*.th")) for sig in sigs)


def fetch_demucs(model: str = DEMUCS_MODEL) -> None:
    """Download (if needed) and load the Demucs model so its weights are cached."""
    from demucs.pretrained import get_model

    get_model(model)
