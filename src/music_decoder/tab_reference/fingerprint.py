from __future__ import annotations

from pathlib import Path


class SongIdentificationFailed(RuntimeError):
    pass


def fingerprint_audio(path: Path) -> str:
    """Compute a Chromaprint fingerprint via pyacoustid."""
    if not path.exists():
        raise SongIdentificationFailed(f"file does not exist: {path}")
    try:
        import acoustid  # pyacoustid
        _duration, fp = acoustid.fingerprint_file(str(path))
        return fp.decode() if isinstance(fp, bytes) else str(fp)
    except Exception as e:
        raise SongIdentificationFailed(str(e)) from e


def lookup_acoustid(api_key: str, fingerprint: str, duration_s: float) -> str | None:
    """Optionally resolve a fingerprint to an AcoustID. Returns None on failure."""
    try:
        import acoustid
        results = list(acoustid.match(api_key, fingerprint, duration_s, parse=True))
        if not results:
            return None
        _score, recording_id, *_ = results[0]
        return str(recording_id)
    except Exception:
        return None
