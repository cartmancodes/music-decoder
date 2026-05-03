from __future__ import annotations

from music_decoder.logging_setup import get_logger

_log = get_logger("models_download")


def _load_basic_pitch() -> None:
    from basic_pitch import ICASSP_2022_MODEL_PATH  # noqa: F401
    _log.info("basic_pitch_loaded")


def _load_crepe() -> None:
    import crepe  # noqa: F401
    _log.info("crepe_loaded")


def _load_demucs() -> None:
    try:
        from demucs.pretrained import get_model
        get_model("htdemucs_6s")
    except Exception as e:
        _log.warning("demucs_load_failed", extra={"error": str(e)})
        return
    _log.info("demucs_loaded")


def warm_models() -> None:
    _load_basic_pitch()
    _load_crepe()
    _load_demucs()
