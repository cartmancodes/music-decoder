from __future__ import annotations

import numpy as np

from music_decoder.logging_setup import get_logger
from music_decoder.types import LoadedAudio, SeparationResult

_log = get_logger("separation")


def _apply_demucs(
    samples: np.ndarray[object, np.dtype[np.float32]], sr: int
) -> np.ndarray[object, np.dtype[np.float32]]:
    """Run Demucs htdemucs_6s and return the guitar stem at the same sample rate."""
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    model = get_model("htdemucs_6s")
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # demucs expects (channels, samples) at the model's native rate
    target_sr = model.samplerate
    if sr != target_sr:
        from librosa import resample

        samples = resample(samples, orig_sr=sr, target_sr=target_sr)
    audio_tensor = torch.from_numpy(samples).float().unsqueeze(0).unsqueeze(0)
    audio_tensor = audio_tensor.repeat(1, 2, 1)  # demucs wants stereo
    audio_tensor = audio_tensor.to(device)

    with torch.no_grad():
        sources = apply_model(model, audio_tensor, split=True, overlap=0.25)
    # find the guitar stem
    sources_names = list(model.sources)
    guitar_idx = sources_names.index("guitar")
    guitar = sources[0, guitar_idx].mean(dim=0).cpu().numpy()
    if target_sr != sr:
        from librosa import resample

        guitar = resample(guitar, orig_sr=target_sr, target_sr=sr)
    if guitar.size > samples.size:
        guitar = guitar[: samples.size]
    return guitar.astype(np.float32)


def isolate_guitar(audio: LoadedAudio) -> SeparationResult:
    if audio.source.declared_kind == "solo_guitar":
        return SeparationResult(
            guitar_samples=None,
            sr=audio.sr,
            skipped_reason="solo_guitar declared",
            bleed_estimate_db=None,
        )
    try:
        guitar = _apply_demucs(audio.samples, audio.sr)
    except Exception as e:
        _log.error("demucs_failed", extra={"error": str(e)})
        return SeparationResult(
            guitar_samples=None,
            sr=audio.sr,
            skipped_reason=f"demucs_failed: {e}",
            bleed_estimate_db=None,
        )
    return SeparationResult(
        guitar_samples=guitar,
        sr=audio.sr,
        skipped_reason=None,
        bleed_estimate_db=None,
    )
