"""The opt-in audio re-rank of fingerstyle candidates (v5 spec 5.4).

The top candidates by symbolic score are rendered and scored by Audiobox
Aesthetics (Meta, ``facebook/audiobox-aesthetics``, CC-BY-4.0), which rates
a clip on four axes from 1 to 10: content enjoyment ``CE``, content
usefulness ``CU``, production complexity ``PC`` and production quality
``PQ``. A candidate's audio score is ``(CE + PQ) / 2`` plus
``SYMBOLIC_WEIGHT`` times its symbolic score min-max normalised to [0, 1]
across the rendered candidates, so the tune still counts when the audio
model can barely tell them apart.

The model runs on the CPU only, one clip at a time, so results repeat.
``torch``, ``torchaudio`` and ``audiobox_aesthetics`` are imported only by
:func:`load_scorer` and :meth:`AestheticScorer.score`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from music_decoder.compose.ml import MLUnavailableError

if TYPE_CHECKING:
    from music_decoder.compose.api import Arrangement

AXES = ("CE", "CU", "PC", "PQ")
# Weight of the normalised symbolic score next to the audio mean (1-10 scale):
# one point, so a clear audio preference wins over the symbolic order.
SYMBOLIC_WEIGHT = 1.0
# Audiobox Aesthetics reads 16 kHz mono in 10 s windows with a 10 s hop.
SAMPLE_RATE = 16000
WINDOW_S = 10


class Scorer(Protocol):
    def score(self, wav_paths: Sequence[Path]) -> list[dict[str, float]]: ...


class AestheticScorer:
    """Audiobox Aesthetics on the CPU. Build it with :func:`load_scorer`."""

    def __init__(self, model: Any) -> None:
        self._model = model

    def score(self, wav_paths: Sequence[Path]) -> list[dict[str, float]]:
        """Each WAV's ``CE``, ``CU``, ``PC`` and ``PQ``, in order.

        The clip is mixed to mono, resampled to 16 kHz and cut into 10 s
        windows; each axis is the window predictions' mean weighted by the
        windows' lengths, as the package's own predictor does.
        """
        import soundfile as sf
        import torch
        import torchaudio
        from audiobox_aesthetics.infer import make_inference_batch

        rows: list[dict[str, float]] = []
        with torch.inference_mode():
            for path in wav_paths:
                data, sr = sf.read(str(path), dtype="float32", always_2d=True)
                wav = torch.from_numpy(data.T.copy()).mean(dim=0, keepdim=True)
                wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=SAMPLE_RATE)
                wavs, masks, weights, _ = make_inference_batch(
                    [wav], WINDOW_S, WINDOW_S, sample_rate=SAMPLE_RATE
                )
                preds = self._model({"wav": torch.stack(wavs), "mask": torch.stack(masks)})
                w = torch.tensor(weights, dtype=torch.float32)
                row: dict[str, float] = {}
                for axis in AXES:
                    norm = self._model.target_transform[axis]
                    values = preds[axis].float() * norm["std"] + norm["mean"]
                    row[axis] = float((values * w).sum() / w.sum())
                rows.append(row)
        return rows


def load_scorer() -> AestheticScorer:
    """Audiobox Aesthetics from the Hugging Face cache, on the CPU.

    Raises :class:`MLUnavailableError` when ``audiobox_aesthetics`` is not
    installed or the weights have not been fetched. Never downloads.
    """
    try:
        import torch
        from audiobox_aesthetics.model.aes import AesMultiOutput
    except ImportError as e:
        raise MLUnavailableError(f"the audio re-rank needs audiobox_aesthetics ({e})") from e
    from music_decoder import assets

    folder = assets.audiobox_dir()
    if folder is None:
        raise MLUnavailableError("the Audiobox Aesthetics weights are not downloaded")
    model = AesMultiOutput.from_pretrained(folder, map_location="cpu")
    model.to(torch.device("cpu"))
    model.eval()
    return AestheticScorer(model)


def choose_by_audio(
    arrangements: Sequence[Arrangement], audio_scores: Sequence[Mapping[str, float]]
) -> int:
    """Index of the best of ``arrangements`` given each one's audio scores:
    ``(CE + PQ) / 2 + SYMBOLIC_WEIGHT * z``, ``z`` the symbolic score
    min-max normalised across them (0 when all are equal). Ties go to the
    lowest index."""
    if len(arrangements) != len(audio_scores) or not arrangements:
        raise ValueError("need one audio score per arrangement, and at least one")
    symbolic = []
    for a in arrangements:
        if a.score is None:
            raise ValueError("the audio re-rank needs scored (fingerstyle) arrangements")
        symbolic.append(a.score)
    low, high = min(symbolic), max(symbolic)
    span = high - low
    totals = [
        (row["CE"] + row["PQ"]) / 2 + SYMBOLIC_WEIGHT * ((s - low) / span if span > 0 else 0.0)
        for row, s in zip(audio_scores, symbolic, strict=True)
    ]
    return max(range(len(totals)), key=totals.__getitem__)  # max() keeps the first best


def rank_by_audio(
    arrangements: Sequence[Arrangement], wav_paths: Sequence[Path], scorer: Scorer
) -> int:
    """Index of the best of ``arrangements``, rendered to ``wav_paths`` and
    scored by ``scorer`` (see :func:`choose_by_audio`)."""
    return choose_by_audio(arrangements, scorer.score(wav_paths))
