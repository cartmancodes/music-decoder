"""The opt-in audio re-rank of fingerstyle candidates (v5 spec 5.4).

The top candidates by symbolic score are rendered and scored by Audiobox
Aesthetics (Meta, ``facebook/audiobox-aesthetics``, CC-BY-4.0), which rates
a clip on four axes from 1 to 10: content enjoyment ``CE``, content
usefulness ``CU``, production complexity ``PC`` and production quality
``PQ``. A candidate's total is ``x / AUDIO_SCALE + SYMBOLIC_WEIGHT * s /
SYMBOLIC_SCALE``, with ``x = (CE + PQ) / 2`` and ``s`` the symbolic score.
Both scales are fixed constants measured on a reference set, so a
candidate's total does not depend on which others are in the shortlist: a
candidate worse than the rest on both signals cannot change the order of the
others (a per-shortlist min-max did). The audio model's raw spread between
candidates of one piece is small (about 0.05-0.3 on its 1-10 scale), so it
is scaled up, else the symbolic order would always win; ``AUDIO_SCALE`` is
at least ``AUDIO_SPAN_FLOOR``, so a gap under the model's resolution cannot
overturn the symbolic order by itself (v5 ruling F4e).

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
# Weight of the scaled symbolic score next to the scaled audio score: half, so
# the audio model decides unless the symbolic gap is the larger one.
SYMBOLIC_WEIGHT = 0.5
# The smallest audio scale (on the model's 1-10 scale): a smaller gap between
# the candidates counts only in proportion (v5 ruling F4e).
AUDIO_SPAN_FLOOR = 0.3
# Fixed scales for the re-rank, measured on 30 reference prompts: for each,
# the top RANK_AUDIO_TOP_K = 4 of 12 melody_engine="tune" fingerstyle
# candidates (the piece seed below, humanize seed the same) were rendered with
# the soundfont and scored. Median span of x = (CE + PQ) / 2 is 0.148 (so
# max(span, 0.3) has median 0.3); median span of the symbolic score is 0.995.
# AUDIO_SCALE = max(median(max(span_x, 0.3)), AUDIO_SPAN_FLOOR) and
# SYMBOLIC_SCALE = median(span_s), each to 3 significant figures: the spans
# the old per-shortlist min-max typically divided by. To re-measure, rebuild
# these prompts and repeat. Reference prompts, one per line:
# id scale     progression            bars tempo tuning     seed
#  0 D:minor  Dm A# F C              4 110.0 Eb         10000
#  1 C:major  C G F G                2 110.0 EADGBE     10001
#  2 E:major  Emaj7 C#m7 F#m7 B7     2 126.0 EADGBE     10002
#  3 A:minor  Am G F G               2  98.0 Eb         10003
#  4 A:major  A E D E                4  86.0 EADGBE     10004
#  5 B:major  G#m E B F#             2 108.0 EADGBE     10005
#  6 D#:major D# Cm G# A# D# Cm Fm A# 2 118.0 EADGBE     10006
#  7 B:minor  Bm Em F#7 Bm           2 118.0 EADGBE     10007
#  8 G#:major G# D# Fm Cm C# G# C# D# 2  66.0 EADGBE     10008
#  9 E:major  E C#m A B              2 104.0 EADGBE     10009
# 10 E:minor  C D Em Em              2  68.0 EADGBE     10010
# 11 E:minor  Em Am B7 Em            2 102.0 EADGBE     10011
# 12 B:minor  Bm Em A D              2 114.0 EADGBE     10012
# 13 C#:minor C#m E B F#m            4  98.0 EADGBE     10013
# 14 B:major  B G#m E F# B G#m C#m F# 2 116.0 EADGBE     10014
# 15 A#:minor A#m D#m G# C# F# D#m F7 A#m 2  78.0 EADGBE     10015
# 16 D:minor  Dm7 Gm7 C Fmaj7        4 126.0 Eb         10016
# 17 A#:minor A#m7 D#m7 G# C#maj7    2 100.0 EADGBE     10017
# 18 C#:major C# G# A#m F#           2  88.0 EADGBE     10018
# 19 D:minor  Dm C A# C              2  80.0 Drop_D     10019
# 20 G#:minor G#m E B F# G#m E C#m D# 2  68.0 EADGBE     10020
# 21 D:major  D A Bm F#m G D G A     2  76.0 EADGBE     10021
# 22 F:major  F Dm Gm C              2 126.0 EADGBE     10022
# 23 F:major  A# C Am Dm             2  82.0 EADGBE     10023
# 24 B:major  B G#m E F# B G#m C#m F# 2  90.0 EADGBE     10024
# 25 B:major  G#m E B F#             4  82.0 EADGBE     10025
# 26 E:minor  Em G D Am              2  96.0 Eb         10026
# 27 A:major  A C#m D E              4 104.0 D_standard 10027
# 28 G:major  Cmaj7 D7 Bm7 Em7       4 122.0 EADGBE     10028
# 29 G#:minor E F# G#m G#m           2  98.0 EADGBE     10029
AUDIO_SCALE = 0.3
SYMBOLIC_SCALE = 0.995
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
    ``x / AUDIO_SCALE + SYMBOLIC_WEIGHT * s / SYMBOLIC_SCALE`` with
    ``x = (CE + PQ) / 2`` and ``s`` the symbolic score. The scales are fixed,
    so a candidate's total does not depend on the others. Ties go to the
    lowest index."""
    if len(arrangements) != len(audio_scores) or not arrangements:
        raise ValueError("need one audio score per arrangement, and at least one")
    totals: list[float] = []
    for a, row in zip(arrangements, audio_scores, strict=True):
        if a.score is None:
            raise ValueError("the audio re-rank needs scored (fingerstyle) arrangements")
        x = (row["CE"] + row["PQ"]) / 2
        totals.append(x / AUDIO_SCALE + SYMBOLIC_WEIGHT * a.score / SYMBOLIC_SCALE)
    return max(range(len(totals)), key=totals.__getitem__)  # max() keeps the first best


def rank_by_audio(
    arrangements: Sequence[Arrangement], wav_paths: Sequence[Path], scorer: Scorer
) -> int:
    """Index of the best of ``arrangements``, rendered to ``wav_paths`` and
    scored by ``scorer`` (see :func:`choose_by_audio`)."""
    return choose_by_audio(arrangements, scorer.score(wav_paths))
