"""The phrase model: a small causal transformer over slot tokens (see phrase_tokens).

The weights live in one numpy file (no pickle). torch is imported only inside the
functions that build or run the network, so importing this module, the compose API or
the CLI never loads it.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from music_decoder.compose.phrase_tokens import (
    CONDITION_KEYS,
    CONDITION_SIZES,
    CONTEXT_SLOTS,
    Encoded,
    figure_hashes,
)

WEIGHTS_PATH = Path(__file__).parent / "data" / "phrase_model.npz"
_PREFIX = "w."


class PhraseModelError(RuntimeError):
    """The phrase model cannot be loaded or used."""


@dataclass(frozen=True)
class PhraseMeta:
    sources: list[str]
    licence: str
    split_seed: int
    pitch_lo: int
    pitch_hi: int
    arch: dict[str, int]
    heldout_bits: float
    surprise: dict[str, tuple[float, float]]  # mode -> (median, IQR) of bits per bar
    figure_length: int
    temperature: float
    nucleus: float

    @property
    def vocab(self) -> int:
        return self.pitch_hi - self.pitch_lo + 2

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> PhraseMeta:
        raw = json.loads(text)
        raw["surprise"] = {k: (float(v[0]), float(v[1])) for k, v in raw["surprise"].items()}
        return cls(**raw)


@cache
def _torch_net_class() -> Any:
    import torch
    import torch.nn.functional as F
    from torch import nn

    class Block(nn.Module):
        def __init__(self, d: int, heads: int, drop: float) -> None:
            super().__init__()
            self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
            self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
            self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
            self.heads, self.drop = heads, drop

        def split(self, x: Any) -> Any:
            b, t, d = x.shape
            q, k, v = self.qkv(self.n1(x)).view(b, t, 3, self.heads, d // self.heads).unbind(2)
            return tuple(z.transpose(1, 2) for z in (q, k, v))

        def finish(self, x: Any, a: Any) -> Any:
            b, t, d = x.shape
            p = self.drop if self.training else 0.0  # residual dropout only
            x = x + F.dropout(self.proj(a.transpose(1, 2).reshape(b, t, d)), p)
            return x + F.dropout(self.ff(self.n2(x)), p)

        def forward(self, x: Any) -> Any:
            q, k, v = self.split(x)
            return self.finish(x, F.scaled_dot_product_attention(q, k, v, is_causal=True))

    class PhraseNet(nn.Module):
        def __init__(self, vocab: int, d: int, layers: int, heads: int, drop: float = 0.0) -> None:
            super().__init__()
            self.vocab = vocab
            self.tok = nn.Embedding(vocab + 1, d)  # + BOS
            self.cond = nn.ModuleDict(
                {k: nn.Embedding(CONDITION_SIZES[k], d) for k in CONDITION_KEYS}
            )
            self.pos = nn.Embedding(CONTEXT_SLOTS, d)
            self.blocks = nn.ModuleList(Block(d, heads, drop) for _ in range(layers))
            self.norm, self.head = nn.LayerNorm(d), nn.Linear(d, vocab)

        def embed(self, prev: Any, cond: Any, start: int = 0) -> Any:
            t = prev.shape[1]
            x = self.tok(prev) + self.pos(torch.arange(start, start + t, device=prev.device))[None]
            for k in CONDITION_KEYS:
                x = x + self.cond[k](cond[k])
            return x

        def forward(self, prev: Any, cond: Any) -> Any:
            x = self.embed(prev, cond)
            for blk in self.blocks:
                x = blk(x)
            return self.head(self.norm(x))

    return PhraseNet


def make_net(arch: Mapping[str, int], vocab: int, drop: float = 0.0) -> Any:
    """A fresh torch network (``arch`` has d, layers, heads)."""
    return _torch_net_class()(vocab, arch["d"], arch["layers"], arch["heads"], drop)


def init_arrays(
    arch: Mapping[str, int], vocab: int, seed: int
) -> dict[str, npt.NDArray[np.float32]]:
    import torch

    torch.manual_seed(seed)
    net = make_net(arch, vocab)
    return {k: v.detach().cpu().numpy().astype(np.float32) for k, v in net.state_dict().items()}


def save_phrase_model(
    path: Path,
    arrays: Mapping[str, npt.NDArray[Any]],
    meta: PhraseMeta,
    figures: npt.NDArray[np.uint32],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {_PREFIX + k: np.asarray(v, dtype=np.float16) for k, v in arrays.items()}
    data["meta"] = np.array([meta.to_json()])
    data["figures"] = np.unique(np.asarray(figures, dtype=np.uint32))
    with open(path, "wb") as fh:
        np.savez(fh, **data)


class PhraseSampler:
    """Next-slot probabilities with cached keys and values: one forward step per slot."""

    def __init__(self, model: PhraseModel, cond: Mapping[str, npt.NDArray[np.int64]]) -> None:
        import torch

        self._m = model
        self._cond = {
            k: torch.from_numpy(np.asarray(cond[k], dtype=np.int64)) for k in CONDITION_KEYS
        }
        self._limit = min(len(self._cond["slot"]), CONTEXT_SLOTS)
        self._k: list[Any] = [None] * len(model.net.blocks)
        self._v: list[Any] = [None] * len(model.net.blocks)
        self._t = 0
        self._prev = model.vocab  # BOS
        self._probs: npt.NDArray[np.float64] | None = None

    def probs(self) -> npt.NDArray[np.float64]:
        if self._probs is None:
            self._probs = self._step()
        return self._probs

    def push(self, token: int) -> None:
        self.probs()
        self._t += 1
        self._prev = int(token)
        self._probs = None

    def _step(self) -> npt.NDArray[np.float64]:
        import torch
        import torch.nn.functional as F

        t = self._t
        if t >= self._limit:
            raise ValueError(f"phrase sampler is past its last slot ({self._limit})")
        net = self._m.net
        with torch.inference_mode():
            prev = torch.tensor([[self._prev]])
            x = net.embed(prev, {k: v[t : t + 1][None] for k, v in self._cond.items()}, start=t)
            for i, blk in enumerate(net.blocks):
                q, k, v = blk.split(x)
                self._k[i] = k if self._k[i] is None else torch.cat([self._k[i], k], 2)
                self._v[i] = v if self._v[i] is None else torch.cat([self._v[i], v], 2)
                x = blk.finish(x, F.scaled_dot_product_attention(q, self._k[i], self._v[i]))
            logp = F.log_softmax(net.head(net.norm(x))[0, 0], -1)
        probs: npt.NDArray[np.float64] = np.exp(logp.numpy().astype(np.float64))
        return probs


class PhraseModel:
    def __init__(
        self,
        path: Path,
        meta: PhraseMeta,
        net: Any,
        figures: npt.NDArray[np.uint32],
        weights_hash: str,
    ) -> None:
        self.path, self.meta, self.net = path, meta, net
        self.figures = figures
        self.weights_hash = weights_hash

    @property
    def vocab(self) -> int:
        return self.meta.vocab

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.net.parameters())

    def slot_bits(self, enc: Encoded) -> npt.NDArray[np.float64]:
        """Bits of each slot's token, read from a cold start at the first slot."""
        return self.slot_bits_many([enc])[0]

    def slot_bits_many(
        self, encs: Sequence[Encoded], batch: int = 64
    ) -> list[npt.NDArray[np.float64]]:
        """:meth:`slot_bits` of each piece, run in batches of pieces of equal length."""
        import torch
        import torch.nn.functional as F

        out: list[npt.NDArray[np.float64] | None] = [None] * len(encs)
        by_len: dict[int, list[int]] = {}
        for i, enc in enumerate(encs):
            n = len(enc.tokens)
            if n > CONTEXT_SLOTS:
                raise ValueError(f"{n} slots is longer than the model's context ({CONTEXT_SLOTS})")
            by_len.setdefault(n, []).append(i)
        for idx in by_len.values():
            for j in range(0, len(idx), batch):
                part = idx[j : j + batch]
                tok = torch.from_numpy(
                    np.stack([np.asarray(encs[i].tokens, dtype=np.int64) for i in part])
                )
                prev = torch.full_like(tok, self.vocab)
                prev[:, 1:] = tok[:, :-1]
                cond = {
                    k: torch.from_numpy(
                        np.stack([np.asarray(encs[i].cond[k], dtype=np.int64) for i in part])
                    )
                    for k in CONDITION_KEYS
                }
                with torch.inference_mode():
                    logp = F.log_softmax(self.net(prev, cond), -1)
                    bits = -logp.gather(-1, tok[..., None])[..., 0] / math.log(2)
                for row, i in zip(bits.numpy().astype(np.float64), part, strict=True):
                    out[i] = row
        return [x for x in out if x is not None]

    def sampler(self, cond: Mapping[str, npt.NDArray[np.int64]]) -> PhraseSampler:
        return PhraseSampler(self, cond)

    def has_figure(self, onsets: Sequence[int], pitches: Sequence[int]) -> bool:
        """True when some run of ``figure_length`` notes matches a training run."""
        hashes = figure_hashes(onsets, pitches, self.meta.figure_length)
        if not len(hashes) or not len(self.figures):
            return False
        at = np.searchsorted(self.figures, hashes).clip(max=len(self.figures) - 1)
        return bool((self.figures[at] == hashes).any())


_CACHE: dict[Path, PhraseModel] = {}


def load_phrase_model(path: Path | None = None) -> PhraseModel:
    """Load (once per path) the weights file; PhraseModelError names the path on failure."""
    path = Path(path or WEIGHTS_PATH).resolve()
    if path in _CACHE:
        return _CACHE[path]
    if not path.is_file():
        raise PhraseModelError(f"phrase model weights not found: {path}")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch is a core dependency
        raise PhraseModelError(
            f"the phrase model at {path} needs torch, which is not installed"
        ) from exc
    try:
        raw = path.read_bytes()
        with np.load(path, allow_pickle=False) as z:
            meta = PhraseMeta.from_json(str(z["meta"][0]))
            state = {
                k[len(_PREFIX) :]: torch.from_numpy(z[k].astype(np.float64))
                for k in z.files
                if k.startswith(_PREFIX)
            }
            figures = np.asarray(z["figures"], dtype=np.uint32)
        net = make_net(meta.arch, meta.vocab).double()
        net.load_state_dict(state)
        net.eval()
    except Exception as exc:
        raise PhraseModelError(f"phrase model weights at {path} are unreadable: {exc}") from exc
    model = PhraseModel(path, meta, net, figures, hashlib.sha256(raw).hexdigest()[:12])
    _CACHE[path] = model
    return model
