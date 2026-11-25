"""Train the phrase model on POP909 and save the weights file (torch needed)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from music_decoder.compose.phrase_model import PhraseMeta, init_arrays, make_net, save_phrase_model
from music_decoder.compose.phrase_tokens import (
    CONDITION_KEYS,
    CONTEXT_SLOTS,
    SLOTS_PER_BAR,
    Encoded,
    eighth_only,
    encode_song,
    figure_hashes,
    phrase_starts,
    song_relative_pitches,
)
from music_decoder.compose.tune_corpus import Song, read_song

SPLIT_SEED = 909
COVER = 0.995
BATCH = 32
LR = 1e-3
DROPOUT = 0.1
EVAL_EVERY = 250
WINDOW_BARS = 8
SOURCES = [
    "POP909 (Wang, Chen, Jiang, Dai, Xia et al., ISMIR 2020), "
    "cleaned release pop909-hsa by Shuqi Dai"
]
LICENCE = "MIT"

Span = tuple[npt.NDArray[np.int64], dict[str, npt.NDArray[np.int64]]]


@dataclass(frozen=True)
class _Piece:
    song: Song
    enc: Encoded
    starts: list[int]


def split(names: Sequence[str], seed: int) -> dict[str, list[str]]:
    """80/10/10 train/valid/held-out by song, in a seeded order."""
    order = np.random.default_rng(seed).permutation(len(names))
    n_tr, n_va = int(0.8 * len(names)), int(0.1 * len(names))

    def pick(idx: npt.NDArray[np.int64]) -> list[str]:
        return [names[int(i)] for i in idx]

    return {
        "train": pick(order[:n_tr]),
        "valid": pick(order[n_tr : n_tr + n_va]),
        "heldout": pick(order[n_tr + n_va :]),
    }


def pitch_range(values: npt.NDArray[np.int64], cover: float = COVER) -> tuple[int, int]:
    """The narrowest [lo, hi] holding at least ``cover`` of ``values``."""
    vals = np.sort(np.asarray(values))
    need = math.ceil(cover * len(vals))
    widths = vals[need - 1 :] - vals[: len(vals) - need + 1]
    i = int(np.argmin(widths))
    return int(vals[i]), int(vals[i + need - 1])


def _songs(root: Path) -> dict[str, Song]:
    out = {}
    for d in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        song = read_song(d)
        if song is not None and song.notes:
            out[d.name] = song
    return out


def _spans(piece: _Piece, overlap: bool) -> list[Span]:
    """Spans of up to 32 bars from capital-letter phrase starts, with no out-of-range note."""
    tok, out, free = piece.enc.tokens, [], 0
    for b in piece.starts:
        if not overlap and b < free:
            continue
        a, e = b * SLOTS_PER_BAR, min(b * SLOTS_PER_BAR + CONTEXT_SLOTS, len(tok))
        if e - a < SLOTS_PER_BAR:
            continue
        if any(
            a // SLOTS_PER_BAR <= x < -(-e // SLOTS_PER_BAR) for x in piece.enc.out_of_range_bars
        ):
            continue
        out.append((tok[a:e], {k: v[a:e] for k, v in piece.enc.cond.items()}))
        free = e // SLOTS_PER_BAR
    return out


def _windows(piece: _Piece) -> list[tuple[int, Span]]:
    """Held-out 8-bar windows at phrase starts: eighth-only, in range."""
    out = []
    for b in piece.starts:
        if not eighth_only(piece.enc, b, WINDOW_BARS):
            continue
        if any(b <= x < b + WINDOW_BARS for x in piece.enc.out_of_range_bars):
            continue
        sl = slice(b * SLOTS_PER_BAR, (b + WINDOW_BARS) * SLOTS_PER_BAR)
        out.append((b, (piece.enc.tokens[sl], {k: v[sl] for k, v in piece.enc.cond.items()})))
    return out


def _batch(items: Sequence[Span], vocab: int, device: Any) -> tuple[Any, dict[str, Any], Any]:
    import torch

    t = max(len(tok) for tok, _ in items)
    prev = torch.full((len(items), t), vocab, dtype=torch.long)
    tgt = torch.full((len(items), t), -100, dtype=torch.long)
    cond = {k: torch.zeros((len(items), t), dtype=torch.long) for k in CONDITION_KEYS}
    for i, (tok, c) in enumerate(items):
        n = len(tok)
        tgt[i, :n] = torch.from_numpy(tok)
        prev[i, 1:n] = torch.from_numpy(tok[:-1])
        for k in CONDITION_KEYS:
            cond[k][i, :n] = torch.from_numpy(c[k])
    return prev.to(device), {k: v.to(device) for k, v in cond.items()}, tgt.to(device)


def _bits(net: Any, items: Sequence[Span], vocab: int, device: Any) -> tuple[float, list[float]]:
    """(bits per bar over all items, each item's bits per bar)."""
    import torch
    import torch.nn.functional as F

    net.eval()
    total, slots, each = 0.0, 0, []
    with torch.no_grad():
        for i in range(0, len(items), 64):
            prev, cond, tgt = _batch(items[i : i + 64], vocab, device)
            logp = F.log_softmax(net(prev, cond).float(), -1)
            m = tgt != -100
            b = -logp.gather(-1, tgt.clamp(min=0)[..., None])[..., 0] * m / math.log(2)
            total += float(b.sum())
            slots += int(m.sum())
            each += (b.sum(1) / (m.sum(1) / SLOTS_PER_BAR)).cpu().tolist()
    return total / max(slots / SLOTS_PER_BAR, 1e-9), each


def _train(
    arch: Mapping[str, int],
    train: list[Span],
    valid: list[Span],
    vocab: int,
    steps: int,
    seed: int,
    device: Any,
) -> Any:
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = make_net(arch, vocab, DROPOUT).to(device)
    net.load_state_dict({k: torch.from_numpy(v) for k, v in init_arrays(arch, vocab, seed).items()})
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.05)
    prev_all, cond_all, tgt_all = _batch(train, vocab, device)
    best, best_state = math.inf, None
    for step in range(1, steps + 1):
        net.train()
        i = torch.as_tensor(rng.integers(len(train), size=BATCH), device=device)
        logits = net(prev_all[i], {k: v[i] for k, v in cond_all.items()})
        loss: Any = F.cross_entropy(
            logits.reshape(-1, vocab), tgt_all[i].reshape(-1), ignore_index=-100
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % EVAL_EVERY == 0 or step == steps:
            v, _ = _bits(net, valid, vocab, device)
            print(f"step {step} loss {loss.item():.3f} valid {v:.3f} bits/bar", flush=True)
            if v < best:
                best = v
                best_state = {k: x.detach().cpu().clone() for k, x in net.state_dict().items()}
    net.load_state_dict(best_state)
    return net


def build(
    root: Path,
    out: Path,
    *,
    arch: Mapping[str, int],
    steps: int,
    seed: int,
    temperature: float,
    nucleus: float,
    figure_length: int,
    device: str | None = None,
) -> PhraseMeta:
    """Split, measure the pitch range, train, measure, and save the weights file."""
    import torch

    dev = torch.device(device or ("mps" if torch.backends.mps.is_available() else "cpu"))
    songs = _songs(root)
    parts = split(sorted(songs), SPLIT_SEED)
    lo, hi = pitch_range(np.concatenate([song_relative_pitches(songs[n]) for n in parts["train"]]))
    vocab = hi - lo + 2
    pieces = {n: _Piece(s, encode_song(s, lo, hi), phrase_starts(s)) for n, s in songs.items()}
    train = [x for n in parts["train"] for x in _spans(pieces[n], overlap=True)]
    valid = [x for n in parts["valid"] for x in _spans(pieces[n], overlap=False)] or train
    held = [x for n in parts["heldout"] for x in _spans(pieces[n], overlap=False)] or valid
    net = _train(dict(arch), train, valid, vocab, steps, seed, dev)
    heldout_bits, _ = _bits(net, held, vocab, dev)
    by_mode: dict[str, list[float]] = {"major": [], "minor": []}
    for n in parts["heldout"]:
        wins = _windows(pieces[n])
        if wins:
            _, each = _bits(net, [w for _, w in wins], vocab, dev)
            for (_, (_, cond)), bits in zip(wins, each, strict=True):
                by_mode["minor" if cond["mode"][0] else "major"].append(bits)
    surprise = {
        m: (float(np.median(v)), float(np.subtract(*np.percentile(v, [75, 25]))))
        for m, v in by_mode.items()
        if v
    }
    figures = np.concatenate(
        [
            figure_hashes(
                [x.onset for x in songs[n].notes], [x.pitch for x in songs[n].notes], figure_length
            )
            for n in parts["train"]
        ]
    )
    meta = PhraseMeta(
        sources=list(SOURCES),
        licence=LICENCE,
        split_seed=SPLIT_SEED,
        pitch_lo=lo,
        pitch_hi=hi,
        arch=dict(arch),
        heldout_bits=float(heldout_bits),
        surprise=surprise,
        figure_length=figure_length,
        temperature=temperature,
        nucleus=nucleus,
    )
    arrays = {k: v.detach().cpu().float().numpy() for k, v in net.state_dict().items()}
    save_phrase_model(Path(out), arrays, meta, figures)
    return meta
