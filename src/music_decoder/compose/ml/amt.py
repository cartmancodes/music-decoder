"""The opt-in melody model for fingerstyle (v5 spec 5.5): the Anticipatory
Music Transformer proposes a melody over the thumb pattern, which the
generator takes as a soft pitch prior.

The model is Stanford CRFM's ``stanford-crfm/music-small-800k`` (Apache-2.0),
run through the ``anticipation`` package (Apache-2.0) at a pinned commit. The
thumb pattern goes in as AMT *controls* on instrument 25 (steel-string
guitar), as in the AMT README's accompaniment example: the model generates
events over them, anticipating each control a few seconds ahead. From what
it generates we keep a monophonic top line (see :func:`top_line`).

Given only a solo-guitar bass and no prompt, the model mostly answers with
silence (rest tokens) or drums, which leave no melody. So it samples under
three constraints on its next-token logits (:class:`_Constrained`), the
same kind ``anticipation``'s own ``safe_logits`` applies:

- rests and drum notes are masked, so every generated event is a pitched
  note;
- a note already generated at the same onset is masked, so the model cannot
  loop on one event without moving time on (it did, for thousands of
  events, once rests were masked);
- after :data:`EVENTS_PER_S` events per second of the piece, the next onset
  is forced past the end, which stops the generation: a bound on the run
  time.

The model runs on the CPU, in eval mode, under ``torch.random.fork_rng()``
seeded with the piece's seed, so the same seed gives the same proposal and
the caller's RNG state is untouched. ``torch``, ``transformers`` and
``anticipation`` are imported only by :func:`load_proposer`,
:meth:`AMTProposer.propose`, :func:`encode_controls` and
:func:`decode_events`.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from functools import partial
from typing import Any

from music_decoder.compose.ml import MelodyProposer, MLUnavailableError
from music_decoder.types import Note

# General MIDI program 25 (0-based): acoustic guitar (steel).
INSTRUMENT = 25
DRUMS = 128  # AMT's instrument code for General MIDI channel 10
# The pitches a proposal may use: G3 to C6, around the melody's reach.
LOW_PITCH, HIGH_PITCH = 55, 84
# Onsets this close to the first onset of a group are one onset.
ONSET_GROUP_S = 0.03
TOP_P = 0.95  # nucleus sampling
# The most events the model may generate per second of the piece (an eighth
# note at 120 bpm is 4 per second); past it the generation stops.
EVENTS_PER_S = 4.0
CONTEXT_VELOCITY = 72  # AMT ignores velocity; anticipation's default
_EPS_S = 1e-9


class AMTProposer:
    """The Anticipatory Music Transformer as a
    :class:`~music_decoder.compose.ml.MelodyProposer`. Build it with
    :func:`load_proposer`."""

    def __init__(self, model: Any) -> None:
        self._model = model

    def propose(
        self, context: Sequence[Note], *, end_s: float, tempo_bpm: float, seed: int
    ) -> list[Note]:
        """A melody over ``context`` up to ``end_s``: the top line of what the
        model generates with ``context`` as its controls. ``tempo_bpm`` is
        unused: AMT works in seconds. AMT's onsets reach 100 s, so a longer
        piece gets a proposal for its first 100 s only."""
        import math

        import torch
        from anticipation import sample
        from anticipation.config import MAX_TIME, TIME_RESOLUTION

        end_s = min(end_s, (MAX_TIME - 1) / TIME_RESOLUTION)
        model = _Constrained(self._model, max_events=math.ceil(EVENTS_PER_S * end_s))
        controls = encode_controls(context)
        with torch.random.fork_rng(devices=[]), _quiet(sample), torch.inference_mode():
            torch.manual_seed(seed)
            events = sample.generate(
                model, start_time=0, end_time=end_s, controls=controls, top_p=TOP_P
            )
        return top_line(decode_events(events))


def load_proposer() -> MelodyProposer:
    """The AMT melody proposer from the Hugging Face cache, on the CPU.

    Raises :class:`MLUnavailableError` when ``anticipation`` or
    ``transformers`` is not installed or the weights have not been fetched.
    Never downloads. The pickled checkpoint is loaded weights-only, so it can
    run no code.
    """
    try:
        import anticipation.sample  # noqa: F401
        import torch
        from transformers import AutoModelForCausalLM
        from transformers.utils import logging as _hf_logging
    except ImportError as e:
        raise MLUnavailableError(
            f"the AMT melody model needs anticipation and transformers ({e})"
        ) from e
    from music_decoder import assets

    folder = assets.amt_dir()
    if folder is None:
        raise MLUnavailableError("the AMT melody model weights are not downloaded")
    hf_logging: Any = _hf_logging  # partly untyped
    verbosity, bar = hf_logging.get_verbosity(), hf_logging.is_progress_bar_enabled()
    hf_logging.set_verbosity_error()  # its load report; missing keys are checked below
    hf_logging.disable_progress_bar()
    try:
        loaded: Any = AutoModelForCausalLM.from_pretrained(
            folder,
            local_files_only=True,
            use_safetensors=False,
            weights_only=True,
            output_loading_info=True,
        )
    finally:
        hf_logging.set_verbosity(verbosity)
        if bar:
            hf_logging.enable_progress_bar()
    model, info = loaded
    if info["missing_keys"] or info["mismatched_keys"]:
        raise MLUnavailableError(
            f"the AMT melody model weights do not fit the model ({info['missing_keys']}, "
            f"{info['mismatched_keys']})"
        )
    model.to(torch.device("cpu"))
    model.eval()
    return AMTProposer(model)


def encode_controls(context: Sequence[Note]) -> list[int]:
    """``context`` as AMT control tokens on :data:`INSTRUMENT`: the notes'
    events through ``anticipation.convert``, each token offset by
    ``CONTROL_OFFSET``."""
    from anticipation import convert
    from anticipation.config import TIME_RESOLUTION
    from anticipation.vocab import CONTROL_OFFSET

    compound: list[int] = []
    for n in sorted(context, key=lambda n: (n.start_s, n.pitch)):
        onset = round(TIME_RESOLUTION * n.start_s)
        duration = max(1, round(TIME_RESOLUTION * n.end_s) - onset)
        compound.extend([onset, duration, n.pitch, INSTRUMENT, CONTEXT_VELOCITY])
    if not compound:
        return []
    events: list[int] = convert.compound_to_events(compound)
    return [CONTROL_OFFSET + tok for tok in events]


def decode_events(events: Sequence[int]) -> list[tuple[int, Note]]:
    """AMT event tokens (no controls) as ``(instrument, note)`` pairs, in
    time order, through ``anticipation.convert``."""
    from anticipation import convert, ops
    from anticipation.config import TIME_RESOLUTION

    tokens = ops.unpad(list(events))
    if not tokens:
        return []
    compound: list[int] = convert.events_to_compound(tokens)
    out: list[tuple[int, Note]] = []
    for k in range(0, len(compound), 5):
        onset, duration, pitch, instrument, velocity = compound[k : k + 5]
        start = onset / TIME_RESOLUTION
        note = Note(
            start_s=start,
            end_s=start + duration / TIME_RESOLUTION,
            pitch=pitch,
            velocity=velocity,
            confidence=1.0,
        )
        out.append((instrument, note))
    out.sort(key=lambda pair: (pair[1].start_s, pair[1].pitch))
    return out


def top_line(decoded: Sequence[tuple[int, Note]]) -> list[Note]:
    """The monophonic top line of decoded ``(instrument, note)`` pairs.

    Drums and pitches outside :data:`LOW_PITCH`..:data:`HIGH_PITCH` are
    dropped. Onsets within :data:`ONSET_GROUP_S` of a group's first onset
    are one onset, of which the highest note is kept (the earliest on a
    tie). Each kept note is cut short at the next kept onset.
    """
    notes = sorted(
        (
            n
            for instrument, n in decoded
            if instrument != DRUMS and LOW_PITCH <= n.pitch <= HIGH_PITCH
        ),
        key=lambda n: n.start_s,
    )
    tops: list[Note] = []
    group_start: float | None = None
    for n in notes:
        if group_start is None or n.start_s - group_start > ONSET_GROUP_S + _EPS_S:
            group_start = n.start_s
            tops.append(n)
        elif n.pitch > tops[-1].pitch:
            tops[-1] = n
    ends = [nxt.start_s for nxt in tops[1:]] + [float("inf")]
    return [
        Note(
            start_s=n.start_s,
            end_s=min(n.end_s, end),
            pitch=n.pitch,
            velocity=n.velocity,
            confidence=n.confidence,
        )
        for n, end in zip(tops, ends, strict=False)
    ]


class _Constrained:
    """``model`` as ``anticipation.sample`` calls it, its next-token logits
    constrained (see the module docstring): no rests, no drums, no note
    twice at one onset, and at most ``max_events`` events.

    ``sample.add_token`` feeds ``[mode] + history + new tokens`` and samples
    an event's time, duration and note in turn; as in ``sample.safe_logits``,
    the slot is the last input position modulo 3 (0 time, 1 duration, 2
    note). Times within one input share one offset, so they compare.
    """

    def __init__(self, model: Any, *, max_events: int) -> None:
        self._model = model
        self._events_left = max_events

    @property
    def device(self) -> Any:
        return self._model.device

    def __call__(self, input_tokens: Any) -> Any:
        from anticipation.config import MAX_PITCH, MAX_TIME
        from anticipation.vocab import CONTROL_OFFSET, NOTE_OFFSET, REST, TIME_OFFSET

        out = self._model(input_tokens)
        logits = out.logits[0, -1]
        tokens: list[int] = input_tokens[0].tolist()
        slot = (len(tokens) - 1) % 3
        if slot == 0:
            if self._events_left <= 0:  # the latest onset there is: past any end
                logits[:] = -float("inf")
                logits[TIME_OFFSET + MAX_TIME - 1] = 0.0
            self._events_left -= 1
        elif slot == 2:
            drums = NOTE_OFFSET + DRUMS * MAX_PITCH
            logits[drums : drums + MAX_PITCH] = -float("inf")
            logits[REST] = -float("inf")
            onset = tokens[-2]
            k = len(tokens) - 5  # the event before this one; events come in time order
            while k >= 1 and (tokens[k] == onset or tokens[k + 2] >= CONTROL_OFFSET):
                if tokens[k] == onset:
                    logits[tokens[k + 2]] = -float("inf")
                k -= 3
        return out


@contextmanager
def _quiet(sample: Any) -> Iterator[None]:
    """``anticipation.sample.generate`` without its tqdm progress bar."""
    from tqdm import tqdm

    original = sample.tqdm
    sample.tqdm = partial(tqdm, disable=True)
    try:
        yield
    finally:
        sample.tqdm = original
