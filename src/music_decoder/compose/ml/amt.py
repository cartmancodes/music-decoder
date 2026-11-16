"""The opt-in melody model for fingerstyle (v5 spec 5.5): the Anticipatory
Music Transformer proposes a melody over the thumb pattern, which the
generator takes as a soft pitch prior.

Not implemented yet: :func:`load_proposer` raises
:class:`~music_decoder.compose.ml.MLUnavailableError`.
"""

from __future__ import annotations

from music_decoder.compose.ml import MelodyProposer, MLUnavailableError


def load_proposer() -> MelodyProposer:
    """The AMT melody proposer on the CPU."""
    raise MLUnavailableError("the AMT melody model is not available yet")
