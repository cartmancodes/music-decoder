"""One-time compatibility shims for basic-pitch's training pipeline.

basic-pitch (Spotify, last shipped train.py update 2024) has been tested
with TensorFlow 2.12. We're on 2.21. Two ``input_shape.rank`` accesses in
``basic_pitch.layers.signal`` fail because newer Keras passes a plain
tuple to ``Layer.build()`` instead of a ``TensorShape``.

This module monkey-patches the affected methods at import time so the
training pipeline runs without modifying upstream files. Idempotent.

The shim is a tiny defensive wrapper: ``getattr(input_shape, "rank", None)
or len(input_shape)``. Functional equivalent on both old and new Keras.
"""
from __future__ import annotations

import logging
from typing import Any

_log = logging.getLogger(__name__)
_PATCHED = False


def _shape_rank(shape: Any) -> int:
    """Return the rank of a TensorShape OR a plain tuple/list."""
    rank = getattr(shape, "rank", None)
    if rank is not None:
        return int(rank)
    return len(shape)


def apply_basic_pitch_shims() -> None:
    """Patch basic_pitch.layers.signal to handle tuple ``input_shape``.

    Safe to call multiple times. Skips silently if basic-pitch isn't installed.
    """
    global _PATCHED
    if _PATCHED:
        return
    try:
        from basic_pitch.layers import signal as bp_signal
    except ImportError:
        _log.warning("basic_pitch not installed; skipping training shims")
        return

    # Patch NormalizedLog.build (line 164: rank = input_shape.rank).
    if hasattr(bp_signal, "NormalizedLog"):
        def patched_normlog_build(self: Any, input_shape: Any) -> None:
            import tensorflow as tf

            self.squeeze_batch = lambda batch: batch
            rank = _shape_rank(input_shape)
            if rank == 4:
                assert input_shape[1] == 1, (
                    "If the rank is 4, the second dimension must be length 1"
                )
                self.squeeze_batch = lambda batch: tf.squeeze(
                    batch, axis=1,
                )
            else:
                assert rank == 3, (
                    f"Only ranks 3 and 4 are supported. Received rank {rank} "
                    f"for {input_shape}."
                )

        bp_signal.NormalizedLog.build = patched_normlog_build

    # Patch Stft.build (line 83: input_shape.rank - 1 inside a lambda's setup).
    # The lambda is constructed during build(); we replace the build() entirely.
    if hasattr(bp_signal, "Stft"):
        original_stft_build = bp_signal.Stft.build

        def patched_stft_build(self: Any, input_shape: Any) -> None:
            import tensorflow as tf

            # Coerce input_shape to expose .rank-equivalent before the original
            # build runs. The cleanest path: temporarily wrap input_shape in
            # tf.TensorShape if it's a tuple/list.
            if not hasattr(input_shape, "rank"):
                wrapped = tf.TensorShape(input_shape)
            else:
                wrapped = input_shape
            original_stft_build(self, wrapped)

        bp_signal.Stft.build = patched_stft_build

    _PATCHED = True
    _log.info("basic_pitch_compat: applied training shims")


# Apply on import so callers can ``import basic_pitch_compat`` and then use
# ``from basic_pitch import train``.
apply_basic_pitch_shims()
