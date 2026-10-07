"""One-time compatibility shims that must be imported BEFORE any madmom import.

madmom 0.16.1 (last release: 2018) has two known incompatibilities with
modern Python + NumPy that the project hasn't shipped a fix for:

1. ``collections.MutableSequence`` and friends moved to ``collections.abc``
   in Python 3.10. madmom imports them from the old location.
2. ``np.float`` / ``np.int`` / ``np.bool`` aliases were removed in NumPy 1.20.
   madmom uses them in dtype declarations.

We patch the affected attributes before any madmom module is imported. The
shim is idempotent. Importing this module is the public API; the function
is also exposed for tests.
"""

from __future__ import annotations

import collections
import collections.abc as _cabc

import numpy as np

_COLLECTION_ALIASES = (
    "MutableSequence",
    "MutableMapping",
    "Mapping",
    "Sequence",
    "Iterable",
    "Callable",
)
_NUMPY_ALIASES: tuple[tuple[str, type], ...] = (
    ("float", float),
    ("int", int),
    ("bool", bool),
)


def apply_madmom_shims() -> None:
    """Apply the compatibility patches. Safe to call multiple times."""
    for name in _COLLECTION_ALIASES:
        if not hasattr(collections, name) and hasattr(_cabc, name):
            setattr(collections, name, getattr(_cabc, name))
    for name, repl in _NUMPY_ALIASES:
        if not hasattr(np, name):
            setattr(np, name, repl)


class _RaggedTolerantNumpy:
    """``numpy`` stand-in for one madmom module whose ``asarray`` accepts ragged input.

    ``DBNDownBeatTrackingProcessor.process`` (madmom 0.16.1) does
    ``np.asarray(results)[:, 1]`` on a list of ``(path_array, log_prob)``
    tuples; NumPy >= 1.24 refuses to build that ragged array. Falling back to
    an object array keeps madmom's own selection logic intact.
    """

    def __getattr__(self, name: str) -> object:
        return getattr(np, name)

    @staticmethod
    def asarray(a: object, *args: object, **kwargs: object) -> object:
        try:
            return np.asarray(a, *args, **kwargs)  # type: ignore[call-overload]
        except ValueError:
            return np.array(a, dtype=object)


def patch_downbeats_numpy() -> None:
    """Make ``madmom.features.downbeats`` tolerate ragged ``asarray`` calls. Idempotent.

    Must be called after ``apply_madmom_shims()`` (i.e. after importing this module).
    """
    import madmom.features.downbeats as downbeats

    if not isinstance(downbeats.np, _RaggedTolerantNumpy):
        downbeats.np = _RaggedTolerantNumpy()


# Apply on import so callers can ``import madmom_compat`` and then ``import madmom``.
apply_madmom_shims()
