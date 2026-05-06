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


# Apply on import so callers can ``import madmom_compat`` and then ``import madmom``.
apply_madmom_shims()
