"""Tests for the madmom compatibility shim."""

from __future__ import annotations

import collections

import numpy as np

from music_decoder.chords import madmom_compat


def test_apply_madmom_shims_idempotent():
    # Calling twice should not error or duplicate.
    madmom_compat.apply_madmom_shims()
    madmom_compat.apply_madmom_shims()
    assert hasattr(collections, "MutableSequence")


def test_collections_aliases_present():
    assert hasattr(collections, "MutableSequence")
    assert hasattr(collections, "MutableMapping")
    assert hasattr(collections, "Mapping")
    assert hasattr(collections, "Sequence")
    assert hasattr(collections, "Iterable")
    assert hasattr(collections, "Callable")


def test_numpy_aliases_are_builtins():
    # The shim sets np.float = float, etc. (builtins, not numpy types).
    # This is what madmom 0.16 expects.
    assert np.float is float  # type: ignore[attr-defined]
    assert np.int is int  # type: ignore[attr-defined]
    assert np.bool is bool  # type: ignore[attr-defined]


def test_importing_madmom_after_shim_works():
    # The actual smoke test: madmom must import without error after the shim.
    import madmom

    assert madmom.__version__ is not None
