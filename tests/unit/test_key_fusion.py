from unittest import mock

import numpy as np

from music_decoder.key import estimate_key
from music_decoder.key.fusion import (
    chord_scores,
    cnn_scores,
    fuse,
    key_index,
)
from music_decoder.types import ChordSegment


def _seg(s: float, e: float, root: str, quality: str) -> ChordSegment:
    return ChordSegment(start_s=s, end_s=e, root=root, quality=quality, confidence=1.0)


def _chroma(pcs: tuple[int, ...]) -> np.ndarray:
    c = np.zeros((12, 8))
    c[list(pcs), :] = 1.0
    return c


def test_key_index_round_trip() -> None:
    assert key_index("C", "major") == 0
    assert key_index("A", "minor") == 19


def test_cnn_scores_reorders_madmom_labels() -> None:
    probs = np.full(24, 1e-3)
    probs[3] = 0.9  # madmom index 3 = "C major"
    assert int(np.argmax(cnn_scores(probs))) == key_index("C", "major")


def test_chord_scores_prefer_the_key_of_a_I_IV_V_progression() -> None:
    segs = [
        _seg(0, 2, "G", "maj"),
        _seg(2, 4, "C", "maj"),
        _seg(4, 6, "D", "7"),
        _seg(6, 8, "G", "maj"),
    ]
    scores = chord_scores(segs)
    assert scores is not None
    assert int(np.argmax(scores)) == key_index("G", "major")


def test_chord_scores_none_without_chords() -> None:
    assert chord_scores([_seg(0, 4, "N", "")]) is None


def test_fuse_sums_zscores_and_ignores_flat_sources() -> None:
    a = np.zeros(24)
    a[key_index("D", "major")] = 1.0
    b = np.zeros(24)
    b[key_index("D", "major")] = 0.5
    b[key_index("A", "major")] = 0.6
    flat = np.ones(24)
    k = fuse([a, b, flat])
    assert (k.tonic, k.mode, k.profile) == ("D", "major", "fusion")
    assert k.margin > 0


def test_estimate_key_fusion_uses_chords_to_break_profile_ambiguity() -> None:
    # C-major-triad chroma is ambiguous between C major and relatives; a
    # G-C-D-G progression should pull the fused estimate to G major.
    segs = [
        _seg(0, 2, "G", "maj"),
        _seg(2, 4, "C", "maj"),
        _seg(4, 6, "D", "maj"),
        _seg(6, 8, "G", "maj"),
    ]
    chroma = _chroma((7, 11, 2, 0, 4, 9, 6))  # G major scale
    with mock.patch("music_decoder.key._resolve_backend", return_value="fusion"):
        k = estimate_key(chroma, chords=segs)
    assert (k.tonic, k.mode) == ("G", "major")
    assert k.profile == "fusion"


def test_estimate_key_fusion_survives_cnn_failure() -> None:
    chroma = _chroma((0, 4, 7))
    with (
        mock.patch("music_decoder.key._resolve_backend", return_value="fusion"),
        mock.patch("music_decoder.key.cnn.cnn_probabilities", side_effect=RuntimeError("x")),
    ):
        k = estimate_key(chroma, samples=np.zeros(22050, np.float32), sr=22050)
    assert (k.tonic, k.mode) == ("C", "major")
