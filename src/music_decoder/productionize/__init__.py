"""Release-quality production of composer output.

``productionize()`` takes what ``compose()`` wrote and renders it as a
boom-bap instrumental: a real multi-mic drum kit, a synthesized bass, a
sampled grand piano and soundfont strings, mixed, mastered to a loudness
target and checked against acceptance tests.
"""

from music_decoder.productionize.api import Production, productionize
from music_decoder.productionize.config import ProductionConfig
from music_decoder.productionize.qc import QcReport
from music_decoder.productionize.song import (
    ProductionError,
    Song,
    hisaab_song,
    song_from_composition,
    song_from_midi,
)

__all__ = [
    "Production",
    "ProductionConfig",
    "ProductionError",
    "QcReport",
    "Song",
    "hisaab_song",
    "productionize",
    "song_from_composition",
    "song_from_midi",
]
