"""Release-quality production of composer output.

``productionize()`` takes what ``compose()`` wrote and renders it as a
mastered instrumental in one of two styles, chosen by
``ProductionConfig.style``:

- ``"boom-bap"`` (the default): a real multi-mic drum kit, a synthesized
  bass, a sampled grand piano and soundfont strings.
- ``"pop-rap"``: synthesized trap drums, an 808, a pad, a pluck and a guitar
  or synth lead, all generated, with no samples to fetch.

Either is mixed, mastered to a loudness target and checked against
acceptance tests.
"""

from music_decoder.productionize.api import Production, productionize
from music_decoder.productionize.config import PopRapConfig, ProductionConfig
from music_decoder.productionize.qc import QcReport
from music_decoder.productionize.song import (
    ProductionError,
    Song,
    hisaab_song,
    song_from_composition,
    song_from_midi,
)

__all__ = [
    "PopRapConfig",
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
