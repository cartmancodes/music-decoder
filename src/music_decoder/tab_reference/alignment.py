from __future__ import annotations

from collections.abc import Iterable

from music_decoder.pipeline.contracts import TabbedNote, TabPosition


def similarity_to_prediction(
    predicted: Iterable[TabbedNote],
    reference: Iterable[TabPosition],
) -> float:
    """Aligned 1:1 by order: matching counts when both string and fret match."""
    pred_list = [(t.position.string, t.position.fret) for t in predicted]
    ref_list = [(p.string, p.fret) for p in reference]
    n = max(len(pred_list), len(ref_list))
    if n == 0:
        return 1.0
    correct = sum(1 for p, r in zip(pred_list, ref_list, strict=False) if p == r)
    return correct / n
