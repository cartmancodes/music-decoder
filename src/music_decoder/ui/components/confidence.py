from __future__ import annotations


def confidence_color(value: float, *, high: float, medium: float) -> str:
    if value >= high:
        return "green"
    if value >= medium:
        return "yellow"
    return "red"
