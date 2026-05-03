from __future__ import annotations

import re

from music_decoder.pipeline.contracts import TabPosition

_LINE_RE = re.compile(r"^[eEBGDADCAGB][^\|]*\|(.*)\|", re.MULTILINE)
_STRING_LETTERS_TO_INDEX = {
    "E": 0, "A": 1, "D": 2, "G": 3, "B": 4, "e": 5,
}


def parse_ascii_tab(text: str) -> list[TabPosition]:
    """Best-effort ASCII-tab parser for standard tuning.

    Returns positions where digits appear in the column content. Handles
    multi-line standard layouts like:
        e|--0--3--|
        B|--1--0--|
        ...
    """
    string_lines: dict[int, str] = {}
    for line in text.splitlines():
        if not line:
            continue
        head = line[:2]
        if head[0] not in _STRING_LETTERS_TO_INDEX:
            continue
        string_idx = _STRING_LETTERS_TO_INDEX[head[0]]
        # Take the substring after the first '|'
        bar_idx = line.find("|")
        if bar_idx < 0:
            continue
        string_lines[string_idx] = line[bar_idx + 1:]

    if not string_lines:
        return []
    positions: list[TabPosition] = []
    width = min(len(s) for s in string_lines.values())
    for col in range(width):
        for s_idx, content in string_lines.items():
            ch = content[col]
            if ch.isdigit():
                # Multi-digit frets: greedy consume
                j = col
                num = ""
                while j < len(content) and content[j].isdigit():
                    num += content[j]
                    j += 1
                positions.append(TabPosition(string=s_idx, fret=int(num)))
    return positions
