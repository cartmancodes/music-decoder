from music_decoder.tabs.render import render_ascii_tab
from music_decoder.types import Note, TabPosition, TabbedNote


def _tabbed(t, p, string, fret):
    return TabbedNote(
        note=Note(start_s=t, end_s=t + 0.25, pitch=p, velocity=80, confidence=1.0),
        position=TabPosition(string=string, fret=fret),
    )


def test_render_ascii_tab_simple():
    notes = [_tabbed(0.0, 64, 5, 0), _tabbed(0.5, 67, 4, 5)]
    out = render_ascii_tab(notes, num_strings=6)
    assert "e|" in out or "E|" in out
    assert "0" in out
    assert "5" in out
