from music_decoder.tab_reference.parser import parse_ascii_tab


def test_parse_simple_eadgbe_tab():
    tab = (
        "e|--0--3--|\n"
        "B|--1--0--|\n"
        "G|--0--0--|\n"
        "D|--2--0--|\n"
        "A|--3--2--|\n"
        "E|--x--3--|\n"
    )
    positions = parse_ascii_tab(tab)
    # First column: open e, B 1, G 0, D 2, A 3, E muted → produces 5 positions
    assert len(positions) >= 5
    assert all(0 <= p.string < 6 for p in positions)
    assert any(p.fret == 3 and p.string == 0 for p in positions)   # E string fret 3 in second column


def test_parse_skips_unparseable_input():
    assert parse_ascii_tab("nothing here") == []
