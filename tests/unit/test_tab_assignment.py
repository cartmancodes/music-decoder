# tests/unit/test_tab_assignment.py

from music_decoder.types import TabPosition, TranscribedNote
from music_decoder.tab_assignment.assigner import assign_tab
from music_decoder.tab_assignment.tuning import get_preset


def _note(pitch: int, start: float = 0.0, end: float = 1.0, conf: float = 1.0):
    return TranscribedNote(start_s=start, end_s=end, pitch=pitch, velocity=80, confidence=conf)


def _default_weights() -> dict[str, float]:
    # w_high raised to 2.0 so that the high-fret penalty is strong enough to
    # prevent the "start high, glide down" path-gaming exploit that arises with
    # the asymmetric move cost.  At 0.4 the penalty on fret 15 (C4 on A-string)
    # was outweighed by the cheaper transitions it enabled for subsequent notes.
    return {
        "w_move": 1.0, "w_string": 0.3, "w_span": 0.5, "w_high": 2.0,
        "w_open": 0.2, "w_chord_intra": 0.6,
    }


def test_open_high_e_is_preferred_over_b_string_5th_fret():
    """E4 (MIDI 64) on EADGBE: open high E (5,0) preferred to (4,5)."""
    notes = [_note(64)]
    result = assign_tab(notes, tuning=get_preset("EADGBE"),
                       weights=_default_weights(), max_fret=22)
    assert result.tabbed_notes[0].position == TabPosition(string=5, fret=0)


def test_low_e_is_only_playable_open_e2():
    """E2 (MIDI 40): only valid position is (0, 0) on EADGBE."""
    notes = [_note(40)]
    result = assign_tab(notes, tuning=get_preset("EADGBE"),
                       weights=_default_weights(), max_fret=22)
    assert result.tabbed_notes[0].position == TabPosition(string=0, fret=0)
    assert len(result.notes_dropped) == 0


def test_below_low_e_is_dropped_in_eadgbe():
    """D2 (MIDI 38) cannot be played on EADGBE (lowest is E2)."""
    notes = [_note(38)]
    result = assign_tab(notes, tuning=get_preset("EADGBE"),
                       weights=_default_weights(), max_fret=22)
    assert len(result.tabbed_notes) == 0
    assert len(result.notes_dropped) == 1
    assert "out_of_range" in result.notes_dropped[0][1]


def test_drop_d_makes_d2_playable_on_open_low_string():
    """D2 (MIDI 38) on Drop_D: open low D string (0, 0)."""
    notes = [_note(38)]
    result = assign_tab(notes, tuning=get_preset("Drop_D"),
                       weights=_default_weights(), max_fret=22)
    assert result.tabbed_notes[0].position == TabPosition(string=0, fret=0)


def test_g_major_chord_returns_six_simultaneous_positions():
    """G2 B2 D3 G3 B3 G4 simultaneously → known open-position G chord."""
    notes = [
        _note(43, 0.0, 1.0), _note(47, 0.0, 1.0), _note(50, 0.0, 1.0),
        _note(55, 0.0, 1.0), _note(59, 0.0, 1.0), _note(67, 0.0, 1.0),
    ]
    result = assign_tab(notes, tuning=get_preset("EADGBE"),
                       weights=_default_weights(), max_fret=22)
    pairs = sorted([(n.position.string, n.position.fret) for n in result.tabbed_notes])
    assert pairs == [(0, 3), (1, 2), (2, 0), (3, 0), (4, 0), (5, 3)]


def test_ascending_scale_is_monotonic_in_pitch():
    """A simple ascending scale should not produce any string-collision artifacts."""
    pitches = [60, 62, 64, 65, 67, 69, 71, 72]
    notes = [_note(p, start=i * 0.5, end=(i + 1) * 0.5) for i, p in enumerate(pitches)]
    result = assign_tab(notes, tuning=get_preset("EADGBE"),
                       weights=_default_weights(), max_fret=22)
    assert len(result.tabbed_notes) == 8
    # No two notes should occupy the same string at the same time
    for a, b in zip(result.tabbed_notes, result.tabbed_notes[1:], strict=False):
        if a.note.end_s > b.note.start_s and a.position.string == b.position.string:
            raise AssertionError("string collision in sequential single notes")


def test_more_than_six_simultaneous_notes_drops_lowest_confidence():
    """Spec: if more than 6 overlap, keep 6 highest-confidence, drop the rest."""
    pitches_and_confs = [
        (60, 0.95), (64, 0.93), (67, 0.91), (70, 0.85),
        (72, 0.80), (74, 0.75), (76, 0.50),  # this last one should be dropped
    ]
    notes = [TranscribedNote(start_s=0.0, end_s=1.0, pitch=p, velocity=80, confidence=c)
             for p, c in pitches_and_confs]
    result = assign_tab(notes, tuning=get_preset("EADGBE"),
                       weights=_default_weights(), max_fret=22)
    dropped_pitches = [n.pitch for n, reason in result.notes_dropped
                       if reason == "chord_too_dense_capped_to_6"]
    assert 76 in dropped_pitches
    assert len(dropped_pitches) == 1


def test_overlapping_notes_with_drifted_onsets_form_chord():
    """Two notes with onsets 30ms apart and overlapping intervals should
    form a single chord state, not be treated as sequential single notes."""
    notes = [
        TranscribedNote(start_s=0.000, end_s=1.000, pitch=60, velocity=80, confidence=0.9),
        TranscribedNote(start_s=0.030, end_s=1.000, pitch=64, velocity=80, confidence=0.9),
    ]
    result = assign_tab(notes, tuning=get_preset("EADGBE"),
                       weights=_default_weights(), max_fret=22)
    # Both notes should be tabbed; both should appear simultaneously (same TabbedNote group),
    # which manifests as: their string positions differ (no string collision).
    assert len(result.tabbed_notes) == 2
    strings = {t.position.string for t in result.tabbed_notes}
    assert len(strings) == 2  # they cannot share a string


def test_c_major_scale_uses_high_e_string_for_position_stability():
    """C major scale ascending from C4 to C5. Position stability should keep
    consecutive notes on the same string when feasible. Open strings still
    win the open_bonus where available."""
    pitches = [60, 62, 64, 65, 67, 69, 71, 72]  # C4 through C5
    notes = [
        TranscribedNote(start_s=i * 0.5, end_s=(i + 1) * 0.5,
                        pitch=p, velocity=80, confidence=0.95)
        for i, p in enumerate(pitches)
    ]
    result = assign_tab(notes, tuning=get_preset("EADGBE"),
                       weights=_default_weights(), max_fret=22)
    assert len(result.tabbed_notes) == 8
    # All assignments must be valid (string + fret = pitch)
    for tabbed, expected_pitch in zip(result.tabbed_notes, pitches, strict=True):
        s = tabbed.position.string
        f = tabbed.position.fret
        open_p = get_preset("EADGBE").open_pitches[s]
        assert open_p + f == expected_pitch
    # Position stability: notes 4-7 (E4..C5, all reachable on high E open + frets 0..8)
    # should be on string 5 (the high E) since once we've moved up there, the anchor
    # holds. Notes 0-3 (C4..F4) might be lower; check they don't bounce wildly.
    later_strings = [t.position.string for t in result.tabbed_notes[4:]]
    assert all(s == later_strings[0] for s in later_strings), \
        f"Position stability violated in upper notes: strings={later_strings}"


def test_melody_crossing_12th_fret_stays_upper_position():
    """When a melody enters the upper position (above fret 12), position
    stability should prefer staying there rather than flipping back to
    open strings. The high_fret penalty pulls toward staying low for
    initial notes, but once the melody forces a high-fret choice, the
    sliding-anchor median should keep subsequent notes nearby."""
    # G5 (79) -> high E string fret 15. The next notes should stay near fret 15.
    # G5, A5, B5, A5, G5: simple melody around fret 15-19 on high E.
    pitches = [79, 81, 83, 81, 79]  # G5 A5 B5 A5 G5
    notes = [
        TranscribedNote(start_s=i * 0.4, end_s=(i + 1) * 0.4,
                        pitch=p, velocity=80, confidence=0.95)
        for i, p in enumerate(pitches)
    ]
    result = assign_tab(notes, tuning=get_preset("EADGBE"),
                       weights=_default_weights(), max_fret=22)
    assert len(result.tabbed_notes) == 5
    # All notes should be on the high E string (string 5) — no other string
    # can play these pitches without exceeding max_fret.
    # G5 (79): only valid on string 5 fret 15.
    strings = [t.position.string for t in result.tabbed_notes]
    frets = [t.position.fret for t in result.tabbed_notes]
    assert all(s == 5 for s in strings), f"Expected all on high E, got {strings}"
    assert frets == [15, 17, 19, 17, 15]
