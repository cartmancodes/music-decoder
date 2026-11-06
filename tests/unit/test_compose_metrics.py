import pytest

from music_decoder.compose import metrics as m
from music_decoder.compose.voicings import voicings_for
from music_decoder.tabs.tuning import STANDARD_EADGBE
from music_decoder.types import ChordSymbol, Note, Scale, TabPosition, VoicedChord

TEMPO = 120.0  # one beat = 0.5 s, one bar = 2 s
C = Scale("C", "major")
A_MINOR = Scale("A", "minor")


def n(start_beats: float, dur_beats: float, pitch: int) -> Note:
    return Note(
        start_s=start_beats * 0.5,
        end_s=(start_beats + dur_beats) * 0.5,
        pitch=pitch,
        velocity=80,
        confidence=1.0,
    )


def prog(*labels: str) -> list[ChordSymbol]:
    return [ChordSymbol.parse(x) for x in labels]


def shape(label: str) -> VoicedChord:
    return voicings_for(ChordSymbol.parse(label), STANDARD_EADGBE)[0]


def test_chord_at_and_metric_position():
    p = prog("C", "G")
    assert m.chord_at(0.0, p, 1, TEMPO).to_label() == "C"
    assert m.chord_at(2.0, p, 1, TEMPO).to_label() == "G"  # bar 2 starts at 2.0 s
    assert m.chord_at(2.0, p, 2, TEMPO).to_label() == "C"  # 2 bars per chord
    assert m.chord_at(99.0, p, 1, TEMPO).to_label() == "G"  # clamped
    assert m.metric_position(0.0, TEMPO) == 0
    assert m.metric_position(1.0, TEMPO) == 4  # beat 3
    assert m.metric_position(2.25, TEMPO) == 1  # bar 2, the "and" of beat 1


def test_strong_beat_chord_tone_ratio():
    # beat 1: E (chord tone of C), beat 2: F (weak, ignored), beat 3: D (non-chord tone)
    mel = [n(0, 1, 64), n(1, 1, 65), n(2, 1, 62)]
    assert m.strong_beat_chord_tone_ratio(mel, prog("C"), 1, TEMPO) == pytest.approx(0.5)
    assert m.strong_beat_chord_tone_ratio([n(1, 1, 60)], prog("C"), 1, TEMPO) is None


def test_interval_ratios_and_max_interval():
    mel = [n(i, 1, p) for i, p in enumerate([60, 62, 62, 64, 72, 71])]
    # intervals: +2, 0, +2, +8, -1
    assert m.step_ratio(mel) == pytest.approx(3 / 5)
    assert m.repeat_ratio(mel) == pytest.approx(1 / 5)
    assert m.large_leap_ratio(mel) == pytest.approx(1 / 5)
    assert m.max_interval(mel) == 8
    assert m.step_ratio([n(0, 1, 60)]) is None


def test_leap_recovery_counts():
    # +7 then -2 (recovered); +5 then +2 (not recovered); final +4 is not a leap
    mel = [n(i, 1, p) for i, p in enumerate([60, 67, 65, 70, 72, 76])]
    assert m.leap_recovery_counts(mel) == (2, 1)


def test_ambitus_and_final_note():
    mel = [n(0, 1, 62), n(1, 1, 79), n(2, 2, 72)]
    assert m.ambitus(mel) == 17
    assert m.final_note_beats(mel, TEMPO) == pytest.approx(2.0)


def test_final_resolution_tonic_or_root():
    assert m.final_resolution([n(0, 4, 72)], C, prog("G", "C"))  # ends on C over C
    assert not m.final_resolution([n(0, 4, 64)], C, prog("G", "C"))  # E over C: not tonic
    assert m.final_resolution([n(0, 4, 67)], C, prog("C", "G"))  # G root (no tonic in G)
    assert not m.final_resolution([], C, prog("C"))


def test_rhythm_variety():
    uniform = [n(i, 0.95, 60) for i in range(8)]
    assert m.distinct_durations(uniform, TEMPO) == 1
    assert m.ioi_entropy(uniform, TEMPO) == pytest.approx(0.0)
    varied = [n(0, 1.5, 60), n(1.5, 0.5, 62), n(2, 1, 64), n(3, 1, 65), n(4, 2, 67)]
    # spacings in eighths: 3, 1, 2, 2 and last note 4 -> {1, 2, 3, 4}
    assert m.distinct_durations(varied, TEMPO) == 4
    assert m.ioi_entropy(varied, TEMPO) == pytest.approx(1.5)


def _bar(bar: int, pitches: list[int], last_beats: float = 0.95) -> list[Note]:
    k = len(pitches)
    step = 4 / k
    notes = [n(bar * 4 + i * step, step * 0.95, p) for i, p in enumerate(pitches[:-1])]
    return [*notes, n(bar * 4 + (k - 1) * step, last_beats, pitches[-1])]


def test_motif_metrics_on_a_four_bar_phrase():
    # layout for 4 bars echoes bar 0 in bar 2
    mel = (
        _bar(0, [60, 62, 64, 62])  # up up down
        + _bar(1, [67])
        + _bar(2, [65, 67, 65, 64])  # up down down
        + _bar(3, [60])
    )
    assert m.motif_pair_coverage(mel, TEMPO, 4) == pytest.approx(1.0)
    assert m.motif_contour_match(mel, TEMPO, 4) == pytest.approx(2 / 3)


def test_motif_metrics_follow_the_five_bar_layout():
    # 5 bars = 3 + 2: the only echo pair is (bar 0, bar 3)
    mel = (
        _bar(0, [60, 62, 64])
        + _bar(1, [65, 64])
        + _bar(2, [62], 2)
        + _bar(3, [64, 65, 67])  # same contour as bar 0
        + _bar(4, [60], 2)
    )
    assert m.motif_pair_coverage(mel, TEMPO, 5) == pytest.approx(1.0)
    assert m.motif_contour_match(mel, TEMPO, 5) == pytest.approx(1.0)


def test_unequal_echo_bars_lower_coverage_and_are_undefined_for_contour():
    mel = _bar(0, [60, 62, 64, 62]) + _bar(1, [67]) + _bar(2, [65, 67]) + _bar(3, [60])
    assert m.motif_pair_coverage(mel, TEMPO, 4) == pytest.approx(0.0)
    assert m.motif_contour_match(mel, TEMPO, 4) is None
    assert m.motif_pair_coverage(_bar(0, [60, 62]), TEMPO, 1) is None  # no pairs at all


def test_phrase_end_long_ratio_uses_the_layout():
    # 5 bars = 3 + 2: phrase ends are bars 2 and 4 (not bar 3)
    mel = [*_bar(0, [60] * 4), *_bar(1, [60] * 4), *_bar(2, [62], 2)]
    mel += [*_bar(3, [60] * 4), *_bar(4, [60], 2)]
    assert m.phrase_end_long_ratio(mel, TEMPO, 5) == pytest.approx(1.0)
    short = [*_bar(0, [60] * 4), *_bar(1, [60] * 4), *_bar(2, [62] * 4)]
    short += [*_bar(3, [60] * 4), *_bar(4, [60], 2)]
    assert m.phrase_end_long_ratio(short, TEMPO, 5) == pytest.approx(0.5)


def test_exposed_chromatic_clash_ratio():
    p = prog("Am", "E7")  # E7 = E G# B D; G# is foreign to A natural minor
    mel = [
        n(0, 1, 67),  # over Am: not counted
        n(4, 1, 67),  # beat 1 over E7, G vs G#: clash
        n(5, 0.5, 69),  # weak, short A: a passing note, not exposed
        n(6, 1, 68),  # beat 3, G# itself is a chord tone: fine
        n(7, 1, 69),  # beat 4 but a whole beat long: exposed A vs G#: clash
    ]
    assert m.exposed_chromatic_clash_ratio(mel, A_MINOR, p, 1, TEMPO) == pytest.approx(2 / 3)
    assert m.exposed_chromatic_clash_ratio(mel, C, prog("C", "G"), 1, TEMPO) is None


def test_chromatic_clash_ignores_chord_tones_next_to_a_foreign_tone():
    # C#maj7 in C major = C# F G# C: C is a chord tone even though C# is foreign
    assert m.exposed_chromatic_clash_ratio([n(0, 2, 72)], C, prog("C#maj7"), 1, TEMPO) == 0.0


def test_leading_tone_resolution_counts():
    p = prog("E7", "Am")
    resolved = [n(0, 4, 68), n(4, 4, 69)]  # G# -> A across E7 -> Am
    assert m.leading_tone_resolution_counts(resolved, A_MINOR, p, 1, TEMPO) == (1, 1)
    falls = [n(0, 4, 68), n(4, 4, 64)]  # G# -> E
    assert m.leading_tone_resolution_counts(falls, A_MINOR, p, 1, TEMPO) == (1, 0)
    same_chord = [n(0, 2, 68), n(2, 2, 69)]  # still inside E7: not eligible
    assert m.leading_tone_resolution_counts(same_chord, A_MINOR, p, 1, TEMPO) == (0, 0)


def test_onset_grid_and_stroke_heads():
    assert m.onset_grid_ratio([0.0, 0.25, 0.5], TEMPO, 2) == pytest.approx(1.0)
    assert m.onset_grid_ratio([0.0, 0.1], TEMPO, 2) == pytest.approx(0.5)
    assert m.stroke_heads([0.0, 0.008, 0.016, 0.25, 0.26]) == [0.0, 0.25]


def test_melody_above_and_exposed_clash():
    acc = [n(0, 4, 48), n(0, 4, 64)]  # C3 + E4 ringing for the bar
    mel = [n(0, 1, 65), n(1, 0.5, 67), n(2, 1, 60)]
    # 65 above 64 (yes), 67 (yes), 60 below 64 (no)
    assert m.melody_above_accompaniment_ratio(mel, acc) == pytest.approx(2 / 3)
    # exposed: 65 (beat 1, F vs E = semitone -> clash), 60 (beat 3, no clash); 67 is weak+short
    assert m.exposed_semitone_clash_ratio(mel, acc, TEMPO) == pytest.approx(0.5)
    assert m.melody_above_accompaniment_ratio(mel, []) == pytest.approx(1.0)


def test_staggered_and_late_entering_accompaniment_is_seen():
    # C4 held for 1.9 s; C3 from 0, C#4 enters 8 ms late (a strummed upper string)
    mel = [Note(start_s=0.0, end_s=1.9, pitch=60, velocity=80, confidence=1.0)]
    acc = [
        Note(start_s=0.0, end_s=2.0, pitch=48, velocity=80, confidence=1.0),
        Note(start_s=0.008, end_s=2.0, pitch=61, velocity=80, confidence=1.0),
    ]
    assert m.melody_above_accompaniment_ratio(mel, acc) == 0.0
    assert m.exposed_semitone_clash_ratio(mel, acc, TEMPO) == 1.0
    entering = [Note(start_s=1.0, end_s=2.0, pitch=61, velocity=80, confidence=1.0)]
    assert m.exposed_semitone_clash_ratio(mel, entering, TEMPO) == 1.0


def test_non_overlapping_accompaniment_is_ignored():
    mel = [Note(start_s=0.0, end_s=1.0, pitch=60, velocity=80, confidence=1.0)]
    before = [Note(start_s=-1.0, end_s=0.0, pitch=61, velocity=80, confidence=1.0)]
    after = [Note(start_s=1.0, end_s=2.0, pitch=61, velocity=80, confidence=1.0)]
    for acc in (before, after):
        assert m.melody_above_accompaniment_ratio(mel, acc) == 1.0
        assert m.exposed_semitone_clash_ratio(mel, acc, TEMPO) == 0.0


def test_final_chord_rings():
    ring = [n(4, 4, 48), n(4.02, 3.98, 52)]
    assert m.final_chord_rings(ring, 2, TEMPO)
    restruck = [n(4, 2, 48), n(6, 2, 48)]
    assert not m.final_chord_rings(restruck, 2, TEMPO)
    assert not m.final_chord_rings([], 2, TEMPO)


def test_essential_tone_coverage():
    d = shape("D")  # xx0232: D3 A3 D4 F#4
    full = [n(0, 4, 50), n(0, 4, 57), n(0, 4, 62), n(0, 4, 66)]
    assert m.essential_tone_coverage([], full, [d], 1, TEMPO, STANDARD_EADGBE) == [True]
    no_third = [n(0, 4, 50), n(0, 4, 57), n(0, 4, 62)]
    assert m.essential_tone_coverage([], no_third, [d], 1, TEMPO, STANDARD_EADGBE) == [False]
    melody_has_it = [n(0, 4, 78)]  # F#5 in the melody supplies the third
    assert m.essential_tone_coverage(melody_has_it, no_third, [d], 1, TEMPO, STANDARD_EADGBE) == [
        True
    ]


def test_hand_shift_rate():
    assert m.hand_shift_rate([5, 7, 8, 9]) == pytest.approx(0.0)
    assert m.hand_shift_rate([5, 7, 12, 13]) == pytest.approx(1 / 3)
    assert m.hand_shift_rate([0, 0, 5]) is None  # open strings do not count


def test_nearest_pitch_proximity():
    v = [shape(c) for c in ("C", "C", "Am")]
    d = m.nearest_pitch_proximity(v, STANDARD_EADGBE)
    assert d is not None and 0.0 < d < 2.0
    assert m.nearest_pitch_proximity(v[:2], STANDARD_EADGBE) is None  # same chord only


def test_nearest_pitch_proximity_is_not_a_voice_leading_score():
    # Documented blind spot: a 3-note shape collapsing onto one doubled pitch
    # scores 0 even though two voices vanished.
    pos = [TabPosition(string=s, fret=-1) for s in range(6)]
    wide = VoicedChord(
        ChordSymbol.parse("C"),
        (
            TabPosition(0, -1),
            TabPosition(1, 3),
            TabPosition(2, 2),
            TabPosition(3, 0),
            *pos[4:],
        ),
    )  # C3 E3 G3
    collapsed = VoicedChord(
        ChordSymbol.parse("Em"),
        (TabPosition(0, -1), TabPosition(1, -1), TabPosition(2, 2), *pos[3:]),
    )  # E3 only
    assert m.nearest_pitch_proximity([wide, collapsed], STANDARD_EADGBE) == 0.0


def test_shape_tone_mismatch_reports_inherited_shape_defects():
    assert m.shape_tone_mismatch(shape("C"), STANDARD_EADGBE) == (frozenset(), frozenset())
    # The canonical C#dim shape (x x 2 3 2 x -> E A# C#) lacks the diminished
    # fifth G and adds A#; H1 cannot see this because it is shape-relative.
    missing, foreign = m.shape_tone_mismatch(shape("C#dim"), STANDARD_EADGBE)
    assert missing == frozenset({7}) and foreign == frozenset({10})
    acc = [n(0, 4, 52), n(0, 4, 58), n(0, 4, 61)]
    assert m.essential_tone_coverage([], acc, [shape("C#dim")], 1, TEMPO, STANDARD_EADGBE) == [True]


def test_cadence_identity_checks_the_held_final_note():
    g7 = shape("G7")  # G2 B2 D3 G3 B3 F4
    melody = [n(0, 1, 67), n(1, 1, 65), n(2, 2, 67)]  # G4, a passing F4, held G4
    full = [n(0, 4, p) for p in (43, 47, 50, 55, 59, 65)]
    assert m.cadence_identity(melody, full, [g7], STANDARD_EADGBE) is True
    no_seventh = [n(0, 4, p) for p in (43, 47, 50, 55, 59)]
    # H1 is satisfied by the passing F4, H2 is not: nothing supplies F under the held G4
    assert m.essential_tone_coverage(melody, no_seventh, [g7], 1, TEMPO, STANDARD_EADGBE) == [True]
    assert m.cadence_identity(melody, no_seventh, [g7], STANDARD_EADGBE) is False
    held_f = [n(0, 2, 67), n(2, 2, 65)]  # the held final note is the seventh itself
    assert m.cadence_identity(held_f, no_seventh, [g7], STANDARD_EADGBE) is True
    assert m.cadence_identity([], full, [g7], STANDARD_EADGBE) is None


def test_cadence_identity_accepts_re_struck_tones_with_articulation_gaps():
    g7 = shape("G7")
    melody = [n(0, 2, 67), n(2, 2, 67)]
    base = [n(0, 4, p) for p in (43, 47, 50, 55, 59)]
    restruck_f = [n(2, 0.95, 65), n(3, 0.95, 65)]  # 95% of the held note
    assert m.cadence_identity(melody, base + restruck_f, [g7], STANDARD_EADGBE) is True
    half_f = [n(2, 0.95, 65)]  # only the first half
    assert m.cadence_identity(melody, base + half_f, [g7], STANDARD_EADGBE) is False
