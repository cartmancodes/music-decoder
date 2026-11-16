import itertools
import math
import statistics

import pytest

from music_decoder.compose import metrics
from music_decoder.compose.melody import (
    CHORD_CHANGE_STEP_BONUS,
    HOOK_EXACT,
    HOOK_NEAR,
    MAX_INTERVAL,
    PRIOR_WEIGHT,
    PRIOR_WIDTH,
    NoteContext,
    _prior_target,
    admissible,
    feasible_sets,
    final_target,
    generate_melody,
    ranked_scores,
    score_pitch,
)
from music_decoder.compose.rhythm import form_labels, phrase_layout
from music_decoder.types import ChordSymbol, Note, Scale

C_MAJOR_PCS = {0, 2, 4, 5, 7, 9, 11}


def test_all_notes_in_scale():
    progression = [ChordSymbol.parse(c) for c in ("C", "Am", "F", "G")]
    notes = generate_melody(
        scale=Scale("C", "major"),
        progression=progression,
        bars_per_chord=1,
        tempo_bpm=120.0,
        seed=42,
    )
    assert notes
    for n in notes:
        assert n.pitch % 12 in C_MAJOR_PCS


def test_deterministic_with_seed():
    progression = [ChordSymbol.parse(c) for c in ("C", "Am", "F", "G")]
    a = generate_melody(
        scale=Scale("C", "major"),
        progression=progression,
        bars_per_chord=1,
        tempo_bpm=120.0,
        seed=42,
    )
    b = generate_melody(
        scale=Scale("C", "major"),
        progression=progression,
        bars_per_chord=1,
        tempo_bpm=120.0,
        seed=42,
    )
    assert [(n.pitch, n.start_s) for n in a] == [(n.pitch, n.start_s) for n in b]


def test_different_seeds_produce_different_melodies():
    progression = [ChordSymbol.parse(c) for c in ("C", "Am", "F", "G")]
    a = generate_melody(
        scale=Scale("C", "major"),
        progression=progression,
        bars_per_chord=1,
        tempo_bpm=120.0,
        seed=1,
    )
    b = generate_melody(
        scale=Scale("C", "major"),
        progression=progression,
        bars_per_chord=1,
        tempo_bpm=120.0,
        seed=2,
    )
    assert [n.pitch for n in a] != [n.pitch for n in b]


def test_strong_beats_favor_chord_tones_statistically():
    # Cmaj7 = {0,4,7,11}; 4 bars all Cmaj7, 4 notes per bar; strong beats are
    # beat 1 and 3 (i.e. notes 0 and 2 of every bar).
    progression = [ChordSymbol.parse("Cmaj7")] * 8
    notes = generate_melody(
        scale=Scale("C", "major"),
        progression=progression,
        bars_per_chord=1,
        tempo_bpm=120.0,
        seed=0,
        notes_per_bar=4,
    )
    chord_tones = {0, 4, 7, 11}
    strong_idxs = [i for i in range(len(notes)) if i % 4 in (0, 2)]
    strong_pcs = [notes[i].pitch % 12 for i in strong_idxs]
    fraction_chord = sum(p in chord_tones for p in strong_pcs) / len(strong_pcs)
    assert fraction_chord >= 0.55, f"strong-beat chord-tone rate {fraction_chord}"


# ---- phrase-planned generator ----------------------------------------------


def _ctx(**overrides):
    base = dict(
        strong=False,
        eighths=1.0,
        chord_pcs=frozenset({0, 4, 7}),
        prev_pitch=72,
        prev_interval=0,
        motif_interval=None,
        target=None,
        cadence_end=False,
    )
    base.update(overrides)
    return NoteContext(**base)


def _gen(scale, labels, bars=1, seed=0, tempo=120.0, **kw):
    return generate_melody(
        scale=scale,
        progression=[ChordSymbol.parse(c) for c in labels],
        bars_per_chord=bars,
        tempo_bpm=tempo,
        seed=seed,
        **kw,
    )


def test_final_target_prefers_tonic_then_root_nearest_reference():
    cands = list(range(62, 80))
    c_major = Scale("C", "major")
    assert final_target(c_major, ChordSymbol.parse("C"), cands, 70) == 72
    assert final_target(c_major, ChordSymbol.parse("Am"), cands, 66) == 72  # Am holds C
    assert final_target(c_major, ChordSymbol.parse("G"), cands, 70) == 67  # no tonic: root G
    with pytest.raises(ValueError):
        final_target(c_major, ChordSymbol.parse("C"), [62, 64], 63)


def test_score_pitch_prefers_chord_tones_on_strong_beats():
    # From C5 (72) over a C chord: E5 (a third, chord tone) beats D5 (a step, non-chord tone)
    assert score_pitch(76, _ctx(strong=True), 72) > score_pitch(74, _ctx(strong=True), 72)
    # ...but on a short weak note the step wins
    assert score_pitch(74, _ctx(), 72) > score_pitch(76, _ctx(), 72)


def test_score_pitch_excludes_leaps_over_an_octave():
    assert score_pitch(85, _ctx(), 72) == -math.inf


def test_score_pitch_keeps_the_final_target_within_an_octave():
    # previous note 72, target 60: 74 is a fine step from 72 but 14 from the target
    ctx = _ctx(target=60)
    assert score_pitch(74, ctx, 70) == -math.inf
    assert score_pitch(72, ctx, 70) > -math.inf


def test_score_pitch_rewards_leap_recovery():
    after_leap_up = _ctx(prev_interval=7)
    assert score_pitch(71, after_leap_up, 72) > score_pitch(74, after_leap_up, 72)


def test_score_pitch_resolves_the_leading_tone_up():
    ctx = _ctx(prev_pitch=71, chord_pcs=frozenset({0, 4, 7}), resolve_up=True)  # B -> C chord
    assert score_pitch(72, ctx, 72) > score_pitch(72, _ctx(prev_pitch=71), 72)
    assert score_pitch(72, ctx, 72) > score_pitch(67, ctx, 72)


def test_score_pitch_follows_the_motif():
    ctx = _ctx(motif_interval=-2)
    assert score_pitch(70, ctx, 72) > score_pitch(74, ctx, 72)


@pytest.mark.parametrize(
    ("key", "labels", "seed"),
    [("E", ("E", "A", "B7", "E"), 283), ("D", ("D", "G", "A", "D"), 5774)],
)
def test_review_reproductions_end_within_an_octave(key, labels, seed):
    notes = _gen(Scale(key, "major"), labels, seed=seed)
    assert metrics.max_interval(notes) <= MAX_INTERVAL


@pytest.mark.parametrize(
    ("key", "mode", "labels"),
    [
        ("E", "major", ("E", "A", "B7", "E")),
        ("D", "major", ("D", "G", "A", "D")),
        ("A", "minor", ("Am", "Dm", "E7", "Am")),
    ],
)
def test_no_interval_exceeds_an_octave_over_400_seeds(key, mode, labels):
    for seed in range(400):
        notes = _gen(Scale(key, mode), labels, seed=seed)
        assert metrics.max_interval(notes) <= MAX_INTERVAL, seed


def test_phrase_melody_has_varied_rhythm_on_the_eighth_grid():
    notes = _gen(Scale("C", "major"), ("C", "G", "Am", "F"), bars=2, seed=5)
    assert metrics.distinct_durations(notes, 120.0) >= 2
    assert metrics.onset_grid_ratio([n.start_s for n in notes], 120.0, 2) == 1.0
    assert notes[-1].end_s <= 8 * 2.0 + 1e-9  # stays within 8 bars at 120 BPM


@pytest.mark.parametrize("seed", range(10))
def test_phrase_melody_resolves_and_stays_in_window(seed):
    notes = _gen(Scale("A", "minor"), ("Am", "Dm", "E7", "Am"), seed=seed)
    assert notes[-1].pitch % 12 == 9  # ends on A
    assert (notes[-1].end_s - notes[-1].start_s) / 0.5 >= 1.5
    assert all(62 <= n.pitch <= 79 for n in notes)


def test_e7_in_a_minor_uses_g_sharp_and_may_keep_a():
    pitch_classes = set()
    for seed in range(40):
        notes = _gen(Scale("A", "minor"), ("Am", "E7"), seed=seed)
        bar2 = [n.pitch % 12 for n in notes if 2.0 <= n.start_s < 4.0]
        assert 7 not in bar2, seed  # no G natural against G#
        pitch_classes.update(bar2)
    assert 8 in pitch_classes  # G# is used
    assert 9 in pitch_classes  # the tonic A is still available over E7


def test_secondary_dominant_in_major_keeps_its_target():
    pitch_classes = set()
    for seed in range(40):
        notes = _gen(Scale("C", "major"), ("C", "D7", "G", "C"), seed=seed)
        bar2 = [n.pitch % 12 for n in notes if 2.0 <= n.start_s < 4.0]  # D7 = D F# A C
        assert 5 not in bar2, seed  # no F natural against F#
        pitch_classes.update(bar2)
    assert 7 in pitch_classes  # G (where D7 is heading) is still available


def test_leading_tone_mostly_resolves_up_into_the_tonic_chord():
    eligible = resolved = 0
    for seed in range(100):
        notes = _gen(Scale("A", "minor"), ("Am", "Dm", "E7", "Am"), seed=seed)
        prog = [ChordSymbol.parse(c) for c in ("Am", "Dm", "E7", "Am")]
        e, r = metrics.leading_tone_resolution_counts(notes, Scale("A", "minor"), prog, 1, 120.0)
        eligible, resolved = eligible + e, resolved + r
    assert eligible >= 10
    assert resolved / eligible >= 0.8


def test_chord_floors_lift_the_melody_over_that_chord():
    floors = [None, 69, None, None]
    for seed in range(20):
        notes = _gen(Scale("D", "major"), ("G", "D", "A", "G"), seed=seed, chord_floors=floors)
        assert all(n.pitch >= 69 for n in notes if 2.0 <= n.start_s < 4.0), seed


def test_a_floor_leaving_too_few_pitches_is_ignored():
    notes = _gen(Scale("C", "major"), ("C",), seed=1, chord_floors=[78])
    assert any(n.pitch < 78 for n in notes)
    with pytest.raises(ValueError):
        _gen(Scale("C", "major"), ("C", "G"), seed=1, chord_floors=[70])


def test_legacy_grid_keeps_uniform_timing():
    notes = _gen(Scale("C", "major"), ("C", "G"), notes_per_bar=4)
    assert len(notes) == 8
    assert [round(n.start_s, 6) for n in notes] == [i * 0.5 for i in range(8)]
    assert notes[-1].pitch % 12 == 7  # ends on the G root (no tonic in G)


def test_single_bar_progression():
    notes = _gen(Scale("D", "major"), ("D",), seed=1)
    assert notes and notes[-1].pitch % 12 == 2


@pytest.mark.parametrize("tempo", [40.0, 220.0])
def test_extreme_tempos_keep_the_grid_and_the_length(tempo):
    notes = _gen(Scale("E", "major"), ("E", "A", "B7"), seed=4, tempo=tempo)
    bar_s = 4 * 60.0 / tempo
    assert metrics.onset_grid_ratio([n.start_s for n in notes], tempo, 2) == 1.0
    assert notes[-1].end_s <= 3 * bar_s + 1e-9
    assert notes[-1].pitch % 12 == 11  # B7 holds no E: ends on the B root


def test_one_note_final_bar_is_approached_within_a_fifth():
    # Legacy grid with one note per bar: bar 2 is the forced target alone.
    # The G7 floor (68) removes G4, so bar 1 is steered toward the final G.
    for seed in range(200):
        notes = _gen(
            Scale("C", "major"),
            ("C", "G7"),
            seed=seed,
            notes_per_bar=1,
            tessitura_center=62,
            chord_floors=[None, 68],
        )
        assert len(notes) == 2
        move = abs(notes[1].pitch - notes[0].pitch)
        assert move <= 7 and move != 6, seed
        assert notes[1].pitch % 12 == 7, seed  # G7 has no C: ends on the root


def test_floor_yields_when_it_would_remove_the_cadence_target():
    # Dsus4 (D G A) contains A, the tonic, but a floor of 70 leaves no A in 62-79
    for seed in range(20):
        notes = _gen(Scale("A", "major"), ("A", "Dsus4"), seed=seed, chord_floors=[None, 70])
        assert notes[-1].pitch % 12 == 9, seed
        assert metrics.max_interval(notes) <= MAX_INTERVAL


def test_floor_yields_when_nothing_above_it_is_reachable():
    # Only possible with a wide custom window: floor 80 leaves 80-91, all more
    # than an octave above a low bar-1 note, so bar 2 must ignore the floor.
    for seed in range(50):
        notes = _gen(
            Scale("C", "major"),
            ("C", "F", "C"),
            seed=seed,
            notes_per_bar=1,
            pitch_high=91,
            tessitura_center=62,
            chord_floors=[None, 80, None],
        )
        assert metrics.max_interval(notes) <= MAX_INTERVAL, seed


# ---- singable leaps (composer v3) ------------------------------------------


def test_admissible_forbids_tritones_sevenths_and_wide_leaps():
    for c in (78, 82, 83, 85, 66, 62, 61):  # +6 +10 +11 +13, -6 -10 -11 from 72
        assert not admissible(c, _ctx())
    assert admissible(79, _ctx()) and admissible(65, _ctx())  # fifths up and down


def test_admissible_large_leap_needs_arpeggiation_and_a_recovery():
    ok = dict(prev_chord_tone=True, recovery_pool=frozenset({74, 76, 77}))
    assert not admissible(80, _ctx(prev_pitch=72, **ok))  # G# is not a C chord tone
    assert admissible(79, _ctx(prev_pitch=67, prev_chord_tone=True, recovery_pool=frozenset({77})))
    assert not admissible(79, _ctx(prev_pitch=67, prev_chord_tone=True, recovery_pool=None))
    assert not admissible(
        79, _ctx(prev_pitch=67, prev_chord_tone=True, recovery_pool=frozenset({76}))
    )
    assert not admissible(
        79, _ctx(prev_pitch=67, prev_chord_tone=False, recovery_pool=frozenset({77}))
    )
    assert not admissible(
        79,
        _ctx(prev_pitch=67, prev_chord_tone=True, chord_change=True, recovery_pool=frozenset({77})),
    )
    assert not admissible(
        79,
        _ctx(prev_pitch=67, prev_chord_tone=True, prev_interval=5, recovery_pool=frozenset({77})),
    )  # no chained leaps
    assert not admissible(
        79, _ctx(prev_pitch=67, prev_chord_tone=True, target=79, recovery_pool=frozenset({77}))
    )  # never in the final bar


def test_admissible_forces_recovery_after_a_large_leap():
    after = _ctx(prev_pitch=76, prev_interval=9)
    assert admissible(74, after) and admissible(75, after)
    assert not admissible(77, after) and not admissible(72, after) and not admissible(76, after)


def test_admissible_respects_the_feasibility_set():
    assert admissible(74, _ctx(allowed=frozenset({74})))
    assert not admissible(71, _ctx(allowed=frozenset({74})))


def test_feasible_sets_walk_back_from_the_goal():
    # review r1 F1: floored final bar of C -> D#m; target D#5 (75), prev E4 (64)
    floored = [69, 70, 72, 74, 75, 77, 78]
    (first,) = feasible_sets(floored, 1, {75})
    assert first == {70, 72, 74, 75, 77, 78}  # 69 is a tritone from 75
    assert not any(abs(c - 64) <= 7 and abs(c - 64) != 6 for c in first)  # dead end: repair
    assert feasible_sets(floored, 0, {75}) == []


def test_ranked_scores_ladder_keeps_r1_and_r2_before_giving_up():
    ctx = _ctx(prev_pitch=64, allowed=frozenset())  # nothing is admissible
    scores, level = ranked_scores([66, 75], ctx, 70)
    assert level == 1 and math.isfinite(scores[0]) and scores[1] == -math.inf  # +11 forbidden
    scores, level = ranked_scores([75], _ctx(prev_pitch=64, allowed=frozenset()), 70)
    assert level == 2 and math.isfinite(scores[0])  # only a forbidden move exists
    assert ranked_scores([66], _ctx(prev_pitch=64), 70)[1] == 0


def test_chord_change_step_bonus():
    step = score_pitch(74, _ctx(chord_change=True), 72) - score_pitch(74, _ctx(), 72)
    assert step == pytest.approx(CHORD_CHANGE_STEP_BONUS)
    assert score_pitch(72, _ctx(chord_change=True), 72) == score_pitch(72, _ctx(), 72)


def _leap_invariants(notes, labels, bars=1, tempo=120.0):
    p = [ChordSymbol.parse(c) for c in labels]
    assert metrics.forbidden_interval_count(notes) == 0
    assert metrics.unjustified_large_leap_count(notes, p, bars, tempo) == 0
    leaps, recovered = metrics.large_leap_recovery_counts(notes)
    assert recovered == leaps
    assert max(metrics.chord_change_moves(notes, p, bars, tempo), default=0) <= 7
    if len(notes) > 1:
        assert abs(notes[-1].pitch - notes[-2].pitch) <= 7


@pytest.mark.parametrize(
    ("key", "mode", "labels"),
    [
        ("C", "major", ("C", "Am", "F", "G")),
        ("A", "minor", ("Am", "Dm", "E7", "Am")),
        ("C", "major", ("Cmaj7", "Am7", "Dm7", "G7")),
        ("C", "major", ("C", "B7", "Em", "G7", "C")),
    ],
)
def test_generated_melodies_obey_the_leap_rules(key, mode, labels):
    for seed in range(100):
        notes = _gen(Scale(key, mode), labels, seed=seed, chord_floors=[None] * len(labels))
        _leap_invariants(notes, labels)


def test_legacy_grid_and_no_floors_obey_the_leap_rules():  # Review Focus 3
    for seed in range(100):
        for npb in (1, 4):
            labels = ("C", "F", "G7", "C")
            _leap_invariants(
                _gen(Scale("C", "major"), labels, seed=seed, notes_per_bar=npb), labels
            )


def test_short_pieces_obey_the_leap_rules():  # Review Focus 4
    for seed in range(100):
        for labels in (("D",), ("C", "G7"), ("A", "Dsus4")):
            floors = [None, 68][: len(labels)]
            notes = _gen(Scale(labels[0][0], "major"), labels, seed=seed, chord_floors=floors)
            _leap_invariants(notes, labels)


def test_legacy_grid_over_no_bars_is_empty():  # v2 accepted this; v3 must not raise
    assert _gen(Scale("C", "major"), (), notes_per_bar=2) == []


PROG = [ChordSymbol.parse(c) for c in ("C", "Am", "G", "C")]  # ends on C, so the cadence is C
ALLOWED = [
    frozenset(range(60, 70)),
    frozenset(range(57, 69)),
    frozenset(range(59, 70)),
    frozenset(range(55, 70)),
]


@pytest.mark.parametrize("seed", range(30))
def test_allowed_pitches_bound_every_note(seed):
    mel = generate_melody(
        scale=Scale("C", "major"),
        progression=PROG,
        tempo_bpm=120.0,
        seed=seed,
        allowed_pitches=ALLOWED,
        final_whole_note=True,
        tessitura_center=64,
    )
    for n in mel:
        assert n.pitch in ALLOWED[min(int(n.start_s // 2.0), 3)]
    assert mel[-1].start_s == pytest.approx(6.0) and mel[-1].pitch % 12 == 0  # one held C
    assert metrics.forbidden_interval_count(mel) == 0
    assert max(metrics.chord_change_moves(mel, PROG, 1, 120.0)) <= 7
    assert abs(mel[-1].pitch - mel[-2].pitch) <= 7


def test_allowed_pitches_argument_checks():
    with pytest.raises(ValueError):
        generate_melody(scale=Scale("C", "major"), progression=PROG, allowed_pitches=ALLOWED[:3])
    with pytest.raises(ValueError):
        generate_melody(
            scale=Scale("C", "major"),
            progression=PROG,
            allowed_pitches=[frozenset(), *ALLOWED[1:]],
        )
    with pytest.raises(ValueError):
        generate_melody(
            scale=Scale("C", "major"),
            progression=PROG,
            allowed_pitches=ALLOWED,
            chord_floors=[None] * 4,
        )
    with pytest.raises(ValueError):
        generate_melody(
            scale=Scale("C", "major"),
            progression=PROG,
            notes_per_bar=4,
            final_whole_note=True,
        )


# ---- v5 feature 3: phrase shaping (spec 2026-11-16 section 5.3) --------------

PROG8 = [ChordSymbol.parse(c) for c in ("C", "Am", "F", "G", "C", "F", "G", "C")]


def test_score_pitch_phrase_shaping_terms():
    def s(candidate, **kw):
        return score_pitch(candidate, _ctx(**kw), 72)

    # the arch target replaces the register gravity
    assert s(76, contour_target=76.0) > s(72, contour_target=76.0)
    assert s(76) < s(72)
    # peak guard: before the window a new high is penalised, inside it rewarded,
    # a tie inside it or a high after it penalised
    assert s(76, phrase_max=74, peak_phase="before") < s(76)
    assert s(76, phrase_max=74, peak_phase="inside") > s(76)
    assert s(74, phrase_max=74, peak_phase="inside") < s(74)
    assert s(76, phrase_max=74, peak_phase="after") < s(76)
    # the forced final note would tie the phrase's peak
    tie = s(74, phrase_max=76, peak_phase="before", final_choices=(76,))
    assert tie < s(74, phrase_max=76, peak_phase="before")


def _hook_gain(candidate, **kw):
    """What the hook term alone adds to ``candidate``'s score (prev pitch 72)."""
    return score_pitch(candidate, _ctx(**kw), 72) - score_pitch(candidate, _ctx(), 72)


def test_hook_term_rewards_exact_and_near_restatements():
    assert _hook_gain(74, hook_interval=2) == pytest.approx(HOOK_EXACT)  # the hook's own interval
    assert _hook_gain(75, hook_interval=2) == pytest.approx(HOOK_NEAR)  # within a semitone
    assert _hook_gain(73, hook_interval=2) == pytest.approx(HOOK_NEAR)
    assert _hook_gain(76, hook_interval=2) == pytest.approx(
        0.0
    )  # two semitones off: no restatement
    assert _hook_gain(70, hook_interval=2) == pytest.approx(0.0)  # the hook's direction reversed


def test_hook_term_in_a_varied_slot_withholds_the_original_interval():
    # A' varies hook interval 2 to 3 (v5 ruling F3d): the original earns nothing
    varied = dict(hook_interval=3, hook_original=2)
    assert _hook_gain(75, **varied) == pytest.approx(HOOK_EXACT)  # the varied interval
    assert _hook_gain(76, **varied) == pytest.approx(HOOK_NEAR)  # its other neighbour
    assert _hook_gain(74, **varied) == pytest.approx(0.0)  # the original interval
    assert score_pitch(74, _ctx(**varied), 72) < score_pitch(75, _ctx(**varied), 72)


def test_a_prime_openings_rarely_restate_the_hook_unchanged():
    identical = 0
    for seed in range(100):
        mel = generate_melody(
            scale=Scale("C", "major"),
            progression=PROG8,
            tempo_bpm=120.0,
            seed=seed,
            phrase_shaping=True,
            final_whole_note=True,
        )
        hook = metrics.hook_cell(mel, 120.0)
        bar4 = [n for n in mel if 8.0 <= n.start_s < 10.0][: len(hook) + 1]  # A' opens bar 4
        identical += tuple(b.pitch - a.pitch for a, b in itertools.pairwise(bar4)) == hook
    # 19 here; 24 when the varied slot still rewarded the original interval (finding I1).
    # The rest: the varied interval was unplayable in every slot left, or not sampled.
    assert identical <= 20


def test_the_a_prime_after_b_echoes_phrase_a_at_16_bars():
    def onsets(mel, bar):  # bar-relative onsets at 120 bpm (2 s bars)
        return [
            round(n.start_s - 2.0 * bar, 6) for n in mel if 2.0 * bar <= n.start_s < 2.0 * (bar + 1)
        ]

    for seed in range(20):
        mel = generate_melody(
            scale=Scale("C", "major"),
            progression=PROG8 * 2,
            tempo_bpm=120.0,
            seed=seed,
            phrase_shaping=True,
            final_whole_note=True,
        )
        assert [onsets(mel, 12 + i) for i in range(3)] == [onsets(mel, i) for i in range(3)]


def test_phrase_shaping_off_is_unchanged():
    kw = dict(scale=Scale("C", "major"), progression=PROG8, tempo_bpm=120.0, seed=5)
    assert generate_melody(**kw) == generate_melody(**kw, phrase_shaping=False)


def test_phrase_shaping_meets_the_melody_targets():
    rows = []
    for seed in range(20):
        mel = generate_melody(
            scale=Scale("C", "major"),
            progression=PROG8,
            tempo_bpm=120.0,
            seed=seed,
            phrase_shaping=True,
            final_whole_note=True,
        )
        rows.append(
            (
                metrics.hook_restatements(mel, 120.0, 8),
                metrics.peak_uniqueness(mel, 120.0, 8),
                metrics.contour_arch(mel, 120.0, 8),
            )
        )
        assert metrics.forbidden_interval_count(mel) == 0
        assert max(metrics.chord_change_moves(mel, PROG8, 1, 120.0)) <= 7
    hook = [r[0] for r in rows]
    peak = [r[1] for r in rows]
    arch = [r[2] for r in rows]
    assert statistics.fmean(hook) >= 0.75
    # v5 ruling F3b: with the v3 singability guards holding, this sample reaches 0.900.
    # v5 ruling F5h: the gate stays 0.85; this calls generate_melody, so best of 12 does
    # not help it, and 0.9 would leave no margin.
    assert statistics.fmean(peak) >= 0.85
    assert statistics.fmean(arch) >= 0.5


def test_phrase_shaping_with_a_one_note_first_bar():
    mel = generate_melody(
        scale=Scale("C", "major"),
        progression=[ChordSymbol.parse("C")],
        seed=1,
        phrase_shaping=True,
        final_whole_note=True,
    )
    assert len(mel) == 1


def test_phrase_shaping_needs_phrase_rhythms():
    with pytest.raises(ValueError):
        generate_melody(
            scale=Scale("C", "major"), progression=PROG8, notes_per_bar=4, phrase_shaping=True
        )


def test_form_labels_follow_the_phrase_layout():
    for n in (1, 2, 3, 4, 5, 7, 8, 9, 16):
        labels = form_labels(n)
        assert len(labels) == len(phrase_layout(n).phrase_lengths) and labels[0] == "A"
    assert form_labels(16)[:4] == ("A", "A'", "B", "A'")
    assert form_labels(8) == ("A", "A'")
    assert form_labels(4) == ("A",)


def test_only_four_or_more_phrases_get_a_b():
    # v5 ruling F3h: three phrases are A A' A', so a 9-12 bar piece ends at home
    for n in range(1, 21):
        has_b = "B" in form_labels(n)
        assert has_b == (len(phrase_layout(n).phrase_lengths) >= 4) == (n >= 13), n
    for n in (9, 10, 11, 12):
        assert form_labels(n) == ("A", "A'", "A'")
    assert form_labels(13) == ("A", "A'", "B", "A'")
    assert form_labels(17) == ("A", "A'", "B", "A'", "A")


def test_the_b_phrase_of_a_16_bar_melody_is_raised_and_has_its_own_rhythm():
    def bars_of(mel, first):  # four bars at 120 bpm (2 s bars)
        return [n for n in mel if 2.0 * first <= n.start_s < 2.0 * (first + 4)]

    def body_rhythms(mel, first):  # each body bar's bar-relative onsets
        return {
            tuple(
                round(n.start_s - 2.0 * b, 6) for n in mel if 2.0 * b <= n.start_s < 2.0 * (b + 1)
            )
            for b in range(first, first + 3)
        }

    lifts = []
    for seed in range(20):
        mel = generate_melody(
            scale=Scale("C", "major"),
            progression=PROG8 * 2,
            tempo_bpm=120.0,
            seed=seed,
            phrase_shaping=True,
            final_whole_note=True,
        )
        a = statistics.fmean(n.pitch for n in bars_of(mel, 0))
        b = statistics.fmean(n.pitch for n in bars_of(mel, 8))
        lifts.append(b - a)
        assert not body_rhythms(mel, 0) & body_rhythms(mel, 8), seed  # B's own body rhythms
    # The B arch sits B_REGISTER_LIFT (2) higher; the notes follow it on average
    # (2.6 here), not in every piece: the hard rules and the chords still decide.
    assert statistics.fmean(lifts) >= 1
    assert sum(lift >= 1 for lift in lifts) >= 15  # 16 of the 20 here


# ---- v5 feature 5: the soft pitch prior (spec 2026-11-16 section 5.5) ---------


def _prior_gain(candidate, target):
    """What the prior term alone adds to ``candidate``'s score."""
    return score_pitch(candidate, _ctx(prior_target=target), 72) - score_pitch(
        candidate, _ctx(), 72
    )


def test_prior_bonus_falls_off_with_octave_aware_distance():
    assert PRIOR_WEIGHT == 2.0 and PRIOR_WIDTH == 2.0
    assert _prior_gain(74, 74) == pytest.approx(PRIOR_WEIGHT)
    assert _prior_gain(74, 74 + 24) == pytest.approx(PRIOR_WEIGHT)  # octaves away: same pull
    assert _prior_gain(76, 74) == pytest.approx(PRIOR_WEIGHT * math.exp(-2 / PRIOR_WIDTH))
    assert _prior_gain(72, 83) == pytest.approx(PRIOR_WEIGHT * math.exp(-1 / PRIOR_WIDTH))
    assert _prior_gain(74, None) == 0.0


def _n(start, end, pitch):
    return Note(start_s=start, end_s=end, pitch=pitch, velocity=80, confidence=1.0)


def test_prior_target_is_the_highest_sounding_note_else_the_next_onset_within_an_eighth():
    prior = [_n(0.0, 1.0, 60), _n(0.5, 1.0, 67), _n(1.3, 1.5, 64), _n(1.3, 1.5, 65)]
    eighth = 0.25
    assert _prior_target(prior, 0.0, eighth) == 60
    assert _prior_target(prior, 0.5, eighth) == 67  # the highest of those sounding
    assert _prior_target(prior, 1.0, eighth) is None  # both end here; 1.3 is over an eighth on
    assert _prior_target(prior, 1.05, eighth) == 65  # the highest at the nearest onset ahead
    assert _prior_target(prior, 1.5, eighth) is None
    assert _prior_target([], 0.0, eighth) is None
