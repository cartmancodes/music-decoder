from dataclasses import replace
from itertools import pairwise

import numpy as np
import pytest

from music_decoder.compose import metrics
from music_decoder.compose import tune as tune_module
from music_decoder.compose.harmony import chord_scale_pcs, scale_pcs
from music_decoder.compose.melody import cadence_pc
from music_decoder.compose.rhythm import split_phrases
from music_decoder.compose.tune import (
    _hold_repeat_share,
    _Piece,
    _replay_starts,
    _Spot,
    _Voice,
    _weights,
    generate_tune,
    plan_rhythm,
    scale_steps,
)
from music_decoder.compose.tune_metrics import tune_features
from music_decoder.compose.tune_stats import load_tune_stats
from music_decoder.types import ChordSymbol, Note, Scale

STATS = load_tune_stats()
C, G, AM, F = (ChordSymbol.parse(x) for x in ("C", "G", "Am", "F"))


def wide(scale, n_bars, low=55, high=72):
    pcs = set(scale_pcs(scale))
    return [frozenset(p for p in range(low, high + 1) if p % 12 in pcs)] * n_bars


def tune(seed=1, scale="C:major", prog=(C, G, AM, F), bars=2, tempo=96.0, allowed=None, **kw):
    s = Scale.parse(scale)
    allowed = allowed if allowed is not None else wide(s, len(prog) * bars)
    return generate_tune(
        scale=s,
        progression=list(prog),
        bars_per_chord=bars,
        tempo_bpm=tempo,
        seed=seed,
        tessitura_center=64,
        allowed_pitches=allowed,
        **kw,
    )


def test_scale_steps():
    c_major = {0, 2, 4, 5, 7, 9, 11}
    assert scale_steps(60, 64, c_major) == 2 and scale_steps(64, 60, c_major) == -2
    assert scale_steps(60, 60, c_major) == 0 and scale_steps(61, 62, c_major) == 1
    assert scale_steps(60, 84, c_major) == 7


def test_same_seed_same_tune_and_seeds_differ():
    assert tune(7) == tune(7) and tune(7) != tune(8)


def test_grid_bars_and_allowed():
    e, bar = 60 / 96 / 2, 4 * 60 / 96
    for seed in range(30):
        notes = tune(seed)
        allowed = wide(Scale.parse("C:major"), 8)
        for n in notes:
            assert abs(n.start_s / e - round(n.start_s / e)) < 1e-6
            b = int(n.start_s / bar + 1e-6)
            assert n.pitch in allowed[b] and n.end_s <= (b + 1) * bar + 1e-9


def test_hard_interval_rules_and_chord_changes():
    bar = 4 * 60 / 96
    for seed in range(60):
        notes = tune(seed, bars=1, prog=(C, G, AM, F, C, G, AM, F))
        for a, b in pairwise(notes):
            size = abs(b.pitch - a.pitch)
            assert size <= 12 and size not in (6, 10, 11)
            if int(a.start_s / bar + 1e-6) != int(b.start_s / bar + 1e-6):
                assert size <= 7  # every bar is a chord change here
    for seed in range(20):  # two bars per chord: the limit applies only where the chord changes
        notes = tune(seed)
        assert max(metrics.chord_change_moves(notes, [C, G, AM, F], 2, 96.0), default=0) <= 7


def test_final_note_is_the_tonic_held_for_the_bar():
    for seed in range(20):
        last = tune(seed, prog=(C, F, G, C))[-1]
        assert last.pitch % 12 == 0 and last.end_s - last.start_s == pytest.approx(
            4 * 60 / 96 * 0.95
        )


@pytest.mark.parametrize("n_chords,bars", [(1, 1), (2, 1), (3, 1), (5, 1), (3, 3), (4, 4)])
def test_odd_lengths(n_chords, bars):
    prog = (C, G, AM, F, C)[:n_chords]
    notes = tune(3, prog=prog, bars=bars)
    assert notes[-1].pitch % 12 == cadence_pc(Scale.parse("C:major"), prog[-1])
    assert notes[-1].end_s - notes[-1].start_s == pytest.approx(4 * 60 / 96 * 0.95)


def test_out_of_key_chords_use_their_chord_scale():
    e7 = ChordSymbol.parse("E7")
    prog = (ChordSymbol.parse("Am"), ChordSymbol.parse("Dm"), e7, ChordSymbol.parse("Am"))
    chromatic = [frozenset(range(55, 73))] * 8
    bar = 4 * 60 / 96
    for seed in range(20):
        for n in tune(seed, scale="A:minor", prog=prog, allowed=chromatic):
            if int(n.start_s / bar + 1e-6) in (4, 5):
                assert n.pitch % 12 != 7  # G natural gives way to G sharp over E7


def test_small_allowed_sets_do_not_crash():
    tiny = [frozenset({60, 62, 64, 65, 67})] * 7 + [frozenset({60, 64, 67})]
    for seed in range(20):
        notes = tune(seed, allowed=tiny)
        assert all(abs(b.pitch - a.pitch) not in (6, 10, 11) for a, b in pairwise(notes))


def test_rejects_bad_allowed_pitches():
    with pytest.raises(ValueError):
        tune(1, allowed=wide(Scale.parse("C:major"), 7))
    with pytest.raises(ValueError):
        tune(1, allowed=[frozenset()] * 8)


def test_it_moves_like_a_real_tune():
    scale = Scale.parse("C:major")
    feats = [tune_features(tune(s), scale, [C, G, AM, F], 2, 96.0) for s in range(80)]
    med = {k: float(np.median([f[k] for f in feats if k in f])) for k in feats[0]}
    assert 0.10 <= med["repeat"] <= 0.40 and med["step"] <= 0.60
    assert med["offbeat"] >= 0.30 and med["figure_recurrence"] >= 0.10
    assert med["rhythm_variety"] >= 0.5 and med["chord_tone_strong"] <= 0.95


def test_prior_pulls_toward_the_proposed_line():
    e = 60 / 96 / 2
    prior = [
        Note(start_s=k * e, end_s=(k + 1) * e, pitch=71, velocity=80, confidence=1.0)
        for k in range(64)
    ]

    def share(p):
        notes = [n for s in range(40) for n in tune(s, pitch_prior=p)]
        return np.mean([n.pitch % 12 == 11 for n in notes])

    assert share(prior) > share(None) + 0.05


def bars_of(notes, tempo=96.0):
    e = 60 / tempo / 2
    out = {}
    for n in notes:
        out.setdefault(round(n.start_s / e) // 8, []).append(n)
    return out


def test_restated_bars_replay_their_source_moves():
    scale = Scale.parse("C:major")
    ladder = set(scale_pcs(scale))
    offsets = [k for length in split_phrases(16) for k in range(length)]
    same = total = 0
    for seed in range(60):
        plan = plan_rhythm(16, 96.0, np.random.default_rng(seed), STATS)
        bars = bars_of(tune(seed, prog=(C, G, AM, F) * 2, allowed=wide(scale, 16)))
        for b, rhythm in enumerate(plan[:-1]):
            if rhythm.restated_from is None or offsets[b] >= 2 or len(rhythm.pattern) < 2:
                continue

            def moves(notes):
                return [scale_steps(x.pitch, y.pitch, ladder) for x, y in pairwise(notes)]

            total += 1
            same += moves(bars[b]) == moves(bars[rhythm.restated_from])
    assert total >= 20 and same / total >= 0.75


def test_phrase_ends_on_a_chord_tone_other_than_the_tonic():
    for seed in range(30):
        bars = bars_of(tune(seed, prog=(F, C, G, C)))  # bar 3, a cadence bar, is over C
        assert bars[3][-1].pitch % 12 in (4, 7)


def test_lengths_and_velocities():
    e = 60 / 96 / 2
    for seed in range(10):
        for notes in bars_of(tune(seed)).values():
            slots = [round(n.start_s / e) % 8 for n in notes]
            for n, slot, nxt in zip(notes, slots, [*slots[1:], 8], strict=True):
                assert n.end_s - n.start_s == pytest.approx(0.95 * (nxt - slot) * e)
                assert n.velocity == (80 if slot == 0 else 75 if slot == 4 else 70)
                assert n.confidence == 1.0


def test_without_a_final_whole_note_the_last_note_is_still_the_cadence():
    for seed in range(20):
        notes = tune(seed, prog=(C, F, G, C), final_whole_note=False)
        assert notes[-1].pitch % 12 == 0
    assert any(len(bars_of(tune(s, final_whole_note=False))[7]) > 1 for s in range(20))


def test_default_stats_are_the_shipped_ones():
    assert tune(5) == tune(5, stats=STATS)


def test_approach_drops_the_chord_scale_before_the_final_approach():
    # Over Am in G major only Eb 63 and F 65 (off the ladder) reach the final Bb 70 smoothly.
    prog = [ChordSymbol.parse(x) for x in ("Bb", "E7", "Am", "Bb")]
    ranges = [range(58, 72), range(58, 72), range(56, 66), range(63, 77)]
    allowed = [frozenset(r) for r in ranges]
    for seed in range(200):
        notes = generate_tune(
            scale=Scale.parse("G:major"),
            progression=prog,
            seed=seed,
            tessitura_center=64,
            allowed_pitches=allowed,
        )
        sizes = [abs(b.pitch - a.pitch) for a, b in pairwise(notes)]
        assert all(s <= 12 and s not in (6, 10, 11) for s in sizes)
        assert sizes[-1] <= 7


def test_approach_drops_the_chord_change_limit_before_the_final_approach(monkeypatch):
    # The tier order is tested with an octave as the general limit; since ruling R10 the
    # shipped general limit equals the chord-change limit (a fifth), so this tier is moot.
    monkeypatch.setattr(tune_module, "MAX_LEAP", 12)
    one_note_bars = {role: {(0,): 10} for role in ("open", "body", "cadence")}
    stats = replace(STATS, rhythm=one_note_bars, rhythm_pairs={r: {} for r in one_note_bars})
    scale = Scale.parse("C:major")
    above = frozenset(p for p in range(67, 80) if p % 12 in set(scale_pcs(scale)))
    allowed = [frozenset({60, 64, 67}), frozenset({67}), above, frozenset({84})]
    for seed in range(20):
        notes = tune(seed, prog=(C, F, G, C), bars=1, allowed=allowed, stats=stats)
        # 79 is the only pitch from which the final C 84 is a smooth move; reaching it from
        # G 67 over the new chord takes an octave, beyond the chord-change limit.
        assert [n.pitch for n in notes][-2:] == [79, 84]


def test_a_repeat_keeps_the_move_rows_share():
    row = {0: 25, 1: 50, -1: 25}  # repeats are a quarter of this row
    pool = [60, 62, 64, 65]
    # Factors that favour the repeat no more than the other candidates (here all 3, the
    # repeat's own move weight inflated to 4) may not raise its share above the row's.
    weights = [3.0, 12.0, 3.0, 3.0]
    _hold_repeat_share(weights, [1.0, 4.0, 1.0, 1.0], [3.0] * 4, pool, 62, row)
    assert weights[1] / sum(weights) == pytest.approx(0.25)
    weights = [2.0, 1.0]
    _hold_repeat_share(weights, [2.0, 1.0], [1.0, 1.0], [60, 62], 62, row)  # moves out of reach
    assert weights[1] / sum(weights) == pytest.approx(0.25)
    # Rulings R11, R11a: the repeat's own factors still count, against the other candidates'
    # factors weighted by their moves: (2 x 1 + 1 x 4 + 1 x 1) / 4 = 1.75 here (the plain mean
    # would be 2). Twice that doubles the odds of repeating (1/3 -> 2/3, a share of 0.4),
    # half of it halves them (a share of 1/7).
    moves, others = [2.0, 1.0, 1.0, 1.0], [1.0, 4.0, 1.0]
    for repeat_factor, expected in ((3.5, 0.4), (0.875, 1 / 7)):
        factors = [others[0], repeat_factor, others[1], others[2]]
        weights = [m * f for m, f in zip(moves, factors, strict=True)]
        _hold_repeat_share(weights, moves, factors, pool, 62, row)
        assert weights[1] / sum(weights) == pytest.approx(expected)
    weights = [1.0, 1.0]
    _hold_repeat_share(weights, [1.0, 1.0], [1.0, 1.0], [60, 64], 62, {0: 25, 1: 75})
    assert weights == [1.0, 1.0]  # no repeat in the pool
    weights = [1.0, 3.0]
    _hold_repeat_share(weights, [1.0, 1.0], [1.0, 3.0], [60, 62], 62, {0: 25})
    assert weights == [1.0, 3.0]  # a row of repeats only


def test_a_repeat_off_the_new_chord_is_less_likely_than_one_on_it():
    # On the downbeat of a new G chord in C major, repeating A (not a chord tone) must be
    # less likely than repeating G (a chord tone); the harmony factor still judges repeats.
    scale = Scale.parse("C:major")
    piece = _Piece(scale=scale, stats=STATS, center=68, prior=None, eighth_s=60 / 96 / 2)
    spot = replace(_spot_for(G, scale), change=True)
    pool = [p for p in range(60, 77) if p % 12 in set(scale_pcs(scale))]

    def repeat_share(prev):
        voice = _Voice(prev=prev, prev_move=1, prev_chord=C)
        weights = _weights(pool, spot, voice, piece)
        return weights[pool.index(prev)] / sum(weights)

    assert repeat_share(69) < 0.6 * repeat_share(67)


def test_repeats_are_as_rare_as_in_real_tunes():
    # Before the repeat kept its row share, the harmony and degree factors made 30% of the
    # moves here repeats (38% on the hand shapes); real tunes have ~20% (band ~0.12-0.28).
    scale = Scale.parse("C:major")
    feats = [tune_features(tune(s), scale, [C, G, AM, F], 2, 96.0) for s in range(80)]
    assert float(np.median([f["repeat"] for f in feats])) <= 0.28


def test_a_replayed_bar_does_not_start_on_the_previous_pitch_by_default():
    # The replay's start used to break ties toward the previous pitch, so a third of the
    # replayed bars here began with a repeat (over 40% on the hand shapes); it is now drawn
    # by the usual weights (14% here).
    scale = Scale.parse("C:major")
    into = repeats = 0
    for seed in range(80):
        plan = plan_rhythm(8, 96.0, np.random.default_rng(seed), STATS)
        bars = bars_of(tune(seed))
        for b, rhythm in enumerate(plan[:-1]):
            if rhythm.restated_from is not None and b - 1 in bars and b in bars:
                into += 1
                repeats += bars[b][0].pitch == bars[b - 1][-1].pitch
    assert into >= 40 and repeats / into <= 0.25
    pool = list(range(60, 73))
    spot_c = _spot_for(C, scale)
    starts = _replay_starts(pool, [2, -2], (0, 2, 4), spot_c)
    assert 60 in starts and 64 in starts and len(starts) > 1  # 60-64-60 and 64-67-64 both tie


def _spot_for(chord, scale):
    return _Spot(
        slot=0,
        onset_s=0.0,
        chord=chord,
        ladder=chord_scale_pcs(scale, chord),
        change=False,
        feasible=None,
        feasible_wide=None,
        phrase_end=False,
    )


def test_no_move_is_larger_than_a_fifth():
    # Ruling R10: a hard filter, so v3's L2 and L3 (no unrecovered or unjustified leap over a
    # fifth) hold by construction, on wide ranges and on narrow, shifting ones alike.
    narrow = [frozenset(range(lo, lo + 15)) for lo in (55, 62, 57, 64, 55, 60, 59, 62)]
    for seed in range(100):
        for allowed in (None, narrow):
            notes = tune(seed, allowed=allowed)
            assert metrics.max_interval(notes) <= 7
            assert metrics.unjustified_large_leap_count(notes, [C, G, AM, F], 2, 96.0) == 0


def test_the_bar_before_the_approach_keeps_the_final_reachable():
    # With no move over a fifth (ruling R10), a line entering the approach bar (Am, 56-65)
    # from high in the bar before could find no smooth path to the final Bb 70 (seed 0 ended
    # 71 -> 64 -> 70, a tritone); the bar before now steers toward the approach's sets too.
    prog = [ChordSymbol.parse(x) for x in ("Bb", "E7", "Am", "Bb")]
    allowed = [frozenset(r) for r in (range(58, 72), range(58, 72), range(56, 66), range(63, 77))]
    for seed in range(300):
        notes = generate_tune(
            scale=Scale.parse("G:major"),
            progression=prog,
            seed=seed,
            tessitura_center=64,
            allowed_pitches=allowed,
        )
        sizes = [abs(b.pitch - a.pitch) for a, b in pairwise(notes)]
        assert all(s <= 7 and s != 6 for s in sizes), seed
