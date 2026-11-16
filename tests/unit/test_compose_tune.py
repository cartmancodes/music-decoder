from dataclasses import replace
from itertools import pairwise

import numpy as np
import pytest

from music_decoder.compose import metrics
from music_decoder.compose.harmony import scale_pcs
from music_decoder.compose.melody import cadence_pc
from music_decoder.compose.rhythm import split_phrases
from music_decoder.compose.tune import generate_tune, plan_rhythm, scale_steps
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


def test_approach_drops_the_chord_change_limit_before_the_final_approach():
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
