# v5 Feature 2: Per-Bar Chord Shapes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let each fingerstyle bar pick its own shape of the chord, chosen by a Viterbi search over the faithful shapes. A repair pass re-voices any bar that still misses chord identity. This fills in the missing sevenths and widens the melody.

**Architecture:**
- A new module, `compose/shapes.py`, builds each bar's candidate shapes, scores them, and picks the sequence with the lowest cost.
- `_arrange_fingerstyle` builds hands from that per-bar sequence and calls the existing `arrange_fingerstyle` with one voicing and one hand per bar, passing `bars_per_chord=1`. Both already support this.
- The repair pass reuses `body_bar` with an alternative hand that reaches the bar's melody notes.
- Feature 1's `shape_dynamics` and `let_ring` still run last.

**Tech Stack:** Python 3.11, numpy, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-11-16-composer-v5-design.md` (§5.2, §6 items 1, 2 and 4).

## Global Constraints

- Activate the venv first: `source .venv/bin/activate`.
- Strum and arpeggio must not change. `tests/unit/test_compose_frozen.py` must pass untouched.
- `compose()`'s signature and the package exports stay frozen. `COMPOSER_VERSION` stays 5.
- The fingerstyle v4 guarantees hold on every matrix and every preset tuning: P1, P2 and P4 are 1.0, P3t is 1.0, P5 is 0, A4 and H2 hold, and the v3 hard melody rules hold. The seeded random-progression property test passes.
- The melody's pitches are never changed for identity. A repaired bar keeps every melody pitch, and only positions may change.
- The feature-2 targets, measured with the defaults:
  - P3 mean is at least 0.99 on every matrix;
  - `F.M6` mean is at least the v4 value + 2 semitones on every matrix;
  - `F.shape_moves` mean is at most 0.5.
- Gates before each commit: `pytest -q tests/unit`, `ruff check src tests`, `ruff format --check src tests`, `mypy --strict src`. Run `pytest -q tests/integration` at the last task.
- If a v4 guarantee fails, stop and report the case. Do not loosen the assertion.
- End every commit message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Branch: `feat/composer-v5`.

## Review Focus

1. **A chord held across several bars** (`bars_per_chord` 2 to 4): it should keep one shape unless a bar needs a repair. The tab must still name the chord on every bar. Test: Task 1's `test_a_held_chord_keeps_its_shape`.
2. **Chords with few candidate shapes** (dim, aug, or awkward roots in Drop C and DADGAD): the search must always return a shape for every bar. The v4 shape is always candidate 0. Test: Task 1's `test_every_quality_and_root_gets_a_shape_in_every_tuning`.
3. **Long progressions** (16 or more bars): Viterbi time grows linearly with bars, and one piece must arrange in under 1 s. Test: Task 2's `test_a_long_piece_arranges_quickly`.
4. **The final bar:** it keeps feature 6's final-voicing logic. Its hand comes from the chosen shape, and the cadence is still voiceable (P5 = 0). This is covered by the matrix gates in Task 2.
5. **Repair that would break playability:** a repaired bar must pass P1 and P2. Otherwise the original bar is kept and counted. Test: Task 2's `test_repair_never_breaks_playability`.

---

### Task 1: Per-bar shape search (`compose/shapes.py`)

**Files:**
- Create: `src/music_decoder/compose/shapes.py`
- Test: `tests/unit/test_compose_shapes.py` (new)

**Interfaces:**
- Consumes:
  - `voicings.voicings_for(chord, tuning)[0]` (the v4 shape);
  - `voicings.ranked_shapes(chord, tuning) -> tuple[tuple[int, ...], ...]` (frets per string, cached);
  - `fingerstyle.hand_shape(voicing, tuning) -> HandShape` (`root`, `alternate`, `fill_strings`, `melody_strings`, `window`, `allowed`);
  - `harmony.essential_pcs`.
- Produces:
  - `MAX_CANDIDATES = 8`
  - `shape_candidates(chord: ChordSymbol, tuning: Tuning) -> tuple[VoicedChord, ...]`: the v4 shape first, then `ranked_shapes` in order, with no duplicate frets. Each must give a valid `hand_shape`, meaning a non-empty `allowed`. At most `MAX_CANDIDATES` are returned.
  - `bar_cost(hand: HandShape, essential: frozenset[int], rank: int) -> float`
  - `move_cost(a: HandShape, b: HandShape) -> float`
  - `choose_bar_shapes(progression: Sequence[ChordSymbol], *, bars_per_chord: int, tuning: Tuning) -> tuple[VoicedChord, ...]`: one shape per bar, found by Viterbi over bars. Ties go to the lower candidate rank.

**Pinned costs:**
- **Identity risk:** for each essential pitch class, take the cheapest of these:
  - 0 if a thumb note (`root` or `alternate`) sounds it;
  - 0.3 if a fill-string note below the shape's top sounding string sounds it;
  - 1.0 if only the top sounding string sounds it.

  Add up the results and multiply by `W_IDENTITY = 3.0`.
- **Reach:** `−0.15 × len(hand.allowed)`.
- **Rank:** `0.4 × rank`, where the v4 shape is rank 0.
- **Register:** `0.05 × |median(sorted(hand.allowed)) − 67|`. 67 is G4.
- **Move:** `move_cost(a, b) = 0.25 × |a.window[0] − b.window[0]|`, which is 0 when the frets are identical.

- [ ] **Step 1: Write the failing tests**

```python
def test_candidates_start_with_the_v4_shape_and_are_faithful():
    for label in ("C", "Dm7", "G7", "F#m", "A#maj7", "Bdim", "Caug", "Esus4"):
        chord = ChordSymbol.parse(label)
        cands = shape_candidates(chord, STANDARD_EADGBE)
        assert cands[0] == voicings_for(chord, STANDARD_EADGBE)[0]
        assert 1 <= len(cands) <= MAX_CANDIDATES
        assert all(not shape_faults([p.fret for p in v.positions], STANDARD_EADGBE, chord) for v in cands)

@pytest.mark.parametrize("tuning", list(PRESETS.values()), ids=list(PRESETS))
def test_every_quality_and_root_gets_a_shape_in_every_tuning(tuning):
    for root in VALID_TONICS:
        for suffix in ("", "m", "7", "maj7", "m7", "dim", "sus4", "aug"):
            prog = [ChordSymbol.parse(root + suffix)]
            assert len(choose_bar_shapes(prog, bars_per_chord=2, tuning=tuning)) == 2

def test_a_held_chord_keeps_its_shape():
    shapes = choose_bar_shapes([ChordSymbol.parse("Dm7")], bars_per_chord=4, tuning=STANDARD_EADGBE)
    assert len({tuple(p.fret for p in v.positions) for v in shapes}) == 1

def test_seventh_chords_prefer_shapes_that_keep_their_seventh_off_the_top_string():
    prog = [ChordSymbol.parse(c) for c in ("Cmaj7", "Fmaj7", "Dm7", "G7")]
    chosen = choose_bar_shapes(prog, bars_per_chord=1, tuning=STANDARD_EADGBE)
    v4 = [voicings_for(c, STANDARD_EADGBE)[0] for c in prog]
    risk = lambda vs: sum(bar_cost(hand_shape(v, STANDARD_EADGBE), essential_pcs(v.chord), 0)
                          + 0.15 * len(hand_shape(v, STANDARD_EADGBE).allowed) for v in vs)
    assert risk(chosen) < risk(v4)      # identity-risk part of the cost drops

def test_choose_is_deterministic():
    prog = [ChordSymbol.parse(c) for c in ("G", "D7", "Em7", "C")]
    assert choose_bar_shapes(prog, bars_per_chord=1, tuning=STANDARD_EADGBE) == \
           choose_bar_shapes(prog, bars_per_chord=1, tuning=STANDARD_EADGBE)
```

- [ ] **Step 2: Run** `pytest -q tests/unit/test_compose_shapes.py`. Expected: FAIL (module missing).
- [ ] **Step 3: Implement** `shapes.py` with the interfaces and pinned costs above. Viterbi runs over bars, with `bar_cost` as the state cost and `move_cost` as the transition cost.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Run the gates and commit**: `feat(compose): per-bar chord shape search`.

---

### Task 2: Wire per-bar shapes into fingerstyle, add the repair pass and metrics

**Files:**
- Modify: `src/music_decoder/compose/api.py` (`Arrangement.bar_voicings`; `_arrange_fingerstyle`)
- Modify: `src/music_decoder/compose/shapes.py` (`repair_identity`)
- Modify: `src/music_decoder/compose/tabtext.py` (`render_fingerstyle_tab` lists each distinct shape once, in order of first use)
- Modify: `src/music_decoder/compose/metrics.py` (`shape_moves`)
- Modify: `src/music_decoder/evaluation/compose_musicality.py` (`F.shape_moves`)
- Create: `tests/fixtures/compose/musicality_v4_fingerstyle.json`
- Test: `tests/unit/test_compose_shapes.py`, `tests/unit/test_compose_musicality.py`, `tests/unit/test_compose_tabtext.py`

**Interfaces:**
- Consumes:
  - Task 1's `choose_bar_shapes` and `shape_candidates`;
  - `fingerstyle.body_bar(hand, melody, *, bar_start_s, beat_s, essential) -> BarPart`;
  - `fingerstyle.arrange_fingerstyle(voicings, melody, hands, *, bars_per_chord, tempo_bpm)`;
  - `fingerstyle.final_bar_pitches(hand, cadence, essential)`;
  - feature 1's `shape_dynamics` and `let_ring`.
- Produces:
  - `Arrangement.bar_voicings: tuple[VoicedChord, ...] = ()`: one per bar for fingerstyle, empty for the other styles. `Arrangement.voicings` stays one per chord, the first bar's shape.
  - `repair_identity(part: FingerstylePart, bar_voicings: Sequence[VoicedChord], progression: Sequence[ChordSymbol], *, bars_per_chord: int, tempo_bpm: float, tuning: Tuning) -> tuple[FingerstylePart, tuple[VoicedChord, ...]]`.
    - For each body bar in `part.identity_misses`, try `shape_candidates(chord)` in cost order, skipping the current shape.
    - Accept the first candidate whose hand reaches every melody pitch of that bar (`pitch in hand.allowed`), whose re-arranged bar meets identity, and whose result keeps P1 and P2 at 1.0 for that bar.
    - On acceptance, replace that bar's notes and its voicing.
    - It returns the updated part, with `identity_misses` recomputed, and the updated bar voicings.
  - `metrics.shape_moves(bar_voicings: Sequence[VoicedChord], tuning: Tuning) -> float | None`: the share of consecutive bar pairs whose hand windows differ in their lowest fret. `None` when there are fewer than 2 bars.
  - The harness key `F.shape_moves`.
- `_arrange_fingerstyle` order:
  1. `bars = choose_bar_shapes(...)`.
  2. `hands = [hand_shape(v, tuning) for v in bars]`.
  3. The allowed set per bar is `hands[bar].allowed`; the last bar's comes from `final_bar_pitches(hands[-1], ...)`.
  4. `generate_melody`, as in v4.
  5. `arrange_fingerstyle(bars, melody, hands, bars_per_chord=1, ...)`.
  6. `repair_identity`.
  7. `shape_dynamics`.
  8. `let_ring`.
  9. Fill `Arrangement`, including `bar_voicings`.
- `compose()` passes `arrangement.bar_voicings` with `bars_per_chord=1` to `render_fingerstyle_tab`, so chord names stay right per bar.
- **The v4 fixture:** check out `master` (6b9cf8a, composer v4) in a temporary `git worktree`. Run `evaluate(ALL_MATRICES)` there and save `{matrix: {"F.M6": mean, "P3": mean}}` to `tests/fixtures/compose/musicality_v4_fingerstyle.json`, with a `"source"` field naming the commit. Remove the worktree afterwards. PYTHONPATH must point at the worktree's `src`, because the venv's editable install points at this checkout.

- [ ] **Step 1: Write the failing tests**

```python
# test_compose_shapes.py
def test_repair_never_breaks_playability():
    for labels in (("Cmaj7", "Fmaj7", "Dm7", "G7"), ("G", "D7", "G"), ("Dm", "A7", "Dm")):
        for seed in range(10):
            prog = [ChordSymbol.parse(c) for c in labels]
            a = arrange(Scale(labels[0].rstrip("m7aj"), "major" if "m" not in labels[0][:2] else "minor"),
                        prog, tempo_bpm=120.0, style="fingerstyle", seed=seed)
            part = [*a.melody_tab, *a.accompaniment_tab]
            assert metrics.playable_moment_ratio(part) == 1.0
            assert metrics.melody_onset_on_top_ratio(a.melody, a.accompaniment) == 1.0
            assert len(a.bar_voicings) == len(prog)

def test_sevenths_keep_their_seventh_more_often_than_v4():
    prog = [ChordSymbol.parse(c) for c in ("Cmaj7", "Fmaj7", "Dm7", "G7")]
    hits = total = 0
    for seed in range(20):
        a = arrange(Scale("C", "major"), prog, bars_per_chord=2, tempo_bpm=70.0, style="fingerstyle", seed=seed)
        ident = metrics.bar_identity(a.accompaniment, prog, 2, 70.0)
        hits += sum(ident); total += len(ident)
    assert hits / total >= 0.95      # v4 sample 13 met 3 of 8 bars

def test_a_long_piece_arranges_quickly():
    prog = [ChordSymbol.parse(c) for c in ("C", "Am7", "Dm7", "G7") * 4]
    t = time.perf_counter()
    arrange(Scale("C", "major"), prog, tempo_bpm=100.0, style="fingerstyle", seed=1)
    assert time.perf_counter() - t < 1.0

# test_compose_tabtext.py
def test_fingerstyle_tab_lists_each_distinct_shape_once(): ...
#   two bars of C (same shape) then G → the Chords block contains "C" once and "G" once

# test_compose_musicality.py (uses the existing all_reports + reserve_report fixtures)
V4_FS = json.loads((Path(__file__).parents[1] / "fixtures" / "compose" / "musicality_v4_fingerstyle.json").read_text())

@pytest.mark.parametrize("matrix", ALL_MATRICES)
def test_per_bar_shapes_meet_the_v5_targets(all_reports, reserve_report, matrix):
    o = {**all_reports, **reserve_report}[matrix]["overall"]
    assert o["P3"]["mean"] >= 0.99
    assert o["F.M6"]["mean"] >= V4_FS[matrix]["F.M6"] + 2
    assert o["F.shape_moves"]["mean"] <= 0.5
```

(`test_repair_never_breaks_playability`'s scale expression is only a shorthand. Write explicit `(key, mode)` pairs in the test.)

- [ ] **Step 2: Run** the new tests. Expected: FAIL.
- [ ] **Step 3: Implement** the wiring, `repair_identity`, the tab change, the metric, the harness key and the v4 fixture.
- [ ] **Step 4: Run** `pytest -q tests/unit tests/integration`. Expected: PASS.
  - If a feature-2 target is not met, record the per-matrix values and report DONE_WITH_CONCERNS. Do not tune the targets.
  - A v4-guarantee failure means stop and report.
- [ ] **Step 5: Run the gates and commit**: `feat(compose): per-bar shapes and identity repair for fingerstyle`.
