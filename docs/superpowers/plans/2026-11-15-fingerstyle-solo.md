# Fingerstyle Solo Part Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `compose(style="fingerstyle")` produce one guitar part a single player can play from the tab. The melody sits on the top strings over a Travis-picked thumb, all inside the held chord shape. Strum and arpeggio stay exactly as they are.

**Architecture:**
- A new module, `compose/fingerstyle.py`, owns the fingerstyle part: what each chord shape can reach, the per-bar thumb, fills and chord identity, and the final bar.
- `arrange()` asks it for each bar's reachable pitches and passes them to `generate_melody` through a new optional argument. It then hands the melody back to the module to place every note on a string and fret.
- `compose()` writes that part to the same two MIDI tracks and renders a new combined tab. Strum and arpeggio take the old code paths untouched; a digest fixture taken before any change proves it.

**Tech Stack:** Python 3.11, numpy, pretty_midi, pytest, ruff, mypy --strict (all already in use; no new dependencies).

**Spec:** `docs/superpowers/specs/2026-11-15-fingerstyle-solo-design.md`. Read it first: every task argues from it, and section numbers (§) below refer to it.

## Global Constraints

- Activate the venv first: `source .venv/bin/activate`. Then use `pytest`, `ruff` and `mypy` as below.
- `compose()`'s signature does not change (`tests/unit/test_public_contract.py`). No new CLI flags, no new dependencies.
- Strum and arpeggio output (melody, accompaniment, melody tab, tab text) stays identical to composer v3, note for note. No strum or arpeggio test may change. Only `composer_version` in the metadata moves.
- `COMPOSER_VERSION = 4`.
- MIDI keeps exactly two tracks named `melody` and `accompaniment`, program `GUITAR_PROGRAM = 25`. Every MIDI note equals a tab note: same start, and pitch = the string's open pitch + fret.
- All onsets are on the eighth-note grid, except the final roll: `FINAL_STAGGER_S = 0.010` per string, at most 50 ms in all.
- Pinned values: `THIN_MARGIN_SEMITONES = 3`; `MAX_FRETTED_SPAN = 3` and `MAX_FINGERS = 4` (from `compose/voicings.py`); the melody floor is G3 = MIDI 55; the thumb is at least a minor third (3 semitones) below the melody.
- Precedence: playability always holds, then the v3 hard melody rules, then chord identity. The melody's pitches are never forced for identity.
- Gates before each commit: `pytest -q tests/unit`, `ruff check src tests`, `ruff format --check src tests`, `mypy --strict src`. Run `pytest -q tests/integration` at Tasks 8 and 10.
- **Hard-failure rule.** Stop and report the case to the human partner, without loosening the assertion, if any of these fails: a hard assertion (playability, melody on top at onsets, tab = MIDI, chord identity, a final-voicing fallback, a v3 hard melody rule, A4, H2) or a productionize test.
  Soft fingerstyle metrics are handled differently: in Task 9, Step 6, they are recorded, not gated.
- End every commit message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. The work stays on branch `feat/fingerstyle-solo`.

## Review Focus

These are the inputs most likely to break a real user's piece, most likely first. Each has a test in the task named after it.

1. **Shapes rooted on the D string** (D, Dm, D7, and many shapes in Drop D and DADGAD). Only two melody strings remain, and the third or seventh sits on the top string (§5.4's own example). Expected: every bar still sounds its third and seventh, and nothing is unplayable. Test: Task 8, `test_d_family_bars_keep_their_identity`.
2. **Barre shapes with no open strings** (F, Bm, A#, C#m). No open melody notes may appear, and fills stay inside the barre. Expected: playable at 100%. Test: Task 8, `test_barre_progressions_are_playable`.
3. **The final chord cannot reach the tonic**, or no re-voicing fits. Expected: the piece still ends on the cadence note over a ringing chord. A fallback is counted, never silent. Tests: Task 6, `test_final_pitches_extend_to_an_unreachable_cadence` and `test_final_bar_falls_back_and_says_so`; Task 8, `test_every_root_and_quality_ends_playably`.
4. **Non-standard tunings** (Drop C, DADGAD, D standard, Eb, Drop D). Lower open pitches move the thumb strings and the G3 floor. Expected: the same 100% playability and identity as standard tuning. Test: Task 8, `test_fingerstyle_in_every_preset_tuning`; Task 9's per-tuning groups.
5. **Two bars per chord and single-chord pieces.** A one-bar piece is only its final bar. With two bars per chord, the shape is reused across a bar line. Expected: both compose, playable, with identity in every bar. Test: Task 8, `test_single_chord_and_two_bars_per_chord`.

---

### Task 1: Freeze strum and arpeggio output before anything changes

**Files:**
- Create: `src/music_decoder/evaluation/compose_frozen.py`
- Create: `tests/fixtures/compose/v3_strum_arpeggio.json` (generated)
- Test: `tests/unit/test_compose_frozen.py`

**Interfaces:**
- Produces: `strum_arpeggio_digests() -> dict[str, str]`. It maps each case to the SHA-256 hex of its output.
  - Key: `f"{tuning.name}|{scale.tonic}:{scale.mode}|{' '.join(c.to_label() for c in progression)}|{bars}|{seed}|{style}"`.
  - Inputs: `edge_inputs()` in `STANDARD_EADGBE`, `tunings_inputs()` in every `PRESETS` tuning, and `validation_inputs()[::10]` in standard tuning.
  - Styles: `"strum"` and `"arpeggio"`.
  - Hash `repr((arr.melody, arr.voicings, accompaniment, arr.melody_tab, tab_text))` with these fields:
    - `arr = arrange(..., style="strum", tempo_bpm=TEMPO_BPM)`;
    - `accompaniment` is `arr.accompaniment` for strum, and `accompany(arr.voicings, arr.melody, ..., style="arpeggio")` for arpeggio;
    - `tab_text = render_composition_tab(arr.melody_tab, arr.voicings, bars_per_chord=bars, tempo_bpm=TEMPO_BPM, tuning=tuning)`.
  - Hash the individual fields, never `repr(arr)`. Task 8 adds fields to `Arrangement`, and those must not change the digest.

- [ ] **Step 1: Write the test**

```python
FIXTURE = Path(__file__).parents[1] / "fixtures" / "compose" / "v3_strum_arpeggio.json"

def test_strum_and_arpeggio_match_composer_v3():
    expected = json.loads(FIXTURE.read_text())
    assert len(expected) == 2 * (110 + 15 * len(PRESETS) + 40)
    assert strum_arpeggio_digests() == expected
```

- [ ] **Step 2: Implement `strum_arpeggio_digests` in `compose_frozen.py`.** Use the helpers in `evaluation/compose_musicality.py` (`edge_inputs`, `tunings_inputs`, `validation_inputs`, `TEMPO_BPM`).
- [ ] **Step 3: Generate the fixture on the unchanged code**

Run: `python -c "import json; from music_decoder.evaluation.compose_frozen import strum_arpeggio_digests as d; open('tests/fixtures/compose/v3_strum_arpeggio.json','w').write(json.dumps(d(), indent=1, sort_keys=True))"`

Run `git status`. Expected: only the three new files are listed; the source tree is otherwise clean.
- [ ] **Step 4: Run** `pytest -q tests/unit/test_compose_frozen.py`. Expected: PASS.
- [ ] **Step 5: Run the gates and commit**: `test(compose): freeze strum and arpeggio output at composer v3`.

---

### Task 2: Playability, onset-on-top, bar identity and tab consistency metrics

**Files:**
- Modify: `src/music_decoder/compose/metrics.py` (add a `# ---- fingerstyle part` section at the end)
- Test: `tests/unit/test_compose_metrics.py`

**Interfaces:**
- Consumes: `fingers_needed`, `MAX_FRETTED_SPAN` and `MAX_FINGERS` from `compose/voicings.py`.
- Produces:
  - `moment_playable(positions: Sequence[TabPosition]) -> bool`. This is §5.1 for one moment:
    - one note per string;
    - the fretted span (frets > 0) is at most `MAX_FRETTED_SPAN`;
    - `fingers_needed(frets) <= MAX_FINGERS`, where `frets` has one entry per string up to the highest string used, and `-1` means not sounding.

    §5.1 rule 4 (no open string under a barre) needs no separate check. In `fingers_needed` an open or silent string ends a barre run, so the split barre costs an extra finger. State this in the docstring.
  - `playable_moment_ratio(part: Sequence[TabbedNote]) -> float | None`. A moment is each distinct onset `t`. The notes sounding at it are those with `start_s <= t + _OVERLAP_EPS_S` and `end_s > t + _OVERLAP_EPS_S`. Returns the share of playable moments, or `None` when there are no notes.
  - `melody_onset_on_top_ratio(melody: Sequence[Note], accompaniment: Sequence[Note]) -> float | None`. For each melody onset `t`, the melody pitch must be higher than every accompaniment note with `start_s < t + _STROKE_WINDOW_S` and `end_s > t + _OVERLAP_EPS_S`. That set includes the final roll's late strings.
  - `bar_identity(notes: Sequence[Note], progression: Sequence[ChordSymbol], bars_per_chord: int, tempo_bpm: float) -> list[bool]`. Per bar: `essential_pcs(chord) <=` the pitch classes of `notes` overlapping the bar (same overlap test as `essential_tone_coverage`).
  - `tab_consistent(notes: Sequence[Note], tab: Sequence[TabbedNote], tuning: Tuning) -> bool`. Two conditions:
    - the multiset of `(round(start_s, 6), pitch)` over `notes` equals the same multiset over the tab's notes;
    - every tab note's pitch equals `tuning.open_pitches[string] + fret`.

- [ ] **Step 1: Write the failing tests.** `P(s, f)` is `TabPosition(string=s, fret=f)`; `tn(...)` and `note(...)` are small local builders.

```python
def test_moment_playable():
    assert metrics.moment_playable([P(1, 3), P(2, 2), P(5, 5)])          # span 3, 3 fingers
    assert not metrics.moment_playable([P(1, 3), P(5, 7)])               # span 4
    assert not metrics.moment_playable([P(5, 0), P(5, 3)])               # two notes, one string
    assert not metrics.moment_playable([P(s, f) for s, f in enumerate((1, 3, 2, 4, 1, 3))])  # 5 fingers
    assert metrics.moment_playable([])

def test_playable_moment_ratio_counts_each_onset():
    thumb = [tn(0.0, 0.95, P(1, 3)), tn(0.0, 0.95, P(2, 2))]
    assert metrics.playable_moment_ratio([*thumb, tn(0.5, 0.9, P(5, 3))]) == 1.0
    assert metrics.playable_moment_ratio([*thumb, tn(0.5, 0.9, P(5, 8))]) == 0.5

def test_melody_onset_on_top_ratio_checks_onsets_only():
    melody = [note(0.0, 0.95, 64)]
    assert metrics.melody_onset_on_top_ratio(melody, [note(0.0, 0.5, 60)]) == 1.0
    assert metrics.melody_onset_on_top_ratio(melody, [note(0.0, 0.5, 67)]) == 0.0
    assert metrics.melody_onset_on_top_ratio(melody, [note(0.25, 0.45, 67)]) == 1.0  # fill over a held note
    assert metrics.melody_onset_on_top_ratio(melody, [note(0.03, 0.5, 67)]) == 0.0   # late roll string

def test_bar_identity_needs_every_essential_tone():
    prog = [ChordSymbol.parse("C"), ChordSymbol.parse("G7")]
    acc = [note(0.0, 0.5, 48), note(0.5, 1.0, 52), note(2.0, 2.5, 43), note(2.5, 3.0, 47)]
    assert metrics.bar_identity(acc, prog, 1, 120.0) == [True, False]    # G7 lacks F

def test_tab_consistent():
    tab = [tn(0.0, 0.4, P(5, 0))]
    assert metrics.tab_consistent([tab[0].note], tab, STANDARD_EADGBE)
    assert not metrics.tab_consistent([note(0.0, 0.4, 65)], tab, STANDARD_EADGBE)
```

- [ ] **Step 2: Run** `pytest -q tests/unit/test_compose_metrics.py`. Expected: FAIL (the functions are not defined).
- [ ] **Step 3: Implement the five functions** with the signatures above.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Run the gates and commit**: `feat(compose): metrics for a one-guitar part (playability, onset on top, bar identity)`.

---

### Task 3: `generate_melody` can be limited to reachable pitches and end on a whole note

**Files:**
- Modify: `src/music_decoder/compose/melody.py` (`generate_melody`, `_phrase_bars`)
- Test: `tests/unit/test_compose_melody.py`

**Interfaces:**
- Produces two new keyword arguments on `generate_melody`, both appended after `chord_floors`:
  - `allowed_pitches: Sequence[frozenset[int]] | None = None`, one set per **bar**;
  - `final_whole_note: bool = False`.
- Behaviour with `allowed_pitches`:
  - `ValueError` if its length is not the number of bars, if any set is empty, or if `chord_floors` is also given.
  - Each bar's candidates are `sorted(p for p in allowed[bar] if p % 12 in chord_scale_pcs(scale, chord))`. If that is empty, use `sorted(allowed[bar])`.
  - `pitch_low`, `pitch_high` and the floors play no part, and `options` has the single candidate list.
  - `final_approach` becomes `sorted(p for p in allowed[-1] if p % 12 == final_pc)`.
  - The rest is unchanged: `admissible`, `ranked_scores`'s fallback ladder, `_plan_bar` and `final_target`.
- Behaviour with `final_whole_note=True`:
  - The last bar's rhythm is replaced by `(8,)` after `plan_bars` has run. It stays a cadence, final bar with `motif_from=None`, and its single slot is strong, with velocity `velocity_strong`.
  - `ValueError` if `notes_per_bar` is also given.
- With neither argument, the output is unchanged; the Task 1 digest test proves it.

- [ ] **Step 1: Write the failing tests**

```python
PROG = [ChordSymbol.parse(c) for c in ("C", "Am", "G", "C")]   # ends on C, so the cadence is C
ALLOWED = [frozenset(range(60, 70)), frozenset(range(57, 69)), frozenset(range(59, 70)), frozenset(range(55, 70))]

@pytest.mark.parametrize("seed", range(30))
def test_allowed_pitches_bound_every_note(seed):
    mel = generate_melody(scale=Scale("C", "major"), progression=PROG, tempo_bpm=120.0,
                          seed=seed, allowed_pitches=ALLOWED, final_whole_note=True, tessitura_center=64)
    for n in mel:
        assert n.pitch in ALLOWED[min(int(n.start_s // 2.0), 3)]
    assert mel[-1].start_s == pytest.approx(6.0) and mel[-1].pitch % 12 == 0   # one held C
    assert metrics.forbidden_interval_count(mel) == 0
    assert max(metrics.chord_change_moves(mel, PROG, 1, 120.0)) <= 7
    assert abs(mel[-1].pitch - mel[-2].pitch) <= 7

def test_allowed_pitches_argument_checks():
    with pytest.raises(ValueError):
        generate_melody(scale=Scale("C", "major"), progression=PROG, allowed_pitches=ALLOWED[:3])
    with pytest.raises(ValueError):
        generate_melody(scale=Scale("C", "major"), progression=PROG, allowed_pitches=ALLOWED,
                        chord_floors=[None] * 4)
    with pytest.raises(ValueError):
        generate_melody(scale=Scale("C", "major"), progression=PROG, notes_per_bar=4,
                        final_whole_note=True)
```

- [ ] **Step 2: Run** `pytest -q tests/unit/test_compose_melody.py`. Expected: the new tests FAIL with `TypeError` (unexpected keyword).
- [ ] **Step 3: Implement** both arguments as described. Add a `final_whole_note` parameter to `_phrase_bars`, or replace the last `_Bar` after it returns.
- [ ] **Step 4: Run** `pytest -q tests/unit/test_compose_melody.py tests/unit/test_compose_frozen.py`. Expected: PASS. The digest test is unchanged.
- [ ] **Step 5: Run the gates and commit**: `feat(compose): optional per-bar pitch limits and a whole-note final bar in the melody generator`.

---

### Task 4: Hand shapes, meaning what each chord shape lets the melody reach

**Files:**
- Create: `src/music_decoder/compose/fingerstyle.py`
- Test: `tests/unit/test_compose_fingerstyle.py`

**Interfaces:**
- Consumes: `moment_playable` (Task 2); `MAX_FRETTED_SPAN` from `voicings.py`.
- Produces:
  - `MELODY_FLOOR = 55` (G3).
  - `@dataclass(frozen=True) class HandShape` with these fields:
    - `voicing: VoicedChord`
    - `tuning: Tuning`
    - `root: TabPosition`: the lowest sounding string.
    - `alternate: TabPosition`: the next sounding string up. It is `root` when the shape sounds only one string.
    - `melody_strings: tuple[int, ...]`: every string index above `alternate.string`, sounded by the shape or not.
    - `fill_strings: tuple[int, ...]`: the shape's sounding strings above `alternate.string`.
    - `window: tuple[int, int]`: the lowest and highest fret a fretted melody note may use.
    - `barre_fret: int | None`
    - `allowed: frozenset[int]`
  - Method `positions(pitch: int) -> tuple[TabPosition, ...]`: the reachable positions of `pitch`, highest string first, or `()` when there are none.
  - `hand_shape(voicing: VoicedChord, tuning: Tuning) -> HandShape`. Window rules (spec §5.5):
    - Let `T` be the fretted (> 0) frets of root and alternate. If `T` is non-empty, `window = (max(1, max(T) - 3), min(T) + 3)`.
    - Otherwise, if the shape has fretted frets `S`, `window = (min(S), min(S) + 3)`. Otherwise `(1, 4)`.
    - `barre_fret` is the shape's lowest fretted fret when the shape sounds no open string and that fret occurs on two or more sounding strings; otherwise `None`. With a barre, `window = (max(window[0], barre_fret), window[1])` and open melody notes are excluded.
  - A position `(s, f)` is reachable when all of these hold:
    - `s in melody_strings`;
    - `f == 0` (if no barre), or `window[0] <= f <= window[1]`;
    - `pitch >= max(MELODY_FLOOR, max(root pitch, alternate pitch) + 3)`;
    - `moment_playable([root, alternate, P(s, f)])`.

    `allowed` is the set of reachable pitches.

- [ ] **Step 1: Write the failing tests.** Shapes are built explicitly with `_shape(label, frets)` → `VoicedChord`. `hand_shape` does not check that a shape is faithful.

```python
def test_open_c_reaches_g3_to_a4_on_the_top_three_strings():
    h = hand_shape(_shape("C", (-1, 3, 2, 0, 1, 0)), STANDARD_EADGBE)
    assert (h.root, h.alternate) == (P(1, 3), P(2, 2))
    assert h.melody_strings == (3, 4, 5) and h.window == (1, 5) and h.barre_fret is None
    assert h.allowed == frozenset(range(55, 70))
    assert h.positions(64) == (P(5, 0), P(4, 5))
    assert h.positions(60) == (P(4, 1), P(3, 5))
    assert h.positions(70) == ()

def test_open_d_keeps_the_melody_a_minor_third_over_the_thumb():
    h = hand_shape(_shape("D", (-1, -1, 0, 2, 3, 2)), STANDARD_EADGBE)
    assert (h.root, h.alternate) == (P(2, 0), P(3, 2)) and h.melody_strings == (4, 5)
    assert h.allowed == frozenset(range(60, 70))            # B3 (59) is under A3 + 3

def test_barre_shape_has_no_open_melody_notes():
    h = hand_shape(_shape("F", (1, 3, 3, 2, 1, 1)), STANDARD_EADGBE)
    assert h.barre_fret == 1 and h.window == (1, 4)
    assert h.allowed == frozenset({*range(56, 64), *range(65, 69)})   # no E4, no G3

def test_open_thumb_widens_the_shape_to_a_four_fret_window():
    h = hand_shape(_shape("A", (-1, 0, 0, 2, 1, 0)), STANDARD_EADGBE)
    assert h.window == (1, 4) and h.allowed == frozenset(range(55, 69))
    assert h.positions(59) == (P(4, 0), P(3, 4))
    h0 = hand_shape(_shape("Em", (0, 0, 0, 0, 0, 0)), STANDARD_EADGBE)
    assert h0.window == (1, 4) and h0.allowed == frozenset(range(55, 69))
```

- [ ] **Step 2: Run** `pytest -q tests/unit/test_compose_fingerstyle.py`. Expected: FAIL (module missing).
- [ ] **Step 3: Implement** `HandShape` and `hand_shape`. The module imports only from `compose.harmony`, `compose.metrics`, `compose.voicings`, `compose.arrangement` (constants and `_thin_below` only) and `types`. Nothing in `arrangement.py` imports it.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Run the gates and commit**: `feat(compose): reachable melody pitches for a held chord shape`.

---

### Task 5: One body bar: string assignment, Travis thumb, fills and chord identity

**Files:**
- Modify: `src/music_decoder/compose/fingerstyle.py`
- Test: `tests/unit/test_compose_fingerstyle.py`

**Interfaces:**
- Consumes: `HandShape` (Task 4); `moment_playable` (Task 2); `THIN_MARGIN_SEMITONES`.
- Produces:
  - `@dataclass(frozen=True) class BarPart` with fields `melody_tab: tuple[TabbedNote, ...]`, `accompaniment_tab: tuple[TabbedNote, ...]` (time-sorted) and `identity_met: bool`.
  - `body_bar(hand: HandShape, melody: Sequence[Note], *, bar_start_s: float, beat_s: float, essential: frozenset[int]) -> BarPart`. `melody` holds the notes that start in this bar. If one of their pitches is not in `hand.allowed`, it raises `ValueError(f"melody pitch {p} is not reachable over {chord}")`.
- Algorithm (spec §5.3–§5.5). Let `e = beat_s / 2`, and call eighth `k` the time `bar_start_s + k*e`.
  1. **Melody strings.** Each note takes `hand.positions(pitch)[0]`, the highest string. §5.5 rule 2 (least fret movement) only breaks ties between positions on the same string, which cannot happen, so it is a no-op. Say so in a comment.
  2. **Thumb.** At `k = 0, 2, 4, 6`, play `root` at `k = 0, 4` and `alternate` at `k = 2, 6`, each lasting `0.95 * beat_s`. The thumb is never thinned: the reachable window already keeps the melody at least 3 semitones above it.
  3. **Fill eligibility.** A fill plays the shape's own note on a `fill_strings` string `s` at eighth `k`, lasting `0.95 * e`. It is eligible when all of these hold:
     - no melody note is sounding on `s` during it;
     - no other fill is at `k`;
     - for every melody note `m` sounding during it, either `p <= m.pitch - THIN_MARGIN_SEMITONES`, or `p % 12` is an essential tone still missing from the bar's accompaniment and `p % 12 != m.pitch % 12`;
     - if a melody note starts at `k`, `p < m.pitch`;
     - `moment_playable` holds for the sounding positions at its onset, fill included.
  4. **Regular fills** at `k = 3` and then `k = 7`, only where no melody note starts.
     - Choose a missing essential tone first. "Missing" is measured against the pitch classes of all four thumb notes plus the fills chosen so far.
     - Otherwise choose the eligible pitch nearest below the melody note sounding then.
     - Otherwise there is no fill.
  5. **Identity, step 2.** For each essential tone still missing, try a fill of that tone at eighths in the order `3, 7, 1, 5, 0, 2, 4, 6`.
  6. **Identity, step 3.** If the tone is still missing, try the same eighths again. At each one, find the melody note sounding on the tone's string. If that note has another reachable position on a different string, and its whole duration stays playable there, move it and retry the fill at that eighth. Move at most one melody note per missing tone.
  7. `identity_met` is whether every tone in `essential` is now in the accompaniment's pitch classes.
- Velocities and note lengths beyond those pinned here are your call. v3 used 72 for the bass and 64 for the pinch.

- [ ] **Step 1: Write the failing tests.** Use 120 BPM, so `beat_s = 0.5`; the bar starts at 0. `C` and `D` are the Task 4 shapes.

```python
def _quarters(*pitches):
    return [note(k * 0.5, k * 0.5 + 0.475, p) for k, p in enumerate(pitches)]

def test_thumb_alternates_on_every_beat():
    bar = body_bar(C_HAND, [note(0.0, 1.9, 64)], bar_start_s=0.0, beat_s=0.5, essential=frozenset({4}))
    thumb = [(t.note.start_s, t.position) for t in bar.accompaniment_tab if t.position.string < 3]
    assert thumb == [(0.0, P(1, 3)), (0.5, P(2, 2)), (1.0, P(1, 3)), (1.5, P(2, 2))]

def test_fills_fall_on_the_and_of_two_and_four_under_the_melody():
    bar = body_bar(C_HAND, _quarters(64, 65, 67, 65), bar_start_s=0.0, beat_s=0.5, essential=frozenset({4}))
    fills = [(t.note.start_s, t.position) for t in bar.accompaniment_tab if t.position.string >= 3]
    assert fills == [(0.75, P(4, 1)), (1.75, P(4, 1))]            # C4, nearest under F4
    assert [t.position for t in bar.melody_tab] == [P(5, 0), P(5, 1), P(5, 3), P(5, 1)]
    assert bar.identity_met

def test_missing_third_is_filled_over_a_held_b_string_note():
    melody = [note(0.0, 0.95, 62), note(1.0, 1.95, 61)]       # D4 and C#4, both on the B string
    bar = body_bar(D_HAND, melody, bar_start_s=0.0, beat_s=0.5, essential=frozenset({6}))
    fills = [(t.note.start_s, t.position) for t in bar.accompaniment_tab if t.position.string >= 4]
    assert fills == [(0.75, P(5, 2))] and bar.identity_met

def test_identity_moves_one_melody_note_to_free_the_top_string():
    melody = [note(0.0, 0.95, 64), note(1.0, 1.95, 64)]       # E4 held on the e string all bar
    bar = body_bar(D_HAND, melody, bar_start_s=0.0, beat_s=0.5, essential=frozenset({6}))
    assert [t.position for t in bar.melody_tab] == [P(4, 5), P(5, 0)]
    assert (0.75, P(5, 2)) in [(t.note.start_s, t.position) for t in bar.accompaniment_tab]
    assert bar.identity_met

def test_every_moment_of_a_bar_is_playable():
    bar = body_bar(C_HAND, _quarters(64, 65, 67, 69), bar_start_s=0.0, beat_s=0.5, essential=frozenset({4}))
    assert metrics.playable_moment_ratio([*bar.melody_tab, *bar.accompaniment_tab]) == 1.0
    assert metrics.melody_onset_on_top_ratio([t.note for t in bar.melody_tab],
                                             [t.note for t in bar.accompaniment_tab]) == 1.0

def test_an_unreachable_melody_pitch_is_rejected():
    with pytest.raises(ValueError, match="not reachable"):
        body_bar(D_HAND, [note(0.0, 0.95, 59)], bar_start_s=0.0, beat_s=0.5, essential=frozenset({6}))
```

- [ ] **Step 2: Run** `pytest -q tests/unit/test_compose_fingerstyle.py`. Expected: the new tests FAIL.
- [ ] **Step 3: Implement** `BarPart` and `body_bar` following the algorithm above.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Run the gates and commit**: `feat(compose): Travis-picked fingerstyle bar with fills that keep the chord's identity`.

---

### Task 6: The final bar and the whole part

**Files:**
- Modify: `src/music_decoder/compose/voicings.py` (add `ranked_shapes`; `search_voicing` delegates to it)
- Modify: `src/music_decoder/compose/fingerstyle.py`
- Test: `tests/unit/test_compose_voicings.py`, `tests/unit/test_compose_fingerstyle.py`

**Interfaces:**
- Produces, in `voicings.py`:
  - `ranked_shapes(chord: ChordSymbol, tuning: Tuning) -> tuple[tuple[int, ...], ...]`. It returns every shape `search_voicing` considers faithful and playable, sorted by its existing key `(omitted tones, lowest fret, -strings, frets)`, with no duplicates. It is wrapped in `@functools.cache`.
  - `search_voicing` returns `ranked_shapes(...)[0]`, or `None` when there are none. Its results do not change.
- Produces, in `fingerstyle.py`:
  - `final_bar_pitches(hand: HandShape, cadence: int) -> frozenset[int]`.
    - When `hand.allowed` has a pitch of pitch class `cadence`, it returns `hand.allowed`.
    - Otherwise it returns `hand.allowed` plus every `cadence`-class pitch on a `melody_strings` string at frets 0–12 that is at least `max(MELODY_FLOOR, thumb high + 3)`.
  - `final_positions(hand: HandShape, pitch: int) -> tuple[TabPosition, ...]`. It returns `hand.positions(pitch)` when that is non-empty. Otherwise it returns every position of `pitch` on a melody string at frets 0–12, highest string first.
  - `final_bar(hand: HandShape, final: Note, *, bar_start_s: float, bar_s: float, essential: frozenset[int]) -> tuple[BarPart, bool]`. The bool is `True` when the fallback was used.
- Algorithm (§5.3, final bar). For each final position `fp` in `final_positions(hand, final.pitch)`, take each candidate in order: the bar's own frets first, then `ranked_shapes(chord, tuning)` with that one removed.
  - The candidate chord is its notes on strings below `fp.string`. Notes at or above the final note's string are left out; this is my call, and it matches the spec's fallback, which "plays the bar's shape without the final note's string".
  - Thin it as v3's `_thin_below` would against the held final note, keeping the lowest note.
  - Accept it when all of these hold:
    - its lowest pitch has the root's pitch class;
    - every essential tone other than `final.pitch % 12` sounds at a pitch below `final.pitch`;
    - `moment_playable(chord positions + [fp])`.
  - The first accepted candidate wins.
  - The fallback, when nothing is accepted, is the bar's own shape below the first `fp`, thinned, with the bool set to `True`.
  - The chord is rolled low to high from `bar_start_s` with `FINAL_STAGGER_S` per string. Every note ends at `bar_start_s + bar_s`, with velocity 80.
  - The melody tab holds `final` at `fp`.
- Produces `@dataclass(frozen=True) class FingerstylePart` with fields `melody_tab: tuple[TabbedNote, ...]`, `accompaniment_tab: tuple[TabbedNote, ...]` (time-sorted), `identity_misses: tuple[int, ...]` (bar indexes) and `final_fallback: bool`.
- Produces `arrange_fingerstyle(voicings: Sequence[VoicedChord], melody: Sequence[Note], hands: Sequence[HandShape], *, bars_per_chord: int, tempo_bpm: float) -> FingerstylePart`.
  - Bars `0 .. n-2` use `body_bar`. Bar `n-1` uses `final_bar` with the last melody note.
  - `ValueError` if the final bar holds anything other than exactly one melody note.

- [ ] **Step 1: Write the failing tests**

```python
# test_compose_voicings.py
@pytest.mark.parametrize("tuning", list(PRESETS.values()), ids=list(PRESETS))
def test_ranked_shapes_start_with_the_searched_voicing(tuning):
    for label in ("C", "D", "F#m", "A#7", "Bdim", "G#maj7", "Esus4", "Caug"):
        chord = ChordSymbol.parse(label)
        ranked = ranked_shapes(chord, tuning)
        assert (ranked[0] if ranked else None) == search_voicing(chord, tuning)
        assert all(_is_playable(s) and not shape_faults(s, tuning, chord) for s in ranked)

# test_compose_fingerstyle.py
def test_d_ending_is_revoiced_with_its_third_under_the_final_note():
    part, fallback = final_bar(D_HAND, note(6.0, 7.9, 62), bar_start_s=6.0, bar_s=2.0, essential=frozenset({6}))
    assert not fallback and part.melody_tab[0].position == P(4, 3)
    acc = part.accompaniment_tab
    assert [t.position for t in acc] == [P(1, 5), P(2, 4), P(3, 2)]      # x-5-4-2-(3)-x
    assert [t.note.start_s for t in acc] == pytest.approx([6.0, 6.01, 6.02])
    assert all(t.note.end_s == pytest.approx(8.0) for t in acc)

def test_final_bar_falls_back_and_says_so(monkeypatch):
    monkeypatch.setattr(fingerstyle, "ranked_shapes", lambda chord, tuning: ())
    part, fallback = final_bar(D_HAND, note(6.0, 7.9, 62), bar_start_s=6.0, bar_s=2.0, essential=frozenset({6}))
    assert fallback and [t.note.pitch for t in part.accompaniment_tab] == [50, 57]

def test_final_pitches_extend_to_an_unreachable_cadence():
    assert final_bar_pitches(D_HAND, 2) == D_HAND.allowed                # D is reachable
    assert final_bar_pitches(D_HAND, 11) == D_HAND.allowed | {71}        # B only at B4
    assert final_positions(D_HAND, 71) == (P(5, 7), P(4, 12))

def test_arrange_fingerstyle_joins_body_and_final_bars():
    voicings = [C_HAND.voicing, D_HAND.voicing]
    melody = [*_quarters(64, 65, 67, 65), note(2.0, 3.9, 62)]
    part = arrange_fingerstyle(voicings, melody, [C_HAND, D_HAND], bars_per_chord=1, tempo_bpm=120.0)
    assert [t.note for t in part.melody_tab] == melody
    assert part.identity_misses == () and not part.final_fallback
    assert metrics.playable_moment_ratio([*part.melody_tab, *part.accompaniment_tab]) == 1.0
```

- [ ] **Step 2: Run** `pytest -q tests/unit/test_compose_voicings.py tests/unit/test_compose_fingerstyle.py`. Expected: the new tests FAIL.
- [ ] **Step 3: Implement** `ranked_shapes` (refactor `search_voicing` onto it), then `final_bar_pitches`, `final_positions`, `final_bar`, `FingerstylePart` and `arrange_fingerstyle`.
- [ ] **Step 4: Run** the same command plus `tests/unit/test_compose_frozen.py`. Expected: PASS. Strum and arpeggio are unchanged.
- [ ] **Step 5: Run the gates and commit**: `feat(compose): fingerstyle final bar with a re-voiced rolled chord`.

---

### Task 7: Combined fingerstyle tab

**Files:**
- Modify: `src/music_decoder/compose/tabtext.py`
- Test: `tests/unit/test_compose_tabtext.py`

**Interfaces:**
- Produces `render_fingerstyle_tab(part: Sequence[TabbedNote], voicings: Sequence[VoicedChord], *, bars_per_chord: int, tempo_bpm: float, tuning: Tuning) -> str`. It returns `"Fingerstyle\n" + systems + "\n\nChords\n" + render_chord_shapes(voicings, tuning)`. `part` is the melody and accompaniment tab together.
- Layout (§5.6):
  - Each bar has 8 eighth columns. A note sits in bar `floor(start / bar_s + 1e-6)` and column `round((start - bar start) / eighth)`, so the final roll's staggered notes land in column 0.
  - A column's width is the widest fret text in it (minimum 1), and every string pads to it with `-`.
  - A bar is written as `"".join("-" + cell for cell in cells) + "-|"`. Each row starts with `label + "|"`, using `string_labels`, top string first.
  - Above each system is a chord-name row, with each bar's chord label starting at that bar's first cell column. Below it is a count row, `1 & 2 & 3 & 4 &`, with each symbol at its cell column. Strip trailing spaces from both rows.
  - Systems hold `BARS_PER_LINE` bars and are joined by a blank line.
  - Held notes are drawn only at their onset. `ValueError` if two notes share a string and a column.

- [ ] **Step 1: Write the failing tests**

```python
SKETCH = """\
   C
e|-0-------3---1---|
B|-------1---------|
G|---------------0-|
D|-----2-------2---|
A|-3-------3-------|
E|-----------------|
   1 & 2 & 3 & 4 &"""

def _sketch_bar():   # spec §5.6: open C at 120 BPM
    return [tabbed(t, s, f) for t, s, f in (
        (0.0, 1, 3), (0.5, 2, 2), (1.0, 1, 3), (1.5, 2, 2),      # thumb
        (0.75, 4, 1), (1.75, 3, 0),                               # fills
        (0.0, 5, 0), (1.0, 5, 3), (1.5, 5, 1))]                   # melody

def test_fingerstyle_tab_matches_the_spec_sketch():
    c = voicings_for(ChordSymbol.parse("C"), STANDARD_EADGBE)
    text = render_fingerstyle_tab(_sketch_bar(), c, bars_per_chord=1, tempo_bpm=120.0, tuning=STANDARD_EADGBE)
    assert text.startswith("Fingerstyle\n" + SKETCH + "\n\nChords\n")

def test_two_digit_frets_widen_their_column_on_every_string():
    text = render_fingerstyle_tab([tabbed(0.0, 5, 10)], voicings_for(ChordSymbol.parse("C"), STANDARD_EADGBE),
                                  bars_per_chord=1, tempo_bpm=120.0, tuning=STANDARD_EADGBE)
    rows = text.split("\n")[2:8]
    assert rows[0].startswith("e|-10-") and all(r.startswith(r[0] + "|----") for r in rows[1:])

def test_fingerstyle_tab_wraps_every_four_bars():
    notes = [tabbed(2.0 * b, 1, 3) for b in range(5)]
    voicings = voicings_for(ChordSymbol.parse("C"), STANDARD_EADGBE) * 5
    text = render_fingerstyle_tab(notes, voicings, bars_per_chord=1, tempo_bpm=120.0, tuning=STANDARD_EADGBE)
    systems = text.split("\n\nChords\n")[0].removeprefix("Fingerstyle\n").split("\n\n")
    assert len(systems) == 2 and systems[0].count("|") == 6 * 5
```

- [ ] **Step 2: Run** `pytest -q tests/unit/test_compose_tabtext.py`. Expected: the new tests FAIL.
- [ ] **Step 3: Implement** `render_fingerstyle_tab`. Leave `render_composition_tab` and `render_melody_tab` untouched.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Run the gates and commit**: `feat(compose): combined fingerstyle tab with rhythm, bar lines and counts`.

---

### Task 8: Wire fingerstyle into `arrange`, `compose` and the MIDI; bump to v4

**Files:**
- Modify: `src/music_decoder/compose/api.py`, `src/music_decoder/compose/arrangement.py`
- Modify tests (each change is listed with its reason below): `tests/unit/test_compose_arrangement.py`, `tests/unit/test_compose_api.py`, `tests/unit/test_compose_samples.py`, `tests/unit/test_productionize_song.py`, `tests/unit/test_productionize_api.py`, `tests/unit/test_productionize_verse.py`, `tests/integration/test_productionize_e2e.py`
- Test: `tests/unit/test_compose_fingerstyle.py`, `tests/unit/test_productionize_song.py`

**Interfaces:**
- Consumes: Tasks 3–7.
- Produces:
  - `Arrangement` gains `accompaniment_tab: tuple[TabbedNote, ...] = ()` and `final_fallback: bool = False`. Both are filled only for fingerstyle.
  - `COMPOSER_VERSION = 4`.
- `arrange(style="fingerstyle")` does the following:
  - Build `hands = [hand_shape(v, tuning) for v in voicings]`.
  - Each bar's allowed set is `hands[bar // bars_per_chord].allowed`. The last bar's is `final_bar_pitches(hands[-1], cadence_pc(scale, progression[-1]))`.
  - `tessitura_center = statistics.median_low(` the concatenation of every bar's `sorted(allowed)` `)`.
  - Call `generate_melody(..., allowed_pitches=..., tessitura_center=..., final_whole_note=True)`, with no `chord_floors`.
  - Then `part = arrange_fingerstyle(...)`. Set `melody_tab = part.melody_tab`, `accompaniment_tab = part.accompaniment_tab`, `accompaniment = tuple(t.note for t in part.accompaniment_tab)` and `final_fallback = part.final_fallback`.
  - Strum and arpeggio follow today's path unchanged.
- `arrangement.py`:
  - Delete `_fingerstyle_bar`.
  - `_accompaniment_notes(..., style="fingerstyle")` raises `ValueError("fingerstyle accompaniment is built with its melody by compose.fingerstyle; use arrange()")`.
  - `build_midi` gains the keyword `accompaniment: Sequence[Note] | None = None`. When it is given, those notes are written as the accompaniment track; otherwise `_accompaniment_notes` is used as today.
- `api.accompany(..., style="fingerstyle")` raises the same `ValueError`.
- `compose()` passes `accompaniment=arrangement.accompaniment` to `build_midi` only for fingerstyle. For fingerstyle, `ascii_tab = render_fingerstyle_tab([*melody_tab, *accompaniment_tab], voicings, ...)`. Strum and arpeggio still use `render_composition_tab`.

**Existing tests that change, and why** (spec §9). No strum or arpeggio case changes.

| Test | Change | Reason |
|---|---|---|
| `test_compose_arrangement.py::test_build_midi_styles_run` | Loop over strum and arpeggio only. Add the fingerstyle case built from `arrange(style="fingerstyle")` with `accompaniment=`. | Fingerstyle needs its arranged accompaniment. |
| `test_compose_arrangement.py`: the `[fingerstyle]` parameter of `test_stroke_heads_sit_on_the_sixteenth_grid`, `test_final_bar_is_one_ringing_chord`, `test_single_bar_accompaniment_is_the_final_chord`, `test_d_major_cadence_keeps_its_third` and `test_final_stroke_keeps_the_seventh_under_the_cadence` | Remove the parameter; re-create each check via `arrange()` in `test_compose_fingerstyle.py` | `_accompaniment_notes` no longer builds fingerstyle from an arbitrary melody (§5.5: the melody comes from reachable pitches). |
| `test_compose_arrangement.py::test_fingerstyle_alternates_the_bass`, `::test_fingerstyle_pinch_keeps_the_third_when_it_is_the_alternate_bass` | Delete them | Replaced by the Task 5 Travis thumb and identity tests (§5.3, §5.4). |
| `test_compose_api.py::test_compose_returns_the_arrangement_and_version` | `== 3` becomes `== 4` | §5.7 version bump. |
| `test_compose_api.py::test_rendered_tab_shows_every_melody_position` | Pass `style="strum"` to both `compose` and `arrange` | The `Melody` block now exists only for strum and arpeggio (§5.6). The fingerstyle tab is tested below. |
| `test_compose_samples.py::test_sample_metadata_reports_composer_version_3` | Rename it to `..._4` and check `== 4` | §5.7. |
| The `build_midi(style="fingerstyle")` fixtures in `test_productionize_song.py`, `test_productionize_api.py`, `test_productionize_verse.py` and `tests/integration/test_productionize_e2e.py` | Add `accompaniment=arr.accompaniment` | `build_midi` needs the fingerstyle accompaniment. Any productionize expectation that then fails falls under the hard-failure rule (stop and report). |

- [ ] **Step 1: Write the failing tests** in `test_compose_fingerstyle.py`. The checks run through the helper below; each test then calls it with its own inputs.

```python
def _assert_one_playable_part(arr, progression, bars=1, tempo=120.0, tuning=STANDARD_EADGBE):
    part = [*arr.melody_tab, *arr.accompaniment_tab]
    assert metrics.playable_moment_ratio(part) == 1.0
    assert metrics.melody_onset_on_top_ratio(arr.melody, arr.accompaniment) == 1.0
    assert all(metrics.bar_identity(arr.accompaniment, progression, bars, tempo))
    assert metrics.tab_consistent([*arr.melody, *arr.accompaniment], part, tuning)
    assert [t.note for t in arr.melody_tab] == list(arr.melody) and not arr.final_fallback
    assert metrics.final_chord_rings(arr.accompaniment, len(progression) * bars, tempo)
    assert metrics.cadence_identity(arr.melody, arr.accompaniment, arr.voicings, tuning)
    assert metrics.forbidden_interval_count(arr.melody) == 0
    assert metrics.unjustified_large_leap_count(arr.melody, progression, bars, tempo) == 0
    assert max(metrics.chord_change_moves(arr.melody, progression, bars, tempo), default=0) <= 7
```

Tests to write, each calling the helper:

| Test | Inputs |
|---|---|
| `test_fingerstyle_arrangement_is_one_playable_part` | `C Am F G`, `Am F C G`, `Cmaj7 Am7 Dm7 G7`, `G D Em C` in their keys; seeds 0–9 |
| `test_d_family_bars_keep_their_identity` | Review Focus 1: `D G A D`, `Dm A7 Dm`, `G D7 G`, `D Bm G A`; seeds 0–9 |
| `test_barre_progressions_are_playable` | Review Focus 2: `F Bb C F`, `Bm G D A`, `C#m A E B` (sharp spelling as the parser needs); seeds 0–4 |
| `test_every_root_and_quality_ends_playably` | Review Focus 3: for `(C major, C)` and `(A minor, Am)`, every root × the `_QUALITY_SUFFIXES` from `test_compose_api.py`, progression `(home, x)`, seed 7 |
| `test_fingerstyle_in_every_preset_tuning` | Review Focus 4: the three `TUNING_PROGRESSIONS` × every `PRESETS` tuning; seeds 1000–1002 |
| `test_single_chord_and_two_bars_per_chord` | Review Focus 5: `("D",)`; `C Am F G` with `bars_per_chord=2`; seeds 0–4 |
| `test_fingerstyle_ends_on_a_held_tonic_over_one_roll` | `D G A D`, seed 9: the last melody note starts at the last bar and is the cadence pitch class; the final accompaniment notes start within 50 ms |
| `test_compose_writes_the_fingerstyle_part_to_midi_and_tab` | `compose()` with `render_wav` mocked. The MIDI has tracks `melody` and `accompaniment`. Their notes equal `arr.melody` and `arr.accompaniment` (start within 1 ms, same pitch). Decoding the tab gives the part's `(bar, column, string, fret)` set exactly. |
| `test_accompany_and_build_midi_refuse_bare_fingerstyle` | Both raise `ValueError` |

Also re-create the five removed `[fingerstyle]` checks through `arrange(style="fingerstyle")` with the same progressions; use seed 9 where the original had none.

Add `test_fingerstyle_bars_keep_their_chord_tones_in_productionize` to `test_productionize_song.py`:

```python
@pytest.mark.parametrize(("key", "mode", "labels", "seed"), [
    ("A", "minor", ("Am", "F", "C", "G"), 7), ("D", "major", ("D", "G", "A7", "D"), 3),
    ("C", "major", ("Cmaj7", "Am7", "Dm7", "G7"), 11)])
def test_fingerstyle_bars_keep_their_chord_tones_in_productionize(tmp_path, key, mode, labels, seed):
    prog = [ChordSymbol.parse(c) for c in labels]
    arr = arrange(Scale(key, mode), prog, tempo_bpm=92.0, style="fingerstyle", seed=seed)
    midi = build_midi(melody=arr.melody, voicings=arr.voicings, bars_per_chord=1, tempo_bpm=92.0,
                      style="fingerstyle", accompaniment=arr.accompaniment, out_path=tmp_path / "c.mid")
    only = SectionSpec("Intro", "intro", drums="none", bass=False, piano_texture="accompaniment_only")
    song = song_from_midi(midi, progression=prog, sections=(only,))
    snap = song.grid.bar_s / 64
    for bar, chord in enumerate(prog):
        t0 = song.grid.time(0, bar)
        heard = {n.pitch % 12 for n in song.piano if t0 - snap <= n.start_s < t0 + song.grid.bar_s - snap}
        assert essential_pcs(chord) <= heard, (labels, bar)
```

- [ ] **Step 2: Run** `pytest -q tests/unit/test_compose_fingerstyle.py tests/unit/test_productionize_song.py`. Expected: the new tests FAIL.
- [ ] **Step 3: Implement** the `api.py` and `arrangement.py` changes above, then apply the table of test changes.
- [ ] **Step 4: Run** `pytest -q tests/unit tests/integration`. Expected: PASS. That includes `test_compose_frozen.py` (strum and arpeggio unchanged) and `test_compose_api.py::test_every_root_and_quality_composes_smoothly`, which now arranges fingerstyle by default. Any hard failure falls under the hard-failure rule.
- [ ] **Step 5: Run the gates and commit**: `feat(compose)!: fingerstyle is one playable guitar part (composer v4)`.

---

### Task 9: Evaluation harness and musicality report

**Files:**
- Modify: `src/music_decoder/evaluation/compose_musicality.py`
- Modify: `tests/unit/test_compose_evaluation.py`, `tests/unit/test_compose_musicality.py`

**Interfaces:**
- `Case` gains `fingerstyle: Arrangement`.
  - `build_cases` builds `arrangement` with `style="strum"`. That melody equals v3's shared melody, so every existing non-fingerstyle metric keeps its value.
  - It builds `fingerstyle` with `style="fingerstyle"`.
  - `accompaniment` is `{"strum": arrangement.accompaniment, "arpeggio": accompany(...), "fingerstyle": fingerstyle.accompaniment}`.
- `matrix_cases(name: str) -> list[Case]` (module level) holds today's `builders` dict. `evaluate` uses it.
- `case_metrics`:
  - Compute the per-style accompaniment metrics (A1–A4, H1, H2) against that style's own melody and tab: `case.fingerstyle.melody` for fingerstyle, `case.arrangement.melody` for the others.
  - Move the melody metrics into `_melody_metrics(melody, melody_tab, case) -> dict[str, Value]`. Emit them unprefixed for the shared melody and with prefix `"F."` for the fingerstyle melody (keys such as `F.M1`, `F.L1`, `F.T1`).
  - Add these keys:
    - `P1`: `playable_moment_ratio` of the fingerstyle melody and accompaniment tabs together;
    - `P2`: `melody_onset_on_top_ratio`;
    - `P3`: the share of `bar_identity` over the fingerstyle accompaniment;
    - `P3m`: the same over the accompaniment plus the melody;
    - `P4`: `tab_consistent`;
    - `P5`: `fingerstyle.final_fallback`.
- `_aggregate` pools `F.M5`, `F.M15`, `F.L2` and `F.L0` the way it pools their unprefixed versions. It skips their raw count keys.

- [ ] **Step 1: Write the failing tests**

```python
# test_compose_evaluation.py
def test_cases_carry_a_separate_fingerstyle_arrangement():
    (case,) = ev.build_cases(ev.edge_inputs()[:1])
    scale, prog, bars, seed = ev.edge_inputs()[0]
    kw = dict(bars_per_chord=bars, tempo_bpm=ev.TEMPO_BPM, seed=seed)
    assert case.arrangement == arrange(scale, prog, style="strum", **kw)
    assert case.fingerstyle == arrange(scale, prog, style="fingerstyle", **kw)
    row = ev.case_metrics(case)
    assert {"P1", "P2", "P3", "P3m", "P4", "P5", "F.M1", "F.L1", "F.M7"} <= set(row)

# test_compose_musicality.py
@pytest.fixture(scope="module")
def reserve_report():
    return evaluate(RESERVE_MATRICES)

@pytest.mark.parametrize("matrix", ALL_MATRICES)
def test_fingerstyle_part_on_every_matrix(all_reports, reserve_report, matrix):
    o = {**all_reports, **reserve_report}[matrix]["overall"]
    for key in ("P1", "P2", "P3", "P4"):
        assert o[key]["min"] == 1.0, key
    assert o["P5"]["max"] == 0.0                                     # no final-voicing fallback
    assert o["F.L1"]["max"] == 0 and o["F.L3"]["max"] == 0
    assert o["F.L2"]["n"] == 0 or o["F.L2"]["pooled"] == 1.0
    assert o["F.L4max"]["max"] <= 7 and o["F.L5"]["max"] <= 7 and o["F.M4max"]["max"] <= 12
    assert o["F.M7"]["min"] == 1.0 and o["F.M8"]["min"] >= 1.5
    assert o["F.M12"]["min"] == 1.0 and o["F.M14"]["min"] == 1.0

def test_fingerstyle_is_playable_in_every_tuning(report):
    for name in PRESETS:
        assert report["tunings"][f"tuning.{name}"]["P1"]["min"] == 1.0, name
```

- [ ] **Step 2: Run** `pytest -q tests/unit/test_compose_evaluation.py`. Expected: FAIL.
- [ ] **Step 3: Implement** the harness changes.
- [ ] **Step 4: Run** `pytest -q tests/unit/test_compose_evaluation.py tests/unit/test_compose_musicality.py`. Expected: PASS, except possibly the soft fingerstyle comparisons handled in Step 6. Hard fingerstyle failures fall under the hard-failure rule.
- [ ] **Step 5: Save the numbers.** Run `python scripts/compose_musicality_report.py` (all matrices; check its `--help` for how to name them) and save the output to `out/fingerstyle-report.txt` for Task 10.
- [ ] **Step 6: Soft fingerstyle metrics** (spec §6 item 7). These are the `*.fingerstyle` keys in `IMPROVE_UP`/`IMPROVE_DOWN`, the A2/A3 absolute targets and `test_relative_clash_reduction`.
  - If one fails, remove `fingerstyle` from that one soft assertion. Add an inline comment citing spec §6 item 7, with the v3 and v4 values.
  - Record it for Task 10's Results.
  - A2 can drop by design: an essential-tone fill may ring above a held melody note between onsets (§5.4). No other assertion may change.
- [ ] **Step 7: Run the gates and commit**: `feat(evaluation): arrange fingerstyle separately and report playability and identity`.

---

### Task 10: Docs, listening samples and the spec's Results

**Files:**
- Modify: `docs/usage.md` (around lines 251–283, 1185 and 1341), `docs/architecture-overview.md` (around line 238), `docs/architecture-technical.md` (around lines 665–680, 880–935 and 1105)
- Modify: `docs/superpowers/specs/2026-11-15-fingerstyle-solo-design.md` (append `## 10. Results`)
- Create (local only; `compose-samples/` is git-excluded): `compose-samples/2026-11-15-fingerstyle/`

- [ ] **Step 1: Update the three docs.**
  - Describe fingerstyle as one playable part: Travis thumb, fills, the melody inside the held shape, the held final note over a roll, and a range of about one octave.
  - Show the combined tab with the §5.6 sketch.
  - Describe the new `Arrangement` fields and `build_midi(accompaniment=)`.
  - Note that `composer_version` is 4.
- [ ] **Step 2: Render 10 samples** through the real CLI: `music-decoder compose ... --style fingerstyle --format json --out <dir>`.
  - Use 10 different keys, both modes, at least the EADGBE, Drop_D and DADGAD tunings, tempos 66–120 and seeds 501–510.
  - Lay out each folder like `compose-samples/2026-11-07-minor-v3/` (`composition.mid`, `composition.wav`, `compose.json`, `tab.txt`, `command.txt`), plus a `README.md` table: id, key, progression, tuning, tempo, seed, P1, identity, and a final-fallback column.
- [ ] **Step 3: Write `## 10. Results`** in the spec:
  - P1–P5 per matrix;
  - the `F.*` soft metrics next to the unprefixed v3 values of the same cases, from `out/fingerstyle-report.txt`;
  - every soft metric moved in Task 9, Step 6, with the reason;
  - any identity miss or final fallback (expected: none);
  - the sample folder path.
- [ ] **Step 4: Run all gates**: `pytest -q tests/unit tests/integration`, `ruff check src tests`, `ruff format --check src tests`, `mypy --strict src`. Expected: all green.
- [ ] **Step 5: Run the opt-in full-song productionize test on a fingerstyle piece** (spec §6 item 8; it downloads about 180 MB of samples and needs about 2.7 GB of RAM).

  Run `MUSIC_DECODER_PRODUCTION_E2E=1 pytest -q tests/integration/test_productionize_e2e.py`. Expected: PASS. Its fixture now builds the fingerstyle accompaniment (Task 8).
- [ ] **Step 6: Commit**: `docs(compose): fingerstyle as one playable part; spec results`.
