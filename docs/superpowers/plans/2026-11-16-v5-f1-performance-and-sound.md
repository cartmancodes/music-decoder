# v5 Feature 1: Expressive Performance and Plucked Sound — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make fingerstyle sound performed: expressive velocities and let-ring note lengths in the MIDI, small timing variation in the rendered audio only, and an opt-in plucked-string guitar renderer.

**Architecture:**
- A new pure module, `compose/performance.py`, post-processes the fingerstyle part after `arrange_fingerstyle`. It does three things: shapes velocities, extends accompaniment lengths under the playability and melody-on-top checks, and applies seeded micro-timing for audio.
- A new `synth/plucked.py` renders `TabbedNote`s with an extended Karplus–Strong voice per note.
- A new `compose.api.compose_with_options(..., sound=...)` takes a `sound` option, and `compose()` delegates to it.

**Tech Stack:** Python 3.11, numpy, scipy (`scipy.signal.lfilter`), pretty_midi, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-11-16-composer-v5-design.md` (§5.1, §5.6, §6 items 1–3 and 8).

## Global Constraints

- Activate the venv first: `source .venv/bin/activate`.
- Strum and arpeggio must not change. `tests/unit/test_compose_frozen.py` must pass untouched.
- `compose()`'s signature and the package exports stay frozen (`tests/unit/test_public_contract.py`). New CLI flags may be added.
- `COMPOSER_VERSION = 5`.
- On every matrix and every preset tuning, the fingerstyle v4 guarantees hold: P1, P2 and P4 are 1.0, P3t is 1.0, P5 is 0, A4 and H2 hold, and the v3 hard melody rules hold. The seeded random-progression property test passes.
- The MIDI file and the tab stay exactly on the eighth-note grid. The only exception is the final roll's `FINAL_STAGGER_S` per string. Micro-timing exists only in the rendered WAV.
- Velocities are clamped to 30–118.
- Gates before each commit: `pytest -q tests/unit`, `ruff check src tests`, `ruff format --check src tests`, `mypy --strict src`. Run `pytest -q tests/integration` at the last task.
- If a v4 guarantee fails, stop and report the case. Do not loosen the assertion.
- End every commit message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Branch: `feat/composer-v5`.

## Review Focus

1. **Very fast tempos** (200–240 BPM, where an eighth is 0.125–0.15 s): let-ring and micro-timing must not reorder onsets or break P1 and P2. Test: Task 1's `test_let_ring_keeps_every_guarantee_at_extreme_tempos`.
2. **Very slow tempos** (40 BPM, where a bar lasts 6 s): let-ring is capped at the bar end, and the plucked renderer must not run out of memory or produce a tail longer than 1.5 s. Tests: the same Task 1 test and Task 3's `test_plucked_render_length`.
3. **Two bars per chord**: ringing stops at each bar line, even when the chord does not change. Test: Task 1's `test_ringing_stops_at_the_bar_line`.
4. **Low tunings** (Drop C, D standard): plucked notes must sound in tune on the lowest strings. Test: Task 3's `test_plucked_pitch_is_in_tune` covers C2 and E4.
5. **`--sound plucked` with strum or arpeggio**: there is no accompaniment tab, so it must still render, taking each note's string as its lowest-fret position. Test: Task 3's `test_plucked_renders_strum_from_midi`.

---

### Task 1: Expressive velocities and let-ring in the fingerstyle part (v5)

**Files:**
- Create: `src/music_decoder/compose/performance.py`
- Modify: `src/music_decoder/compose/api.py` (`_arrange_fingerstyle`; `COMPOSER_VERSION = 5`)
- Modify: `src/music_decoder/compose/metrics.py` (two metrics)
- Modify: `src/music_decoder/evaluation/compose_musicality.py` (keys `F.vel_std`, `F.ring`)
- Test: `tests/unit/test_compose_performance.py` (new); update the version checks in `tests/unit/test_compose_api.py` and `tests/unit/test_compose_samples.py`

**Interfaces:**
- Consumes:
  - `FingerstylePart` (`melody_tab`, `accompaniment_tab`, `identity_misses`, `final_fallback`) from `compose/fingerstyle.py`;
  - `metrics.moment_playable(positions)`;
  - `rhythm.phrase_layout(n_bars).phrase_lengths`;
  - `harmony.essential_pcs`;
  - `THIN_MARGIN_SEMITONES` from `compose/arrangement.py`.
- Produces:
  - `shape_dynamics(part: FingerstylePart, *, n_bars: int, tempo_bpm: float) -> FingerstylePart`
  - `let_ring(part: FingerstylePart, progression: Sequence[ChordSymbol], *, bars_per_chord: int, tempo_bpm: float) -> FingerstylePart`
  - `metrics.velocity_spread(notes: Sequence[Note]) -> float | None`: population standard deviation of velocities; `None` when empty.
  - `metrics.mean_length_beats(notes: Sequence[Note], tempo_bpm: float) -> float | None`
  - `COMPOSER_VERSION = 5`

**Pinned values** (velocity is in MIDI units):

| Note | Base | Adjustments |
|---|---|---|
| Melody | 78 | +6 on eighths 0 and 4; phrase swell |
| Final melody note | 84 | none |
| Thumb, beats 1 and 3 | 66 | none |
| Thumb, beats 2 and 4 | 58 | none |
| Fill | 50 | none |
| Final roll | 70 | none |

- **Phrase swell:** add `round(10 * w)` to the base, where `w` rises linearly from 0 at the phrase's first note to 1 at its peak, then falls linearly to 0 at its last note. The peak is the phrase's highest note, the first one if tied. Phrases come from `phrase_layout(n_bars)`.
- **Margin rule:** after the above, any melody note less than 8 above the loudest accompaniment note sounding at its onset is raised to that velocity + 8.
- **Clamp:** every velocity to 30–118.
- **Role:** thumb, fill and roll are told apart by string and timing. Thumb notes are on the hand's root or alternate strings at eighths 0, 2, 4 and 6. The roll is the final bar's accompaniment. Everything else is a fill. If finer roles are needed, `arrange_fingerstyle` may tag notes; that is your call.

**Let-ring algorithm:**
- For each accompaniment note outside the final bar, in time order:
  - the candidate end is `min(next onset on the same string (any note of the part), the bar's end)`;
  - walk the part's onsets `t` with `start < t < candidate end`;
  - cut the end to `t` at the first `t` where any of these holds:
    - (a) `moment_playable` fails for the notes sounding at `t` with this note extended;
    - (b) a melody note starts at `t` with pitch `m`, and `p >= m`;
    - (c) a melody note starts at `t`, `p > m - THIN_MARGIN_SEMITONES`, and `p % 12` is not an essential tone of the bar's chord.
- A note is never made shorter than it is in v4.
- Final-bar notes are unchanged.
- `melody_tab` is unchanged, apart from its velocities.

- [ ] **Step 1: Write the failing tests.** Build parts with `arrange(..., style="fingerstyle")`. After this task, `arrange` applies `shape_dynamics` and then `let_ring`.

```python
def _fs(labels, key="C", mode="major", seed=3, bars=1, tempo=120.0, tuning=STANDARD_EADGBE):
    prog = [ChordSymbol.parse(c) for c in labels]
    return prog, arrange(Scale(key, mode), prog, bars_per_chord=bars, tempo_bpm=tempo,
                         style="fingerstyle", tuning=tuning, seed=seed)

def test_melody_is_at_least_8_louder_than_the_accompaniment_under_it():
    _, a = _fs(("C", "Am", "F", "G"))
    for m in a.melody:
        under = [n.velocity for n in a.accompaniment if n.start_s <= m.start_s + 1e-4 < n.end_s]
        assert not under or m.velocity >= max(under) + 8

def test_velocities_spread_and_stay_in_range():
    _, a = _fs(("C", "Am", "F", "G", "C", "F", "G", "C"))
    notes = [*a.melody, *a.accompaniment]
    assert metrics.velocity_spread(notes) >= 8
    assert all(30 <= n.velocity <= 118 for n in notes)

def test_accompaniment_rings_longer_than_v4():
    _, a = _fs(("C", "Am", "F", "G"))
    assert metrics.mean_length_beats(a.accompaniment, 120.0) >= 1.2   # v4 ≈ 0.8

def test_ringing_stops_at_the_bar_line():
    _, a = _fs(("C", "Am"), bars=2)
    bar = 2.0
    for n in a.accompaniment:
        assert int((n.start_s + 1e-6) // bar) == int((n.end_s - 1e-6) // bar) or n.start_s >= 3 * bar - 1e-6

@pytest.mark.parametrize("tempo", [40.0, 200.0, 240.0])
def test_let_ring_keeps_every_guarantee_at_extreme_tempos(tempo):
    for labels in (("C", "Am", "F", "G"), ("D", "G", "A7", "D"), ("F", "A#", "C", "F")):
        prog, a = _fs(labels, key=labels[0], tempo=tempo)
        part = [*a.melody_tab, *a.accompaniment_tab]
        assert metrics.playable_moment_ratio(part) == 1.0
        assert metrics.melody_onset_on_top_ratio(a.melody, a.accompaniment) == 1.0
        assert metrics.tab_consistent([*a.melody, *a.accompaniment], part, STANDARD_EADGBE)
        assert metrics.final_chord_rings(a.accompaniment, len(prog), tempo)

def test_onsets_stay_on_the_grid():
    _, a = _fs(("C", "Am", "F", "G"))
    heads = metrics.stroke_heads([n.start_s for n in a.accompaniment])
    assert metrics.onset_grid_ratio(heads, 120.0, 2) == 1.0
    assert metrics.onset_grid_ratio([n.start_s for n in a.melody], 120.0, 2) == 1.0
```

- [ ] **Step 2: Run** `pytest -q tests/unit/test_compose_performance.py`. Expected: FAIL (no module or attribute).
- [ ] **Step 3: Implement.**
  - `velocity_spread` and `mean_length_beats` in `metrics.py`.
  - `shape_dynamics` and `let_ring` in `performance.py`, with the pinned values and algorithm above.
  - In `_arrange_fingerstyle`, apply them in that order after `arrange_fingerstyle`. Fill `melody`, `melody_tab`, `accompaniment` and `accompaniment_tab` from the result. `melody` must equal `[t.note for t in melody_tab]`.
  - Set `COMPOSER_VERSION = 5`.
  - In `_fingerstyle_part_metrics`, add `F.vel_std` (`velocity_spread` over melody and accompaniment) and `F.ring` (`mean_length_beats` over the accompaniment).
  - Update the version checks: `test_compose_api.py`'s `== 4` becomes `== 5`, and `test_compose_samples.py`'s `..._4` test becomes `..._5` with `== 5`. List both in the report as version-bump changes.
- [ ] **Step 4: Run** `pytest -q tests/unit`. Expected: PASS, including `test_compose_frozen.py`, `test_compose_fingerstyle.py` (the property test now exercises let-ring) and `test_compose_musicality.py`.
  - If a soft `*.fingerstyle` musicality assertion moves (A2 or A3), follow the v4 rule: drop only that key from that one assertion, with a comment citing v5 spec §5.1 and both values.
  - A hard failure means stop and report.
- [ ] **Step 5: Run the gates and commit**: `feat(compose): expressive velocities and let-ring for fingerstyle (composer v5)`.

---

### Task 2: Micro-timing in the audio, `compose_with_options` and `--sound`

**Files:**
- Modify: `src/music_decoder/compose/performance.py` (add `humanize`)
- Modify: `src/music_decoder/compose/api.py` (`compose_with_options`; `compose` delegates)
- Modify: `src/music_decoder/cli/main.py` (`--sound`)
- Test: `tests/unit/test_compose_performance.py`, `tests/unit/test_compose_api.py`, `tests/unit/test_cli_main.py`

**Interfaces:**
- Consumes: Task 1; `build_midi(..., accompaniment=)`; `render_wav(midi_path, out_path)`.
- Produces:
  - `humanize(notes: Sequence[Note], *, seed: int, max_offset_s: float = 0.006) -> tuple[Note, ...]`.
    - Onsets within 60 ms of a stroke's first onset (`metrics.stroke_heads` grouping) form one stroke, and all its notes shift by the same offset, so the roll's stagger is kept.
    - Offsets are uniform in `[-max_offset_s, +max_offset_s]`, from `numpy.random.default_rng(seed)`, one per stroke in time order.
    - Lengths are kept. No onset moves below 0.
  - `compose_with_options(scale, progression, *, bars_per_chord=1, tempo_bpm=100.0, style="fingerstyle", tuning=STANDARD_EADGBE, seed=None, out_dir, sound: Literal["soundfont", "plucked"] = "soundfont") -> Composition`.
    - `compose()` keeps its exact signature and returns `compose_with_options(...)` with `sound="soundfont"`.
    - For fingerstyle, the WAV is rendered from a temporary performance MIDI: `build_midi` over the humanized melody and accompaniment, with `seed if seed is not None else 0`, written next to `composition.mid` and deleted after rendering.
    - `composition.mid` itself is the on-grid part.
    - Strum and arpeggio render exactly as today.
    - `metadata["sound"] = sound`.
  - CLI `compose --sound [soundfont|plucked]`, default `soundfont`, passed through to `compose_with_options`.

- [ ] **Step 1: Write the failing tests**

```python
def test_humanize_keeps_strokes_together_and_bounded():
    _, a = _fs(("C", "G"))
    notes = (*a.melody, *a.accompaniment)
    h = humanize(notes, seed=7)
    assert humanize(notes, seed=7) == h and humanize(notes, seed=8) != h
    for orig, new in zip(notes, h, strict=True):
        assert abs(new.start_s - orig.start_s) <= 0.006 + 1e-9
        assert new.end_s - new.start_s == pytest.approx(orig.end_s - orig.start_s)
    roll = sorted((n for n in h if n.start_s >= 2.0 - 0.01), key=lambda n: n.start_s)
    gaps = [b.start_s - a_.start_s for a_, b in zip(roll, roll[1:])]
    assert all(g == pytest.approx(0.010, abs=1e-6) or g == pytest.approx(0, abs=1e-6) for g in gaps)

def test_compose_writes_on_grid_midi_and_records_the_sound(tmp_path):
    with mock.patch("music_decoder.compose.api.render_wav") as render:
        render.side_effect = lambda midi_path, out_path: out_path.write_bytes(b"RIFF")
        result = compose_with_options(Scale("C", "major"), _PROG, seed=1, out_dir=tmp_path)
    pm = pretty_midi.PrettyMIDI(str(result.midi_path))
    starts = [n.start for i in pm.instruments for n in i.notes]
    assert metrics.onset_grid_ratio(metrics.stroke_heads(starts), 100.0, 2) == 1.0
    assert result.metadata["sound"] == "soundfont" and result.metadata["composer_version"] == 5
    assert not list(result.midi_path.parent.glob("*performance*"))   # temp MIDI removed

def test_compose_delegates_to_compose_with_options(tmp_path):
    with mock.patch("music_decoder.compose.api.render_wav") as render:
        render.side_effect = lambda midi_path, out_path: out_path.write_bytes(b"RIFF")
        a = compose(Scale("C", "major"), _PROG, seed=1, out_dir=tmp_path / "a")
        b = compose_with_options(Scale("C", "major"), _PROG, seed=1, out_dir=tmp_path / "b")
    assert a.melody_notes == b.melody_notes and a.ascii_tab == b.ascii_tab

def test_cli_compose_accepts_sound(tmp_path):   # in test_cli_main.py; mock render_wav the same way
    r = CliRunner().invoke(main, ["compose", "--scale", "C:major", "--progression", "C G",
                                  "--sound", "soundfont", "--seed", "1", "--out", str(tmp_path),
                                  "--format", "json"])
    assert r.exit_code == 0 and '"sound": "soundfont"' in r.output
```

- [ ] **Step 2: Run** the new tests. Expected: FAIL.
- [ ] **Step 3: Implement** `humanize`, `compose_with_options`, the `compose()` delegation and the CLI flag. `sound="plucked"` raises `NotImplementedError` until Task 3.
- [ ] **Step 4: Run** `pytest -q tests/unit`. Expected: PASS, including `test_public_contract.py`.
- [ ] **Step 5: Run the gates and commit**: `feat(compose): micro-timed audio and a --sound option`.

---

### Task 3: Plucked-string renderer

**Files:**
- Create: `src/music_decoder/synth/plucked.py`
- Modify: `src/music_decoder/compose/api.py` (`sound="plucked"` path)
- Test: `tests/unit/test_synth_plucked.py` (new); `tests/unit/test_compose_api.py`

**Interfaces:**
- Consumes: `TabbedNote`, `Tuning`, Task 2's `humanize` and `compose_with_options`.
- Produces:
  - `render_plucked(notes: Sequence[TabbedNote], tuning: Tuning, out_path: Path, *, seed: int = 0, sr: int = 44100) -> Path`. It writes a 16-bit mono WAV.
  - `tab_for_midi_notes(notes: Sequence[Note], tuning: Tuning) -> tuple[TabbedNote, ...]`: each note on its lowest-fret string, used when no tab exists.
  - In `compose_with_options(sound="plucked")`:
    - fingerstyle renders `humanize`d `[*melody_tab, *accompaniment_tab]`;
    - strum and arpeggio render `tab_for_midi_notes` of the MIDI file's notes;
    - the `composition.mid` file is unchanged.

**Voice:**
- **Pitch:** `f0 = 440 · 2^((pitch − 69)/12)`, using `tuning.open_pitches[string] + fret`.
- **Excitation:** noise from `default_rng(seed + note index)`. It is low-passed more at low velocity. A pluck-position comb subtracts a copy delayed by `round(0.18 · N)`.
- **String loop:** use `scipy.signal.lfilter` with a denominator that has `−g/2` at lags `N` and `N+1`, where `N = round(sr/f0 − 0.5)`. Choose `g` so that the T60 decay is about 4.5 s for open low strings and about 2 s for fretted notes above fret 7 on the top strings. You may add an all-pass for tuning or stiffness.
- **End of note:** at `end_s`, fade out over 40 ms, as a finger mute.
- **Body:** mix the summed voices with the output of three resonant biquads (about 100, 200 and 400 Hz), at a mix you choose.
- **Output:**
  - peak-normalise to 0.9 full scale, as in `synthesize_midi_to_wav`;
  - the length is the last note's end plus a release tail of at most 1.5 s;
  - the same seed gives byte-identical output.

- [ ] **Step 1: Write the failing tests**

```python
def _one(pitch_string, fret, dur=1.5, vel=80, tuning=STANDARD_EADGBE):
    pitch = tuning.open_pitches[pitch_string] + fret
    return TabbedNote(note=Note(start_s=0.0, end_s=dur, pitch=pitch, velocity=vel, confidence=1.0),
                      position=TabPosition(string=pitch_string, fret=fret))

@pytest.mark.parametrize(("tuning", "string", "fret"), [(DROP_C, 0, 0), (STANDARD_EADGBE, 5, 0)])
def test_plucked_pitch_is_in_tune(tmp_path, tuning, string, fret):   # C2 and E4
    out = render_plucked([_one(string, fret, tuning=tuning)], tuning, tmp_path / "n.wav")
    y, sr = soundfile.read(out)
    f0 = float(np.median(librosa.yin(y[: sr], fmin=50, fmax=1000, sr=sr)))
    target = 440 * 2 ** ((tuning.open_pitches[string] + fret - 69) / 12)
    assert abs(1200 * np.log2(f0 / target)) <= 10

def test_plucked_is_deterministic_and_does_not_clip(tmp_path):
    notes = [_one(1, 3), _one(5, 0)]
    a = render_plucked(notes, STANDARD_EADGBE, tmp_path / "a.wav", seed=3).read_bytes()
    b = render_plucked(notes, STANDARD_EADGBE, tmp_path / "b.wav", seed=3).read_bytes()
    assert a == b
    y, _ = soundfile.read(tmp_path / "a.wav", dtype="int16")
    assert np.abs(y).max() <= int(0.9 * 32767) + 1

def test_plucked_render_length(tmp_path):
    out = render_plucked([_one(0, 0, dur=6.0)], STANDARD_EADGBE, tmp_path / "s.wav")
    y, sr = soundfile.read(out)
    assert 6.0 <= len(y) / sr <= 7.5

@pytest.mark.parametrize("tuning_name", list(PRESETS))
def test_compose_with_plucked_sound_in_every_tuning(tmp_path, tuning_name):
    r = compose_with_options(Scale("D", "major"), [ChordSymbol.parse(c) for c in ("D", "G", "A", "D")],
                             tuning=PRESETS[tuning_name], seed=2, out_dir=tmp_path, sound="plucked")
    y, _ = soundfile.read(r.wav_path)
    assert len(y) > 0 and r.metadata["sound"] == "plucked"

def test_plucked_renders_strum_from_midi(tmp_path):
    r = compose_with_options(Scale("C", "major"), _PROG, style="strum", seed=1, out_dir=tmp_path,
                             sound="plucked")
    assert soundfile.info(r.wav_path).duration > 0
```

- [ ] **Step 2: Run** `pytest -q tests/unit/test_synth_plucked.py`. Expected: FAIL.
- [ ] **Step 3: Implement** `render_plucked`, `tab_for_midi_notes` and the `sound="plucked"` path.
- [ ] **Step 4: Run** `pytest -q tests/unit tests/integration`. Expected: PASS. The plucked tests must take under about 20 s in total; shorten the test notes if they take longer.
- [ ] **Step 5: Run the gates and commit**: `feat(synth): string-aware plucked guitar renderer (--sound plucked)`.
