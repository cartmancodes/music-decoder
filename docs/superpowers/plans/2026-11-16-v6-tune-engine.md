# Composer v6 Tune Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the fingerstyle melody generator with an engine whose rhythm, moves and note choices are learned from real melodies, gated against real-melody statistics, with the v5 generator kept selectable.

**Architecture:** A script turns the cleaned POP909 dataset into one small statistics file shipped in the package. A new module, `compose/tune.py`, plans a rhythm from real bar patterns and then picks pitches from learned move, chord-tone and degree tables under the existing guitar constraints. One feature function measures both the corpus and generated melodies, so the gates and the best-of-12 score compare like with like.

**Tech Stack:** Python 3.11, numpy 1.26.4, stdlib only otherwise; pytest, ruff, mypy --strict.

**Spec:** `docs/superpowers/specs/2026-11-16-composer-v6-melody-design.md`

## Global Constraints

- No new runtime dependency. `numpy` stays at 1.26.4.
- All randomness is `numpy.random.default_rng(seed)`; the same seed gives the same music.
- `compose()` keeps its signature. `melody_engine="rules"` must give the v5 output for a seed, note for note.
- Every melody note is on the eighth grid, inside `allowed_pitches[bar]`, and ends before its bar line.
- Hard interval rules: no move over 7 semitones, none of 6, 10 or 11 semitones; the move into a new chord and into the final note is at most 7.
- Only statistics leave a dataset. The dataset itself is never copied into the repo.
- `ruff check`, `ruff format --check` (line length 100) and `mypy --strict src` stay clean.
- Never stage `spikes/`, `out/` or `compose-samples/`.
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Work on branch `feat/composer-v6`, created from `master`.

## Review Focus

1. Chords outside the key (E7 in A minor, a flat-seven major chord): the bar's scale ladder changes, and a move counted in scale steps must still be defined. Test in Task 5.
2. Odd lengths (1, 2, 3, 5, 9 bars): every bar gets a role, a one-bar piece is one whole note, the final-note rule holds. Tests in Tasks 4 and 5.
3. Small allowed sets (5 pitches or fewer in a bar, as barre shapes and alternate tunings give): no crash, hard rules hold, replay gives up cleanly. Test in Task 5; Drop D, DADGAD and D standard in Task 6.
4. A chord change every bar against two or four bars per chord: the chord-change limit applies at each change and nowhere else. Test in Task 5.
5. Tempos 40 and 200, `seed=None`, and seventh, dim, aug and sus4 chords: the density clamp holds and a melody comes out. Test in Task 6.

## File Structure

| File | Responsibility |
|---|---|
| `src/music_decoder/compose/tune_metrics.py` (new) | Eighth-grid melody features and the distance from the real-melody bands. |
| `src/music_decoder/compose/tune_corpus.py` (new) | Reads the cleaned POP909 folder; builds the statistics. |
| `scripts/build_tune_stats.py` (new) | Thin command line around `tune_corpus`. |
| `src/music_decoder/compose/data/tune_stats.json` (new) | The statistics, committed. |
| `src/music_decoder/compose/tune_stats.py` (new) | Typed loader for the statistics file. |
| `src/music_decoder/compose/tune.py` (new) | The engine: rhythm plan, pitch plan, `generate_tune`. |
| `src/music_decoder/compose/api.py`, `scoring.py`, `cli/main.py` | `melody_engine` wiring, score, flag, version 6. |
| `src/music_decoder/evaluation/compose_tune.py` (new) | The tune matrix and its report. |
| `spikes/preference-pilot/extract_vocal_lines.py` (new, untracked) | Vocal lines of the pilot songs. Throwaway. |

## Controller steps (not implementer tasks)

- **Before Task 2:** `git clone --depth 1 https://github.com/Dsqvival/hierarchical-structure-analysis ~/.music-decoder/datasets/pop909-hsa` (26 MB, MIT). Pass `~/.music-decoder/datasets/pop909-hsa/POP909` to the implementer as the dataset path.
- **After Task 3's review:** start `extract_vocal_lines.py` on all songs in the background (three workers, about four hours). Do not wait for it.
- **After Task 6's review (listening checkpoint, spec 5.7):** generate the ten pieces of `compose-samples/2026-11-16-v5-master/01`..`10` (commands in each `command.txt`) with `--melody-engine tune` and again with `--melody-engine rules`, convert to MP3, send them to the user, and carry on with Task 7.
- **Before Task 7:** the gates are tuned against one final statistics file. If the vocal-line run has finished, rebuild the statistics with `--pilot` and commit the file (`feat(compose): tune statistics with the pilot pitch data`). If it has not, wait for it up to six hours from its start; after that, record in the ledger that the file is POP909-only and the merge is a follow-up. Tell the Task 7 implementer which it is.

---

### Task 1: Melody features on the eighth grid

**Files:**
- Create: `src/music_decoder/compose/tune_metrics.py`
- Test: `tests/unit/test_compose_tune_metrics.py`

**Interfaces:**
- Produces:

```python
Mode = Literal["major", "minor"]
FEATURES: tuple[str, ...] = (
    "repeat", "step", "third", "leap", "notes_per_second", "rhythm_variety", "rhythm_lag2",
    "rhythm_lag4", "figure_recurrence", "offbeat", "late_entry", "chord_tone_strong",
    "pentatonic", "range",
)

@dataclass(frozen=True)
class GridNote:
    slot: int   # eighths from the start of the excerpt
    pitch: int  # MIDI

@dataclass(frozen=True)
class Band:
    low: float
    mid: float
    high: float

def grid_features(notes: Sequence[GridNote], *, n_bars: int, tonic_pc: int, mode: Mode,
                  chord_tones: Sequence[frozenset[int]], seconds_per_bar: float) -> dict[str, float]: ...
def tune_features(melody: Sequence[Note], scale: Scale, progression: Sequence[ChordSymbol],
                  bars_per_chord: int, tempo_bpm: float) -> dict[str, float]: ...
def band_distance(features: Mapping[str, float], bands: Mapping[str, Band]) -> float: ...
```

`chord_tones` has one set per eighth slot (`n_bars * 8` entries). Definitions, all over 8-bar windows (bars 0–7, 8–15, …; a trailing window shorter than 8 bars is ignored unless the piece itself is shorter than 8 bars, when the whole piece is the one window) and then averaged over windows:

- `repeat`, `step`, `third`, `leap`: share of moves between consecutive notes of 0, 1–2, 3–4, and 5 or more semitones.
- `notes_per_second`: notes ÷ (bars in the window × `seconds_per_bar`).
- A bar's pattern is the sorted tuple of `slot % 8` of its onsets. `rhythm_variety`: distinct non-empty patterns ÷ non-empty bars. `rhythm_lag2`, `rhythm_lag4`: share of bar pairs (b, b−2) or (b, b−4), both non-empty, with equal patterns; 0.0 with no pair.
- `figure_recurrence`: over the signed semitone moves, the share of 3-move figures that occur more than once in the window; 0.0 with fewer than 4 notes.
- `offbeat`: share of onsets on odd slots. `late_entry`: share of the window's bars with no onset on slot 0.
- `chord_tone_strong`: share of the notes on slots 0 and 4 whose pitch class is in that slot's `chord_tones`; the key is left out when there is no such note.
- `pentatonic`: share of notes with `(pitch - tonic_pc) % 12` in `{0, 2, 4, 7, 9}` (major) or `{0, 3, 5, 7, 10}` (minor).
- `range`: highest minus lowest pitch.

`band_distance` sums, over the features present in both mappings, `max(0, low - v, v - high) / max(high - low, floor)`, with `floor` 1.0 for `range`, 0.2 for `notes_per_second` and 0.05 for every other feature.

- [ ] **Step 1: Write the failing tests**

```python
def _notes(pairs):
    return [GridNote(slot=s, pitch=p) for s, p in pairs]

C_MAJOR = [frozenset({0, 4, 7})] * 64  # eight bars of C

def test_interval_shares():
    f = grid_features(_notes([(0, 60), (2, 60), (4, 62), (6, 65), (8, 72)]), n_bars=8, tonic_pc=0,
                      mode="major", chord_tones=C_MAJOR, seconds_per_bar=2.0)
    assert (f["repeat"], f["step"], f["third"], f["leap"]) == (0.25, 0.25, 0.25, 0.25)
    assert f["notes_per_second"] == 5 / 16.0 and f["range"] == 12.0

def test_rhythm_features():
    bars = {0: (0, 2, 4), 1: (0, 3), 2: (0, 2, 4), 3: (1, 5), 4: (0, 2, 4)}
    notes = _notes([(b * 8 + s, 60) for b, slots in bars.items() for s in slots])
    f = grid_features(notes, n_bars=8, tonic_pc=0, mode="major", chord_tones=C_MAJOR, seconds_per_bar=2.0)
    assert f["rhythm_variety"] == 3 / 5
    assert f["rhythm_lag2"] == 2 / 3 and f["rhythm_lag4"] == 1.0
    assert f["late_entry"] == 4 / 8  # bar 3 and the three empty bars
    assert f["offbeat"] == 3 / 13

def test_figure_recurrence_counts_repeated_three_move_figures():
    up = [60, 62, 64, 65]
    notes = _notes(list(enumerate(up + up + [60, 67, 59])))
    f = grid_features(notes, n_bars=8, tonic_pc=0, mode="major", chord_tones=C_MAJOR, seconds_per_bar=2.0)
    assert f["figure_recurrence"] == 4 / 8  # (2, 2, 1) and (2, 1, -5) each occur twice among eight figures

def test_chord_tone_strong_and_pentatonic():
    f = grid_features(_notes([(0, 60), (4, 62), (8, 65), (9, 71)]), n_bars=8, tonic_pc=0, mode="major",
                      chord_tones=C_MAJOR, seconds_per_bar=2.0)
    assert f["chord_tone_strong"] == 1 / 3 and f["pentatonic"] == 0.5

def test_chord_tone_strong_is_omitted_without_strong_notes():
    f = grid_features(_notes([(1, 60), (3, 62)]), n_bars=8, tonic_pc=0, mode="major",
                      chord_tones=C_MAJOR, seconds_per_bar=2.0)
    assert "chord_tone_strong" not in f

def test_sixteen_bars_average_two_windows():
    notes = _notes([(0, 60), (2, 60)] + [(64, 60), (66, 62)])
    f = grid_features(notes, n_bars=16, tonic_pc=0, mode="major", chord_tones=C_MAJOR * 2, seconds_per_bar=2.0)
    assert f["repeat"] == 0.5  # 1.0 in the first window, 0.0 in the second

def test_tune_features_matches_grid_features():
    scale, prog = Scale.parse("C:major"), [ChordSymbol.parse("C"), ChordSymbol.parse("G")]
    e = 60.0 / 120 / 2
    melody = [Note(start_s=k * e, end_s=k * e + 0.2, pitch=p, velocity=80, confidence=1.0)
              for k, p in [(0, 60), (3, 62), (32, 67)]]
    f = tune_features(melody, scale, prog, bars_per_chord=4, tempo_bpm=120.0)
    assert f["notes_per_second"] == 3 / 16.0 and f["chord_tone_strong"] == 1.0  # C over C, G over G

def test_band_distance():
    bands = {"repeat": Band(0.1, 0.2, 0.3), "range": Band(12, 14, 16), "offbeat": Band(0.4, 0.4, 0.4)}
    assert band_distance({"repeat": 0.2, "range": 14.0}, bands) == 0.0
    assert band_distance({"repeat": 0.5, "range": 10.0, "offbeat": 0.5, "leap": 0.9}, bands) == pytest.approx(1.0 + 0.5 + 2.0)
```

- [ ] **Step 2: Run them and see them fail**

Run: `pytest -q tests/unit/test_compose_tune_metrics.py` — Expected: import error.

- [ ] **Step 3: Implement `tune_metrics.py`** to the interfaces and definitions above. `tune_features` converts each note's `start_s` to a slot with `round(start_s / eighth_s)` and builds `chord_tones` from `harmony.chord_pcs` of the chord of each bar.

- [ ] **Step 4: Run the tests** — Expected: pass. Run `ruff check`, `ruff format --check`, `mypy --strict src`.

- [ ] **Step 5: Commit** — `git add src/music_decoder/compose/tune_metrics.py tests/unit/test_compose_tune_metrics.py && git commit -m "feat(compose): eighth-grid melody features and band distance"`

---

### Task 2: Statistics from POP909

**Files:**
- Create: `src/music_decoder/compose/tune_corpus.py`, `scripts/build_tune_stats.py`, `src/music_decoder/compose/tune_stats.py`, `src/music_decoder/compose/data/tune_stats.json`
- Test: `tests/unit/test_compose_tune_corpus.py`, `tests/unit/test_compose_tune_stats.py`, fixture folder `tests/fixtures/compose/pop909_mini/` (two hand-written songs in the dataset's format)

**Interfaces:**
- Consumes: `grid_features`, `GridNote`, `Band`, `Mode`, `FEATURES` (Task 1).
- Produces:

```python
# tune_stats.py
Pattern = tuple[int, ...]                       # eighth slots with an onset, ascending, never empty
MetricClass = Literal["strong", "beat", "off"]  # slots 0,4 / 2,6 / odd
Role = Literal["open", "body", "cadence"]
START = 99                                       # "previous move" of a chain's first note

def metric_class(slot: int) -> MetricClass: ...

@dataclass(frozen=True)
class TuneStats:
    rhythm: Mapping[Role, Mapping[Pattern, int]]
    rhythm_pairs: Mapping[Role, Mapping[Pattern, Mapping[Pattern, int]]]  # previous -> this -> count
    steps: Mapping[MetricClass, Mapping[int, Mapping[int, int]]]          # previous move -> move -> count
    steps_base: Mapping[MetricClass, Mapping[int, int]]
    chord_tone: Mapping[MetricClass, float]
    degrees: Mapping[Mode, tuple[float, ...]]        # 12 shares, by semitone above the tonic
    final_degrees: Mapping[Mode, tuple[float, ...]]  # same, for phrase-final notes
    notes_per_second: Band
    bands: Mapping[Mode, Mapping[str, Band]]
    meta: Mapping[str, Any]

def load_tune_stats() -> TuneStats: ...   # cached; reads compose/data/tune_stats.json via importlib.resources

# tune_corpus.py
def read_song(folder: Path) -> Song | None: ...               # None when a file is missing or unparsable
def build_stats(songs: Iterable[Song], pilot: Mapping[str, Any] | None = None,
                *, min_pattern_count: int = 5) -> dict[str, Any]: ...
```

Dataset format (per song folder): `melody.txt` lines `pitch duration` in sixteenths, pitch 0 = rest; `finalized_chord.txt` lines `name [pc, pc, ...] root beats`; `analyzed_key.txt` like `Gb:maj` or `A:min`; `tempo.txt` one integer; `human_label1.txt` like `i4A4B8A4b4o3` (letter, bars; capital letters are melodic phrases).

Rules of the eighth-grid view (spec 5.1):

- A bar is 16 sixteenths. It is *pure* when all its onsets are on even sixteenths; its pattern is the tuple of `sixteenth // 2`. Bars with no onset have no pattern.
- Roles come from the capital-letter phrases: first bar `open`, last bar `cadence`, others `body`; a one-bar phrase is `cadence`.
- `rhythm[role]` counts pure bars; `rhythm_pairs[role][previous][this]` counts a pure bar of that role whose immediately preceding bar is pure and non-empty. After counting, patterns seen fewer than `min_pattern_count` times in total are removed everywhere (the tests on the mini fixture pass `min_pattern_count=1`).
- Moves: scale index of a note = position of `(pitch - tonic) % 12` in the mode's scale (`major` 0 2 4 5 7 9 11, `minor` 0 2 3 5 7 8 10) plus 7 per octave. A note off the scale, off the eighth grid, or after a rest of a bar or more ends the chain. A move is the index difference clipped to −7..7; the first note of a chain has previous move `START` and is counted only in `steps_base`.
- `chord_tone[class]`: over on-grid notes, the share whose pitch class is in the chord sounding at their onset.
- `degrees` and `final_degrees` (last note of each capital phrase), per mode, each summing to 1.
- Windows: eight bars from the start of a capital 8-bar phrase, or of two consecutive capital 4-bar phrases; kept when every non-empty bar is pure and the window has at least 12 notes. `bands[mode][feature]` is the 25th, 50th and 75th percentile of `grid_features` over the mode's windows, with `seconds_per_bar = 240 / tempo`; `notes_per_second` is the same over all windows.
- `meta`: `{"sources": [...], "songs": n, "windows": {"major": n, "minor": n}, "built": "YYYY-MM-DD"}`.
- `pilot` (Task 3) is stored under `"pilot"` and otherwise ignored here.

- [ ] **Step 1: Write the fixture and the failing tests.** Two songs under `tests/fixtures/compose/pop909_mini/001` and `002`: `001` in `C:maj`, tempo 120, label `A4A4`, eight bars of melody on even sixteenths with a known content (bar 0 `60 2, 60 2, 62 4, 64 8`; bars repeating so that pattern `(0, 1, 2, 4)` occurs 6 times, and the two phrase-final bars are `67 16`); `002` in `A:min`, tempo 90, label `A4A4`, eight bars that all have notes, exactly one of which has an onset on an odd sixteenth. Assertions:

```python
def test_read_song_parses_the_dataset_format(mini):
    song = read_song(mini / "001")
    assert (song.tonic_pc, song.mode, song.tempo) == (0, "major", 120)
    assert song.phrases == (("A", 0, 4), ("A", 4, 4))

def test_roles_patterns_and_pairs(mini):
    stats = build_stats([read_song(mini / "001")], min_pattern_count=1)
    assert stats["rhythm"]["open"] == {"0,1,2,4": 2} and stats["rhythm"]["cadence"] == {"0": 2}
    assert stats["rhythm_pairs"]["body"]["0,1,2,4"]["0,1,2,4"] == 4

def test_impure_bars_are_left_out(mini):
    stats = build_stats([read_song(mini / "002")], min_pattern_count=1)
    assert sum(sum(v.values()) for v in stats["rhythm"].values()) == 7

def test_moves_are_scale_steps(mini):
    stats = build_stats([read_song(mini / "001")], min_pattern_count=1)
    assert stats["steps"]["beat"]["0"]["1"] >= 1      # 60 -> 62 after a repeat, landing on slot 2
    assert stats["degrees"]["major"][0] > 0 and abs(sum(stats["degrees"]["major"]) - 1) < 1e-9

def test_bands_use_grid_features(mini):
    stats = build_stats([read_song(mini / "001")], min_pattern_count=1)
    assert set(stats["bands"]["major"]) == set(FEATURES)

def test_shipped_stats_load_and_are_sane():
    s = load_tune_stats()
    assert s.meta["songs"] >= 800 and min(s.meta["windows"].values()) >= 100
    assert sum(len(v) for v in s.rhythm.values()) >= 60
    assert all(p and p == tuple(sorted(p)) and 0 <= p[0] and p[-1] <= 7 for r in s.rhythm.values() for p in r)
    assert 0.6 <= s.chord_tone["strong"] <= 0.95 and s.chord_tone["strong"] > s.chord_tone["off"]
    assert 0.10 <= s.bands["major"]["repeat"].mid <= 0.35
    assert load_tune_stats() is s
```

(JSON keys: a pattern is its slots joined by commas; moves are decimal strings. The loader converts both.)

- [ ] **Step 2: Run them and see them fail.**

- [ ] **Step 3: Implement `tune_corpus.py`, `tune_stats.py` and `scripts/build_tune_stats.py`** (`--pop909 DIR --out PATH [--pilot FILE]`; the script prints the song and window counts and the bands).

- [ ] **Step 4: Build the real file.** Run: `python scripts/build_tune_stats.py --pop909 ~/.music-decoder/datasets/pop909-hsa/POP909 --out src/music_decoder/compose/data/tune_stats.json`. Expected: about 900 songs; put the printed bands in the task report.

- [ ] **Step 5: Run the tests, ruff, mypy; commit** the four source files, the JSON, the fixture and the tests: `feat(compose): real-melody statistics from POP909`.

---

### Task 3: Pitch statistics from the pilot dataset

**Files:**
- Create (untracked, never staged): `spikes/preference-pilot/extract_vocal_lines.py`
- Modify: `src/music_decoder/compose/tune_corpus.py` (pilot merge), `scripts/build_tune_stats.py`
- Test: `tests/unit/test_compose_tune_corpus.py`

**Interfaces:**
- Produces: `pilot_stats(lines: Iterable[Mapping[str, Any]]) -> dict[str, Any]` in `tune_corpus.py`, and the merge inside `build_stats(songs, pilot=...)`.

The spike script reuses `features.py` in the same folder (`track_beats`, `estimate_key`, `vocals_stem`, `transcribe`; 90 s from 30 s in, as the pilot did) for every file in `data/audio/`, and appends one JSON line per song to `data/vocal_lines_<worker>.jsonl`: `{"id", "tonic_pc", "mode", "bpm", "notes": [[eighth_slot, pitch], ...]}`. Cleaning: drop notes shorter than 0.12 s; keep the highest note of those starting within 50 ms; slot = nearest tracked eighth (two per beat); songs with a vocal share of 0.05 or less, or fewer than 30 notes left, are skipped. Usage `extract_vocal_lines.py <worker> <n_workers> [limit]`, resumable like `run_audio.py`.

`pilot_stats` returns `{"songs": n, "interval_classes": {"repeat", "step", "third", "leap"}, "pentatonic": {"major", "minor"}, "degrees": {"major": [12], "minor": [12]}, "range": [q25, q50, q75]}`, where the first two and `range` are medians over songs.

Merge in `build_stats` when `pilot` is given and has at least 100 songs: `degrees[mode]` becomes the mean of the two sources, renormalised; the bands of `repeat`, `step`, `third`, `leap`, `pentatonic` and `range` are widened to `min(low, pilot value)` .. `max(high, pilot value)` (for `range`, the pilot quartiles); `meta["sources"]` gains the pilot entry. Rhythm and move tables are unchanged.

- [ ] **Step 1: Write the failing tests** (synthetic lines only):

```python
def test_pilot_stats_from_lines():
    line = {"id": "x", "tonic_pc": 0, "mode": "major", "bpm": 100, "notes": [[0, 60], [1, 60], [2, 62], [3, 67]]}
    p = pilot_stats([line] * 3)
    assert p["songs"] == 3 and p["interval_classes"] == {"repeat": 1 / 3, "step": 1 / 3, "third": 0.0, "leap": 1 / 3}
    assert p["pentatonic"]["major"] == 1.0 and p["degrees"]["major"][0] == 0.5

def test_merge_widens_pitch_bands_and_averages_degrees(mini):
    base = build_stats([read_song(mini / "001")], min_pattern_count=1)
    pilot = {"songs": 150, "interval_classes": {"repeat": 0.9, "step": 0.0, "third": 0.0, "leap": 0.0},
             "pentatonic": {"major": 0.5, "minor": 0.5}, "degrees": {"major": [1.0] + [0.0] * 11, "minor": [1.0] + [0.0] * 11},
             "range": [5.0, 9.0, 30.0]}
    merged = build_stats([read_song(mini / "001")], pilot=pilot, min_pattern_count=1)
    assert merged["bands"]["major"]["repeat"][2] == 0.9 and merged["bands"]["major"]["range"][2] == 30.0
    assert merged["degrees"]["major"][0] == pytest.approx((base["degrees"]["major"][0] + 1.0) / 2)
    assert merged["rhythm"] == base["rhythm"]

def test_small_pilot_is_ignored(mini):
    small = {"songs": 20, "interval_classes": {"repeat": 0.9, "step": 0, "third": 0, "leap": 0},
             "pentatonic": {"major": 0.5, "minor": 0.5}, "degrees": {"major": [1.0] + [0.0] * 11, "minor": [1.0] + [0.0] * 11},
             "range": [5.0, 9.0, 30.0]}
    assert build_stats([read_song(mini / "001")], pilot=small, min_pattern_count=1)["bands"] == build_stats([read_song(mini / "001")], min_pattern_count=1)["bands"]
```

- [ ] **Step 2: See them fail. Step 3: Implement** `pilot_stats`, the merge and `--pilot FILE...` (one or more JSONL files).
- [ ] **Step 4: Write the spike script and run it on 3 songs** (`python extract_vocal_lines.py 0 1 3` from `spikes/preference-pilot/`, with the repo's venv). Expected: three JSON lines; report the note counts and seconds per song. Do not run the full set; the controller does.
- [ ] **Step 5: Tests, ruff, mypy; commit** only the tracked files: `feat(compose): merge pilot pitch statistics into the tune statistics`.

---

### Task 4: Rhythm plan

**Files:**
- Create: `src/music_decoder/compose/tune.py`
- Test: `tests/unit/test_compose_tune_rhythm.py`

**Interfaces:**
- Consumes: `TuneStats`, `Pattern`, `Role`, `load_tune_stats` (Task 2); `rhythm.split_phrases`, `rhythm.form_labels`.
- Produces:

```python
P_RESTATE_RHYTHM = 0.5
DENSITY_SIGMA = 1.5
DENSITY_MIN, DENSITY_MAX = 3.0, 6.5
PAIR_MIN_COUNT = 20

@dataclass(frozen=True)
class BarRhythm:
    pattern: Pattern
    role: Role
    restated_from: int | None   # the bar whose pattern this bar copied

def bar_roles(n_bars: int) -> list[Role]: ...
def plan_rhythm(n_bars: int, tempo_bpm: float, rng: np.random.Generator, stats: TuneStats,
                *, final_whole_note: bool = True) -> list[BarRhythm]: ...
```

Behaviour (spec 5.2, rhythm plan): roles from `split_phrases` (first bar of a phrase `open`, last `cadence`, others `body`; a one-bar phrase `cadence`). Target onsets = `stats.notes_per_second.mid * 240 / tempo_bpm` clamped to 3.0–6.5. A bar of a phrase labelled `A'` whose role is `open` or `body`, and whose same-numbered bar in the first phrase exists with the same role, copies that bar's pattern with probability `P_RESTATE_RHYTHM` (`restated_from` = that bar). Any other bar draws from `rhythm[role]` with weight `p × exp(−(len(pattern) − target)² / (2 × 1.5²))`, where `p` is `0.5 × P(pattern | role, previous) + 0.5 × P(pattern | role)` when the previous bar's pattern has at least 20 following bars in `rhythm_pairs[role]`, else `P(pattern | role)`. Bar 0 draws only from patterns containing slot 0. With `final_whole_note` the last bar is `(0,)`, role `cadence`.

- [ ] **Step 1: Write the failing tests**

```python
STATS = load_tune_stats()

def plan(n, tempo=96.0, seed=1):
    return plan_rhythm(n, tempo, np.random.default_rng(seed), STATS)

@pytest.mark.parametrize("n,roles", [
    (1, ["cadence"]), (2, ["open", "cadence"]), (4, ["open", "body", "body", "cadence"]),
    (5, ["open", "body", "cadence", "open", "cadence"]),
    (8, ["open", "body", "body", "cadence"] * 2),
])
def test_bar_roles(n, roles):
    assert bar_roles(n) == roles

def test_plan_shape_and_determinism():
    a, b = plan(8), plan(8)
    assert a == b and len(a) == 8 and a[-1].pattern == (0,) and 0 in a[0].pattern
    assert all(bar.pattern in STATS.rhythm[bar.role] for bar in a[:-1])

@pytest.mark.parametrize("n", [1, 2, 3, 5, 9, 12, 16])
def test_every_length_plans(n):
    assert len(plan(n)) == n

def test_slow_tempos_are_denser():
    def mean(tempo):
        plans = [plan(8, tempo, s) for s in range(60)]
        return np.mean([len(bar.pattern) for p in plans for bar in p[:-1]])
    assert mean(66.0) > mean(120.0) + 0.5

def test_restatement_only_in_a_prime_and_about_half_the_time():
    plans = [plan(8, seed=s) for s in range(200)]
    assert all(bar.restated_from is None for p in plans for bar in p[:4])
    share = np.mean([p[k].restated_from == k - 4 for p in plans for k in (4, 5, 6)])
    assert 0.4 <= share <= 0.6
    assert all(p[k].pattern == p[k - 4].pattern for p in plans for k in (4, 5, 6) if p[k].restated_from is not None)

def test_rhythms_vary_like_real_tunes():
    variety = [len({bar.pattern for bar in plan(8, seed=s)}) for s in range(100)]
    assert np.median(variety) >= 4

def test_without_final_whole_note_the_last_bar_is_a_cadence_pattern():
    p = plan_rhythm(8, 96.0, np.random.default_rng(3), STATS, final_whole_note=False)
    assert p[-1].pattern in STATS.rhythm["cadence"]
```

- [ ] **Step 2: See them fail. Step 3: Implement** `bar_roles` and `plan_rhythm` in `tune.py`. **Step 4:** tests, ruff, mypy. **Step 5: Commit** `feat(compose): rhythm plan from real bar patterns`.

---

### Task 5: Pitch plan and `generate_tune`

**Files:**
- Modify: `src/music_decoder/compose/tune.py`
- Test: `tests/unit/test_compose_tune.py`

**Interfaces:**
- Consumes: `plan_rhythm`, `BarRhythm` (Task 4); `TuneStats`, `metric_class`, `START` (Task 2); from `melody.py`: `feasible_sets`, `smooth`, `final_target`, `cadence_pc`, `_prior_target`, `PRIOR_WEIGHT`, `PRIOR_WIDTH`; from `harmony.py`: `chord_pcs`, `chord_scale_pcs`, `tonic_pc`.
- Produces:

```python
P_RESTATE_CELL = 0.5
DEGREE_EXPONENT = 0.5
REGISTER_SIGMA = 5.0          # semitones
CONTEXT_MIN_COUNT = 20
MAX_REPLAY_REPLACEMENTS = 2

def scale_steps(prev: int, cand: int, pcs: Collection[int]) -> int: ...
def generate_tune(*, scale: Scale, progression: Sequence[ChordSymbol], bars_per_chord: int = 1,
                  tempo_bpm: float = 100.0, seed: int | None = None, tessitura_center: int,
                  allowed_pitches: Sequence[frozenset[int]], final_whole_note: bool = True,
                  pitch_prior: Sequence[Note] | None = None, stats: TuneStats | None = None) -> list[Note]: ...
```

The ladder for a note is the bar's chord scale, `chord_scale_pcs(scale, chord)` (the key's scale for chords in the key; spec 5.2).

`scale_steps`: the signed number of ladder pitches (pitches of any octave whose class is in `pcs`) passed going from `prev` to `cand`, counting `cand` if it is on the ladder and not `prev`; 0 when equal; clipped to −7..7.

Behaviour is spec 5.2 ("Pitch plan", "Cell memory", "Phrase ends", "Note lengths"), with these points fixed:

- Candidates of a note: `allowed_pitches[bar]` restricted to `chord_scale_pcs(scale, chord)`; all of `allowed_pitches[bar]` if that is empty.
- Weight = move × harmony × degree × register × prior, as in the spec. The move factor uses `stats.steps[class][previous move]` when its total is at least 20, else `stats.steps_base[class]`; a move with no count gets 0.5 (in counts) so nothing has weight zero. The first note of the piece uses `steps_base` with previous move `START`, treating its move as 0.
- Hard filters before weighting: interval size at most 12 and not 6, 10 or 11; at a chord change, and for the final note, at most 7. If the filters leave no candidate, drop the chord-scale restriction first, then the chord-change limit; never the interval sizes.
- The final note is `final_target(scale, progression[-1], candidates of the last bar, reference=previous pitch)`; the bar before it keeps each note inside `feasible_sets(candidates, notes left, {final})`. The final note lasts the whole bar (`4 * beat_s * 0.95`).
- The last note of a non-final phrase: chord tones only (if any is a candidate), weighted by `stats.final_degrees[mode]` instead of `degrees`, the tonic excluded when another chord tone is a candidate.
- Cell memory: a bar whose pattern was used by an earlier bar replays that bar's moves (the most recent such bar; for a bar with `restated_from` set, that bar) with probability `P_RESTATE_CELL`, or always when `restated_from` is set and the bar is the first or second of its phrase. The start pitch is the candidate giving the most chord tones on the bar's slots 0 and 4, ties to the one nearest the previous pitch. A replayed pitch that is not a candidate or fails a hard filter is sampled normally instead; after 2 such replacements the rest of the bar is sampled.
- Notes: `start_s = (bar * 8 + slot) * eighth_s`; `end_s = start_s + 0.95 × (time to the next onset in the bar, or to the bar's end)`; velocity 80 on slot 0, 75 on slot 4, 70 elsewhere; `confidence=1.0`.
- Validation mirrors `generate_melody`: `ValueError` when `allowed_pitches` has the wrong length or an empty set.

- [ ] **Step 1: Write the failing tests**

```python
STATS = load_tune_stats()
C, G, AM, F = (ChordSymbol.parse(x) for x in ("C", "G", "Am", "F"))

def wide(scale, n_bars, low=55, high=72):
    pcs = set(scale_pcs(scale))
    return [frozenset(p for p in range(low, high + 1) if p % 12 in pcs)] * n_bars

def tune(seed=1, scale="C:major", prog=(C, G, AM, F), bars=2, tempo=96.0, allowed=None, **kw):
    s = Scale.parse(scale)
    allowed = allowed if allowed is not None else wide(s, len(prog) * bars)
    return generate_tune(scale=s, progression=list(prog), bars_per_chord=bars, tempo_bpm=tempo, seed=seed,
                         tessitura_center=64, allowed_pitches=allowed, **kw)

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
        for a, b in zip(notes, notes[1:]):
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
        assert last.pitch % 12 == 0 and last.end_s - last.start_s == pytest.approx(4 * 60 / 96 * 0.95)

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
        assert all(abs(b.pitch - a.pitch) not in (6, 10, 11) for a, b in zip(notes, notes[1:]))

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
    prior = [Note(start_s=k * e, end_s=(k + 1) * e, pitch=71, velocity=80, confidence=1.0) for k in range(64)]
    def share(p):
        notes = [n for s in range(40) for n in tune(s, pitch_prior=p)]
        return np.mean([n.pitch % 12 == 11 for n in notes])
    assert share(prior) > share(None) + 0.05
```

- [ ] **Step 2: See them fail. Step 3: Implement** the pitch plan and `generate_tune`. **Step 4:** tests, ruff, mypy. **Step 5: Commit** `feat(compose): tune engine pitch plan and generate_tune`.

---

### Task 6: Wire the engine in

**Files:**
- Modify: `src/music_decoder/compose/api.py`, `src/music_decoder/compose/scoring.py`, `src/music_decoder/cli/main.py`, `src/music_decoder/evaluation/compose_musicality.py`
- Test: `tests/unit/test_compose_tune_wiring.py`, existing compose tests, new fixture `tests/fixtures/compose/rules_engine_melodies.json`

**Interfaces:**
- Consumes: `generate_tune` (Task 5), `tune_features`, `band_distance` (Task 1), `load_tune_stats` (Task 2).
- Produces:
  - `MelodyEngine = Literal["tune", "rules"]` in `api.py`; `melody_engine: MelodyEngine` keyword on `arrange_candidates` and `arrange` (default `"rules"`, so existing callers are unchanged) and on `compose_with_options` (default `"tune"`); `compose()` passes nothing and so uses `"tune"`.
  - `score_arrangement(arr, progression, *, bars_per_chord, tempo_bpm, tuning, scale: Scale | None = None, melody_engine: MelodyEngine = "rules")`; `W_CORPUS = 2.0`. For `"tune"` (which requires `scale`): `W_IDENTITY × identity + W_ABOVE × above + W_SHAPE_MOVES × moves + W_FALLBACK × fallback − W_CORPUS × band_distance(tune_features(...), stats.bands[scale.mode])`.
  - `metadata["melody_engine"]`; `COMPOSER_VERSION = 6`; CLI option `--melody-engine [tune|rules]`, default `tune`.
  - `evaluation/compose_musicality.build_cases` arranges fingerstyle with `melody_engine="rules"`.

- [ ] **Step 1: Capture the rules engine before touching it.** Write `tests/fixtures/compose/rules_engine_melodies.json` from the current code: for seeds 11–16, `arrange(Scale.parse("C:major"), [C, G, Am, F], bars_per_chord=2, tempo_bpm=96.0, seed=seed, candidates=12)` → the melody as `[start_s, end_s, pitch]` triples (rounded to 6 places) and the chosen candidate. Commit it alone: `test(compose): pin the rules engine's melodies`.

- [ ] **Step 2: Write the failing tests**

```python
PROG = [ChordSymbol.parse(c) for c in ("C", "G", "Am", "F")]
SCALE = Scale.parse("C:major")

def test_rules_engine_is_unchanged(rules_fixture):
    for seed, want in rules_fixture.items():
        arr = arrange(SCALE, PROG, bars_per_chord=2, tempo_bpm=96.0, seed=int(seed), candidates=12, melody_engine="rules")
        assert [[round(n.start_s, 6), round(n.end_s, 6), n.pitch] for n in arr.melody] == want["melody"]
        assert arr.chosen_candidate == want["chosen_candidate"]

def test_arrange_defaults_to_rules_and_compose_to_tune(tmp_path):
    assert arrange(SCALE, PROG, seed=3) == arrange(SCALE, PROG, seed=3, melody_engine="rules")
    c = compose(SCALE, PROG, seed=3, out_dir=tmp_path / "a")
    assert c.metadata["melody_engine"] == "tune" and c.metadata["composer_version"] == 6
    r = compose_with_options(SCALE, PROG, seed=3, out_dir=tmp_path / "b", melody_engine="rules")
    assert r.metadata["melody_engine"] == "rules"
    assert [n.pitch for n in c.melody_notes] != [n.pitch for n in r.melody_notes]

@pytest.mark.parametrize("tuning", sorted(PRESETS))
@pytest.mark.parametrize("key,mode", [("C", "major"), ("A", "minor"), ("F#", "minor"), ("G#", "major")])
def test_guitar_guarantees_hold_with_the_tune_engine(tuning, key, mode):
    # build I-V-vi-IV or i-VI-III-VII in the key with evaluation.compose_musicality.realize
    for seed in (1, 2, 3):
        arr = arrange(scale, prog, bars_per_chord=2, tempo_bpm=96.0, tuning=PRESETS[tuning], seed=seed,
                      candidates=4, melody_engine="tune")
        assert metrics.playable_moment_ratio((*arr.melody_tab, *arr.accompaniment_tab)) == 1.0
        assert metrics.melody_onset_on_top_ratio(arr.melody, arr.accompaniment) == 1.0
        assert metrics.tab_consistent((*arr.melody, *arr.accompaniment), (*arr.melody_tab, *arr.accompaniment_tab), PRESETS[tuning])
        assert metrics.forbidden_interval_count(arr.melody) == 0 and metrics.max_interval(arr.melody) <= 12

@pytest.mark.parametrize("tempo", [40.0, 200.0])
@pytest.mark.parametrize("chords", ["Cmaj7 Am7 Dm7 G7", "C Csus4 Caug Bdim"])
def test_extreme_tempos_and_chord_qualities(tempo, chords):
    prog = [ChordSymbol.parse(c) for c in chords.split()]
    assert arrange(SCALE, prog, tempo_bpm=tempo, seed=None, candidates=2, melody_engine="tune").melody

def test_tune_score_prefers_real_melody_statistics():
    cands = arrange_candidates(SCALE, PROG, bars_per_chord=2, tempo_bpm=96.0, seed=5, candidates=12, melody_engine="tune")
    stats = load_tune_stats()
    dist = [band_distance(tune_features(a.melody, SCALE, PROG, 2, 96.0), stats.bands["major"]) for a in cands]
    assert dist[0] <= float(np.median(dist))
    assert [a.score for a in cands] == sorted((a.score for a in cands), reverse=True)

def test_cli_flag(tmp_path):
    # invoke `compose --scale C:major --progression "C G Am F" --melody-engine rules --seed 3 --format json --out <tmp>`
    # with click's CliRunner; assert metadata.melody_engine == "rules"; without the flag it is "tune".
    ...
```

(`PRESETS` is `music_decoder.tabs.tuning.PRESETS`; every preset is covered.)

- [ ] **Step 3: See them fail.**

- [ ] **Step 4: Implement.** `_arrange_fingerstyle` takes `melody_engine` and calls `generate_tune` (same `scale`, `progression`, `bars_per_chord`, `tempo_bpm`, `seed`, `tessitura_center=center`, `allowed_pitches=allowed`, `pitch_prior`) or `generate_melody` as today. Thread the parameter through `arrange_candidates`, `arrange`, `compose_with_options` and the CLI; pass `scale` and `melody_engine` to `score_arrangement`. If a bar without an onset on beat 1 breaks the arranger, the repair, let-ring or the tab, fix that at its cause with a test of its own; do not make the engine avoid such bars.

- [ ] **Step 5: Make the existing suite pass.** Existing tests that assert v5 melody behaviour through `compose()` or `compose_with_options()` pass `melody_engine="rules"` (or go through `arrange`, whose default is `"rules"`); tests of guitar guarantees and output files stay engine-neutral. List every test changed, with the reason, in the report. Run `pytest -q tests/unit tests/integration`, ruff, mypy.

- [ ] **Step 6: Commit** `feat(compose): tune engine as the fingerstyle default (composer v6)`.

---

### Task 7: The tune harness, the gates and the tuning

**Files:**
- Create: `src/music_decoder/evaluation/compose_tune.py`
- Modify: `src/music_decoder/compose/tune.py` (constants only, unless a gate shows a defect)
- Test: `tests/unit/test_compose_tune_gates.py`

**Interfaces:**
- Produces:

```python
DEV_KEYS_MAJOR = ("C", "G", "D", "A#", "E", "F")
DEV_KEYS_MINOR = ("A", "E", "D", "C", "F#", "B")
HOLDOUT_KEYS_MAJOR = ("A", "D#", "B", "G#")
HOLDOUT_KEYS_MINOR = ("G", "C#", "F", "A#")
MAJOR_PROGRESSIONS = ("I V vi IV", "I vi IV V", "I IV V I")
MINOR_PROGRESSIONS = ("i VI III VII", "i iv V i", "i VII VI V")
TEMPOS = (72.0, 96.0, 120.0)
DEV_SEEDS, HOLDOUT_SEEDS = (5000, 5001), (6000, 6001)

@dataclass(frozen=True)
class TuneCase:
    scale: Scale; progression: tuple[ChordSymbol, ...]; bars_per_chord: int; tempo_bpm: float; seed: int
    arrangement: Arrangement          # the best of 12, as compose() writes it
    features: dict[str, float]        # tune_features of candidate 0, before any selection

def tune_cases(matrix: Literal["dev", "holdout"]) -> list[TuneCase]: ...
def tune_report(cases: Sequence[TuneCase]) -> dict[str, dict[str, dict[str, float]]]: ...
```

Each matrix: every key × its mode's three progressions × the three tempos × the two seeds at `bars_per_chord=2` (8 bars), plus the first four keys of each mode × the first progression × tempo 96 × the two seeds at `bars_per_chord=4` (16 bars); all with `arrange_candidates(..., candidates=12, melody_engine="tune")`, standard tuning. `arrangement` is the first (best) candidate; `features` are those of the candidate whose `chosen_candidate` is 0, so T1 and T2 measure the engine before selection (spec 5.5: the score picks by the same bands, so gating its pick would pass by construction). Progressions are realised with `compose_musicality.realize`. `tune_report` gives, per mode and per feature, `{"q25", "median", "q75"}` over the cases' `features`.

`KNOWN_MISSES: dict[tuple[str, str], str]` in the test module maps `(mode, feature)` to the ruling's one-line reason. It starts empty. Only the controller adds to it, after a BLOCKED-ON-RULING report.

- [ ] **Step 1: Write the gate tests** (module-scoped fixtures build each matrix once):

```python
STATS = load_tune_stats()
SPREAD_FEATURES = ("repeat", "notes_per_second", "rhythm_variety", "figure_recurrence")
KNOWN_MISSES: dict[tuple[str, str], str] = {}

@pytest.mark.parametrize("mode", ["major", "minor"])
@pytest.mark.parametrize("feature", FEATURES)
def test_t1_medians_sit_in_the_real_melody_bands(dev_report, mode, feature):
    if (mode, feature) in KNOWN_MISSES:
        pytest.xfail(KNOWN_MISSES[(mode, feature)])
    band = STATS.bands[mode][feature]
    assert band.low <= dev_report[mode][feature]["median"] <= band.high

@pytest.mark.parametrize("mode", ["major", "minor"])
@pytest.mark.parametrize("feature", SPREAD_FEATURES)
def test_t2_pieces_are_not_all_alike(dev_report, mode, feature):
    band, got = STATS.bands[mode][feature], dev_report[mode][feature]
    assert got["q75"] - got["q25"] >= 0.5 * (band.high - band.low)

def test_t3_guitar_guarantees(dev_cases):
    for c in dev_cases:
        a = c.arrangement
        assert metrics.playable_moment_ratio((*a.melody_tab, *a.accompaniment_tab)) == 1.0
        assert metrics.melody_onset_on_top_ratio(a.melody, a.accompaniment) == 1.0
        assert metrics.tab_consistent((*a.melody, *a.accompaniment), (*a.melody_tab, *a.accompaniment_tab), STANDARD_EADGBE)
        assert metrics.forbidden_interval_count(a.melody) == 0 and metrics.max_interval(a.melody) <= 12
        assert metrics.final_note_beats(a.melody, c.tempo_bpm) >= 1.5
        assert metrics.final_resolution(a.melody, c.scale, c.progression)
        assert max(metrics.chord_change_moves(a.melody, c.progression, c.bars_per_chord, c.tempo_bpm), default=0) <= 7
    triad_identity = [...]  # P3 over triad bars, as compose_musicality._fingerstyle_part_metrics computes P3t
    assert np.mean(triad_identity) >= 0.99
    # the move into the final note is at most 7 semitones (the function behind the L5 gate)

def test_known_misses_stay_few():
    assert all(sum(m == mode for m, _ in KNOWN_MISSES) <= 2 for mode in ("major", "minor"))

def test_t4_same_seed_same_music():
    a, b = tune_cases("dev")[:3], tune_cases("dev")[:3]
    assert [c.arrangement for c in a] == [c.arrangement for c in b]

def test_t5_every_key_and_mode_composes():
    for tonic in VALID_TONICS:
        for mode in ("major", "minor"):
            # I-V-vi-IV or i-VI-III-VII, 8 bars, seed 1, candidates=2
            assert arrange(...).melody

def test_holdout_meets_t1_and_t3(holdout_report, holdout_cases): ...  # the same assertions as T1 and T3
```

(The repo has no acceptance marker and runs with `--strict-markers`; the holdout test is a plain test whose fixtures build the holdout matrix, as `test_holdout_acceptance` does in `test_compose_musicality.py`. Write it last, after the tuning of Step 3 is finished.)

- [ ] **Step 2: Run the dev gates and record every miss** (feature, mode, value, band) in the report.

- [ ] **Step 3: Tune.** Adjust only `P_RESTATE_RHYTHM`, `P_RESTATE_CELL`, `DEGREE_EXPONENT`, `REGISTER_SIGMA`, `DENSITY_SIGMA` and `W_CORPUS` until T1 and T2 pass on the dev matrix with T3 intact. If a feature still misses, find the cause in the engine (for example the hand shape bending replayed figures) and fix that, with a unit test in `test_compose_tune.py`. A band that cannot be met together with a guitar guarantee is reported as BLOCKED-ON-RULING with the measured values and the cause; do not edit the statistics file, the gate or `KNOWN_MISSES` to pass, and do not start a melody-aware shape choice (a non-goal of the spec).

- [ ] **Step 4: Run the holdout test once.** Report its values whatever they are.

- [ ] **Step 5: Full suite, ruff, mypy; commit** `test(compose): gate the tune engine against real-melody statistics`.

---

### Task 8: Documents, samples, results

**Files:**
- Modify: `docs/usage.md`, `docs/architecture-overview.md`, `docs/architecture-technical.md`, `docs/superpowers/specs/2026-11-16-composer-v6-melody-design.md` (add "10. Results")
- Create (untracked): `compose-samples/2026-11-16-v6/`

- [ ] **Step 1: Usage and architecture.** Document `--melody-engine`, what the tune engine does in plain words, where the statistics come from (sources, licences, how to rebuild: the clone command and the script), and that strum and arpeggio use the rules engine.
- [ ] **Step 2: Samples.** Re-run the 28 commands of `compose-samples/2026-11-16-v5-master/*/command.txt` (the tune engine is now the default) into `compose-samples/2026-11-16-v6/`, each folder with `composition.mid`, `composition.wav`, `compose.json`, `tab.txt`, `command.txt`; add `NN-rules` for pieces 01–10 with `--melody-engine rules`; write a `README.md` with the feature table per piece beside the bands.
- [ ] **Step 3: Results.** In the spec's new section 10: each success criterion with its measured value, every ruling and tuning change, the dev and holdout tables (feature, v5 rules engine, tune engine, band), whether the statistics include the pilot data, and what remains open.
- [ ] **Step 4: Verify** the docs' commands by running them; full suite, ruff, mypy. **Step 5: Commit** the docs only: `docs(compose): composer v6 tune engine, usage and results`.
