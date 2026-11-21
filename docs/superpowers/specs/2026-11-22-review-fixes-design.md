# Review fixes: take manifest, stable audio re-rank, scored endings — design

Status: approved (22 Nov 2026). This is a small change: three bug fixes from an outside review. Each one was reproduced on master `5ba3da2`.

## Why

An outside reviewer found three bugs. Each makes the tool pick or play something other than what the composition says.

1. **The ending chord depends on how a take is opened.**
   - Example: `Dm G C Am` in C major at 96 bpm, seed 7. Given the `Composition` object, the production ends on C. Given the take folder, it ends on D. Both boom-bap and pop-rap are affected.
   - Cause: `compose` writes only `composition.mid` and `composition.wav`, so a folder carries no key. The production then guesses home from the loop's first chord.
2. **A dominated candidate can flip the audio re-rank.**
   - `choose_by_audio` min-max scales the audio and symbolic scores across the current shortlist. A beat B, but adding a C that is worse than both on both scores made B win.
   - The audio scores were A 6.1, B 6.2, C 6.0. The symbolic scores were A 1.0, B 0.0, C -10.0.
3. **The melody's ending is never scored.**
   - `grid_features` averages over whole 8-bar windows and drops a shorter tail. For a 10- or 12-bar pass, the last 2 or 4 bars, including the cadence, never reach the candidate score.

## Goals

- **A take folder plays the same as its `Composition`.** That covers the same home chord, progression, scale and tempo.
- **Adding a candidate that is worse than two others on both scores never changes their order.**
- **Candidate scoring sees the end of every pass.**

## Non-goals

- A separate cadence measure. That would be a hand-written melody rule; v5 rules were rejected.
- Changing the corpus statistics (`tune_stats`), or how real tunes are measured.
- The reviewer's larger suggestions: listener labels, loop endings, articulation, and hook-aware shapes. Those are separate decisions.

## Anti-goals

- Changing boom-bap or pop-rap output for a `Composition` source.
- Changing strum or arpeggio output, which is frozen at v3.
- Breaking standalone MIDI files that have no manifest. They keep today's inference.

## Approaches

### Bug 1

- **Chosen:** a JSON manifest beside the MIDI. This is the reviewer's suggestion. The metadata is already a dict on `Composition`, so it is cheap to write, and it is readable by people and other tools.
- **Rejected:** key and progression embedded as MIDI meta events. That works for standalone MIDI too, but pretty_midi's text-event support is thin and the result is opaque to people.

### Bug 2

- **Chosen:** fixed scales, measured once on a reference set.
- **Rejected:** per-shortlist rank-based scores. Those still depend on the shortlist.

### Bug 3

- **Chosen:** add a window anchored at the last bar. Every window stays 8 bars, so the corpus bands still apply.
- **Rejected:** a shorter tail window. That would be compared with 8-bar bands it doesn't match.

## Design

### Bug 1: the take manifest

- **Writing it.** `compose_with_options` writes `composition.json` beside `composition.mid`.
  - The file is `{"manifest_version": 1, **metadata}`.
  - `metadata` is the exact dict the returned `Composition` carries.
- **Reading it.** `productionize`'s folder and `.mid` paths read `composition.json` when it sits beside the MIDI. It supplies the same tempo, progression, scale, title and key that a `Composition` source gets through `composition_kwargs`.
  - An explicit `--progression`, `--key` or `--title` still wins.
  - Without the file, nothing changes.
  - A manifest whose `manifest_version` isn't 1, or that fails to parse, raises `ProductionError` with a clear message.
- **Tests.**
  - A parity test for boom-bap and pop-rap, using `Dm G C Am` in C major: the folder, the `.mid` and the `Composition` give the same home chord and the same `Song` / `PopRapSong`.
  - A test that a bare MIDI with no manifest behaves as before.

### Bug 2: fixed re-rank scales

`choose_by_audio` scores each candidate as `x / AUDIO_SCALE + SYMBOLIC_WEIGHT * s / SYMBOLIC_SCALE`.
- `x` is `(CE + PQ) / 2` and `s` is the symbolic score. Shifting the scores by a constant doesn't change the order, so no minimum is subtracted.
- `SYMBOLIC_WEIGHT` stays 0.5.
- **Measuring the scales.** Use a fixed reference set of 30 fingerstyle prompts with default settings: the keys, progressions, tempos and seeds are listed in the code comment.
  - `AUDIO_SCALE` is the median over the reference shortlists of `max(span of x, 0.3)`.
  - `SYMBOLIC_SCALE` is the median span of `s`.
  - These are the spans today's code effectively divides by, so typical choices stay close to today's.
- **Tests.**
  - The reviewer's case: appending a dominated candidate keeps A ahead of B.
  - A property test over random scores: appending a candidate that is worse than every existing one on both scores never changes the order of the others.

### Bug 3: scoring the last bars

- `grid_features` gains a keyword `tail_window: bool = False`. When it is True and `n_bars > 8` and `n_bars % 8 != 0`, it adds one more 8-bar window starting at bar `n_bars - 8`, so the last bar is always covered.
- `tune_features`, used for candidate scoring and the evaluation report, passes True. Corpus extraction in `tune_corpus.py` keeps False, so the bands are unchanged.
- **Tests:**
  - 9-, 12- and 15-bar cases: two melodies with the same first 8 bars and different endings get different features.
  - For 8 and 16 bars, the features are unchanged.

### Composer version

- `COMPOSER_VERSION` goes from 6 to 7, because the same seed can now give a different fingerstyle piece. Fixes 2 and 3 change the winner for some prompts.
- The two tests that pin the version are updated on purpose.

### Docs

- The usage guide's "Composition output" lists `composition.json`.
- The `productionize` section says a take folder's manifest supplies the key, progression and tempo.

## Left to the builder

- Where the manifest helpers live (`productionize/song.py` next to `composition_kwargs` is suggested).
- The exact reference-set prompts, as long as they span major and minor keys, at least five progressions, tempos 66–126 and several bars per chord.
