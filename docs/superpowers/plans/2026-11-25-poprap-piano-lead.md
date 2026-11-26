# Pop-rap Piano Lead Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `productionize --style pop-rap --lead piano` plays the composer's melody on the boom-bap sampled grand piano.

**Architecture:** Factor the core of `instruments/piano.py` so it takes notes, bar length, tail start and leveling spans rather than a boom-bap `Song`. `render_piano(song, ...)` becomes a thin wrapper, so boom-bap is unchanged. `poprap_mix._render_lead` gets a piano branch fed by a `SampleStore`, which `api._productionize_poprap` now receives.

**Tech Stack:** Python 3.11, numpy, click, pytest.

**Spec:** `docs/superpowers/specs/2026-11-25-poprap-piano-lead-design.md`

## Global Constraints

- `PopRapLead = Literal["guitar", "synth", "piano"]`; the default lead stays `"guitar"`.
- `PopRapMixConfig.lead_piano_dbfs = -17.0`.
- Hook-2 double: piano lead → guitar; guitar lead → synth; synth lead → guitar (unchanged).
- Guitar and synth renders stay bit-identical; boom-bap renders stay bit-identical.
- The piano plays the melody's exact notes (pitch, onset); `lead_notes` is not changed.
- The piano lead compresses against the sections with `lead != "none"`; notes at or after `song.body_s` get `tail_release_tau_s`.
- With the piano lead, metadata `credits == [PIANO_CREDIT]`; otherwise `[]`.
- CLI `--samples-dir` help: `"sample folder (boom-bap, and the pop-rap piano lead)"`.
- No commits: the user asked for none; leave changes in the working tree.

## Review Focus

- Missing piano samples with `download=False`: a `SampleError` raised before any output file is written.
- A melody with no notes in any lead section (empty lead): a silent lead stem, not a crash or divide-by-zero in the compressor (`rms_over` of silence is 0).
- A melody note longer than its bar: the pedal rule ends it at the bar end, the same as boom-bap. Pinned by reusing `plan_voices` semantics.
- Buffer lengths that round down (`int(n/sr*sr) != n`): the piano lead must fit `n` exactly, like the other leads. Add `"piano"` to the existing parametrized test.
- `--lead piano` with the boom-bap style: ignored, as `--lead synth` is today (no new error).

---

### Task 1: Piano core that takes plain notes

**Files:**
- Modify: `src/music_decoder/productionize/instruments/piano.py`
- Test: `tests/unit/test_productionize_piano.py` (extend the existing file)

**Interfaces:**
- Produces:
  - `plan_note_voices(notes: Sequence[NoteEvent], *, bar_s: float, tail_start_s: float, cfg: PianoConfig) -> list[Voice]`. The body of today's `plan_voices`; `plan_voices(song, cfg)` calls it with `song.piano, song.grid.bar_s, song.tail_start_s`.
  - `render_piano_notes(notes, *, bar_s: float, tail_start_s: float, level_spans: Sequence[tuple[int, int]], sr: int, cfg: PianoConfig, store: SampleStore, n_samples: int) -> Audio`. The body of today's `render_piano`, with the compressor reference `dsp.rms_over(out, level_spans)`. `render_piano(song, ...)` passes the verse spans.
  - `piano_note_refs(notes, *, bar_s, tail_start_s, cfg) -> set[SampleRef]`: the refs `render_piano_notes` will load, for an up-front `store.ensure`.

- [ ] **Step 1:** Write `test_plan_voices_matches_plan_note_voices`: for `tiny_song()`, `plan_voices(song, cfg) == plan_note_voices(song.piano, bar_s=song.grid.bar_s, tail_start_s=song.tail_start_s, cfg=cfg)`.
- [ ] **Step 2:** Write `test_render_piano_is_unchanged_by_the_split`: `render_piano(song, cfg, store, n)` equals `render_piano_notes(song.piano, ..., level_spans=[verse spans], ...)` with `np.array_equal`, using `fake_store`.
- [ ] **Step 3:** Write `test_render_piano_notes_with_no_notes_is_silent`: empty notes → `np.zeros((n, 2))`.
- [ ] **Step 4:** Run `.venv/bin/pytest tests/unit/test_productionize_piano.py -q`. Expected: the new tests FAIL (names not defined).
- [ ] **Step 5:** Implement the three functions and turn `plan_voices` / `render_piano` into wrappers.
- [ ] **Step 6:** Run `.venv/bin/pytest tests/unit/test_productionize_piano.py tests/unit/test_productionize_mix.py -q`. Expected: PASS.

### Task 2: Piano lead in the pop-rap mix, config and API

**Files:**
- Modify: `src/music_decoder/productionize/config.py` (`PopRapLead`, `lead_piano_dbfs`)
- Modify: `src/music_decoder/productionize/poprap_mix.py` (`render_poprap_mix(..., store: SampleStore | None = None)`, `_render_lead`, level target)
- Modify: `src/music_decoder/productionize/api.py` (pass `SampleStore(samples_dir)` to pop-rap; `ensure` the piano refs before `out_dir.mkdir`; credits)
- Test: `tests/unit/test_productionize_poprap_mix.py`, `tests/unit/test_productionize_api.py`

**Interfaces:**
- Consumes: Task 1's `render_piano_notes`, `piano_note_refs`.
- Produces: `render_poprap_mix(song, cfg, work_dir, progress=None, store=None)`. `store` is required only when `cfg.lead == "piano"`; otherwise it raises `ProductionError("the piano lead needs a sample store")`.

`PopRapConfig` gains `piano_lead: PianoConfig = PianoConfig()` (the boom-bap piano's defaults).

Piano lead rendering in `_render_lead`:
- `main` notes go to `render_piano_notes` with `bar_s=song.grid.bar_s`, `tail_start_s=song.body_s`, `level_spans=[song.section_span(i) for i, s in enumerate(song.sections) if s.lead != "none"]`, `sr` and `cfg.piano_lead`, then `_fit(clip, n)`.
- The stereo clip is used as is, with no `constant_power_pan`.
- The double is `_lead("guitar", double, ...)`.

- [ ] **Step 1:** Add `"piano"` to the parametrize list of `test_lead_fits_buffer_lengths_that_round_down`, passing a `fake_store` that holds the piano refs of the song's lead notes. The helper is `fake_lead_store(root, song)` in `tests/productionize_helpers.py`; it writes `write_piano_sample` for each ref of `piano_note_refs`.
- [ ] **Step 2:** Write `test_piano_lead_plays_the_melody_where_the_guitar_does`: render with lead piano and with lead guitar. Over each section, the lead stem has energy (RMS > 1e-4) in exactly the same sections, the verse lead energy is below 1e-6 of the hook's, and the other stems (`drums`, `808`, `pad`, `pluck`, `fx`) are `np.array_equal` across the two renders.
- [ ] **Step 3:** Write `test_piano_lead_needs_a_store`: `render_poprap_mix(song, replace(CFG, lead="piano"), tmp)` with no store raises `ProductionError`.
- [ ] **Step 4:** Write `test_piano_lead_levels_at_minus_17`: `PopRapMixConfig().lead_piano_dbfs == -17.0`, and the rendered lead stem's RMS over the lead sections is within 0.5 dB of -17 dBFS.
- [ ] **Step 5:** In `test_productionize_api.py`, write `test_poprap_piano_metadata_credits_the_piano`: with `lead="piano"` and an offline fake store, `metadata["lead"] == "piano"` and `metadata["credits"] == [PIANO_CREDIT]`. Write `test_poprap_piano_missing_samples_fail_before_writing`: with `SampleStore(empty_dir, download=False)` it raises `SampleError` and `out_dir` does not exist. The api takes `samples_dir` plus a test seam: monkeypatch `api.SampleStore` to build `download=False` stores.
- [ ] **Step 6:** Run the new tests. Expected: FAIL.
- [ ] **Step 7:** Implement the config, mix and API changes.
- [ ] **Step 8:** Run `.venv/bin/pytest tests/unit/test_productionize_poprap_mix.py tests/unit/test_productionize_api.py tests/unit/test_productionize_config.py -q`. Expected: PASS.

### Task 3: CLI, docs and full verification

**Files:**
- Modify: `src/music_decoder/cli/main.py` (`--lead` choices `["guitar", "synth", "piano"]`, cast, `--samples-dir` help)
- Modify: `docs/usage.md` (pop-rap section: the piano lead, its double, samples)
- Test: `tests/unit/test_cli_productionize.py`, `tests/integration/test_productionize_e2e.py`

- [ ] **Step 1:** Extend `test_productionize_style_and_lead_reach_the_config` with `--lead piano` → `config.poprap.lead == "piano"`.
- [ ] **Step 2:** Add `"piano"` to `test_poprap_render_passes_every_check`'s parametrize. It uses the real sample store and is skipped when the samples are not present locally, through a new `needs_piano_samples` marker in `tests/productionize_helpers.py` (checks `SampleStore().missing(piano_refs(PianoConfig()))` is empty).
- [ ] **Step 3:** Implement the CLI changes and update `docs/usage.md`.
- [ ] **Step 4:** Run the full suite, `.venv/bin/pytest -q`. Expected: all pass (or the same skips as on master).
- [ ] **Step 5:** Run `music-decoder productionize <a take> --style pop-rap --lead piano --out <tmp>` on a real take and confirm the acceptance checks print.
