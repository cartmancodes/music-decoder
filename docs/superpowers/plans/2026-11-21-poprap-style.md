# Pop-rap Production Style Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `music-decoder productionize SOURCE --style pop-rap [--lead guitar|synth]` renders any composed piece as a mastered half-time trap / pop-rap instrumental that sounds like spike take A, keeping our melody, chords, key and tempo.

**Architecture:**
- **What's new:** a second render path inside `music_decoder.productionize`:
  - synthesized trap drums, 808, pad, pluck and a lead;
  - a fixed song layout (intro, hook, verse, hook, outro);
  - its own bus mix.
- **What's shared with boom-bap:** composer-file reading, master, export, JSON report and acceptance checks.
- **Unchanged:** boom-bap stays the default, and its output does not change by a single sample.

**Tech Stack:** Python 3.11, numpy, scipy.signal, pretty_midi, the FluidSynth CLI, click, pytest, mypy --strict, ruff.

**Spec:** `docs/superpowers/specs/2026-11-21-poprap-style-design.md`

**Spike (the reference sound and every number):** `/Users/shubhojeet/Projects/music-decoder/compose-samples/2026-11-21-poprap-1871/poprap_spike.py`.
- The file is untracked and is not in a worktree, so read it by this absolute path.
- Port its synthesis formulas as they are, except where this plan says otherwise.
- Never copy the spike into the repo.

## Global Constraints

- **Boom-bap unchanged:** every existing test passes unmodified, except the two mechanical signature updates Task 7 names. That includes `make test-production` for boom-bap.
- **Numbers live in config:** every level, pattern step, filter frequency, time constant, gain and send is a field of a frozen dataclass in `productionize/config.py`.
  - **Ruling:** the fine micro-shape constants inside a single hit (the clap's three burst offsets 0 / 9 / 19 ms, the click lengths) may be UPPER_CASE module constants with a comment.
- **Deterministic:** the same input and config give identical audio. Use `PopRapConfig.seed`, with a separate `np.random.default_rng((seed, stream, index))` per hit, so adding a hit never changes another.
- **No new downloads, no new dependencies.**
  - The guitar renders with the FluidSynth CLI.
  - Soundfont: `GuitarLeadConfig.soundfont` when that path exists, else `music_decoder.synth._resolve_soundfont_path()`, else `ProductionError`.
  - The synth lead needs neither FluidSynth nor a soundfont.
- **Master unchanged:** `master()` with `MasterConfig()` (−10.5 LUFS, −1.8 dBTP); MP3 genre tag "Hip-Hop".
- **Sample rate:** 44 100 Hz (`SAMPLE_RATE`).
- **Each task ends green:** `ruff check src tests`, `mypy --strict src` and `pytest -q tests/unit` all pass.
- **Commits** end with the line `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Never stage** `spikes/`, `out/` or `compose-samples/`.

## Review Focus

1. **A pass longer than 16 bars** (8 chords × 4 bars = 32): the hook and verse are 1 pass (32 bars) and the 8-bar intro cuts the pass short. Nothing breaks. Pinned in Task 5.
2. **A melody note that rings past the end of its pass or section:** it is cut at the section end and never sounds into the verse. Pinned in Task 5.
3. **A slow piece (66 bpm):** the pop-rap render completes, every sample is finite, and the hook holds 16 bars. Pinned in Task 6.
4. **Guitar lead without FluidSynth or a soundfont:** a clear `ProductionError` before any file is written. The synth lead still renders. Pinned in Task 4.
5. **Pop-rap given a ready `Song`** (`--hisaab`, or `hisaab_song()` in Python): a clear usage error or `ProductionError`, never a crash deep in the mix. Pinned in Task 8.

---

### Task 1: Shared composer-file reading

**Files:**
- Modify: `src/music_decoder/productionize/song.py` (`song_from_midi`, `song_from_composition`)
- Test: `tests/unit/test_productionize_song.py`

**Interfaces:**
- Produces, in `song.py`:

```python
@dataclass(frozen=True)
class ComposerParts:
    melody: tuple[NoteEvent, ...]          # sorted by onset, times within one pass
    accompaniment: tuple[NoteEvent, ...]
    beat_s: float
    n_bars: int                            # bars in one pass
    root_pcs: tuple[int, ...]              # per bar
    bar_roots: tuple[int, ...]             # per bar, folded into arrange.bass_low..bass_high
    harmonies: tuple[BarHarmony, ...]      # per bar (verse.bar_harmonies)
    voicings: tuple[tuple[int, ...], ...]  # per bar (verse.voice_led_voicings)
    home: BarHarmony                       # verse.tonic_harmony of the key
    home_voicing: tuple[int, ...]          # verse.resolve_voicing(home, voicings[-1], cfg)

    def bar_of(self, note: NoteEvent) -> int: ...  # the bar a note starts in (with the 1/64-bar snap)

def read_composer_parts(midi_path: Path, *, tempo_bpm: float | None = None,
                        progression: Sequence[ChordSymbol] | None = None,
                        scale: Scale | None = None, key: str = "",
                        arrange: ComposerArrangeConfig | None = None) -> ComposerParts: ...

def composition_kwargs(composition: Composition) -> dict[str, Any]: ...
# -> {"tempo_bpm", "progression", "scale", "title", "key"}, exactly as song_from_composition derives them today
```

- [ ] **Step 1: Write the failing tests.** Use the `composer_midi` fixture already in the file.
  - `test_composer_parts_match_the_song`: for the fixture,
    - `parts.bar_roots == song_from_midi(path).bar_roots`;
    - `len(parts.harmonies) == len(parts.voicings) == parts.n_bars`;
    - `parts.melody` is non-empty and sorted by `start_s`;
    - `parts.home.root` is the tonic pitch class of the fixture's scale.
  - `test_composition_kwargs_parse_the_metadata`:
    - a `Composition` with metadata `{"scale": "A:minor", "progression": ["Am","F","C","G"], "tempo_bpm": 92}`
    - gives `key == "Am"`, `title == "Composition in A minor"`, `tempo_bpm == 92.0`,
    - and `[c.to_label() for c in progression] == ["Am","F","C","G"]`.
- [ ] **Step 2: Run them to make sure they fail.** Run `pytest tests/unit/test_productionize_song.py -q`. Expected: ImportError on `read_composer_parts`.
- [ ] **Step 3: Implement `read_composer_parts` and `composition_kwargs`.**
  - Move the reading half of `song_from_midi` (from loading the file through `voicings`, plus the tonic and home voicing) into `read_composer_parts`.
  - Make `song_from_midi` call it.
  - Make `song_from_composition` call `composition_kwargs`.
  - No behaviour change; the existing song, api and e2e-gate tests are the proof.
- [ ] **Step 4: Run** `pytest -q tests/unit && mypy --strict src && ruff check src tests`. Expected: all pass, with no existing test edited.
- [ ] **Step 5: Commit.** Message: `refactor(productionize): share composer-file reading`.

### Task 2: Pop-rap config

**Files:**
- Modify: `src/music_decoder/productionize/config.py`
- Test: `tests/unit/test_productionize_config.py`

**Interfaces:**
- Produces, in `config.py`. Every default is the spike's value. Field names are exact; add others as needed, in the same style.

```python
ProductionStyle = Literal["boom-bap", "pop-rap"]
PopRapLead = Literal["guitar", "synth"]

@dataclass(frozen=True)
class PopRapFormConfig:
    intro_bars: int = 8; section_target_bars: int = 16; outro_bars: int = 4
    final_chord_bars: int = 2; intro_hats_from_bar: int = 4; riser_bars: int = 2
    tail_s: float = 3.0; fade_out_s: float = 3.0; fade_out_curve: float = 2.0

@dataclass(frozen=True)
class TrapDrumsConfig:
    kick_steps_even: tuple[int, ...] = (0, 6, 11); kick_steps_odd: tuple[int, ...] = (0, 3, 10, 14)
    clap_step: int = 8; clap_gain: float = 0.9
    ghost_clap_step: int = 15; ghost_clap_gain: float = 0.35          # bars with b % 4 == 3
    hat_odd_bar_16ths_from: int = 12; hat_accent: float = 1.0; hat_soft: float = 0.7; hat_pan: float = 0.25
    roll_hits: int = 6; roll_start_beat: int = 3; roll_velocity: float = 0.5; roll_velocity_step: float = 0.08
    open_hat_step: int = 14; open_hat_velocity: float = 0.8; open_hat_pan: float = -0.2  # bars with b % 4 == 2
    crash_pan: float = -0.3; riser_gain: float = 0.5
    sub_gain: float = 0.9; sub_punch_ratio: float = 4.5; sub_punch_s: float = 0.025; sub_decay_s: float = 1.4
    sub_choke_s: float = 0.06; sub_drive: float = 2.4; sub_tail_s: float = 0.25
    sidechain_depth: float = 0.55; sidechain_release_s: float = 0.18

@dataclass(frozen=True)
class PadConfig:
    detune_semitones: tuple[float, ...] = (-0.12, -0.05, 0.0, 0.06, 0.13)
    lowpass_hz: float = 1600.0; attack_s: float = 0.35; decay_s: float = 1.0
    sustain: float = 0.85; release_s: float = 0.6; release_tail_s: float = 0.8; final_chord_gain: float = 0.9

@dataclass(frozen=True)
class PluckConfig:
    decay_s: float = 0.35; max_length_s: float = 1.2; tail_s: float = 0.4
    square_level: float = 0.15; lowpass_hz: float = 2600.0; pan_spread: float = 0.35

@dataclass(frozen=True)
class SynthLeadConfig:
    octave: int = 12; detune_semitones: float = 0.09; second_saw: float = 0.8; sub_square: float = 0.25
    glide_s: float = 0.035; vibrato_semitones: float = 0.18; vibrato_hz: float = 5.2
    gate_s: float = 0.008; lowpass_hz: float = 3200.0

@dataclass(frozen=True)
class GuitarLeadConfig:
    octave: int = 0; program: int = 29; velocity: int = 100; soundfont: str = ""; fluidsynth_gain: float = 0.6
    mid_band_hz: tuple[float, float] = (500.0, 2500.0); mid_gain: float = 1.5; dry_gain: float = 0.6
    drive: float = 2.2; lowpass_hz: float = 4800.0; highpass_hz: float = 120.0

@dataclass(frozen=True)
class LeadBusConfig:
    delay_beats: float = 0.75; delay_first: float = 0.28; delay_second: float = 0.18
    plate_send: float = 0.22; double_gain: float = 0.35; double_pan: float = 0.3

@dataclass(frozen=True)
class PopRapMixConfig:
    drums_dbfs: float = -14.5; sub_dbfs: float = -20.0; pad_dbfs: float = -27.0; pluck_dbfs: float = -27.5
    fx_dbfs: float = -30.0; lead_guitar_dbfs: float = -18.0; lead_synth_dbfs: float = -16.5
    peak_ratio: float = 2.6                     # drums and 808 soft-limited at this multiple of their RMS
    drums_plate_send: float = 0.12; pad_hall_send: float = 0.35; pluck_hall_send: float = 0.30
    fx_hall_send: float = 0.40; sub_highpass_hz: float = 28.0; silence: float = 1e-5

@dataclass(frozen=True)
class PopRapConfig:
    lead: PopRapLead = "guitar"; seed: int = 1871
    form: PopRapFormConfig = PopRapFormConfig(); drums: TrapDrumsConfig = TrapDrumsConfig()
    pad: PadConfig = PadConfig(); pluck: PluckConfig = PluckConfig()
    synth_lead: SynthLeadConfig = SynthLeadConfig(); guitar_lead: GuitarLeadConfig = GuitarLeadConfig()
    lead_bus: LeadBusConfig = LeadBusConfig(); mix: PopRapMixConfig = PopRapMixConfig()

# ProductionConfig gains: style: ProductionStyle = "boom-bap"; poprap: PopRapConfig = PopRapConfig()
```

- [ ] **Step 1: Write the failing tests.**
  - `test_poprap_defaults_are_the_spike_take_a`:
    - `ProductionConfig().style == "boom-bap"`;
    - `ProductionConfig().poprap.lead == "guitar"`;
    - `cfg.poprap.mix.drums_dbfs == -14.5`;
    - `cfg.poprap.mix.lead_guitar_dbfs == -18.0`;
    - `cfg.poprap.drums.kick_steps_odd == (0, 3, 10, 14)`;
    - `cfg.poprap.form.section_target_bars == 16`.
  - Extend `test_config_is_frozen` with an assignment to `ProductionConfig().poprap.lead`, which must raise.
- [ ] **Step 2: Run** `pytest tests/unit/test_productionize_config.py -q`. Expected: FAIL (no attribute `poprap`).
- [ ] **Step 3: Implement the dataclasses above.**
- [ ] **Step 4: Run** the global check commands. Expected: PASS.
- [ ] **Step 5: Commit.** Message: `feat(productionize): pop-rap config`.

### Task 3: Trap drums, 808 and fx

**Files:**
- Create: `src/music_decoder/productionize/instruments/trap.py`
- Test: `tests/unit/test_productionize_trap.py`

**Interfaces:**
- Produces mono `npt.NDArray[np.float64]` clips, all with signature-first `sr: int` and `rng: np.random.Generator` where noise is used. Port each from the spike's function of the same name.

```python
def sub_808(pitch: int, length_s: float, cfg: TrapDrumsConfig, sr: int, rng: np.random.Generator) -> NDArray  # spike kick_808
def kick_punch(sr: int, rng: np.random.Generator) -> NDArray
def clap(sr: int, rng: np.random.Generator) -> NDArray
def hat(open_: bool, velocity: float, sr: int, rng: np.random.Generator) -> NDArray
def crash(sr: int, rng: np.random.Generator) -> NDArray
def riser(length_s: float, sr: int, rng: np.random.Generator) -> NDArray
```

- [ ] **Step 1: Write the failing tests.**
  - `test_sub_808_sounds_at_its_pitch`:
    - for `sub_808(37, 1.0, cfg, SR, rng)`, the strongest FFT bin between 20 and 200 Hz, taken over 0.1–0.9 s, is within 2 Hz of `midi_to_hz(37)` (34.6 Hz);
    - length is `int((1.0 + cfg.sub_tail_s) * SR)`.
  - `test_sub_808_is_choked_after_its_length`: RMS over the last 0.1 s is at least 30 dB below RMS over 0.1–0.5 s.
  - `test_hits_are_deterministic_per_generator`: each function called twice with `np.random.default_rng(5)` gives identical arrays.
  - `test_hit_lengths`:
    - clap 0.45 s, closed hat 0.045 s, open hat 0.32 s, crash 2.5 s, `riser(2.0, ...)` 2.0 s;
    - all finite, peak > 0.
  - `test_riser_rises`: RMS of its last quarter is greater than 4 × the RMS of its first quarter.
- [ ] **Step 2: Run** `pytest tests/unit/test_productionize_trap.py -q`. Expected: FAIL (module missing).
- [ ] **Step 3: Implement `trap.py`**, porting the spike's formulas. Read every number that `TrapDrumsConfig` names from it.
- [ ] **Step 4: Run** the global check commands. Expected: PASS.
- [ ] **Step 5: Commit.** Message: `feat(productionize): trap drums, 808 and riser`.

### Task 4: Pad, pluck and leads

**Files:**
- Create: `src/music_decoder/productionize/instruments/fluid.py`
- Create: `src/music_decoder/productionize/instruments/synths.py`
- Modify: `src/music_decoder/productionize/instruments/strings.py` (to use `fluid.py`; its public functions keep their signatures)
- Test: `tests/unit/test_productionize_synths.py`

**Interfaces:**
- Produces:

```python
# fluid.py: shared FluidSynth CLI rendering
def find_soundfont(preferred: str, part: str) -> Path   # preferred path if it exists, else _resolve_soundfont_path(), else ProductionError naming `part`
def fluidsynth_command(gain: float, soundfont: Path, midi: Path, wav: Path, sr: int) -> list[str]  # the strings' exact argv
def render_notes(notes: Sequence[NoteEvent], program: int, tempo_bpm: float, soundfont: Path, gain: float,
                 work_dir: Path, stem: str, sr: int, n_samples: int) -> Audio   # stereo, zero-padded/cut to n_samples
# strings.resolve_soundfont(cfg) and strings.fluidsynth_command(cfg, ...) stay as thin wrappers.
# Keep the strings' current preference: cfg.soundfont first, then the app soundfont, with the same warning log.

# synths.py
def saw(freq_hz: NDArray, sr: int, phase0: float = 0.0) -> NDArray   # PolyBLEP band-limited saw, freq may glide (spec ruling)
def pad_chord(pitches: Sequence[int], length_s: float, cfg: PadConfig, sr: int, rng: np.random.Generator) -> NDArray
def pluck(pitch: int, length_s: float, velocity: float, cfg: PluckConfig, sr: int) -> NDArray
def synth_lead(notes: Sequence[NoteEvent], total_s: float, cfg: SynthLeadConfig, sr: int) -> NDArray
def guitar_lead(notes: Sequence[NoteEvent], total_s: float, cfg: GuitarLeadConfig, tempo_bpm: float,
                work_dir: Path, sr: int) -> NDArray      # mono (mean of FluidSynth's L/R), normalized, then the amp chain
```

- **Notes:** lead `notes` are at the melody's pitch; each lead adds its own `cfg.octave`.
- **Pad clips:** `int((length_s + cfg.release_tail_s) * sr)` samples long, with the release starting at `length_s`.
- **`guitar_lead` errors:** a missing `fluidsynth` binary or no soundfont raises `ProductionError` before rendering.

- [ ] **Step 1: Write the failing tests.**
  - `test_saw_is_band_limited`: a 3 kHz `saw` at 44.1 kHz has less than −40 dB of energy above 20 kHz relative to the fundamental, and its fundamental is within 1 % of 3 kHz.
  - `test_pad_chord_holds_then_releases`:
    - `pad_chord((61, 64, 68, 49), 2.0, ...)` is `int(2.8 * SR)` long;
    - RMS over 1.0–1.9 s is greater than 5 × RMS over the last 0.1 s.
  - `test_pluck_decays`: RMS over the first 0.2 s is greater than 4 × RMS over 0.8–1.0 s.
  - `test_synth_lead_plays_an_octave_up_and_rests`:
    - one note of pitch 57 from 0.5 to 1.5 s, `total_s=2.5`;
    - the dominant pitch over 0.8–1.4 s is within 3 % of 440 Hz (57 + 12 = A4);
    - RMS over 2.0–2.5 s is below 1e-3 × the note's RMS.
  - `test_guitar_lead_needs_fluidsynth`: with `shutil.which` patched to return None, `guitar_lead(...)` raises `ProductionError` whose message names fluidsynth.
  - `test_guitar_lead_needs_a_soundfont`:
    - with `soundfont` set to a missing path and `music_decoder.synth._resolve_soundfont_path` patched to return None, it raises `ProductionError`;
    - skip when the fluidsynth binary is absent.
  - `test_guitar_lead_renders` (skip without the fluidsynth binary or a soundfont): two notes give a finite clip of `int(total_s * SR)` samples with peak > 0.
  - `tests/unit/test_productionize_strings.py` stays unedited and passes.
- [ ] **Step 2: Run** `pytest tests/unit/test_productionize_synths.py -q`. Expected: FAIL (module missing).
- [ ] **Step 3: Implement `fluid.py` and `synths.py`**, and rewire `strings.py` to use `fluid.py`. Port the spike's `pad_chord`, `pluck`, `synth_lead` and `guitar_lead`, replacing `bl_saw` with `saw` and `pretty_midi.fluidsynth` with `render_notes`.
- [ ] **Step 4: Run** the global check commands. Expected: PASS.
- [ ] **Step 5: Commit.** Message: `feat(productionize): pad, pluck, synth and guitar leads`.

### Task 5: Pop-rap layout

**Files:**
- Create: `src/music_decoder/productionize/poprap.py`
- Test: `tests/unit/test_productionize_poprap.py`

**Interfaces:**
- Consumes: `ComposerParts`, `read_composer_parts` and `composition_kwargs` (Task 1); `PopRapConfig` (Task 2).
- Produces, in `poprap.py`:

```python
PopRapDrums = Literal["none", "full", "drop"]          # "drop": full, but silent in the last bar (QC treats it as a break)
PopRapLeadPart = Literal["none", "melody", "melody_double", "tail"]

@dataclass(frozen=True)
class PopRapSection:
    name: str; role: SectionRole; start_bar: int; bars: int
    drums: PopRapDrums; lead: PopRapLeadPart
    hats_from_bar: int | None = None; riser: bool = False; crash: bool = False; rolls: bool = False

@dataclass(frozen=True)
class PopRapSong:
    title: str; key: str; grid: Grid      # Grid(beat_s, bars_per_section=parts.n_bars)
    sections: tuple[PopRapSection, ...]; parts: ComposerParts; source: str = ""
    reference: QcReference | None = None
    @property
    def body_s(self) -> float: ...        # total bars * bar_s (the final chord is in the tail)
    def section_span(self, index: int) -> tuple[int, int]: ...   # samples, from start_bar/bars
    @property
    def verse_sections(self) -> tuple[int, ...]: ...             # role == "verse"
    def pass_bar(self, bar_in_section: int) -> int: ...          # bar_in_section % parts.n_bars

@dataclass(frozen=True)
class DrumHit:
    time_s: float; kind: Literal["kick", "clap", "hat", "open_hat", "crash", "riser"]
    velocity: float = 1.0; pitch: int = 0; length_s: float = 0.0   # kick: 808 pitch and length to the next kick/bar end
    pan: float = 0.0

@dataclass(frozen=True)
class PadEvent:
    start_s: float; length_s: float; pitches: tuple[int, ...]; gain: float = 1.0

def hook_bars(pass_bars: int, target: int) -> int     # max(1, round(target / pass_bars)) * pass_bars
def poprap_form(pass_bars: int, cfg: PopRapFormConfig) -> tuple[PopRapSection, ...]
def poprap_song(midi_path: Path, *, cfg: PopRapConfig, tempo_bpm: float | None = None,
                progression: Sequence[ChordSymbol] | None = None, scale: Scale | None = None,
                title: str = "Composition (Instrumental)", key: str = "") -> PopRapSong
def drum_hits(song: PopRapSong, cfg: TrapDrumsConfig) -> list[DrumHit]
def pad_events(song: PopRapSong, cfg: PopRapConfig) -> list[PadEvent]
def pluck_notes(song: PopRapSong) -> list[NoteEvent]        # accompaniment once per pass, absolute times, cut at section end
def lead_notes(song: PopRapSong) -> tuple[list[NoteEvent], list[NoteEvent]]   # (main, hook-2 double), melody pitch
```

- **Form** (the spec's table):
  - Intro: `drums="none"`, `hats_from_bar=cfg.intro_hats_from_bar`, `riser=True`.
  - Hook 1: `"full"`, `lead="melody"`, `crash=True`.
  - Verse: `"drop"`, `lead="none"`, `rolls=True`, `crash=True`.
  - Hook 2: `"full"`, `lead="melody_double"`, `crash=True`.
  - Outro: `"none"`, `lead="tail"`.
- **Drum patterns:**
  - Bar parity, `b % 4` and steps follow the spike's 0-based counting within each section.
  - A kick's `length_s` runs to the next kick in the bar, else to the bar's end.
  - Kick and clap times are exactly on the 16th grid (no humanizing).
- **Pad:**
  - Each event is the bar's `parts.voicings[pass_bar]` plus the harmony root folded into `[arrange.rh_low - 12, arrange.rh_low - 1]`.
  - A new event starts at each section start and wherever the voicing changes, and lasts until the next event or the section end.
  - Final event: `parts.home_voicing` plus its root, at `body_s`, for `final_chord_bars`, with `gain=cfg.pad.final_chord_gain`.

- [ ] **Step 1: Write the failing tests.**
  - **Fixture:** build composer MIDI the way `test_productionize_api.composition_dir` does, with A minor, `Am F C G`, `bars_per_chord=2`, 120 bpm: an 8-bar pass.
  - `test_hook_bars_rule`: `hook_bars(8, 16) == 16`, `hook_bars(4, 16) == 16`, `hook_bars(12, 16) == 12`, `hook_bars(32, 16) == 32`, `hook_bars(6, 16) == 18`.
  - `test_form_of_an_8_bar_pass`:
    - `[(s.name, s.bars) for s in poprap_form(8, PopRapFormConfig())]` equals `[("Intro", 8), ("Hook 1", 16), ("Verse", 16), ("Hook 2", 16), ("Outro", 4)]`;
    - `start_bar`s are 0, 8, 24, 40, 56.
  - `test_form_of_a_32_bar_pass`: the bar counts are 8, 32, 32, 32, 4 (Review Focus 1).
  - `test_lead_is_the_melody_once_per_pass_and_silent_in_the_verse`:
    - `main` holds `len(parts.melody) * (16 // 8) * 2` hook notes plus the outro notes that start before bar 4 of the outro;
    - `main` has no note inside the verse span;
    - every `main` pitch is in `{n.pitch for n in parts.melody}`;
    - `double` notes all fall inside hook 2 and match hook 2's `main` notes one to one.
  - `test_notes_are_cut_at_the_section_end` (Review Focus 2): with a fixture melody whose last note ends past the pass (extend it by 2 bars before building the song), no `main` note's `end_s` exceeds its section's end.
  - `test_no_drums_in_the_verse_drop_bar_or_the_outro`: no `DrumHit` falls in the last bar of the verse, in the outro, or in intro bars 0–3. Intro hits from bar 4 on are only `hat` and `riser`.
  - `test_kicks_and_claps_are_on_the_16th_grid`: every kick and clap `time_s / grid.step_s` is within 1e-9 of an integer. Claps sit at step 8 of every drum bar.
  - `test_kick_carries_the_bar_root`: each kick's `pitch == parts.bar_roots[song.pass_bar(b)]`.
  - `test_pad_follows_the_harmony_and_ends_home`:
    - every pad event's pitch classes are a subset of that bar's `harmonies[...].tones`;
    - the last event starts at `body_s` with the pitch classes of `parts.home_voicing`.
- [ ] **Step 2: Run** `pytest tests/unit/test_productionize_poprap.py -q`. Expected: FAIL (module missing).
- [ ] **Step 3: Implement `poprap.py`** as described above.
- [ ] **Step 4: Run** the global check commands. Expected: PASS.
- [ ] **Step 5: Commit.** Message: `feat(productionize): pop-rap song layout`.

### Task 6: Pop-rap mix

**Files:**
- Create: `src/music_decoder/productionize/poprap_mix.py`
- Test: `tests/unit/test_productionize_poprap_mix.py`

**Interfaces:**
- Consumes: Tasks 3–5; `mix.MixResult` and `mix.mono_low_end` (existing).
- Produces:

```python
POPRAP_STEMS = ("drums", "808", "pad", "pluck", "fx", "lead")
def poprap_length(song: PopRapSong, cfg: PopRapFormConfig) -> int     # samples: body_s + tail_s
def render_poprap_mix(song: PopRapSong, cfg: PopRapConfig, work_dir: Path,
                      progress: ProgressCallback | None = None) -> MixResult
# MixResult.stems has exactly POPRAP_STEMS (levelled, post-send, pre-master);
# MixResult.drum_onsets_s holds the kick and clap times only (hats excluded: triplet rolls).
```

- **Port the spike's `main()` bus stage:**
  - The sidechain envelope from kick times ducks the pad and pluck.
  - The lead bus gets the delay and the plate.
  - The hook-2 double is the other lead, at its own octave, at `double_gain` and `double_pan`.
- **Sends:**
  - drums get the plate send;
  - the 808 is high-passed;
  - pad, pluck and fx get the hall send.
- **Levels:** each bus is set to its `*_dbfs` by RMS over samples where `|L|+|R| > silence`. Drums and 808 are then `dsp.soft_limit` at `peak_ratio × target gain`.
- **Sum:** `mono_low_end(sum, MixConfig(), sr)`, then a fade over `fade_out_s` with `(1 - t) ** fade_out_curve`.
- **Lead level:** `lead_guitar_dbfs` or `lead_synth_dbfs`, by `cfg.lead`.
- **Progress:** call it with `"drums"`, `"808"`, `"pad"`, `"pluck"`, `"lead"`, `"mix"`.

- [ ] **Step 1: Write the failing tests.**
  - **Setup:** use the Task 5 fixture with `cfg = replace(PopRapConfig(lead="synth"), form=replace(PopRapFormConfig(), intro_bars=2, section_target_bars=8, outro_bars=1))` so the test runs fast.
  - `test_stems_and_length`: the stem keys equal `POPRAP_STEMS`; every stem and the mix are `(poprap_length, 2)` and finite.
  - `test_drums_lead_the_mix_and_the_808_sits_under_them`: over the two hook spans, the drums stem's energy is greater than every other stem's, and the 808's is less than the drums'.
  - `test_drum_onsets_are_kicks_and_claps`: their count equals the kick and clap `DrumHit`s from `drum_hits`.
  - `test_render_is_deterministic`: two renders give identical mixes.
  - `test_the_mix_fades_to_silence`: the peak of the last 50 ms is below −80 dBFS.
  - `test_slow_piece_renders` (Review Focus 3): the same fixture at 66 bpm (synth lead, short form) renders finite audio with hook 1 holding 8 bars.
  - `test_guitar_mix_renders` (skip without fluidsynth or a soundfont): `lead="guitar"` renders with a non-silent lead stem.
- [ ] **Step 2: Run** `pytest tests/unit/test_productionize_poprap_mix.py -q`. Expected: FAIL (module missing).
- [ ] **Step 3: Implement `poprap_mix.py`** as described above.
- [ ] **Step 4: Run** the global check commands. Expected: PASS, with the new file's tests under 60 s in total.
- [ ] **Step 5: Commit.** Message: `feat(productionize): pop-rap mix`.

### Task 7: Checks for both song types

**Files:**
- Modify: `src/music_decoder/productionize/qc.py` (`run_qc`)
- Test: `tests/unit/test_productionize_qc.py`

**Interfaces:**
- Produces, in `qc.py`:

```python
class SectionLike(Protocol):
    @property
    def drums(self) -> str: ...
class Timeline(Protocol):           # read-only properties so frozen dataclasses (Song, PopRapSong) satisfy it
    @property
    def grid(self) -> Grid: ...
    @property
    def sections(self) -> Sequence[SectionLike]: ...
    @property
    def reference(self) -> QcReference | None: ...
    @property
    def body_s(self) -> float: ...
    @property
    def verse_sections(self) -> tuple[int, ...]: ...
    def section_span(self, index: int) -> tuple[int, int]: ...

def run_qc(song: Timeline, wav_path, mp3_path, drum_onsets_s, cfg, master_cfg, *, tail_s: float,
           ffmpeg: Path | None = None, drum_grid: Literal["8th", "16th"] = "8th") -> QcReport
```

- **Drum grid check:** `drum_grid="16th"` checks against `grid.step_s`, and its value text reads `"N of M hits on a 16th note"`. The `"8th"` text is unchanged.
- **Groove continuation:** the rule (`drums in ("light", "full")` on both sides) is unchanged. A `"drop"` or `"none"` section is a break for free.
- **Section spans:** run_qc already reads them from `song.section_span`. Verify that nothing else assumes equal section lengths.

- [ ] **Step 1: Write the failing test.**
  - `test_run_qc_accepts_a_poprap_song_and_a_16th_grid`:
    - render the Task 6 short-form synth mix;
    - master it with `master()` and write it with `with_lead_in` and `write_wav(..., "PCM_24")`;
    - run `run_qc(song, wav, None, mix.drum_onsets_s, QcConfig(), MasterConfig(), tail_s=cfg.form.tail_s, drum_grid="16th")`.
  - Then assert:
    - the `"format"`, `"drum grid"`, `"DC offset"` and `"loudness (WAV)"` checks pass;
    - the drum-grid value ends with `"on a 16th note"`;
    - the `"section RMS (dBFS)"` check's `passed is None`.
- [ ] **Step 2: Run** `pytest tests/unit/test_productionize_qc.py -q`. Expected: FAIL (unexpected keyword `drum_grid`).
- [ ] **Step 3: Implement** the Protocol and the `drum_grid` parameter.
- [ ] **Step 4: Run** the global check commands. Expected: PASS. All the boom-bap qc and api tests are unchanged.
- [ ] **Step 5: Commit.** Message: `feat(productionize): checks accept pop-rap songs`.

### Task 8: API and CLI

**Files:**
- Modify: `src/music_decoder/productionize/api.py`
- Modify: `src/music_decoder/productionize/__init__.py` (docstring mentions both styles; export `PopRapConfig`)
- Modify: `src/music_decoder/cli/main.py` (`cli_productionize`)
- Test: `tests/unit/test_productionize_api.py`, `tests/unit/test_cli_productionize.py`

**Interfaces:**
- Consumes: `poprap_song`, `composition_kwargs`, `render_poprap_mix` and `run_qc(..., drum_grid=)`.
- Produces:
  - `resolve_poprap_song(source: Song | Composition | Path | str, *, cfg: PopRapConfig, progression=None, title=None, key=None) -> PopRapSong`.
    - It raises `ProductionError("pop-rap needs a composer file or composition, not a ready Song")` for a `Song`.
    - Folder and file handling match `resolve_song`, as do the title and key overrides.
  - `productionize(...)` keeps its signature and branches on `cfg.style`. Pop-rap:
    - uses no `SampleStore`;
    - mixes with `render_poprap_mix`;
    - uses `tail_s = cfg.poprap.form.tail_s`;
    - runs QC with `drum_grid="16th"`;
    - writes stems `stem_<name>.wav` for `POPRAP_STEMS`.
  - JSON `metadata` gains `"style"` and `"lead"`.
    - For boom-bap, `lead` is `None`.
    - For pop-rap, `verse_style` is `None`, `percussion` is `[]` and `credits` is `[]`.
    - `duration_s` uses the style's tail.
  - Factor the shared master, export and report code into one private helper. Do not duplicate it.
- **CLI options:**
  - `--style` takes `boom-bap` or `pop-rap` (default boom-bap, shown).
  - `--lead` takes `guitar` or `synth` (default guitar, shown, help "pop-rap lead sound").
  - They set `replace(config, style=..., poprap=replace(config.poprap, lead=...))`.
- **CLI usage errors** (`click.UsageError`), with pop-rap:
  - `--hisaab`;
  - any `--percussion`;
  - `--verse` given on the command line (`ctx.get_parameter_source("verse") is ParameterSource.COMMANDLINE`).

- [ ] **Step 1: Write the failing tests.**
  - `test_poprap_production_writes_master_mp3_stems_and_report` (`needs_ffmpeg`): `productionize(composition_dir, out_dir=..., config=` short-form synth `ProductionConfig(style="pop-rap", poprap=...))`. Assert:
    - the WAV and MP3 exist;
    - `set(stem_paths) == set(POPRAP_STEMS)`;
    - `metadata["style"] == "pop-rap"` and `metadata["lead"] == "synth"`;
    - the JSON file's `metadata.style` is `"pop-rap"`;
    - `qc` is not None;
    - the `"drum grid"` check passed.
  - `test_poprap_rejects_a_ready_song` (Review Focus 5): `productionize(tiny_song(), ..., config=ProductionConfig(style="pop-rap"))` raises `ProductionError` and leaves `out_dir` empty or absent.
  - `test_poprap_progression_must_divide_the_bars`: a 3-chord `progression` on the 4-bar fixture raises `ProductionError`.
  - `test_boom_bap_metadata_has_style_and_no_lead`: the existing fake-store render path reports `style == "boom-bap"` and `lead is None`.
  - CLI, mocking `productionize` as the existing tests do:
    - `test_productionize_style_and_lead_reach_the_config`: `--style pop-rap --lead synth` gives `config.style == "pop-rap"` and `config.poprap.lead == "synth"`. Without them the config is `"boom-bap"` with lead `"guitar"`.
    - `test_poprap_rejects_boom_bap_only_options`: `--style pop-rap` with `--percussion shaker`, with `--hisaab`, and with `--verse held` each exit non-zero with a usage error. With no `--verse`, it runs.
- [ ] **Step 2: Run** `pytest tests/unit/test_productionize_api.py tests/unit/test_cli_productionize.py -q`. Expected: the new tests fail.
- [ ] **Step 3: Implement** the API and CLI changes above.
- [ ] **Step 4: Run** the global check commands. Expected: PASS.
- [ ] **Step 5: Commit.** Message: `feat(productionize): --style pop-rap and --lead`.

### Task 9: Full render test and docs

**Files:**
- Modify: `tests/integration/test_productionize_e2e.py` (add pop-rap cases; keep its opt-in markers)
- Modify: `README.md` (Usage and Library)
- Modify: `docs/usage.md` (`productionize` section)
- Modify: `docs/architecture-overview.md` and `docs/architecture-technical.md` (one paragraph each where productionize is described)

- [ ] **Step 1: Write the opt-in tests.**
  - `test_poprap_render_passes_every_check[guitar|synth]`:
    - compose with `arrange(Scale("A","minor"), Am F C G, tempo_bpm=120.0, seed=7)` and `build_midi(..., bars_per_chord=2, ...)`, as the file's existing composer case does;
    - `productionize` with `ProductionConfig(style="pop-rap", poprap=PopRapConfig(lead=lead))` and the default form.
  - Assert:
    - `qc.passed`;
    - `master.limiter_max_reduction_db <= 3.0`;
    - over the two hook spans of the stems (shifted by nothing: the stems have no lead-in), the drums' energy share is in [0.40, 0.60] and the 808's in [0.10, 0.25], where share = stem energy / sum of the six stems' energy.
- [ ] **Step 2: Run** `MUSIC_DECODER_PRODUCTION_E2E=1 pytest -q tests/integration/test_productionize_e2e.py -k poprap`. Expected: PASS. If a share falls outside its range, report it and stop; do not retune the spike's levels.
- [ ] **Step 3: Update the README.**
  - In Usage, under the productionize line, add:

    ```bash
    # Or as an upbeat pop-rap instrumental: trap drums, 808, and your melody on electric guitar
    music-decoder productionize ./out/<take-folder> --style pop-rap --out ./release --name my_track
    ```

  - In Library, add:

    ```python
    from music_decoder.productionize import ProductionConfig, PopRapConfig
    poprap = ProductionConfig(style="pop-rap", poprap=PopRapConfig(lead="guitar"))
    release = productionize(comp, out_dir=Path("./release"), name="my_track", config=poprap)
    ```

- [ ] **Step 4: Update `docs/usage.md`.**
  - Add `[--style boom-bap | pop-rap] [--lead guitar | synth]` to the synopsis, with option bullets for each.
  - Add a "Pop-rap style" subsection with:
    - the spec's layout table;
    - the stems list;
    - the note that it suits pieces around 100–130 bpm (the drums are half-time trap at the piece's own tempo);
    - the note that `--percussion`, `--verse` and `--hisaab` are boom-bap only;
    - the note that the guitar uses the installed soundfont through FluidSynth, while `--lead synth` needs neither.
  - Add one paragraph each to the two architecture docs.
- [ ] **Step 5: Run** `pytest -q tests/unit && ruff check src tests && mypy --strict src`. Expected: PASS.
- [ ] **Step 6: Commit.** Message: `docs(productionize): pop-rap style; opt-in full render test`.
