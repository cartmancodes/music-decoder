# Pop-rap production style — design

Status: approved (21 Nov 2026). Next step: implementation plan.

## Intent

The user asked for an upbeat pop and rap mix of composed piece 1871. The
reference was a half-time trap / pop-rap track. The melody, chords and key
had to stay ours. A throwaway spike produced two takes. The user liked
take A best: the melody on overdriven electric guitar over trap drums and
an 808. They want this as a feature anyone can run on any composed piece,
and documented in the README.

Success: running `music-decoder productionize <take> --style pop-rap` on a
composed piece gives a mastered track that sounds like take A, and the
README says how.

The spike: `compose-samples/2026-11-21-poprap-1871/poprap_spike.py` in the
main checkout. It is untracked and never committed. Its README holds the
measurements. Take A is the reference sound.

## Goals

- A second production style, `pop-rap`, next to today's boom-bap sound,
  which stays the default and is unchanged.
- Works on any composer output `productionize` accepts: a folder, a
  `composition.mid`, or what `compose()` returned. Any key, progression,
  bars per chord and tempo.
- Our melody, chords, key and tempo, unchanged. Only the instruments and
  the song layout are new.
- Two lead sounds: `guitar` (take A, the default) and `synth` (take B).
- The same outputs as today: 24-bit WAV, 320 kbps MP3, float raw mix, one
  float stem per element, a JSON report and the acceptance checks.
- README and usage guide document it.

## Non-goals

- Vocals or a rap. The verse leaves room for one; nobody records it here.
- Changing the composer, or picking the style automatically.
- Other new styles (rock, EDM, ...). The code should make a third style
  easy, but none is built.
- Bit-identical audio to the spike's take A. The sound and the measured
  balance must match; the random noise in each hit may differ.

## Anti-goals (a failure even if it runs)

- Changing a single sample of the boom-bap output. Its existing tests,
  including the opt-in full render, must still pass.
- Changing our melody, chords or key. The lead plays the composer's melody
  notes, and the pad, pluck and 808 follow its chords.
- A mix where the lead is buried or the 808 swamps the drums. Take A
  measured, from Demucs, about: drums 48 %, bass 18 % of the energy.
- Downloading anything new. Pop-rap is synthesized, apart from the guitar,
  which uses the soundfont already installed for the strings.

## Constraints

- Python, inside `music_decoder.productionize`. The DSP helpers live in
  `productionize.dsp`, and `master()`, `with_lead_in`, `write_wav` and
  `encode_mp3` in `productionize.master`. All are reused as they are.
- Every number lives in a frozen config dataclass, as `config.py` already
  says ("a different sound is a different `ProductionConfig`").
- Deterministic: the same input and config give the same audio.
- The guitar renders through the FluidSynth CLI, the same way the strings
  do (`instruments/strings.py`: `resolve_soundfont`, `fluidsynth_command`).
  It needs no pyfluidsynth.
- Master targets are unchanged: −10.5 LUFS, −1.8 dBTP ceiling.
- MIT licence. No borrowed code.

## Approaches

1. **A style option inside `productionize`. Chosen.** It reuses the
   reading of composer files, mastering, export, the JSON report and the
   checks. Only the instruments, the layout and the mix are new. Cost:
   `run_qc` and the composer-file reading must accept a second kind of
   song.
2. **A separate `music-decoder poprap` command.** Rejected because it
   duplicates the export, master and check code, and the README would
   have two production commands.
3. **Commit the spike under `spikes/`.** Rejected because it is
   hard-coded to piece 1871 (its chords, 120 bpm, 2 bars a chord) and is
   not a feature.

## Design

### Interface

- **CLI:** `music-decoder productionize SOURCE --style pop-rap [--lead guitar|synth] --out DIR ...`
  - `--style` takes `boom-bap` (default) or `pop-rap`.
  - `--lead` takes `guitar` (default) or `synth`, and only matters for
    pop-rap.
  - These options combine with pop-rap: `--out`, `--name`, `--title`,
    `--key`, `--progression`, `--no-qc`, `--format`.
  - `--hisaab` or `--percussion` with pop-rap is a usage error.
  - `--verse` set explicitly with pop-rap is a usage error. Its default
    is ignored.
  - `--samples-dir` is accepted and unused.
- **Library:** `ProductionConfig` gains `style: Literal["boom-bap", "pop-rap"] = "boom-bap"`
  and `poprap: PopRapConfig`, where `PopRapConfig.lead` is
  `Literal["guitar", "synth"] = "guitar"`. The signature of
  `productionize(...)` does not change.
- **Source:** pop-rap needs composer notes (a melody and an
  accompaniment). A ready `Song`, such as `hisaab_song()`, raises
  `ProductionError`.

### Reading the composition

The two styles read a composer file the same way. Factor that reading out
of `song_from_midi` into one helper that both use. It returns:

- the melody and accompaniment notes;
- the beat length and bars per pass;
- each bar's chord root pitch class (from `--progression` when given,
  else from the accompaniment);
- each bar's harmony (`bar_harmonies`);
- each bar's voice-led right-hand voicing (`voice_led_voicings`);
- the home chord (`_tonic`, `tonic_harmony`).

`song_from_midi` keeps its behaviour exactly; the existing song tests pin
it.

### Layout (in bars)

| Section | Bars | Lead | Drums and 808 | Pad and pluck | Other |
|---|---|---|---|---|---|
| Intro | 8 | none | hats from bar 5 | yes | riser in the last 2 bars |
| Hook 1 | P | melody | yes | yes | crash on bar 1 |
| Verse | P | **none** (room for a rap) | yes, with hat rolls; none in its last bar | yes | crash on bar 1 |
| Hook 2 | P | melody plus a quiet double | yes | yes | crash on bar 1 |
| Outro | 4 | melody's first 4 bars | none | yes | — |
| Final chord | 2 bars held | — | — | home chord on the pad | — |

- **Hook and verse length.** P is the whole number of passes closest to
  16 bars, at least 1. For 1871 (8-bar pass) that is 16 bars. For a
  4-bar pass it is 16, and for a 12-bar pass it is 12.
- **Intro and outro.** They are fixed in bars, and their chords follow the
  pass from bar 0. The intro has no melody, so it never cuts a phrase.
- **Melody.** It repeats once per pass, as in the composition; a note
  that crosses the end of a section is cut there.
- **The double in hook 2.** The other lead sound at its own usual
  octave, at 0.35 of the main lead's level, panned 0.3. Each lead always
  plays at its own octave: the guitar at the melody's pitch, the synth an
  octave up. So take A gets the synth an octave above the guitar, and
  take B gets the guitar an octave below the synth.
- **Bar counting.** Bars are counted from 0 within each section. "Even
  bar", "every 4th bar" (`b % 4 == 3`) and "bars 3, 7, ..." (`b % 4 == 2`)
  all follow the spike's 0-based rule.

### Instruments

All values are the spike's.

- **808** (`kick_808`): a sine that drops in pitch onto the bar's root.
  The root is folded into MIDI 28–39, as the boom-bap bass does, which
  gives C#2, A1, F#1 and G#1 for 1871. It is saturated and choked at the
  next kick.
- **Kick layer** (`kick_punch`): a short kick on the drum bus at each 808
  hit, so the hit reads apart from the note.
- **Clap** on step 8 (beat 3) of every bar; a ghost clap at step 15 every
  4th bar.
- **Kick steps:** 0, 6, 11 on even bars and 0, 3, 10, 14 on odd bars,
  counted within the section.
- **Hats:**
  - 8ths;
  - 16ths on steps 12–15 of odd bars;
  - a triplet roll on beat 4 every 4th bar, and every bar in the verse;
  - an open hat on step 14 of bars 3, 7, ...
  - The crash marks section starts, and a band-pass noise riser leads
    into hook 1.
- **Pad:** five detuned band-limited saws per note, a 1600 Hz low-pass
  and an ADSR. It plays the bar's voice-led voicing plus the root an
  octave down. A new chord starts only when the harmony changes. The
  final chord is the home chord.
- **Pluck:** a soft triangle and square tone playing the composer's
  accompaniment notes, panned at random within ±0.35.
- **Leads:**
  - `guitar`: GM program 29 (Overdriven Guitar) through FluidSynth, then
    a 500–2500 Hz mid push, drive 2.2, a 4800 Hz low-pass and a 120 Hz
    high-pass, at the melody's pitch.
  - `synth`: two detuned saws and a sub square, a 35 ms glide and light
    vibrato, an octave above the melody.
  - Both go through a dotted-8th ping-pong delay and 0.22 of the plate
    reverb.
- **Sidechain:** the pad and pluck duck by 55 % at each kick, recovering
  over 180 ms.

### Mix and master

- **Buses:** drums (plus 0.12 plate), 808 (high-passed at 28 Hz), pad
  (plus 0.35 hall), pluck (plus 0.30 hall), fx (plus 0.40 hall) and lead.
- **Levels.** Each bus is set to a fixed RMS over its non-silent samples:
  - drums −14.5 dBFS;
  - 808 −20.0;
  - pad −27.0;
  - pluck −27.5;
  - fx −30.0;
  - lead −18.0 for the guitar, −16.5 for the synth.
- **Peaks.** Drums and 808 are soft-limited at 2.6 × their RMS, so the
  master limiter barely works. Take A's limiter peaked at 1.5–1.7 dB.
- **Sum.** The buses are summed, the low end is made mono, and the mix
  fades out. It then goes through today's `master()`, lead-in and export.
- **MP3 genre tag:** "Hip-Hop" (unchanged).
- **Stems:** `drums`, `808`, `pad`, `pluck`, `fx`, `lead`.
- **JSON metadata** gains `style` and `lead`.

### Checks

`run_qc` runs on pop-rap with the checks that make sense for it:

- **Unchanged checks:** format and length; loudness and true peak of the
  WAV and MP3; full-scale samples; first sound; tail; DC; stereo and mono
  fold-down; low-band correlation; tempo (octave errors allowed, since
  there is no reference master); beat jitter.
- **Section RMS and octave bands** are reported without a pass/fail, as
  for any song without a reference master.
- **Section-boundary level checks** apply only where the groove carries
  on. The verse's drop bar and the outro count as breaks.
- **Drum grid:** kick and clap onsets must sit on the 16th grid. The hats
  are left out because the rolls are triplets by design.

Sections now differ in length, so `run_qc` reads section spans from the
song instead of assuming equal sections. The boom-bap results are
unchanged.

### Tests

- **Unit tests:**
  - Each instrument is deterministic and has the right length and
    channels; the 808 sounds at the bar root's pitch.
  - Layout: the lead is silent in the verse; there are no drums in the
    verse's last bar or the outro; the lead's notes are the composer's
    melody, repeated per pass; the pad's chord tones are the bar's
    harmony; the final chord is the home chord; P follows the rule above.
  - Mix: in the pre-master stems the drums carry the most energy, and the
    808 carries less than the drums.
  - The config and CLI options flow through; the usage errors above
    trigger.
  - The boom-bap tests pass untouched.
- **Opt-in full render** (`MUSIC_DECODER_PRODUCTION_E2E=1`): a composed
  piece rendered as pop-rap with both leads.
  - Every pass/fail check passes.
  - The limiter peaks at 3 dB or less.
  - The drums carry 40–60 % of the energy and the 808 10–25 %, measured
    from the pre-master stems.

### Docs

- **README Usage:** the pop-rap command line. **README Library:** the
  config example.
- **`docs/usage.md`:**
  - `--style` and `--lead` in the `productionize` synopsis and options;
  - a pop-rap section with the layout table;
  - the note that it suits pieces around 100–130 bpm, since the drums are
    half-time trap at the piece's own tempo.
- **Architecture docs:** one paragraph each.

## Calls made after the design review

- **No soundfont for the guitar lead:** raise `ProductionError`, worded
  like the strings error. The synth lead needs no soundfont.
- **Tempo outside 100–130 bpm:** only a note in the docs; no check.
- **Energy shares in the full-render test:** measured over the whole
  pre-master stems.

## Left to the builder

- Module layout:
  - suggested `productionize/instruments/trap.py` for the drums, 808 and
    fx;
  - `productionize/instruments/synths.py` for the pad, pluck and leads;
  - `productionize/poprap.py` for the layout and mix.
- The pop-rap song type, and how `run_qc` accepts both kinds of song (a
  shared protocol is suggested).
- Seeding: one config seed, with a separate generator per hit or per
  instrument, so adding a hit doesn't change the others.
- How the tests build small composer files (the existing helpers write
  them).
