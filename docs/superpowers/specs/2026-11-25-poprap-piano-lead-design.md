# Pop-rap piano lead — design

Approved by the user on 2026-11-25 (short design; small change).

## Intent

The user is making a pop-rap version of the reference song "Hisaab", keeping
its melody. Its original melody is played on piano, and they asked that "our
script should also generate with piano like original melody in lead". So
`productionize --style pop-rap` gains a third lead sound: the sampled grand
piano the boom-bap / Hisaab production already plays.

## Goals

- `music-decoder productionize SOURCE --style pop-rap --lead piano`, and
  `PopRapConfig(lead="piano")` in the library.
- The piano plays the composer's melody note for note, in the same sections
  and at the same pitch as the guitar lead: Hook 1, Hook 2 and the outro.
  The verse keeps no lead (room for a rap).
- Same piano as boom-bap: `PianoConfig`, the Salamander samples, and the
  pedal and re-strike rules of `plan_voices`.

## Non-goals and anti-goals

- No change to the guitar or synth leads, the layout, the drums or the mix of
  the other buses. Their renders stay bit-identical.
- Never alter the melody (pitches, onsets) to suit the piano.
- No new samples or downloads beyond the piano samples boom-bap already uses.

## Approaches

1. **Reuse the boom-bap piano renderer (picked).** Split `render_piano` so its
   core takes notes, bar length, tail start and the spans to level against,
   instead of a boom-bap `Song`. It sounds exactly like the original Hisaab
   piano, and the tested pedal rules come with it.
2. FluidSynth piano (GM program 0) through the guitar-lead path. Cheapest
   code, but a different, thinner piano than the original; rejected because
   the user asked for the original's piano.
3. A new pop-rap-only piano sampler. Duplicates `piano.py`; rejected.

## Decisions (the user approved these as proposed)

- **Sound:** the boom-bap piano chain (samples, pedal to the bar end, re-strike
  damping, EQ, compression, soft limit), then the existing pop-rap lead bus
  (ping-pong delay and plate) like the other leads.
- **Level:** `lead_piano_dbfs = -17.0` (between guitar -18.0 and synth -16.5).
- **Hook-2 double:** a second piano. The proposal was the guitar. After hearing
  it the user said "the electric guitar part should have been also in piano",
  then "keep the second part also in piano" (2026-11-25, chose version A). So the
  double is a second piano at the melody's pitch, mono, at the double's gain and
  pan, each note held for its written length (no pedal), as the guitar played
  it. The piano lead needs no FluidSynth. The guitar lead's double stays the
  synth, the synth's the guitar.
- **Peak limit (builder's call, found by a real render):** the piano lead is
  soft-limited at 4x its RMS (`lead_piano_peak_ratio`), like the drums and 808;
  its 18 dB crest otherwise made the master miss its loudness target.
- **Layout:** unchanged.

## Calls made by the builder (Claude)

- **Compression reference:** boom-bap compresses against the verses' RMS, but
  the pop-rap verse has no lead. The piano lead levels against the sections
  that have a lead (`lead != "none"`).
- **Tail:** notes from the final chord on (`song.body_s`) get the piano's tail
  release, as boom-bap's tail notes do. The lead has no notes there today, so
  this only fixes the rule.
- **Stereo:** first kept the piano's own stereo image; a real render then fell
  under the stereo-correlation floor (0.74 < 0.75) once the double was gone, so
  the piano lead is summed to mono and centred like the guitar and synth.
- **Samples:** `--samples-dir` (and `productionize(samples_dir=...)`) now also
  reach pop-rap; the piano lead fetches any missing samples on first use, as
  boom-bap does, before anything is written. The help text changes from
  "pop-rap needs no samples" to "pop-rap needs them only for the piano lead".
- **Credits:** with the piano lead, the report's credits list the Salamander
  piano (`PIANO_CREDIT`), as boom-bap does.

## Left to the builder

Function names, how the piano core is factored, and test shapes, within the
plan.
