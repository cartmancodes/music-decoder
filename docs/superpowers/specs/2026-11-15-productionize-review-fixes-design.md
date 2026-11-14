# Productionize review fixes — short design (PR #3)

**Date:** 2026-11-15 · **Status:** approved in chat (all four fixes)

## Intent

Review PR #3 (`music_decoder.productionize`) as an audio engineer, producer and
senior developer, and fix what should be done better. The review found four
changes worth making before merge; taste-level notes are reported, not changed.

## Goals

1. **Limiter without gain jumps.** `master.limiter_gain` takes
   `min(held, smooth(held))`; at the edge of each hold region the gain jumps
   (measured: up to 4.8 % / 0.42 dB in one sample, 37 such jumps in 2 s of a
   bursty test signal) — clicks/distortion exactly where limiting is heaviest.
   The smoothed hold alone already stays under the ceiling (every sample of the
   centred average of width `ahead` sees a hold window of ±`ahead`), so use it
   alone. Verified: max step 0.29 %, same true-peak ceiling and gain reduction.
2. **Composer songs end on the home chord.** The final 8-beat chord was the
   loop's last chord in the composer's guitar shape (a `Cmaj7 Am7 Dm7 G7` take
   ended on G7 as G2 B2 D3 …; `Am F C G` ends on G). It becomes the tonic,
   voiced like the verses (left-hand root + fifth, right-hand chord tones moving
   least from the last voice-led chord); the final strings chord takes its top
   voices. Tonic: the scale when known, else the key tag (`--key`, e.g. `Am`),
   else the loop's first bar. The chord is the progression's chord on the tonic
   when it has one (so `Cmaj7` stays a maj7), else the scale's tonic triad.
   The Hisaab reference song is unchanged.
3. **Render tests on request only.** `tests/integration/test_productionize_e2e.py`
   downloads ~180 MB and renders two songs (~2.7 GB RAM each) on every
   `make test`, and fails offline. Skip unless `MUSIC_DECODER_PRODUCTION_E2E=1`;
   new `make test-production` runs them.
4. **Samples in setup.** `setup.sh` already pre-downloads models so nothing
   downloads on first use; it now also runs `music-decoder fetch-samples`
   (drum + percussion samples; piano still on demand). `--no-models` skips it.

## Non-goals

Strings double reverb, piano tanh saturation, the Linux default strings
soundfont path (falls back correctly), beat-jitter tracker resolution, stems
being pre-fade: reported in the review, not changed.

## Approaches

- Limiter: (a) smoothed hold only — chosen, keeps the look-ahead design and
  provably holds the ceiling; (b) add a separate attack ramp — more code, same
  result.
- Ending: (a) tonic chord voice-led from the last verse voicing — chosen;
  (b) repeat the first bar's composer shape — still the muddy guitar voicing;
  (c) leave the loop's last chord — the current unresolved ending.
- Tests: (a) env-var opt-in + make target — chosen; (b) `-m "not slow"` in
  `make test` — changes the meaning of an existing target for every suite.

## Left to the builder

Exact test signals, function names, docs wording.
