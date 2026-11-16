# Music Decoder — Plain-English Overview

Other docs: [Technical](architecture-technical.md) · **Overview (this file)** · [Usage](usage.md)

This document explains what Music Decoder is, what it does, and what
happens behind the scenes — without any code or jargon. If you're a
musician, a product manager, or just curious, this is the right doc.

If you want to see the source code or the algorithms in detail, read the
[Technical](architecture-technical.md) doc. If you want to install and
run it, read the [Usage](usage.md) doc.

---

## What is Music Decoder?

Music Decoder is a small, local tool that listens to a song and tells you
what it's playing. You can also flip it around and have it write a new
piece of music for you. Everything runs on your own laptop — no servers,
no accounts, no upload of your audio anywhere.

There are four things it does. Each one is a self-contained feature.

---

## Feature 1: What chords is this song using?

You give it any song you have on disk (an MP3, a WAV, or a FLAC), or
paste a YouTube link. It listens to the recording and prints out a
**chord progression** — the sequence of chords the song moves through,
like "C, A minor, F, G", along with the time stamps where each chord
begins and ends.

Why this matters: if you've ever wanted to play along with a song but
can't find a tab, or you want to figure out the harmonic skeleton of a
piece, this is the answer in about a minute.

---

## Feature 2: Where do I put my fingers on the guitar?

Same input — a song or a YouTube link — but now the answer is
**guitar tablature** ("tab"). Tab is the guitar-specific shorthand
that says, for each note: which string to play, and which fret to press.
The output is the standard six-line ASCII tab guitarists already know
how to read, plus a graphical fretboard view in the desktop app.

You can pick the tuning you actually have your guitar in (standard,
Drop D, half-step down, Drop C, DADGAD, etc.) and the tool will pick
fingerings that work for that tuning.

Why this matters: instead of staring at a chord chart and guessing
which voicing the player used, you get one specific fingering for the
entire song.

---

## Feature 3: Write me a piece of music

This is the flip side. Instead of giving it a song, you give it a
**scale** (which set of notes the piece should use — for example
"C major" means the white keys on a piano) and a **chord progression**
(the sequence of chords you want, like "Cmaj7, Am7, Dm7, G7"). The
tool writes a short, simple piece in that key over those chords:

- A **melody line** — a sequence of single notes, like a singer would
  sing.
- A **chord accompaniment** — the chords played underneath the
  melody. You can pick whether the chords are strummed (one strike per
  chord), arpeggiated (rolled out one note at a time), or fingerpicked
  (a thumb-and-fingers pattern).

You get four files back:

1. A **MIDI file** — a digital score you can open in any music
   software (GarageBand, Logic, Reaper, MuseScore...) and edit further.
2. A **WAV file** — the actual sound, ready to play.
3. An **ASCII tab** — the guitar fingering for the melody.
4. A **JSON file** with the full structured data, if you want to feed
   it into another program.

Give the same inputs and the same "seed" (a single number that picks
the random sequence) and you'll get the same piece back. Change the
seed, get a different piece.

Why this matters: it's a fast way to sketch an idea — pick a scale and
some chords, hit a button, hear it. No music theory degree required.

**What version 5 of the composer adds (fingerstyle only).** Fingerstyle is
the one style meant to be played by a person, and version 5 makes it sound
more like one:

- It is **performed**. Notes are no longer all equally loud: the melody
  sits above the accompaniment, each phrase swells toward its high point
  and relaxes into the end, and the bass notes ring on instead of stopping
  early. In the WAV (never in the MIDI or the tab) every note also lands a
  few thousandths of a second early or late, the way a hand does.
- It can use a **plucked-string sound** (`--sound plucked`): a built-in
  model of a plucked guitar string, in place of the soundfont guitar.
- Each bar may use a **different shape of its chord**, chosen so the
  chord's defining notes are still heard and the melody has room to move.
- The tune has a **hook** (its first few notes come back at the start of
  each phrase, once with one small change), each phrase **rises to one
  clear high note and falls**, and longer pieces go A, A', B, A'.
- The tool writes **12 pieces and keeps the best** by a score built from its
  own measures of a good tune. Optionally, a pretrained model that rates
  how enjoyable a clip sounds picks among the best four (`--rank-audio`),
  and a pretrained melody model, the Anticipatory Music Transformer,
  proposes the tune that the composer's rules then shape
  (`--melody-model amt`). Those two models are an optional download of
  about 930 MB.

---

## Feature 4: Make that piece sound like a record

The composition from Feature 3 is a sketch: a guitar-like melody and
accompaniment. `productionize` turns it into a finished instrumental you
could release. It lays the sketch out as a song (intro, hook, four
verses, second hook, outro), plays it on a sampled grand piano, adds a
real recorded drum kit playing a steady hip-hop backbeat, a deep bass
that follows the chords, and strings that swell in near the end. Then it
masters the result to streaming loudness and writes a WAV, an MP3 and
the separate instrument tracks.

It also checks its own work: loudness, peaks, timing of every drum hit,
and whether the track still sounds right in mono. The checks catch
technical faults; whether it sounds good is still for your ears.

---

## How does it know how to do this?

A handful of pieces of pre-trained AI software do the heavy lifting.
None of it was trained by us; we use it as-is, locally:

- **Demucs** — a model that Facebook AI Research published. It listens
  to a full mix and pulls the guitar out of it, the way you might
  isolate a single voice in a crowded room.
- **basic-pitch** — a model that Spotify published. It listens to
  audio and writes down which notes are being played at which moments.
- **madmom DeepChroma** — a chord-recognition model from a research
  group in Vienna. It identifies which chord is sounding at each point
  in time.

Around those three models, the tool has its own logic — written by
hand, no learning involved — for things like:

- Figuring out the **key** of a song (whether it's in "G major" or
  "E minor" overall) by comparing the audio's pitch profile to
  textbook pitch profiles for each key.
- Picking the **best fingering** for a tab, by trying every legal
  position on the guitar neck and choosing the one that minimizes
  hand movement and finger stretch.
- **Composing the melody** from a chord progression in four-bar
  phrases: rhythms from a small pattern library with a repeating
  motif, each pitch picked by weighted dice that favour chord tones on
  strong beats and small steps, and a final cadence on the home note.

The composition side is *not* AI-generated by default. It's a deterministic
recipe: same inputs, same output every time. Two optional pretrained models
can help the fingerstyle composer, and both are off unless you ask:

- **Audiobox Aesthetics** — a model Meta published (CC-BY-4.0, about
  415 MB). It listens to a clip and rates how enjoyable and how well
  produced it sounds. With `--rank-audio` it picks among the composer's
  best four pieces.
- **The Anticipatory Music Transformer** — a model Stanford CRFM
  published (Apache-2.0, about 512 MB). Given a bass line it writes a
  melody. With `--melody-model amt` the composer treats that melody as a
  soft hint and still follows its own rules for every note.

Both need the optional `ml` install (`pip install -e ".[ml]"`) and
`music-decoder fetch-models --composer-models`. They run on the CPU and give
the same answer for the same seed.

---

## What happens when you click "Analyze"?

Imagine you've just dropped a recording of a song into the app and
hit "Analyze". Here's the story of what happens, in order, in plain
English. The whole thing usually takes under a minute on a laptop.

1. **Read the audio**. The file (or the YouTube stream) gets converted
   into a long list of numbers — basically the waveform — so the rest
   of the tool can work with it. If the file is broken or silent, the
   tool stops and says so.

2. **Pull the guitar out of the mix**. If the recording is a full
   band, the tool runs Demucs to isolate the guitar from the drums,
   bass, vocals, and everything else. If you told the tool "this is
   already just a guitar", it skips this step.

3. **Find the beat**. A pre-trained neural beat tracker (madmom)
   figures out the tempo (in beats per minute), where each beat lands
   — like a tapping foot — and which beats start a bar (3/4 or 4/4).

4. **Build a pitch profile**. It produces a moving picture of which
   musical notes are sounding loudest at each moment — a "chroma"
   reading. (*Chroma* — a 12-dimensional fingerprint of the audio,
   one number per pitch class, that ignores which octave a note is
   in.)

5. **Identify each chord**. A pre-trained chord-recognition model
   (madmom's deep-chroma network) labels the major and minor chords
   over time; if it isn't available, a simpler template matcher with a
   *Viterbi pass* (a standard smoother, so a four-beat chord isn't
   reported as four one-beat chords) takes over.

6. **Identify the key**. Four independent clues each vote for all 24
   major and minor keys: two textbook pitch fingerprints
   (Krumhansl-Kessler and Temperley) compared against the song's average
   pitch profile, a neural key classifier, and how well the chords from
   step 5 fit each key (I, IV and V chords count most). The votes are put
   on a common scale and added up. The clues make different kinds of
   mistakes, so together they are right more often than any one alone.

7. **Transcribe the notes**. basic-pitch listens to the guitar
   recording and writes down a list of every individual note:
   pitch, start time, end time, loudness.

8. **Lay them out on the guitar neck**. For each note, the tool finds
   every possible (string, fret) pair that produces that pitch in the
   chosen tuning. Notes struck together (within 50 ms) are treated as
   one chord shape. It then runs an A* search — the same algorithm GPS
   apps use to find the shortest route — over the sequence of notes,
   choosing fingerings that minimize hand movement, finger stretch,
   and high-fret jumps. (*A\** — a method for finding the cheapest
   path through a maze of choices.)

9. **Bundle everything up**. The result is a structured object: the
   key, the tempo, the chord progression with timestamps, the tab,
   and metadata about how the run was configured.

The desktop app shows a progress bar that ticks through these stages.
The command-line version prints results when it's done.

---

## What happens when you click "Compose"?

You give the tool a scale, a chord progression, and a few options
(tempo, style, optional random seed). Here's the story:

1. **Pick a chord shape for each chord**. For every chord in the
   progression, the tool looks up a hand-curated guitar fingering for
   that chord and adjusts it for the tuning you chose. You get back
   the most playable position — usually open chords for common keys,
   barre chords otherwise.

2. **Write the melody, phrase by phrase**. The tool splits the piece into
   four-bar phrases, the way a song verse is built:
   - Each bar gets a rhythm from a small library of patterns: quarter notes,
     pairs of eighths, dotted rhythms, held notes. The third bar of a phrase
     repeats the first bar's rhythm and its rising-and-falling shape, and the
     next phrase answers the first one with the same rhythms. That
     repetition is what makes a tune feel like a tune.
   - Every phrase ends on a long note. The very last note lands on the home
     note of the key (or, if the last chord doesn't contain it, on that
     chord's root), so the piece sounds finished.
   - For each note it scores every candidate pitch and rolls weighted dice.
     Notes that belong to the current chord get a strong preference on the
     strong beats (beats 1 and 3). Small steps are preferred over jumps, and
     after a big jump the line steps back the other way. The "leading tone"
     (the note just below home) rises to the home note when the harmony
     comes home. The line stays in a comfortable register above the chord.
   - If a chord borrows a note from outside the scale (like G# in an E7
     chord in A minor), the melody borrows it too, in place of the note it
     replaces (G), instead of clashing.
   The "seed" you pass is the only source of randomness: same seed,
   same melody, every time.

3. **Add the accompaniment**. Underneath the melody, the tool plays the
   chord on every bar in your chosen pattern. The patterns are: a pop strum
   (down, down-up, up-down-up), an eighth-note arpeggio, or fingerstyle.
   The strum and the arpeggio leave out chord notes that would crowd the
   melody, but never drop the notes in the chord shape that make it
   major, minor or a seventh. Each ends with one chord left ringing.

   Fingerstyle works differently, because it is meant to be one guitar part
   that one person can play. The fretting hand holds the chord shape, the
   thumb plays an alternating bass (a Travis pattern), small fills pick
   chord tones on free strings, and the melody is chosen only from notes
   that shape can reach on the top strings (so its range is about one
   octave). The last bar is one held melody note over a rolled chord whose
   notes all sound below it. The melody is written first for the other
   styles, but for fingerstyle the shape comes first and the melody follows
   it.

   Version 5 refines the fingerstyle part in four ways. First, the shape
   can change from bar to bar: the tool searches for the series of shapes
   that keep the chord recognisable, reach the melody's notes and need the
   least hand movement, then repairs any bar that still lacks an essential
   chord tone. Second, the melody gets a **hook** (the first few notes
   return at every phrase start, with one change in the answer), a rising
   and falling **arch** with one unique high note per phrase, and the form
   A A' B A' for pieces of 16 bars or more. Third, the part is
   **performed**: louder melody, a swell over each phrase, bass notes that
   ring until the string is plucked again or the bar ends. Fourth, the tool
   writes 12 whole pieces from 12 seeds derived from yours and **keeps the
   best** by a weighted score (identity of the chords, melody on top,
   chord tones on strong beats, stepwise motion, rhythm variety, hook,
   peak, arch, range; minus repeated notes, hand movement and a thinned
   final chord). With the optional models, an audio-taste model can
   re-rank the best four, and a melody model can propose the tune.

4. **Render the audio**. The melody and accompaniment together get
   written to a MIDI file (a digital score) and then synthesized to a
   WAV file (actual sound). The MIDI is the master copy; the WAV is
   for instant playback. For fingerstyle the WAV is rendered from a copy
   of the MIDI whose note onsets are nudged by a few milliseconds, so it
   sounds played; the MIDI file and the tab stay exactly on the grid.
   `--sound plucked` renders from the tab instead, with a built-in
   plucked-string model.

5. **Write the tab**. For strum and arpeggio, the melody notes get
   placed onto the guitar neck by the same fingering search used for
   analysis, so the line stays in one hand position where it can. The tab
   shows the melody note by note, then the chord shapes under their names,
   in the kind of ASCII tab you might see in any guitar magazine. For
   fingerstyle the tab shows the whole part (melody and accompaniment) in
   rhythm, eight columns to a bar with a count row under each line, so you
   can play it straight from the page. Each note has one string and fret,
   and the MIDI file plays exactly those notes.

6. **Save everything**. The four files (MIDI, WAV, ASCII tab, JSON
   metadata) end up in a per-take folder named with a timestamp and a
   short hash of your inputs, so you can keep multiple takes side by
   side without overwriting.

---

## What does the tool not do?

A few things the tool deliberately stays out of:

- **It doesn't host anything online**. It's a tool you run on your
  own machine. No accounts, no cloud uploads.
- **It doesn't keep a history of past songs**. Re-running on the
  same input gives you the same result; the tool doesn't remember
  what you analyzed yesterday. The YouTube downloads it makes do
  get cached, so you don't re-download the same video twice.
- **It only handles guitar**. No piano transcription, no drum tabs,
  no horn parts. The composition output is also guitar-shaped — six
  strings, six-string-friendly voicings.
- **It only knows major and minor scales**. Modal scales (Dorian,
  Mixolydian, etc.) and exotic scales (whole-tone, harmonic minor)
  aren't in this version.
- **It isn't a fine-tuning sandbox**. The AI models are used as
  shipped; there's no training, no model improvement loop. That includes
  the two optional composer models (Audiobox Aesthetics and the
  Anticipatory Music Transformer): they are downloaded, used and never
  trained, and the default composer needs neither.

---

## When does it work well, and when does it struggle?

It does best on:

- Recordings where the guitar is clearly in front of the mix, or solo
  guitar recordings.
- Songs in standard genres with diatonic chords (most pop, rock,
  folk, jazz standards).
- Tempos between roughly 60 and 180 beats per minute.
- Clear recordings, not lo-fi cassette rips.

It struggles with:

- Heavily distorted guitar — the harmonics confuse the chord and
  pitch detectors.
- Songs that change key dramatically partway through.
- Very short clips (under one second).
- Polyphonic guitar with more than six simultaneous notes.

The output also includes confidence numbers. If the key estimate has a
low correlation, or the chord segments have low confidence, treat the
result as a starting point and not gospel.

---

## Where do the files end up?

By default, everything lives under a hidden folder called
`.music-decoder` in your home directory:

- The **YouTube cache** is at `~/.music-decoder/yt_cache/`. Each
  video you've ever analyzed sits here as a WAV file named after its
  YouTube ID. You can wipe this folder safely; the tool will
  re-download as needed.
- **Composition output** is at `~/.music-decoder/compositions/`,
  with one folder per piece you've generated.

You can change the location with an environment variable, or by
editing the runtime configuration file. See the [Usage](usage.md)
guide.

---

## What's next?

If you'd like to actually try it, head to the [Usage](usage.md)
guide for installation and copy-paste examples.

If you'd like to read the code or change something, the
[Technical](architecture-technical.md) doc lays out exactly what
lives where.
