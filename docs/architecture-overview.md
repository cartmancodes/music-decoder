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

There are three things it does. Each one is a self-contained feature.

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
- **Composing the melody** from a chord progression by stepping
  through the bars and rolling weighted dice — chord tones on the
  strong beats, scale tones on the weak ones, with a soft preference
  for staying close to the previous note.

The composition side is *not* AI-generated. It's a deterministic
recipe: same inputs, same output every time.

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

3. **Find the beat**. The tool figures out the tempo (in beats per
   minute) and where each beat lands, like a tapping foot.

4. **Build a pitch profile**. It produces a moving picture of which
   musical notes are sounding loudest at each moment — a "chroma"
   reading. (*Chroma* — a 12-dimensional fingerprint of the audio,
   one number per pitch class, that ignores which octave a note is
   in.)

5. **Identify the key**. It compares the average pitch profile of the
   whole song against textbook fingerprints for all 24 major and minor
   keys, and picks the closest match. Two different fingerprints are
   tried (Krumhansl-Kessler and Temperley); if they agree, the result
   is high-confidence.

6. **Identify each chord**. For each beat in the song, the tool looks
   at the pitch profile during that beat and asks "which chord does
   this look most like?" — comparing against templates for major,
   minor, dominant 7th, major 7th, minor 7th, diminished, suspended,
   and augmented chords across all 12 keys. Then it smooths the
   sequence so a chord that lasts four beats doesn't get reported as
   four separate one-beat chords (this is called a *Viterbi pass* — a
   standard dynamic-programming smoother).

7. **Transcribe the notes**. basic-pitch listens to the guitar
   recording and writes down a list of every individual note:
   pitch, start time, end time, loudness.

8. **Lay them out on the guitar neck**. For each note, the tool finds
   every possible (string, fret) pair that produces that pitch in the
   chosen tuning. It then runs an A* search — the same algorithm GPS
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

2. **Write the melody, bar by bar**. The tool walks through the
   progression and, for each beat, rolls weighted dice:
   - On strong beats (beats 1 and 3 in 4/4 time), it prefers a chord
     tone — a note that's already in the chord that's sounding.
   - On weak beats (beats 2 and 4), it prefers any note from the
     scale.
   - It also nudges itself toward small steps — usually the next note
     is within a couple of half-steps of the previous one, the way a
     singable melody behaves.
   The "seed" you pass is the only source of randomness: same seed,
   same melody, every time.

3. **Add the accompaniment**. Underneath the melody, the tool plays
   the chord on every bar, in your chosen pattern (block strum,
   ascending arpeggio, or fingerstyle bass-and-treble alternation).

4. **Render the audio**. The melody and accompaniment together get
   written to a MIDI file (a digital score) and then synthesized to a
   WAV file (actual sound). The MIDI is the master copy; the WAV is
   for instant playback.

5. **Write the tab**. The melody notes get placed onto the guitar
   neck so you can play them. The result is an ASCII tab string —
   what you might see in any guitar magazine.

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
  shipped; there's no training, no model improvement loop.

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
