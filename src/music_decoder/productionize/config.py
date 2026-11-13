"""Every number the production chain uses: grid, sections, levels, EQ.

The defaults are the "Hisaab v3" recipe (boom-bap drums, sine bass, sampled
grand piano, soundfont strings, clipper + true-peak limiter master). Nothing
in the DSP, instrument, mix or master modules hard-codes a level, a filter
frequency or a time constant: they all read one of the frozen dataclasses
below, so a different sound is a different ``ProductionConfig``.

What is *not* here is the song itself (grid tempo, section list, notes):
that is a :class:`music_decoder.productionize.song.Song`, built either from
a composer output or from a note file.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

SAMPLE_RATE = 44_100

DrumMode = Literal["none", "light", "full", "outro"]
SectionRole = Literal["intro", "hook", "verse", "outro"]
PianoTexture = Literal["as_composed", "held_chords", "accompaniment_only"]
ChordVoicing = Literal["shapes", "voice_led"]
VerseMelody = Literal["full", "skeleton", "guide_line"]
EqKind = Literal["peak", "low_shelf", "high_shelf"]


# ---- Grid and form ---------------------------------------------------------


@dataclass(frozen=True)
class Grid:
    """A straight 4/4 grid: sections of whole bars, bars of 16th steps."""

    beat_s: float
    bars_per_section: int = 4
    beats_per_bar: int = 4
    steps_per_bar: int = 16
    sample_rate: int = SAMPLE_RATE

    @property
    def bar_s(self) -> float:
        return self.beat_s * self.beats_per_bar

    @property
    def section_s(self) -> float:
        return self.bar_s * self.bars_per_section

    @property
    def step_s(self) -> float:
        return self.bar_s / self.steps_per_bar

    @property
    def tempo_bpm(self) -> float:
        return 60.0 / self.beat_s

    def time(self, section: int, bar: int = 0, step: float = 0.0) -> float:
        """Seconds of ``step`` (16ths) in ``bar`` of ``section`` (all 0-based)."""
        return section * self.section_s + bar * self.bar_s + step * self.step_s

    def samples(self, seconds: float) -> int:
        return round(seconds * self.sample_rate)


@dataclass(frozen=True)
class SectionSpec:
    """What plays in one section. ``role`` drives crashes, levels and QC."""

    name: str
    role: SectionRole
    drums: DrumMode = "full"
    bass: bool = True
    strings: bool = False
    reverb_boost: bool = False
    # Used only when the notes come from the composer (see song.py).
    piano_texture: PianoTexture = "as_composed"
    piano_velocity_offset: int = 0
    # For "held_chords" only: the composer's guitar shapes or piano voicings,
    # and the whole melody, its notes on beats 1 and 3, or one note a bar.
    chord_voicing: ChordVoicing = "shapes"
    verse_melody: VerseMelody = "full"


# Slope of a straight-line fit to the original loop's 13 note onsets.
HISAAB_BEAT_S = 0.7487363
HISAAB_GRID = Grid(beat_s=HISAAB_BEAT_S, bars_per_section=4)
# Bass roots per bar: F#1, B1, C#2, F#1 (F#m, Bm, C#7, F#m).
HISAAB_BAR_ROOTS: tuple[int, ...] = (30, 35, 37, 30)
HISAAB_SECTIONS: tuple[SectionSpec, ...] = (
    SectionSpec("Intro", "intro", drums="none", bass=False, reverb_boost=True),
    SectionSpec("Hook 1", "hook", drums="light"),
    SectionSpec("Verse 1", "verse"),
    SectionSpec("Verse 2", "verse"),
    SectionSpec("Verse 3", "verse"),
    SectionSpec("Verse 4", "verse"),
    SectionSpec("Hook 2", "hook", strings=True),
    SectionSpec("Outro", "outro", drums="outro", bass=False, strings=True, reverb_boost=True),
)

# The same form for composer output. The composer writes one pass over the
# progression; each section replays it with the texture named here.
# ``composer_sections`` swaps in another verse style (see VERSE_STYLES).
COMPOSER_SECTIONS: tuple[SectionSpec, ...] = (
    SectionSpec(
        "Intro",
        "intro",
        drums="none",
        bass=False,
        reverb_boost=True,
        piano_texture="accompaniment_only",
        piano_velocity_offset=-14,
    ),
    SectionSpec("Hook 1", "hook", drums="light"),
    SectionSpec("Verse 1", "verse", piano_texture="held_chords"),
    SectionSpec("Verse 2", "verse", piano_texture="held_chords"),
    SectionSpec("Verse 3", "verse", piano_texture="held_chords"),
    SectionSpec("Verse 4", "verse", piano_texture="held_chords"),
    SectionSpec("Hook 2", "hook", strings=True),
    SectionSpec(
        "Outro",
        "outro",
        drums="outro",
        bass=False,
        strings=True,
        reverb_boost=True,
        piano_texture="held_chords",
        piano_velocity_offset=-12,
    ),
)


# Verse styles for composer songs: (voicing, melody) of verses 1, 2, 3, 4
# (a shorter list repeats).
VERSE_STYLES: dict[str, tuple[tuple[ChordVoicing, VerseMelody], ...]] = {
    # The composer's chord shapes held under the whole hook melody.
    "held": (("shapes", "full"),),
    # Piano voicings with smooth voice leading; one held note a bar from a
    # stepwise line over the chord tones (the Hisaab verse).
    "guide-line": (("voice_led", "guide_line"),),
    # Piano voicings; the hook melody's notes on beats 1 and 3 only.
    "skeleton": (("voice_led", "skeleton"),),
    # Verses 1-2 as guide-line, verses 3-4 as skeleton: the verse builds.
    "build": (
        ("voice_led", "guide_line"),
        ("voice_led", "guide_line"),
        ("voice_led", "skeleton"),
        ("voice_led", "skeleton"),
    ),
}


def composer_sections(verse_style: str = "held") -> tuple[SectionSpec, ...]:
    """The composer form with ``verse_style`` (a key of VERSE_STYLES) in its verses."""
    if verse_style not in VERSE_STYLES:
        raise ValueError(f"Unknown verse style {verse_style!r}; choose from {sorted(VERSE_STYLES)}")
    plan = VERSE_STYLES[verse_style]
    out, verse = [], 0
    for section in COMPOSER_SECTIONS:
        if section.role == "verse":
            voicing, melody = plan[verse % len(plan)]
            section = replace(section, chord_voicing=voicing, verse_melody=melody)
            verse += 1
        out.append(section)
    return tuple(out)


@dataclass(frozen=True)
class ComposerArrangeConfig:
    """How composer notes become the piano and strings parts."""

    # "held_chords" texture: the bar's chord, rolled low to high, held.
    held_chord_velocity: int = 62
    held_chord_beats: float = 3.9
    roll_steps: float = 1.0
    # Final chord after the last section.
    tail_beats: float = 8.0
    tail_roll_steps: float = 1.5
    tail_velocity: int = 78
    # Strings: one chord per bar, the top voices of the bar's chord.
    strings_voices: int = 4
    strings_velocity: int = 56
    strings_beats: float = 4.0
    # The strings do not decay on their own, so their final chord ends by the
    # time the mix fade-out starts (MixConfig: 7 s tail - 2.5 s fade). Held
    # for ``tail_beats`` at slow tempos they are still loud in the last 50 ms.
    strings_tail_max_s: float = 4.5
    # Bass roots are folded into this MIDI range (E1..D#2).
    bass_low: int = 28
    bass_high: int = 39
    # Verse style (a key of VERSE_STYLES) when no section list is given.
    verse_style: str = "held"
    # Voice-led piano voicings: left-hand root (+ fifth) from E2 up, and
    # right-hand chord tones (always the third and seventh) in G3..A4.
    lh_low: int = 40
    rh_low: int = 55
    rh_high: int = 69
    rh_voices: int = 3
    # Guide line: one note a bar at this 16th, at least this far above the
    # bar's chord and no higher than the hook melody.
    line_step: float = 10.0
    line_beats: float = 1.4
    line_velocity: int = 72
    line_min_gap: int = 3
    # Skeleton: the melody notes sounding at these 16ths of each bar.
    skeleton_steps: tuple[float, ...] = (0.0, 8.0)


# ---- Shared DSP blocks -----------------------------------------------------


@dataclass(frozen=True)
class EqBand:
    """One RBJ cookbook biquad. Shelves ignore ``q`` (slope 1)."""

    kind: EqKind
    freq_hz: float
    gain_db: float
    q: float = 1.0


@dataclass(frozen=True)
class CompressorConfig:
    """Stereo-linked feed-forward compressor; threshold = ``rms_multiple`` x RMS."""

    rms_multiple: float
    ratio: float
    attack_s: float
    release_s: float
    block: int = 32


@dataclass(frozen=True)
class ReverbConfig:
    """Synthetic impulse response: decaying band-limited noise."""

    t60_s: float
    pre_delay_s: float
    bright: float
    seed_left: int
    seed_right: int
    length_t60: float = 1.3
    band_edges_hz: tuple[float, ...] = (150.0, 300.0, 600.0, 1200.0, 2400.0, 4800.0, 9000.0)
    # Decay-time factor per band; the top two also scale with ``bright``.
    band_decay: tuple[float, ...] = (1.15, 1.1, 1.0, 0.85, 0.65, 0.45)
    bright_bands: tuple[int, ...] = (4, 5)
    build_up_s: float = 0.04
    build_up_curve: float = 0.7
    early_reflections: int = 7
    early_min_s: float = 0.004
    early_max_s: float = 0.045
    early_gain: float = 0.6


HALL = ReverbConfig(t60_s=2.3, pre_delay_s=0.025, bright=1.0, seed_left=11, seed_right=12)
PLATE = ReverbConfig(t60_s=1.0, pre_delay_s=0.012, bright=1.3, seed_left=21, seed_right=22)


# ---- Drums -----------------------------------------------------------------


@dataclass(frozen=True)
class DrumPiece:
    """One kit piece: which samples, how the mics are summed, the envelope."""

    folder: str
    articulation: str
    mic_weights: tuple[tuple[str, float], ...]
    round_robins: tuple[int, ...] = ()  # empty: files have no ``_rrN`` suffix
    hold_s: float = 0.0
    tau_s: float = 0.0
    cut_s: float = 0.0  # 0: unshaped
    fade_s: float = 0.010


@dataclass(frozen=True)
class ThumpConfig:
    """Sine under every kick, so the low end is the same on every hit."""

    gain: float = 0.4
    base_hz: float = 50.0
    sweep_hz: float = 45.0
    sweep_tau_s: float = 0.018
    attack_s: float = 0.003
    decay_tau_s: float = 0.075
    length_s: float = 0.220


@dataclass(frozen=True)
class DrumsConfig:
    kick: DrumPiece = DrumPiece(
        folder="kick",
        articulation="kick_snoff",
        mic_weights=(("kickmic", 1.0), ("oh", 3.0), ("room", 0.8)),
        round_robins=(1, 2, 3, 4),
        hold_s=0.07,
        tau_s=0.16,
        cut_s=0.8,
    )
    snare: DrumPiece = DrumPiece(
        folder="snare",
        articulation="snare_center",
        mic_weights=(("snaremic", 1.0), ("oh", 0.8), ("room", 0.4)),
        hold_s=0.12,
        tau_s=0.22,
        cut_s=1.2,
    )
    hat_closed: DrumPiece = DrumPiece(
        folder="hh",
        articulation="hh_closed",
        mic_weights=(("oh", 1.0), ("room", 0.3)),
        round_robins=(1, 2, 3, 4),
        hold_s=0.05,
        tau_s=0.07,
        cut_s=0.4,
    )
    hat_open: DrumPiece = DrumPiece(
        folder="hh",
        articulation="hh_open",
        mic_weights=(("oh", 1.0), ("room", 0.3)),
        round_robins=(1, 2, 3),
    )
    crash: DrumPiece = DrumPiece(
        folder="crash",
        articulation="crash_crash",
        mic_weights=(("oh", 1.0), ("room", 0.45)),
        round_robins=(1, 2, 3, 4),
        hold_s=0.3,
        tau_s=1.6,
        cut_s=6.0,
    )
    thump: ThumpConfig = ThumpConfig()

    # Building one hit from the mics.
    source_rate: int = 48_000
    onset_threshold: float = 0.05
    onset_pre_roll_s: float = 0.0015

    # Velocity layers.
    kick_main_layer: int = 3
    kick_other_layer: int = 2
    snare_layers: tuple[int, ...] = (18, 19, 20, 21, 22, 23, 24)
    snare_light_layers: tuple[int, ...] = (12, 13, 14, 15, 16)
    hat_layer: int = 3
    hat_light_layer: int = 2
    hat_open_layer: int = 2
    crash_hook_layer: int = 2
    crash_other_layer: int = 1

    # Pattern, in 16th steps of a bar. Bars cycle through ``kick_steps``; the
    # open hat replaces the closed hat on the last bar of a section.
    kick_steps: tuple[tuple[int, ...], ...] = ((0, 10), (0, 6, 10))
    kick_main_step: int = 0
    kick_other_db: float = -3.0
    kick_light_db: float = -2.0
    snare_steps: tuple[int, ...] = (4, 12)
    snare_light_db: float = -2.5
    hat_steps: tuple[int, ...] = (0, 2, 4, 6, 8, 10, 12, 14)
    hat_gains_db: tuple[float, ...] = (0.0, -6.0, -2.5, -7.0, -0.5, -6.0, -2.5, -5.0)
    hat_light_db: float = -1.5
    open_hat_step: int = 14
    open_hat_db: float = 0.0
    open_hat_choke_after_s: float = 0.030
    open_hat_choke_fade_s: float = 0.070
    crash_hook_db: float = 0.0
    crash_light_hook_db: float = -1.0
    crash_other_db: float = -1.5
    outro_kick_db: float = -2.0

    # Reverse crash swell into each hook.
    swell_source_s: float = 2.5
    swell_beats: float = 1.0
    swell_gain: float = 0.5
    swell_round_robin: int = 1

    # Variation.
    humanize_time_s: float = 0.002
    humanize_gain_db: float = 0.6
    seed: int = 2026

    # Processing. Snare, hats and crash levels are verse RMS relative to the kick.
    kick_highpass_hz: float = 32.0
    kick_eq: tuple[EqBand, ...] = (
        EqBand("peak", 60.0, 1.5, 1.0),
        EqBand("peak", 350.0, -4.0, 1.2),
        EqBand("peak", 3000.0, 3.0, 0.8),
    )
    kick_peak_dbfs: float = -11.5
    kick_peak_percentile: float = 99.9
    snare_highpass_hz: float = 110.0
    snare_eq: tuple[EqBand, ...] = (
        EqBand("peak", 210.0, 1.0, 1.0),
        EqBand("peak", 900.0, -1.5, 1.2),
        EqBand("peak", 3500.0, 4.0, 0.8),
        EqBand("high_shelf", 7000.0, 5.0),
    )
    snare_drive: float = 1.2
    snare_db: float = -1.5
    snare_plate_send_db: float = -15.0
    hats_highpass_hz: float = 500.0
    hats_eq: tuple[EqBand, ...] = (EqBand("high_shelf", 8000.0, 3.5),)
    hats_lowpass_hz: float = 15_000.0
    hats_db: float = -6.0
    hats_left_gain: float = 0.85
    crash_highpass_hz: float = 350.0
    crash_eq: tuple[EqBand, ...] = (EqBand("high_shelf", 10_000.0, -1.5),)
    crash_db: float = -13.0
    bus_compressor: CompressorConfig = CompressorConfig(
        rms_multiple=2.5, ratio=2.0, attack_s=0.012, release_s=0.140
    )
    bus_drive: float = 1.5


# ---- Bass ------------------------------------------------------------------


@dataclass(frozen=True)
class BassConfig:
    """One continuous-phase sine oscillator; notes join without restarts."""

    # (16th step, accent) of each note in a bar.
    events: tuple[tuple[int, float], ...] = ((0, 1.0), (10, 0.6))
    glide_s: float = 0.012
    accent_tau_s: float = 0.12
    accent_length_s: float = 0.350
    amp_base: float = 0.8
    amp_accent: float = 0.2
    release_tau_s: float = 0.9
    release_beats: float = 3.2
    release_curve: float = 0.5
    gate_smooth_s: float = 0.020
    # v3 tone: harmonics so the bass shows on phone speakers (quiet: 0.10 / 1.4).
    second_harmonic: float = 0.20
    drive: float = 2.0
    highpass_hz: float = 28.0
    lowpass_hz: float = 500.0
    duck_depth: float = 0.3
    duck_tau_s: float = 0.06
    duck_length_s: float = 0.200


# ---- Piano -----------------------------------------------------------------


@dataclass(frozen=True)
class PianoConfig:
    """Sampler over the Salamander Grand Piano V3."""

    # Sampled pitches: minor thirds from A0 (MIDI 21) to C8 (108).
    sample_pitches: tuple[int, ...] = tuple(range(21, 109, 3))
    source_rate: int = 48_000
    max_sample_s: float = 13.0
    resample_max_denominator: int = 400
    velocity_scale: float = 1.05
    velocity_max: int = 104
    # Upper velocity bound of layers 1..13 (layers 14-16 are not used).
    layer_bounds: tuple[int, ...] = (26, 34, 36, 43, 46, 50, 56, 64, 72, 80, 88, 96, 104)
    trim_threshold: float = 0.04
    trim_pre_roll_s: float = 0.002
    # Pedal: a note rings to the end of its bar, less this, then releases.
    pedal_lift_s: float = 0.020
    # An onset this close before a bar line belongs to the next bar.
    bar_snap_s: float = 0.060
    release_tau_s: float = 0.22
    tail_release_tau_s: float = 0.8
    restrike_delay_s: float = 0.030
    restrike_tau_s: float = 0.03
    release_length_taus: float = 6.0
    release_curve: float = 0.5
    side_gain: float = 0.8
    highpass_hz: float = 125.0
    # The big shelf is intended: the soft Salamander layers are dark.
    eq: tuple[EqBand, ...] = (
        EqBand("peak", 300.0, -2.5, 1.0),
        EqBand("peak", 2200.0, 3.0, 0.7),
        EqBand("high_shelf", 3000.0, 9.5),
    )
    compressor: CompressorConfig = CompressorConfig(
        rms_multiple=1.8, ratio=3.0, attack_s=0.008, release_s=0.220
    )
    limit_percentile: float = 99.9
    limit_headroom: float = 1.15
    reverb_send_db: float = -13.0
    reverb_band_hz: tuple[float, float] = (250.0, 7000.0)
    reverb_boost: float = 1.5
    reverb_boost_smooth_s: float = 1.0


# ---- Strings ---------------------------------------------------------------


@dataclass(frozen=True)
class StringsConfig:
    """General MIDI strings rendered by the FluidSynth command-line tool."""

    program: int = 49  # String Ensemble 2 (0-based)
    soundfont: str = "/usr/share/sounds/sf2/FluidR3_GM.sf2"
    fluidsynth_gain: float = 0.6
    highpass_hz: float = 180.0
    lowpass_hz: float = 4500.0
    fade_in_s: float = 1.5
    fade_in_curve: float = 0.7
    reverb_send_db: float = -8.0


# ---- Optional percussion ---------------------------------------------------


@dataclass(frozen=True)
class PercussionVoice:
    """One sample set: ``{name}_rr{N}_Close.wav`` under ``perc/close/{folder}``."""

    folder: str
    name: str
    round_robins: int
    cut_s: float
    align: Literal["onset", "peak"] = "onset"


@dataclass(frozen=True)
class PercussionHit:
    step: int
    voice: int  # index into PercussionOption.voices
    gain_db: float


@dataclass(frozen=True)
class PercussionOption:
    name: str
    voices: tuple[PercussionVoice, ...]
    pattern: tuple[PercussionHit, ...]
    highpass_hz: float
    eq: tuple[EqBand, ...]
    pan: float
    level_dbfs: float
    hall_send_db: float


def _shaker_pattern() -> tuple[PercussionHit, ...]:
    hits = []
    for step in range(16):
        if step % 2:
            hits.append(PercussionHit(step, 1, -6.0))  # up stroke
        else:
            hits.append(PercussionHit(step, 0, 0.0 if step % 4 == 0 else -2.5))
    return tuple(hits)


SHAKER = PercussionOption(
    name="shaker",
    voices=(
        PercussionVoice("shaker", "LShaker_Shake1D", 7, 0.25, "peak"),
        PercussionVoice("shaker", "LShaker_Shake1U", 7, 0.25, "peak"),
    ),
    pattern=_shaker_pattern(),
    highpass_hz=2000.0,
    eq=(),
    pan=-0.35,
    level_dbfs=-38.5,
    hall_send_db=-16.0,
)
TAMBOURINE = PercussionOption(
    name="tambourine",
    voices=(PercussionVoice("tambourine", "Tamb1_Shake", 10, 0.5, "peak"),),
    pattern=(PercussionHit(4, 0, 0.0), PercussionHit(12, 0, 0.0)),
    highpass_hz=1200.0,
    eq=(),
    pan=0.3,
    level_dbfs=-40.0,
    hall_send_db=-12.0,
)
CONGAS = PercussionOption(
    name="congas",
    voices=(
        PercussionVoice("conga", "Conga_22_HitN_51_100", 8, 0.7),  # open
        PercussionVoice("conga", "Conga_17_HitHM1_1_70", 8, 0.5),  # muted
        PercussionVoice("tumba", "Tumba_24_HitN_41_70", 8, 0.8),
    ),
    pattern=(
        PercussionHit(2, 1, -5.0),
        PercussionHit(6, 0, 0.0),
        PercussionHit(8, 1, -6.0),
        PercussionHit(14, 2, -1.0),
    ),
    highpass_hz=110.0,
    eq=(EqBand("peak", 400.0, -2.0, 1.0), EqBand("peak", 4000.0, 2.0, 1.0)),
    pan=-0.3,
    level_dbfs=-35.0,
    hall_send_db=-13.0,
)
PERCUSSION_OPTIONS: dict[str, PercussionOption] = {o.name: o for o in (SHAKER, TAMBOURINE, CONGAS)}


@dataclass(frozen=True)
class PercussionLayer:
    """An option switched on for some sections (0-based section indices)."""

    option: str
    sections: tuple[int, ...] = (2, 3, 4, 5, 6)


@dataclass(frozen=True)
class PercussionConfig:
    layers: tuple[PercussionLayer, ...] = ()  # off in v3
    onset_threshold: float = 0.05
    onset_pre_roll_s: float = 0.0015
    peak_window_s: float = 0.010
    fade_s: float = 0.020


# ---- Mix and master --------------------------------------------------------


@dataclass(frozen=True)
class MixConfig:
    """Bus levels: RMS in dBFS over the verses, in the pre-master mix."""

    drums_dbfs: float = -28.4
    bass_dbfs: float = -15.0  # -17.5 default + 2.5 dB in v3
    piano_dbfs: float = -22.8  # dry; its hall return comes on top
    strings_dbfs: float = -31.0  # dry, over the first section with strings
    mono_below_hz: float = 110.0
    mono_filter_order: int = 4
    air: EqBand = EqBand("high_shelf", 9000.0, 1.5)
    tail_s: float = 7.0
    fade_out_s: float = 2.5
    fade_out_curve: float = 1.5


@dataclass(frozen=True)
class MasterConfig:
    subsonic_hz: float = 25.0
    subsonic_order: int = 4
    glue: CompressorConfig = CompressorConfig(
        rms_multiple=2.2, ratio=1.6, attack_s=0.030, release_s=0.250, block=64
    )
    clip_ratio: float = 0.85
    clip_percentile: float = 99.9
    clip_knee: float = 0.75
    silence_floor: float = 1e-4
    oversample: int = 4
    target_lufs: float = -10.5
    ceiling_dbtp: float = -1.8
    ceiling_margin: float = 0.985  # room for MP3 overs
    lookahead_s: float = 0.005
    release_s: float = 0.090
    limiter_block: int = 32
    loudness_iterations: int = 5
    loudness_tolerance_lu: float = 0.1
    lead_in_s: float = 0.12
    mp3_bitrate: str = "320k"
    genre: str = "Hip-Hop"


# ---- Acceptance checks -----------------------------------------------------


@dataclass(frozen=True)
class QcReference:
    """Measured values of a reference master that a rebuild must match."""

    section_rms_dbfs: tuple[float, ...]
    section_rms_tolerance_db: float = 1.0
    octave_band_db: tuple[float, ...] = ()
    octave_band_tolerance_db: float = 2.0
    mp3_true_peak_dbtp: float | None = None  # the rebuild's MP3 may not peak higher


# Hisaab v3: intro, hook 1, verses (x4), hook 2, outro.
HISAAB_V3_REFERENCE = QcReference(
    section_rms_dbfs=(-17.2, -9.6, -9.8, -9.8, -9.8, -9.8, -9.5, -16.6),
    octave_band_db=(-15.3, -11.6, -23.3, -25.1, -23.4, -29.6, -33.6, -37.0, -37.4, -39.3),
    mp3_true_peak_dbtp=-1.43,
)


@dataclass(frozen=True)
class QcConfig:
    loudness_tolerance_lu: float = 0.3
    mp3_true_peak_max_dbtp: float = -1.0  # delivery ceiling for lossy files
    first_sound_dbfs: float = -60.0
    first_sound_tolerance_s: float = 0.010
    tail_window_s: float = 0.050
    tail_peak_dbfs: float = -80.0
    dc_offset_max: float = 0.001
    stereo_correlation_min: float = 0.75
    mono_fold_down_min_db: float = -1.0
    low_band_hz: float = 110.0
    low_correlation_min: float = 0.99
    tempo_tolerance_bpm: float = 0.5
    beat_jitter_max_s: float = 0.008
    last_beat_ratio: tuple[float, float] = (0.85, 1.05)
    first_beat_ratio: tuple[float, float] = (0.9, 1.1)
    drum_grid_tolerance_s: float = 0.003
    octave_centres_hz: tuple[float, ...] = (
        31.25,
        62.5,
        125.0,
        250.0,
        500.0,
        1000.0,
        2000.0,
        4000.0,
        8000.0,
        15000.0,
    )
    octave_reference_lufs: float = -10.0
    octave_fft: int = 2048


# ---- The whole recipe ------------------------------------------------------


@dataclass(frozen=True)
class ProductionConfig:
    drums: DrumsConfig = DrumsConfig()
    bass: BassConfig = BassConfig()
    piano: PianoConfig = PianoConfig()
    strings: StringsConfig = StringsConfig()
    percussion: PercussionConfig = PercussionConfig()
    hall: ReverbConfig = HALL
    plate: ReverbConfig = PLATE
    mix: MixConfig = MixConfig()
    master: MasterConfig = MasterConfig()
    qc: QcConfig = QcConfig()
    arrange: ComposerArrangeConfig = field(default_factory=ComposerArrangeConfig)


def with_percussion(
    config: ProductionConfig, layers: tuple[PercussionLayer, ...]
) -> ProductionConfig:
    """``config`` with the given percussion layers switched on."""

    for layer in layers:
        if layer.option not in PERCUSSION_OPTIONS:
            raise ValueError(
                f"Unknown percussion option {layer.option!r}; "
                f"choose from {sorted(PERCUSSION_OPTIONS)}"
            )
    return replace(config, percussion=replace(config.percussion, layers=layers))
