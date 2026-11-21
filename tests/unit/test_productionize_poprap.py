from itertools import pairwise

import pretty_midi
import pytest

from music_decoder.compose.api import arrange
from music_decoder.compose.arrangement import build_midi
from music_decoder.productionize.config import PopRapConfig, PopRapFormConfig, TrapDrumsConfig
from music_decoder.productionize.poprap import (
    drum_hits,
    hook_bars,
    lead_notes,
    pad_events,
    pluck_notes,
    poprap_form,
    poprap_song,
)
from music_decoder.types import ChordSymbol, Scale

PROGRESSION = [ChordSymbol.parse(c) for c in ("Am", "F", "C", "G")]
CFG = PopRapConfig()
DRUMS = TrapDrumsConfig()


def _write_composition(path, *, extend_melody_s: float = 0.0):
    arr = arrange(Scale("A", "minor"), PROGRESSION, bars_per_chord=2, tempo_bpm=120.0, seed=7)
    build_midi(
        melody=arr.melody,
        voicings=arr.voicings,
        bars_per_chord=2,
        tempo_bpm=120.0,
        style="fingerstyle",
        accompaniment=arr.accompaniment,
        out_path=path,
    )
    if extend_melody_s:
        pm = pretty_midi.PrettyMIDI(str(path))
        melody = next(i for i in pm.instruments if i.name == "melody")
        last = max(melody.notes, key=lambda n: n.end)
        last.end = 8 * 2.0 + extend_melody_s  # past the 8-bar pass (2 s a bar)
        pm.write(str(path))
    return path


@pytest.fixture
def song(tmp_path):
    midi = _write_composition(tmp_path / "composition.mid")
    return poprap_song(
        midi,
        cfg=CFG,
        progression=PROGRESSION,
        scale=Scale("A", "minor"),
        key="Am",
    )


def _bar_of(song, time_s):
    return int(time_s // song.grid.bar_s + 1e-9)


def _span_s(song, index):
    section = song.sections[index]
    bar_s = song.grid.bar_s
    return section.start_bar * bar_s, (section.start_bar + section.bars) * bar_s


def test_hook_bars_rule():
    assert hook_bars(8, 16) == 16
    assert hook_bars(4, 16) == 16
    assert hook_bars(12, 16) == 12
    assert hook_bars(32, 16) == 32
    assert hook_bars(6, 16) == 18


def test_form_of_an_8_bar_pass():
    form = poprap_form(8, PopRapFormConfig())
    assert [(s.name, s.bars) for s in form] == [
        ("Intro", 8),
        ("Hook 1", 16),
        ("Verse", 16),
        ("Hook 2", 16),
        ("Outro", 4),
    ]
    assert [s.start_bar for s in form] == [0, 8, 24, 40, 56]
    assert [s.role for s in form] == ["intro", "hook", "verse", "hook", "outro"]


def test_form_of_a_32_bar_pass():
    form = poprap_form(32, PopRapFormConfig())
    assert [s.bars for s in form] == [8, 32, 32, 32, 4]


def test_song_reads_the_file_and_names_itself(song, tmp_path):
    assert song.source == "composer file composition.mid"
    assert song.key == "Am"
    assert song.title == "Composition (Instrumental)"
    assert song.grid.bars_per_section == song.parts.n_bars == 8
    assert song.body_s == pytest.approx(60 * song.grid.bar_s)
    assert song.verse_sections == (2,)
    assert song.pass_bar(13) == 5
    start, end = song.section_span(1)
    assert (start, end) == (song.grid.samples(8 * 2.0), song.grid.samples(24 * 2.0))


def test_lead_is_the_melody_once_per_pass_and_silent_in_the_verse(song):
    parts = song.parts
    main, double = lead_notes(song)
    outro_start, _ = _span_s(song, 4)
    outro_notes = [n for n in parts.melody if n.start_s < 4 * song.grid.bar_s - 1e-9]
    assert len(main) == len(parts.melody) * (16 // 8) * 2 + len(outro_notes)
    verse_start, verse_end = _span_s(song, 2)
    assert not [n for n in main if verse_start <= n.start_s < verse_end]
    assert {n.pitch for n in main} <= {n.pitch for n in parts.melody}
    hook2_start, hook2_end = _span_s(song, 3)
    assert double
    assert all(hook2_start <= n.start_s < hook2_end for n in double)
    hook2_main = [n for n in main if hook2_start <= n.start_s < hook2_end]
    assert double == hook2_main
    assert outro_start > hook2_end - 1e-9


def test_notes_are_cut_at_the_section_end(tmp_path):
    midi = _write_composition(tmp_path / "long.mid", extend_melody_s=4.0)
    song = poprap_song(midi, cfg=CFG, progression=PROGRESSION, scale=Scale("A", "minor"))
    main, _ = lead_notes(song)
    pluck = pluck_notes(song)
    spans = [_span_s(song, i) for i in range(len(song.sections))]
    for notes in (main, pluck):
        assert notes
        for n in notes:
            span = next(s for s in spans if s[0] <= n.start_s < s[1])
            assert n.end_s <= span[1] + 1e-9
    assert any(n.end_s == pytest.approx(spans[1][1]) for n in main)  # actually cut


def test_pluck_covers_every_section_and_keeps_velocity(song):
    notes = pluck_notes(song)
    velocities = {n.velocity for n in song.parts.accompaniment}
    assert {n.velocity for n in notes} <= velocities
    for i in range(len(song.sections)):
        start, end = _span_s(song, i)
        assert any(start <= n.start_s < end for n in notes)
    assert max(n.start_s for n in notes) < song.body_s


def test_no_drums_in_the_verse_drop_bar_or_the_outro(song):
    hits = drum_hits(song, DRUMS)
    bar_s = song.grid.bar_s
    verse_start, verse_end = _span_s(song, 2)
    outro_start, outro_end = _span_s(song, 4)
    assert not [h for h in hits if verse_end - bar_s <= h.time_s < verse_end and h.kind != "riser"]
    assert any(verse_start <= h.time_s < verse_end - bar_s for h in hits)
    assert not [h for h in hits if outro_start <= h.time_s < outro_end]
    intro = [h for h in hits if h.time_s < 8 * bar_s]
    assert not [h for h in intro if h.time_s < 4 * bar_s and h.kind != "riser"]
    assert {h.kind for h in intro if h.time_s >= 4 * bar_s} <= {"hat", "riser"}
    risers = [h for h in hits if h.kind == "riser"]
    assert len(risers) == 1
    assert risers[0].time_s == pytest.approx(6 * bar_s)
    assert risers[0].length_s == pytest.approx(2 * bar_s)
    assert risers[0].velocity == DRUMS.riser_gain


def test_kicks_and_claps_are_on_the_16th_grid(song):
    hits = drum_hits(song, DRUMS)
    step_s = song.grid.step_s
    on_grid = [h for h in hits if h.kind in ("kick", "clap")]
    assert on_grid
    for h in on_grid:
        ratio = h.time_s / step_s
        assert abs(ratio - round(ratio)) < 1e-9
    claps = [h for h in hits if h.kind == "clap" and h.velocity == DRUMS.clap_gain]
    assert claps
    assert all(round(h.time_s / step_s) % 16 == DRUMS.clap_step for h in claps)
    ghosts = [h for h in hits if h.kind == "clap" and h.velocity == DRUMS.ghost_clap_gain]
    assert ghosts
    for h in ghosts:
        bar = _bar_of(song, h.time_s)
        section = next(s for s in song.sections if s.start_bar <= bar < s.start_bar + s.bars)
        assert (bar - section.start_bar) % 4 == 3
        assert round(h.time_s / step_s) % 16 == DRUMS.ghost_clap_step


def test_kick_carries_the_bar_root_and_runs_to_the_next_kick(song):
    hits = drum_hits(song, DRUMS)
    kicks = [h for h in hits if h.kind == "kick"]
    assert kicks
    step_s = song.grid.step_s
    for h in kicks:
        bar = _bar_of(song, h.time_s)
        section = next(s for s in song.sections if s.start_bar <= bar < s.start_bar + s.bars)
        b = bar - section.start_bar
        assert h.pitch == song.parts.bar_roots[song.pass_bar(b)]
        steps = DRUMS.kick_steps_even if b % 2 == 0 else DRUMS.kick_steps_odd
        step = round(h.time_s / step_s) % 16
        later = [s for s in steps if s > step]
        assert h.length_s == pytest.approx(((later[0] if later else 16) - step) * step_s)
        assert h.velocity == 1.0


def test_hats_rolls_open_hat_and_crash(song):
    hits = drum_hits(song, DRUMS)
    bar_s, step_s = song.grid.bar_s, song.grid.step_s
    hook = [h for h in hits if 8 * bar_s <= h.time_s < 24 * bar_s]
    # bar 0 of hook 1: a crash, 8th hats, accents on the beat
    first = [h for h in hook if h.time_s < 8 * bar_s + bar_s]
    assert [h.kind for h in first].count("crash") == 1
    crash = next(h for h in first if h.kind == "crash")
    assert (crash.time_s, crash.pan) == (8 * bar_s, DRUMS.crash_pan)
    hats = [h for h in first if h.kind == "hat"]
    assert [round((h.time_s - 8 * bar_s) / step_s) for h in hats] == list(range(0, 16, 2))
    assert [h.velocity for h in hats[:3]] == [DRUMS.hat_accent, DRUMS.hat_soft, DRUMS.hat_accent]
    assert all(h.pan == DRUMS.hat_pan for h in hats)
    # bar 1 (odd): 8ths to step 10, then 16ths from step 12
    odd = [h for h in hook if bar_s * 9 <= h.time_s < bar_s * 10 and h.kind == "hat"]
    assert [round((h.time_s - 9 * bar_s) / step_s) for h in odd] == [
        0,
        2,
        4,
        6,
        8,
        10,
        12,
        13,
        14,
        15,
    ]
    # bar 3 (b % 4 == 3): a triplet roll of six from beat 3, spaced beat/6
    beat = song.grid.beat_s
    bar3 = [h for h in hook if 11 * bar_s <= h.time_s < 12 * bar_s and h.kind == "hat"]
    expected = [
        (11 * bar_s + 3 * beat + k * beat / 6, DRUMS.roll_velocity + DRUMS.roll_velocity_step * k)
        for k in range(DRUMS.roll_hits)
    ]
    for time_s, velocity in expected:
        assert any(
            abs(h.time_s - time_s) < 1e-9 and h.velocity == pytest.approx(velocity) for h in bar3
        )
    verse_bar = [h for h in hits if 24 * bar_s <= h.time_s < 25 * bar_s and h.kind == "hat"]
    assert len(verse_bar) == 8 + DRUMS.roll_hits  # the verse rolls in every drum bar
    # bar 2 (b % 4 == 2): an open hat at step 14
    open_hats = [h for h in hook if h.kind == "open_hat"]
    assert [round((h.time_s - 8 * bar_s) / step_s) % 16 for h in open_hats] == [14] * len(open_hats)
    assert all(
        h.velocity == DRUMS.open_hat_velocity and h.pan == DRUMS.open_hat_pan for h in open_hats
    )
    assert len(open_hats) == 4


def test_pad_follows_the_harmony_and_ends_home(song):
    events = pad_events(song, CFG)
    parts = song.parts
    bar_s = song.grid.bar_s
    body = events[:-1]
    assert body
    for e in body:
        bar = int(e.start_s // bar_s + 1e-9)
        pass_bar = song.pass_bar(
            bar
            - next(s.start_bar for s in song.sections if s.start_bar <= bar < s.start_bar + s.bars)
        )
        assert {p % 12 for p in e.pitches} <= parts.harmonies[pass_bar].tones
    final = events[-1]
    assert final.start_s == pytest.approx(song.body_s)
    assert final.length_s == pytest.approx(CFG.form.final_chord_bars * bar_s)
    assert final.gain == CFG.pad.final_chord_gain
    assert {p % 12 for p in final.pitches} == {p % 12 for p in parts.home_voicing} | {
        parts.home.root
    }
    # events start at each section start and tile the song without gaps
    starts = {round(e.start_s, 9) for e in body}
    assert {round(_span_s(song, i)[0], 9) for i in range(5)} <= starts
    for a, b in pairwise(body):
        assert a.start_s + a.length_s == pytest.approx(b.start_s)
