import re
from pathlib import Path
from unittest import mock

import pretty_midi
import pytest
import soundfile

from music_decoder.chords.templates import QUALITIES, ROOTS, chord_label
from music_decoder.compose import metrics
from music_decoder.compose.api import (
    COMPOSE_TAB_WEIGHTS,
    COMPOSER_VERSION,
    _melody_floor,
    _melody_tab,
    arrange,
    compose,
    compose_with_options,
)
from music_decoder.compose.melody import MAX_INTERVAL, generate_melody
from music_decoder.compose.tabtext import render_melody_tab
from music_decoder.compose.voicings import shape_faults, voicings_for
from music_decoder.errors import InvalidProgressionError, InvalidScaleError
from music_decoder.tabs import _DEFAULT_WEIGHTS
from music_decoder.tabs.tuning import PRESETS, STANDARD_EADGBE, get_preset
from music_decoder.types import VALID_TONICS, ChordSymbol, Note, Scale


def test_compose_returns_paths_and_data(tmp_path):
    with mock.patch("music_decoder.compose.api.render_wav") as render:
        render.side_effect = lambda midi_path, out_path: out_path.write_bytes(b"RIFFWAV")
        result = compose(
            scale=Scale("C", "major"),
            progression=[ChordSymbol.parse(c) for c in ("Cmaj7", "Am7", "Dm7", "G7")],
            tempo_bpm=120.0,
            bars_per_chord=1,
            style="strum",
            seed=7,
            out_dir=tmp_path,
        )
    assert result.midi_path.exists()
    assert result.wav_path.exists()
    assert result.ascii_tab.strip() != ""
    assert len(result.melody_notes) > 0
    assert len(result.chord_voicings) == 4
    assert result.metadata["seed"] == 7


def test_compose_invalid_progression_raises():
    with pytest.raises(InvalidProgressionError):
        compose(
            scale=Scale("C", "major"),
            progression=[],  # empty
            out_dir=Path("/tmp"),
        )


def test_compose_invalid_scale_raises():
    # non-Literal mode is only constructible by going around the parser:
    bad = Scale(tonic="C", mode="dorian")  # type: ignore[arg-type]
    with pytest.raises(InvalidScaleError):
        compose(scale=bad, progression=[ChordSymbol.parse("C")], out_dir=Path("/tmp"))


# ---- arrange(): the filesystem-free core of compose() ----------------------

_PROG = [ChordSymbol.parse(c) for c in ("Cmaj7", "Am7", "Dm7", "G7")]


def _chords(*labels):
    return [ChordSymbol.parse(c) for c in labels]


def _decode(block):
    """(string, fret) per melody note from a rendered melody tab block."""
    out = []
    for system in block.split("\n\n"):
        rows = system.splitlines()[::-1]
        hits = []
        for s, row in enumerate(rows):
            body = row.split("|", 1)[1]
            hits.extend((m.start(), s, int(m.group())) for m in re.finditer(r"\d+", body))
        out.extend((s, f) for _, s, f in sorted(hits))
    return out


def test_arrange_is_deterministic_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    a = arrange(Scale("C", "major"), _PROG, tempo_bpm=120.0, style="strum", seed=3)
    b = arrange(Scale("C", "major"), _PROG, tempo_bpm=120.0, style="strum", seed=3)
    assert a == b
    assert list(tmp_path.iterdir()) == []
    assert len(a.voicings) == 4 and a.accompaniment


def test_arrange_reads_no_configuration():
    with (
        mock.patch("music_decoder.config.hyperparameters.load_hyperparameters") as load,
        mock.patch("music_decoder.tabs.assign_tabs") as configurable,
    ):
        arrange(Scale("C", "major"), _PROG, seed=3)
    load.assert_not_called()
    configurable.assert_not_called()


def test_arrange_ignores_the_analysis_hyperparameters_file(tmp_path, monkeypatch):
    before = arrange(Scale("G", "major"), _PROG, seed=5)
    other = tmp_path / "hp.yaml"
    other.write_text("id: test\ntab_assignment:\n  weights: {w_move: 9.0}\n  max_fret: 5\n")
    monkeypatch.setenv("MUSIC_DECODER_HYPERPARAMETERS", str(other))
    assert arrange(Scale("G", "major"), _PROG, seed=5) == before


def test_pinned_tab_weights_match_the_analysis_defaults():
    assert COMPOSE_TAB_WEIGHTS == _DEFAULT_WEIGHTS


def test_compose_returns_the_arrangement_and_version(tmp_path):
    with mock.patch("music_decoder.compose.api.render_wav"):
        result = compose(scale=Scale("C", "major"), progression=_PROG, seed=1, out_dir=tmp_path)
    # compose() uses the tune engine since composer v6.
    arr = arrange(Scale("C", "major"), _PROG, seed=1, candidates=12, melody_engine="tune")
    assert result.melody_notes == arr.melody
    assert result.chord_voicings == arr.voicings
    assert result.metadata["composer_version"] == COMPOSER_VERSION == 6


def test_arrange_tabs_every_melody_note_in_a_playable_register():
    arr = arrange(Scale("G", "major"), _PROG, bars_per_chord=2, seed=9)
    assert [tn.note for tn in arr.melody_tab] == list(arr.melody)
    for tn in arr.melody_tab:
        assert STANDARD_EADGBE.open_pitches[tn.position.string] + tn.position.fret == tn.note.pitch
    assert max(tn.position.fret for tn in arr.melody_tab) <= 15


def test_melody_tab_falls_back_when_astar_drops_a_note():
    melody = [
        Note(start_s=0.0, end_s=0.4, pitch=64, velocity=80, confidence=1.0),
        Note(start_s=0.5, end_s=0.9, pitch=67, velocity=80, confidence=1.0),
    ]
    empty = mock.Mock(tabbed_notes=[])
    with mock.patch("music_decoder.compose.api.assign_tab", return_value=empty):
        tab = _melody_tab(melody, STANDARD_EADGBE)
    assert [(t.position.string, t.position.fret) for t in tab] == [(5, 0), (5, 3)]


@pytest.mark.parametrize("bars", [0, -1])
def test_bars_per_chord_below_one_raises(bars):
    with pytest.raises(InvalidProgressionError):
        arrange(Scale("C", "major"), _PROG, bars_per_chord=bars)


@pytest.mark.parametrize("tuning_name", ["Drop_D", "DADGAD", "D_standard"])
def test_arrange_in_other_tunings_tabs_every_note(tuning_name):
    tuning = PRESETS[tuning_name]
    arr = arrange(Scale("D", "major"), _chords("D", "G", "A", "D"), tuning=tuning, seed=2)
    assert len(arr.melody_tab) == len(arr.melody)
    for tn in arr.melody_tab:
        assert tuning.open_pitches[tn.position.string] + tn.position.fret == tn.note.pitch


def test_melody_floor_sits_a_minor_third_above_the_defining_tones():
    d = voicings_for(ChordSymbol.parse("D"), STANDARD_EADGBE)[0]  # D3 A3 D4 F#4
    assert _melody_floor(d, STANDARD_EADGBE) == 66 + 3
    g7 = voicings_for(ChordSymbol.parse("G7"), STANDARD_EADGBE)[0]  # G2 B2 D3 G3 B3 F4
    assert _melody_floor(g7, STANDARD_EADGBE) == 65 + 3  # the seventh F4 is the highest


def test_d_major_cadence_has_its_third_under_the_melody():
    # review reproduction: D G A D, seed 9
    arr = arrange(Scale("D", "major"), _chords("D", "G", "A", "D"), tempo_bpm=120.0, seed=9)
    final_melody = [n for n in arr.melody if n.start_s >= 6.0]
    final_acc = [n for n in arr.accompaniment if n.start_s >= 6.0 - 1e-9]
    assert 6 in {n.pitch % 12 for n in final_acc}  # F# is heard
    assert max(n.pitch for n in final_acc) < min(n.pitch for n in final_melody)
    assert metrics.final_resolution(arr.melody, Scale("D", "major"), _chords("D", "G", "A", "D"))


@pytest.mark.parametrize(
    ("key", "labels", "bars", "seed"),
    [
        ("C", ("C", "G", "Am", "F"), 1, 0),  # review reproduction
        ("A#", ("A#", "Gm", "D#", "F", "Cm", "F", "A#", "A#"), 2, 1004),  # 16 bars
    ],
)
def test_rendered_tab_shows_every_melody_position(tmp_path, key, labels, bars, seed):
    with mock.patch("music_decoder.compose.api.render_wav"):
        result = compose(
            scale=Scale(key, "major"),
            progression=_chords(*labels),
            bars_per_chord=bars,
            tempo_bpm=120.0,
            style="strum",
            seed=seed,
            out_dir=tmp_path,
        )
    arr = arrange(
        Scale(key, "major"),
        _chords(*labels),
        bars_per_chord=bars,
        tempo_bpm=120.0,
        style="strum",
        seed=seed,
    )
    melody_block = result.ascii_tab.split("\n\nChords\n")[0].removeprefix("Melody\n")
    expected = [(tn.position.string, tn.position.fret) for tn in arr.melody_tab]
    assert _decode(melody_block) == expected
    assert melody_block == render_melody_tab(
        arr.melody_tab, n_bars=len(labels) * bars, tempo_bpm=120.0, tuning=STANDARD_EADGBE
    )


@pytest.mark.parametrize(
    ("key", "mode", "labels", "seed"),
    [
        ("C", "major", ("C", "Am", "F", "G7"), 16),  # review r2: 65 -> 79 jump
        ("C", "major", ("C", "G", "Am", "G7"), 16),
        ("A", "major", ("A", "Dsus4"), 0),  # review r2: no cadence target
        ("A", "minor", ("Am", "Dsus4"), 0),
        ("A", "major", ("Dsus4",), 0),
    ],
)
def test_review_reproductions_compose_within_the_invariants(key, mode, labels, seed):
    progression = _chords(*labels)
    arr = arrange(Scale(key, mode), progression, tempo_bpm=120.0, seed=seed)
    assert metrics.max_interval(arr.melody) <= MAX_INTERVAL
    assert metrics.final_resolution(arr.melody, Scale(key, mode), progression)
    assert all(
        metrics.essential_tone_coverage(
            arr.melody, arr.accompaniment, arr.voicings, 1, 120.0, STANDARD_EADGBE
        )
    )


@pytest.mark.parametrize(
    "labels",
    [("C", "Am", "F", "G7"), ("C", "G", "Am", "G7"), ("C", "F", "Csus4"), ("C", "F", "Dm7")],
)
def test_floor_coupled_sweep_keeps_the_octave_limit(labels):
    # the generator with the floors arrange() would pass, without the A* cost
    progression = _chords(*labels)
    voicings = [voicings_for(c, STANDARD_EADGBE)[0] for c in progression]
    floors = [_melody_floor(v, STANDARD_EADGBE) for v in voicings]
    for seed in range(150):
        notes = generate_melody(
            scale=Scale("C", "major"), progression=progression, seed=seed, chord_floors=floors
        )
        assert metrics.max_interval(notes) <= MAX_INTERVAL, seed
        assert metrics.final_resolution(notes, Scale("C", "major"), progression), seed


_QUALITY_SUFFIXES = ("", "m", "7", "maj7", "m7", "dim", "sus4", "aug")

# Spec 5.3 allows a counted, listed final-voicing fallback when no voicing
# exists. (key, mode, labels, seed, tuning) of the one such case: no shape the
# design allows voices D#dim's F# and A under a D# on the B or e string.
FINAL_FALLBACK_CASES = frozenset({("C", "major", ("C", "D#dim"), 7, "EADGBE")})


@pytest.mark.parametrize(("key", "mode", "home"), [("C", "major", "C"), ("A", "minor", "Am")])
def test_every_chord_quality_on_every_root_composes(key, mode, home):
    scale = Scale(key, mode)
    for root in VALID_TONICS:
        for suffix in _QUALITY_SUFFIXES:
            progression = _chords(home, root + suffix)
            arr = arrange(scale, progression, tempo_bpm=120.0, seed=7)
            label = root + suffix
            assert metrics.max_interval(arr.melody) <= MAX_INTERVAL, label
            assert metrics.final_resolution(arr.melody, scale, progression), label
            fallback = (key, mode, (home, label), 7, STANDARD_EADGBE.name) in FINAL_FALLBACK_CASES
            assert arr.final_fallback is fallback, label
            if fallback:
                continue  # the counted fallback: H1 is not checked (spec 5.3)
            coverage = metrics.essential_tone_coverage(
                arr.melody, arr.accompaniment, arr.voicings, 1, 120.0, STANDARD_EADGBE
            )
            assert all(coverage), label


@pytest.mark.parametrize("style", ["arpeggio", "strum", "fingerstyle"])
@pytest.mark.parametrize(
    ("key", "mode", "labels", "seed"),
    [
        ("C", "major", ("C", "Am", "F", "G7"), 5),  # review r3: G7 ending lost its F
        ("A", "major", ("A", "Dsus4"), 1),  # review r3: Dsus4 ending lost its G
        ("A", "minor", ("Am", "Dsus4"), 1),
        ("C", "major", ("C", "Fm", "C", "G7"), 1006),  # from the edge matrix
        ("A", "major", ("A", "Dsus4"), 1002),
    ],
)
def test_sustained_cadence_keeps_the_chord_identity(key, mode, labels, seed, style):
    progression = _chords(*labels)
    arr = arrange(Scale(key, mode), progression, tempo_bpm=120.0, style=style, seed=seed)
    assert metrics.cadence_identity(arr.melody, arr.accompaniment, arr.voicings, STANDARD_EADGBE)
    assert metrics.final_chord_rings(arr.accompaniment, len(labels), 120.0)
    assert metrics.max_interval(arr.melody) <= MAX_INTERVAL


@pytest.mark.parametrize("style", ["arpeggio", "strum", "fingerstyle"])
@pytest.mark.parametrize(
    ("tuning_name", "labels"),
    [
        ("Drop_C", ("C", "G", "Am", "G7", "Gmaj7", "C")),
        ("Drop_D", ("C", "G7", "C")),
        ("DADGAD", ("D", "A#", "D#dim", "A#sus4", "Bmaj7", "D")),
        ("Eb", ("C", "C7", "F", "B7", "C")),
    ],
)
def test_searched_voicings_compose_with_their_identity_heard(tuning_name, labels, style):
    tuning = get_preset(tuning_name)
    arr = arrange(Scale(labels[0], "major"), _chords(*labels), style=style, tuning=tuning, seed=3)
    assert all(
        shape_faults([p.fret for p in v.positions], tuning, v.chord) == () for v in arr.voicings
    )
    assert all(
        metrics.essential_tone_coverage(
            arr.melody, arr.accompaniment, arr.voicings, 1, 100.0, tuning
        )
    )
    assert metrics.cadence_identity(arr.melody, arr.accompaniment, arr.voicings, tuning)


# ---- singable leaps (composer v3): real floors from arrange() --------------


def _api_leap_invariants(arr, labels, tempo=100.0):
    mel, p = arr.melody, _chords(*labels)
    assert metrics.forbidden_interval_count(mel) == 0
    assert metrics.unjustified_large_leap_count(mel, p, 1, tempo) == 0
    leaps, recovered = metrics.large_leap_recovery_counts(mel)
    assert recovered == leaps
    assert max(metrics.chord_change_moves(mel, p, 1, tempo), default=0) <= 7
    assert abs(mel[-1].pitch - mel[-2].pitch) <= 7


def test_review_r1_major_seventh_is_repaired():
    arr = arrange(Scale("C", "major"), _chords("C", "D#m"), seed=19)  # v2 and rev. 1: 64 -> 75
    _api_leap_invariants(arr, ("C", "D#m"))
    assert arr.melody[-1].pitch % 12 == 3  # still ends on D#, the root of D#m


@pytest.mark.parametrize(
    ("key", "mode", "labels", "seed"),
    [
        ("C", "major", ("C", "D#m"), 20),
        ("C", "major", ("C", "D#m7"), 20),
        ("A", "minor", ("Am", "D#m"), 20),
        ("A", "minor", ("Am", "D#m7"), 20),
    ],
)
def test_revision_1_sweep_failures_are_repaired(key, mode, labels, seed):
    _api_leap_invariants(arrange(Scale(key, mode), _chords(*labels), seed=seed), labels)


def test_every_root_and_quality_composes_smoothly():  # spec section 6.4 bounded sweep
    for key, mode, home in (("C", "major", "C"), ("A", "minor", "Am")):
        for root in ROOTS:
            for quality in QUALITIES:
                x = chord_label(root, quality)
                for labels in ((home, x), (home, x, home)):
                    for seed in range(5):
                        arr = arrange(Scale(key, mode), _chords(*labels), seed=seed)
                        _api_leap_invariants(arr, labels)


def test_compose_writes_on_grid_midi_and_records_the_sound(tmp_path):
    with mock.patch("music_decoder.compose.api.render_wav") as render:
        render.side_effect = lambda midi_path, out_path: out_path.write_bytes(b"RIFF")
        result = compose_with_options(Scale("C", "major"), _PROG, seed=1, out_dir=tmp_path)
    pm = pretty_midi.PrettyMIDI(str(result.midi_path))
    starts = [n.start for i in pm.instruments for n in i.notes]
    assert metrics.onset_grid_ratio(metrics.stroke_heads(starts), 100.0, 2) == 1.0
    assert result.metadata["sound"] == "soundfont" and result.metadata["composer_version"] == 6
    assert not list(result.midi_path.parent.glob("*performance*"))  # temp MIDI removed


def test_compose_renders_fingerstyle_from_the_humanized_performance_midi(tmp_path):
    rendered = []

    def fake(midi_path, out_path):
        rendered.append(midi_path)
        assert midi_path.exists()
        out_path.write_bytes(b"RIFF")

    with mock.patch("music_decoder.compose.api.render_wav", side_effect=fake):
        result = compose_with_options(Scale("C", "major"), _PROG, seed=1, out_dir=tmp_path)
    assert rendered[0] != result.midi_path and "performance" in rendered[0].name
    assert not rendered[0].exists()


def test_compose_renders_strum_from_composition_midi(tmp_path):
    rendered = []
    with mock.patch("music_decoder.compose.api.render_wav") as render:
        render.side_effect = lambda m, o: (rendered.append(m), o.write_bytes(b"RIFF"))
        result = compose_with_options(
            Scale("C", "major"), _PROG, style="strum", seed=1, out_dir=tmp_path
        )
    assert rendered == [result.midi_path]


@pytest.mark.parametrize("tuning_name", list(PRESETS))
def test_compose_with_plucked_sound_in_every_tuning(tmp_path, tuning_name):
    r = compose_with_options(
        Scale("D", "major"),
        [ChordSymbol.parse(c) for c in ("D", "G", "A", "D")],
        tuning=PRESETS[tuning_name],
        seed=2,
        out_dir=tmp_path,
        sound="plucked",
    )
    y, _ = soundfile.read(r.wav_path)
    assert len(y) > 0 and r.metadata["sound"] == "plucked"


def test_plucked_renders_strum_from_midi(tmp_path):
    r = compose_with_options(
        Scale("C", "major"), _PROG, style="strum", seed=1, out_dir=tmp_path, sound="plucked"
    )
    assert soundfile.info(r.wav_path).duration > 0


def test_plucked_sound_leaves_composition_midi_unchanged(tmp_path):
    with mock.patch("music_decoder.compose.api.render_wav") as render:
        render.side_effect = lambda midi_path, out_path: out_path.write_bytes(b"RIFF")
        a = compose_with_options(Scale("C", "major"), _PROG, seed=1, out_dir=tmp_path / "a")
    b = compose_with_options(
        Scale("C", "major"), _PROG, seed=1, out_dir=tmp_path / "b", sound="plucked"
    )
    assert a.midi_path.read_bytes() == b.midi_path.read_bytes()
    assert a.ascii_tab == b.ascii_tab
    assert not list(b.midi_path.parent.glob("*performance*"))


def test_compose_delegates_to_compose_with_options(tmp_path):
    with mock.patch("music_decoder.compose.api.render_wav") as render:
        render.side_effect = lambda midi_path, out_path: out_path.write_bytes(b"RIFF")
        a = compose(Scale("C", "major"), _PROG, seed=1, out_dir=tmp_path / "a")
        b = compose_with_options(Scale("C", "major"), _PROG, seed=1, out_dir=tmp_path / "b")
    assert a.melody_notes == b.melody_notes and a.ascii_tab == b.ascii_tab


_PROG8 = _chords("C", "Am", "F", "G", "C", "F", "G", "C")


def test_compose_records_the_choice(tmp_path):
    with mock.patch("music_decoder.compose.api.render_wav") as render:
        render.side_effect = lambda midi_path, out_path: out_path.write_bytes(b"RIFF")
        r = compose_with_options(
            Scale("C", "major"), _PROG8, seed=4, out_dir=tmp_path, melody_engine="rules"
        )
    m = r.metadata
    assert m["candidates"] == 12 and isinstance(m["chosen_candidate"], int)
    assert isinstance(m["score"], float) and m["final_fallback"] is False
    assert isinstance(m["identity_misses"], list)
    best = arrange(Scale("C", "major"), _PROG8, seed=4, candidates=12)
    assert m["chosen_candidate"] == best.chosen_candidate and m["score"] == best.score
    assert r.melody_notes == best.melody


def test_compose_strum_records_one_candidate(tmp_path):
    with mock.patch("music_decoder.compose.api.render_wav") as render:
        render.side_effect = lambda midi_path, out_path: out_path.write_bytes(b"RIFF")
        r = compose_with_options(
            Scale("C", "major"), _PROG8, style="strum", seed=4, out_dir=tmp_path
        )
    m = r.metadata
    assert m["candidates"] == 1 and m["chosen_candidate"] == 0 and m["score"] is None
    assert m["final_fallback"] is False and m["identity_misses"] is None  # not counted


def _silent_render(midi_path, out_path):
    out_path.write_bytes(b"RIFF")


@pytest.mark.parametrize("style", ["fingerstyle", "strum"])
def test_compose_records_the_seed_it_used_and_rebuilds_from_it(tmp_path, style):
    with mock.patch("music_decoder.compose.api.render_wav", side_effect=_silent_render):
        given = compose_with_options(
            Scale("C", "major"), _PROG8, style=style, seed=4, out_dir=tmp_path, candidates=2
        )
        assert given.metadata["seed"] == 4 and given.metadata["seed_used"] == 4
        drawn = compose_with_options(
            Scale("C", "major"), _PROG8, style=style, out_dir=tmp_path, candidates=2
        )
        used = drawn.metadata["seed_used"]
        assert drawn.metadata["seed"] is None and isinstance(used, int)
        again = compose_with_options(
            Scale("C", "major"), _PROG8, style=style, seed=used, out_dir=tmp_path, candidates=2
        )
    assert again.melody_notes == drawn.melody_notes and again.ascii_tab == drawn.ascii_tab
    assert again.midi_path.read_bytes() == drawn.midi_path.read_bytes()


def test_compose_validates_before_loading_any_model(tmp_path, monkeypatch):
    from music_decoder.compose.ml import aesthetics, amt

    def fail():
        raise AssertionError("no model may load for invalid arguments")

    monkeypatch.setattr(aesthetics, "load_scorer", fail)
    monkeypatch.setattr(amt, "load_proposer", fail)
    for kw in ({"rank_audio": True}, {"melody_model": "amt"}):
        with pytest.raises(ValueError, match="candidates must be >= 1"):
            compose_with_options(Scale("C", "major"), _PROG8, out_dir=tmp_path, candidates=0, **kw)
        with pytest.raises(InvalidProgressionError):
            compose_with_options(Scale("C", "major"), [], out_dir=tmp_path, **kw)
        with pytest.raises(InvalidProgressionError):
            compose_with_options(
                Scale("C", "major"), _PROG8, bars_per_chord=0, out_dir=tmp_path, **kw
            )
    assert list(tmp_path.iterdir()) == []


def _compose_manifest_take(tmp_path):
    with mock.patch("music_decoder.compose.api.render_wav") as render:
        render.side_effect = lambda midi_path, out_path: out_path.write_bytes(b"RIFFWAV")
        return compose(
            Scale("C", "major"),
            [ChordSymbol.parse(c) for c in ("Dm", "G", "C", "Am")],
            tempo_bpm=96.0,
            seed=7,
            out_dir=tmp_path,
        )


def test_compose_writes_a_manifest(tmp_path):
    import json

    result = _compose_manifest_take(tmp_path)
    manifest = result.midi_path.with_name("composition.json")
    assert manifest.is_file()
    data = json.loads(manifest.read_text())
    assert data["manifest_version"] == 1
    expected = json.loads(json.dumps(result.metadata))
    assert {k: data[k] for k in result.metadata} == expected
