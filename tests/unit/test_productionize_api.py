import json
import shutil
from dataclasses import replace

import pytest
import soundfile as sf

from music_decoder.compose.api import arrange
from music_decoder.compose.arrangement import build_midi
from music_decoder.productionize import Production, ProductionError, productionize
from music_decoder.productionize.api import resolve_poprap_song, resolve_song
from music_decoder.productionize.config import PopRapConfig, PopRapFormConfig, ProductionConfig
from music_decoder.productionize.dsp import gain_to_db
from music_decoder.productionize.master import ceiling_gain
from music_decoder.productionize.poprap import PopRapSong
from music_decoder.productionize.poprap_mix import POPRAP_STEMS
from music_decoder.productionize.samples import SampleError
from music_decoder.types import ChordSymbol, Composition, Scale
from tests.productionize_helpers import SR, fake_store, tiny_song

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
PROGRESSION = [ChordSymbol.parse(c) for c in ("Am", "F", "C", "G")]


@pytest.fixture
def composition_dir(tmp_path):
    folder = tmp_path / "take"
    folder.mkdir()
    arr = arrange(Scale("A", "minor"), PROGRESSION, tempo_bpm=92.0, seed=7)
    build_midi(
        melody=arr.melody,
        voicings=arr.voicings,
        bars_per_chord=1,
        tempo_bpm=92.0,
        style="fingerstyle",
        accompaniment=arr.accompaniment,
        out_path=folder / "composition.mid",
    )
    return folder


def test_resolve_song_accepts_a_folder_a_file_and_a_song(composition_dir):
    from_dir = resolve_song(composition_dir)
    from_file = resolve_song(str(composition_dir / "composition.mid"))
    assert from_dir.piano == from_file.piano
    song = tiny_song()
    assert resolve_song(song) is song


def test_resolve_song_applies_title_key_and_progression(composition_dir):
    song = resolve_song(composition_dir, progression=PROGRESSION, title="Night Drive", key="Am")
    assert (song.title, song.key) == ("Night Drive", "Am")
    assert song.bar_roots == (33, 29, 36, 31)


def test_resolve_song_accepts_a_composition(composition_dir):
    composition = Composition(
        midi_path=composition_dir / "composition.mid",
        wav_path=composition_dir / "composition.wav",
        ascii_tab="",
        melody_notes=(),
        chord_voicings=(),
        metadata={"scale": "A:minor", "progression": ["Am", "F", "C", "G"], "tempo_bpm": 92.0},
    )
    assert resolve_song(composition).key == "Am"


def test_missing_source_is_a_clear_error(tmp_path):
    with pytest.raises(ProductionError, match="no composer MIDI file"):
        resolve_song(tmp_path / "nothing")
    with pytest.raises(ProductionError, match="no composer MIDI file"):
        resolve_song(tmp_path)  # a folder without composition.mid


def test_missing_samples_are_reported_before_anything_is_written(tmp_path):
    from music_decoder.productionize import api

    class Offline(api.SampleStore):
        def __init__(self, root=None):
            super().__init__(tmp_path / "empty", download=False)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(api, "SampleStore", Offline)
        with pytest.raises(SampleError, match="fetch-samples"):
            productionize(tiny_song(), out_dir=tmp_path / "out")
    assert not list((tmp_path / "out").glob("*.wav"))


@needs_ffmpeg
def test_productionize_writes_master_mp3_mix_stems_and_report(tmp_path):
    song = tiny_song()
    samples_dir = tmp_path / "samples"
    fake_store(samples_dir, song)
    stages = []
    result = productionize(
        song,
        out_dir=tmp_path / "out",
        name="tiny",
        samples_dir=samples_dir,
        progress=lambda stage, fraction: stages.append(stage),
    )
    assert isinstance(result, Production)
    assert stages == ["drums", "bass", "piano", "strings", "mix", "master", "export", "checks"]

    master = sf.info(str(result.wav_path))
    assert result.wav_path.name == "tiny.wav"
    assert (master.samplerate, master.subtype, master.channels) == (SR, "PCM_24", 2)
    assert master.duration == pytest.approx(song.body_s + 7.0 + 0.12, abs=1e-3)
    assert sf.info(str(result.mp3_path)).format == "MP3"
    assert sf.info(str(result.raw_mix_path)).subtype == "FLOAT"
    assert result.raw_mix_path.name == "pro_raw.wav"
    assert set(result.stem_paths) == {
        "drums", "bass", "piano", "strings", "kick", "snare", "hats", "crash"
    }  # fmt: skip
    for path in result.stem_paths.values():
        info = sf.info(str(path))
        assert info.subtype == "FLOAT" and info.frames == sf.info(str(result.raw_mix_path)).frames

    assert result.master.integrated_lufs == pytest.approx(-10.5, abs=0.3)
    assert result.master.true_peak_dbtp <= -1.92
    assert result.qc is not None
    by_name = {c.name: c for c in result.qc.checks}
    for name in ("format", "loudness (WAV)", "true peak (WAV)", "samples at full scale (MP3)",
                 "DC offset", "drum grid", "first sound above -60 dBFS"):  # fmt: skip
        assert by_name[name].passed, by_name[name]
    assert by_name["section RMS (dBFS)"].passed is None  # no reference master for this song

    report = json.loads((tmp_path / "out" / "tiny.json").read_text())
    assert report["metadata"]["title"] == "Tiny"
    assert report["metadata"]["credits"] == [
        "Piano: Salamander Grand Piano V3 by Alexander Holm (CC BY 3.0)"
    ]
    assert report["qc"]["checks"] and report["master"]["iterations"] >= 1


@needs_ffmpeg
def test_checks_can_be_skipped(tmp_path):
    song = tiny_song()
    fake_store(tmp_path / "samples", song)
    result = productionize(
        song, out_dir=tmp_path / "out", samples_dir=tmp_path / "samples", check=False
    )
    assert result.qc is None and result.wav_path.name == "production.wav"


SHORT_FORM = replace(PopRapFormConfig(), intro_bars=2, section_target_bars=8, outro_bars=1)
POPRAP = ProductionConfig(
    style="pop-rap", poprap=replace(PopRapConfig(lead="synth"), form=SHORT_FORM)
)


def _composition(folder):
    return Composition(
        midi_path=folder / "composition.mid",
        wav_path=folder / "composition.wav",
        ascii_tab="",
        melody_notes=(),
        chord_voicings=(),
        metadata={"scale": "A:minor", "progression": ["Am", "F", "C", "G"], "tempo_bpm": 92.0},
    )


def test_resolve_poprap_song_accepts_a_folder_a_file_and_a_composition(composition_dir):
    cfg = PopRapConfig()
    from_dir = resolve_poprap_song(composition_dir, cfg=cfg)
    from_file = resolve_poprap_song(str(composition_dir / "composition.mid"), cfg=cfg)
    assert isinstance(from_dir, PopRapSong) and from_dir.parts == from_file.parts
    from_composition = resolve_poprap_song(_composition(composition_dir), cfg=cfg)
    assert from_composition.key == "Am" and from_composition.grid.tempo_bpm == pytest.approx(92.0)


def test_resolve_poprap_song_applies_title_key_and_progression(composition_dir):
    song = resolve_poprap_song(
        composition_dir, cfg=PopRapConfig(), progression=PROGRESSION, title="Night Drive", key="Em"
    )
    assert (song.title, song.key) == ("Night Drive", "Em")
    assert song.parts.bar_roots == (33, 29, 36, 31)


def test_resolve_poprap_song_missing_source_is_a_clear_error(tmp_path):
    with pytest.raises(ProductionError, match="no composer MIDI file"):
        resolve_poprap_song(tmp_path, cfg=PopRapConfig())


def test_poprap_rejects_a_ready_song(tmp_path):
    with pytest.raises(ProductionError, match="pop-rap needs a composer file or composition"):
        productionize(tiny_song(), out_dir=tmp_path / "out", config=POPRAP)
    assert not (tmp_path / "out").exists()


def test_poprap_progression_must_divide_the_bars(composition_dir, tmp_path):
    three = PROGRESSION[:3]
    with pytest.raises(ProductionError):
        productionize(composition_dir, out_dir=tmp_path / "out", config=POPRAP, progression=three)


@needs_ffmpeg
def test_poprap_production_writes_master_mp3_stems_and_report(composition_dir, tmp_path):
    stages = []
    result = productionize(
        composition_dir,
        out_dir=tmp_path / "out",
        name="rap",
        config=POPRAP,
        progression=PROGRESSION,
        progress=lambda stage, fraction: stages.append(stage),
    )
    assert result.wav_path.is_file() and result.mp3_path.is_file()
    assert set(result.stem_paths) == set(POPRAP_STEMS)
    assert result.stem_paths["808"].name == "stem_808.wav"
    assert stages[-3:] == ["master", "export", "checks"]
    assert result.metadata["style"] == "pop-rap" and result.metadata["lead"] == "synth"
    assert result.metadata["verse_style"] is None
    assert result.metadata["percussion"] == [] and result.metadata["credits"] == []
    expected = result.metadata["duration_s"]
    assert sf.info(str(result.wav_path)).duration == pytest.approx(expected, abs=1e-3)
    assert result.qc is not None
    assert {c.name: c for c in result.qc.checks}["drum grid"].passed
    report = json.loads((tmp_path / "out" / "rap.json").read_text())
    assert report["metadata"]["style"] == "pop-rap"


@needs_ffmpeg
def test_boom_bap_metadata_has_style_and_no_lead(tmp_path):
    song = tiny_song()
    fake_store(tmp_path / "samples", song)
    result = productionize(
        song, out_dir=tmp_path / "out", samples_dir=tmp_path / "samples", check=False
    )
    assert result.metadata["style"] == "boom-bap" and result.metadata["lead"] is None
    assert list(result.metadata)[:9] == [
        "title", "verse_style", "key", "tempo_bpm", "duration_s",
        "sections", "notes", "percussion", "credits",
    ]  # fmt: skip
    report = json.loads((tmp_path / "out" / "production.json").read_text())
    assert report["metadata"]["lead"] is None


def _spy_master(monkeypatch):
    from music_decoder.productionize import api

    ceilings = []
    real = api.master

    def spy(mix, cfg, sr):
        ceilings.append(cfg.ceiling_dbtp)
        return real(mix, cfg, sr)

    monkeypatch.setattr(api, "master", spy)
    return api, ceilings


@needs_ffmpeg
def test_poprap_mp3_guard_remasters_with_a_lower_ceiling(composition_dir, tmp_path, monkeypatch):
    api, ceilings = _spy_master(monkeypatch)
    measured = iter([-0.5, -1.4])  # hot, then passing
    monkeypatch.setattr(api, "mp3_true_peak_dbtp", lambda mp3, cfg: next(measured))
    result = productionize(
        composition_dir, out_dir=tmp_path / "out", config=POPRAP, progression=PROGRESSION
    )
    margin = POPRAP.poprap.mp3_guard_margin_db
    threshold = POPRAP.qc.mp3_true_peak_max_dbtp - margin
    excess = -0.5 - threshold
    assert len(ceilings) == 2
    assert ceilings[0] == POPRAP.master.ceiling_dbtp
    assert ceilings[1] == pytest.approx(ceilings[0] - (excess + margin), abs=1e-12)
    assert result.metadata["mp3_guard_passes"] == 1
    final = replace(POPRAP.master, ceiling_dbtp=ceilings[1])
    assert result.master.true_peak_dbtp == pytest.approx(
        gain_to_db(ceiling_gain(final)), abs=0.01
    )  # the report is the second master's
    report = json.loads((tmp_path / "out" / "production.json").read_text())
    assert report["metadata"]["mp3_guard_passes"] == 1
    assert report["master"]["true_peak_dbtp"] == pytest.approx(result.master.true_peak_dbtp)


@needs_ffmpeg
def test_poprap_mp3_guard_stops_after_its_attempts(composition_dir, tmp_path, monkeypatch):
    api, ceilings = _spy_master(monkeypatch)
    monkeypatch.setattr(api, "mp3_true_peak_dbtp", lambda mp3, cfg: -0.5)
    result = productionize(
        composition_dir,
        out_dir=tmp_path / "out",
        config=POPRAP,
        progression=PROGRESSION,
        check=False,
    )
    assert len(ceilings) == 1 + POPRAP.poprap.mp3_guard_attempts
    assert result.metadata["mp3_guard_passes"] == POPRAP.poprap.mp3_guard_attempts


@needs_ffmpeg
def test_boom_bap_never_measures_or_remasters_the_mp3(tmp_path, monkeypatch):
    api, ceilings = _spy_master(monkeypatch)

    def fail(mp3, cfg):
        raise AssertionError("boom-bap must not run the MP3 guard")

    monkeypatch.setattr(api, "mp3_true_peak_dbtp", fail)
    song = tiny_song()
    fake_store(tmp_path / "samples", song)
    result = productionize(
        song, out_dir=tmp_path / "out", samples_dir=tmp_path / "samples", check=False
    )
    assert len(ceilings) == 1
    assert "mp3_guard_passes" not in result.metadata
