import json
import shutil

import pytest
import soundfile as sf

from music_decoder.compose.api import arrange
from music_decoder.compose.arrangement import build_midi
from music_decoder.productionize import Production, ProductionError, productionize
from music_decoder.productionize.api import resolve_song
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
