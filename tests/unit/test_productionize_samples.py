import io
import itertools

import numpy as np
import pytest
import soundfile as sf

from music_decoder.productionize import samples
from music_decoder.productionize.config import SHAKER, DrumsConfig, PianoConfig
from music_decoder.productionize.samples import (
    RoundRobin,
    SampleError,
    SampleRef,
    SampleStore,
    drum_ref,
    drum_refs,
    drum_takes,
    note_name,
    percussion_ref,
    percussion_refs,
    piano_ref,
    piano_refs,
    resample,
)


def test_note_names():
    assert [note_name(p) for p in (21, 30, 60, 66, 75, 108)] == [
        "A0",
        "F#1",
        "C4",
        "F#4",
        "D#5",
        "C8",
    ]


def test_piano_ref_escapes_the_sharp_in_the_url_and_stores_it_as_s():
    ref = piano_ref(66, 8)
    assert ref.url.endswith("/Samples/F%234v8.flac")
    assert ref.path == "piano/Fs4v8.flac"
    assert samples.SALAMANDER_COMMIT in ref.url
    assert piano_ref(60, 1).path == "piano/C4v1.flac"


def test_drum_refs_follow_the_library_layout():
    cfg = DrumsConfig()
    kick = drum_ref(cfg.kick, "oh", 3, 1)
    assert kick.url.endswith("/Samples/oh/kick/oh_kick_snoff_vl3_rr1.flac")
    assert kick.path == "drums/oh_kick_snoff_vl3_rr1.flac"
    snare = drum_ref(cfg.snare, "snaremic", 18, None)  # the snare has no round robins
    assert snare.url.endswith("/Samples/snaremic/snare/snaremic_snare_center_vl18.flac")
    hat = drum_ref(cfg.hat_open, "room", 2, 3)
    assert hat.url.endswith("/Samples/room/hh/room_hh_open_vl2_rr3.flac")


def test_drum_refs_cover_every_take_once():
    refs = drum_refs(DrumsConfig())
    paths = [r.path for r in refs]
    assert len(paths) == len(set(paths))
    # kick 2 layers x 4 x 3 mics, snare 12 layers x 3 mics, hats 2x4x2 + 3x2, crash 2x4x2
    assert len(paths) == 24 + 36 + 16 + 6 + 16


def test_drum_takes():
    cfg = DrumsConfig()
    assert drum_takes(cfg.kick, (3,)) == [(3, 1), (3, 2), (3, 3), (3, 4)]
    assert drum_takes(cfg.snare, (12, 13)) == [(12, None), (13, None)]


def test_percussion_refs():
    ref = percussion_ref(SHAKER.voices[0], 7)
    assert ref.url.endswith("/Samples/perc/close/shaker/LShaker_Shake1D_rr7_Close.wav")
    assert ref.path == "perc/LShaker_Shake1D_rr7_Close.wav"
    assert len(percussion_refs(["shaker", "tambourine", "congas"])) == 48


def test_piano_refs_default_to_every_sampled_pitch_and_13_layers():
    cfg = PianoConfig()
    assert len(cfg.sample_pitches) == 30
    assert len(piano_refs(cfg)) == 30 * 13
    assert len(piano_refs(cfg, [30, 36, 30])) == 2 * 13


def test_round_robin_never_repeats_a_take():
    rng = np.random.default_rng(2026)
    for n in (2, 3, 4, 7):
        robin = RoundRobin(n, rng)
        picks = [robin.next() for _ in range(500)]
        assert all(a != b for a, b in itertools.pairwise(picks))
        assert set(picks) == set(range(n))
    assert [RoundRobin(1, rng).next() for _ in range(3)] == [0, 0, 0]
    with pytest.raises(ValueError):
        RoundRobin(0, rng)


def test_resample_changes_rate_and_pitch():
    sr = 48_000
    t = np.arange(sr) / sr
    x = np.sin(2 * np.pi * 440.0 * t)[:, None]
    same_pitch = resample(x, sr, 44_100)
    assert len(same_pitch) == 44_100
    up = resample(x, sr, 44_100, semitones=1)
    assert len(up) == pytest.approx(44_100 / 2 ** (1 / 12), rel=2e-3)
    spectrum = np.abs(np.fft.rfft(up[:, 0] * np.hanning(len(up))))
    peak_hz = np.argmax(spectrum) * 44_100 / len(up)
    assert peak_hz == pytest.approx(440.0 * 2 ** (1 / 12), rel=5e-3)
    assert resample(x, sr, sr) is x


def test_store_reports_missing_samples_when_offline(tmp_path):
    store = SampleStore(tmp_path, download=False)
    with pytest.raises(SampleError, match="fetch-samples"):
        store.ensure([piano_ref(60, 1)])


def test_store_loads_what_is_on_disk(tmp_path):
    ref = SampleRef(url="unused", path="piano/C4v1.flac")
    (tmp_path / "piano").mkdir()
    sf.write(str(tmp_path / ref.path), np.full((48_000, 2), 0.25), 48_000, subtype="PCM_24")
    store = SampleStore(tmp_path, download=False)
    assert store.ensure([ref]) == 0
    data, rate = store.load(ref, max_seconds=0.5)
    assert rate == 48_000 and data.shape == (24_000, 2)
    assert data[0, 0] == pytest.approx(0.25, abs=1e-6)


def test_store_downloads_missing_samples(tmp_path, monkeypatch):
    payload = io.BytesIO()
    sf.write(payload, np.zeros((100, 2)), 48_000, format="FLAC")
    fetched = []

    def fake_urlopen(url, timeout):
        fetched.append(url)
        return io.BytesIO(payload.getvalue())

    monkeypatch.setattr(samples.urllib.request, "urlopen", fake_urlopen)
    store = SampleStore(tmp_path)
    refs = [piano_ref(60, 1), piano_ref(66, 2), piano_ref(60, 1)]
    assert store.ensure(refs) == 2
    assert sorted(fetched) == sorted({r.url for r in refs})
    assert (tmp_path / "piano" / "Fs4v2.flac").exists()
    assert store.ensure(refs) == 0  # idempotent


def test_a_bogus_download_leaves_no_file_behind(tmp_path, monkeypatch):
    monkeypatch.setattr(
        samples.urllib.request, "urlopen", lambda url, timeout: io.BytesIO(b"<html>404</html>")
    )
    store = SampleStore(tmp_path)
    with pytest.raises(SampleError, match=r"not a \.flac file"):
        store.ensure([piano_ref(60, 1)])
    assert not list(tmp_path.rglob("*.flac")) and not list(tmp_path.rglob("*.part"))


def test_default_samples_dir_follows_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSIC_DECODER_SAMPLES_DIR", str(tmp_path / "s"))
    assert samples.default_samples_dir() == tmp_path / "s"
    monkeypatch.delenv("MUSIC_DECODER_SAMPLES_DIR")
    monkeypatch.setenv("MUSIC_DECODER_DATA_DIR", str(tmp_path / "data"))
    assert samples.default_samples_dir() == tmp_path / "data" / "samples"
