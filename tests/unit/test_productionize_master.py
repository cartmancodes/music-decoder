import shutil

import numpy as np
import pytest
import soundfile as sf

from music_decoder.productionize import dsp
from music_decoder.productionize.config import MasterConfig
from music_decoder.productionize.master import (
    ceiling_gain,
    encode_mp3,
    limiter_gain,
    loudness,
    master,
    mp3_command,
    soft_clip,
    with_lead_in,
    write_wav,
)
from music_decoder.productionize.song import ProductionError

SR = 44_100
CFG = MasterConfig()


def _music(seconds=8.0, seed=0, level=0.1):
    """Bass tone, a mid tone, filtered noise and a few loud drum-like hits."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    x = 0.6 * np.sin(2 * np.pi * 55 * t) + 0.3 * np.sin(2 * np.pi * 440 * t)
    stereo = np.stack([x, x], axis=1) + 0.2 * dsp.butter(
        rng.standard_normal((n, 2)), "lowpass", 4000.0, SR
    )
    for beat in np.arange(0.5, seconds - 0.5, 0.5):
        a = int(beat * SR)
        hit = rng.standard_normal((2000, 2)) * np.exp(-np.arange(2000) / 300.0)[:, None]
        stereo[a : a + 2000] += 3.0 * hit
    return stereo * level


def test_ceiling_is_minus_1_8_dbtp_with_a_margin():
    assert ceiling_gain(CFG) == pytest.approx(10 ** (-1.8 / 20) * 0.985)
    assert dsp.gain_to_db(ceiling_gain(CFG)) == pytest.approx(-1.93, abs=0.005)


def test_soft_clip_is_linear_below_the_knee_and_folds_peaks_under_the_threshold():
    t = np.arange(SR) / SR
    x = dsp.to_stereo(0.3 * np.sin(2 * np.pi * 200 * t))
    bump = slice(SR // 2, SR // 2 + 22)  # a half-millisecond transient, far above the rest
    x[bump] += 2.0 * np.hanning(22)[:, None]
    out = soft_clip(x, CFG)
    threshold = CFG.clip_ratio * np.percentile(np.abs(x), CFG.clip_percentile)
    assert threshold == pytest.approx(0.85 * 0.3, rel=0.02)
    quiet = np.abs(x[:, 0]) < 0.9 * CFG.clip_knee * threshold
    quiet[SR // 2 - 200 : SR // 2 + 200] = False
    assert np.allclose(out[quiet], x[quiet], atol=2e-3)
    assert np.abs(out).max() < 1.1 * threshold
    assert out.shape == x.shape


def test_soft_clip_passes_silence():
    x = np.zeros((1000, 2))
    assert soft_clip(x, CFG) is x


def test_limiter_keeps_the_true_peak_under_the_ceiling():
    x = _music(level=0.5)
    ceiling = ceiling_gain(CFG)
    assert dsp.true_peak(x) > ceiling
    gain = limiter_gain(x, ceiling, CFG, SR)
    assert gain.shape == (len(x),) and gain.max() <= 1.0 and gain.min() < 1.0
    assert dsp.true_peak(x * gain[:, None]) <= ceiling * 1.02


def test_limiter_looks_ahead_and_releases():
    x = np.full((SR, 2), 0.1)
    x[SR // 2] = [2.0, 2.0]
    ceiling = 0.8
    gain = limiter_gain(x, ceiling, CFG, SR)
    ahead = int(CFG.lookahead_s * SR)
    assert gain[SR // 2] <= ceiling / 2.0 + 1e-9
    assert gain[SR // 2 - ahead // 2] < 0.7  # already turning down before the peak
    assert gain[SR // 2 - 3 * ahead] == pytest.approx(1.0)
    assert gain[SR // 2 + int(0.03 * SR)] < 0.95  # 90 ms release: still recovering
    assert gain[-1] == pytest.approx(1.0, abs=1e-2)
    assert np.all(limiter_gain(np.full((1000, 2), 0.1), ceiling, CFG, SR) == 1.0)


def test_master_hits_the_loudness_target_under_the_ceiling():
    out, report = master(_music(), CFG, SR)
    assert out.shape == _music().shape
    assert loudness(out, SR) == pytest.approx(CFG.target_lufs, abs=0.3)
    assert report.integrated_lufs == pytest.approx(loudness(out, SR))
    assert dsp.true_peak(out) <= ceiling_gain(CFG) * 1.0001
    assert report.true_peak_dbtp <= -1.93 + 0.01
    assert 1 <= report.iterations <= CFG.loudness_iterations
    assert report.limiter_max_reduction_db >= report.limiter_p95_reduction_db >= 0.0


def test_master_is_independent_of_the_mix_level():
    quiet, _ = master(_music(level=0.01), CFG, SR)
    loud, _ = master(_music(level=0.3), CFG, SR)
    assert loudness(quiet, SR) == pytest.approx(loudness(loud, SR), abs=0.2)


def test_master_removes_subsonics():
    x = _music()
    x += 0.2 * np.sin(2 * np.pi * 8 * np.arange(len(x)) / SR)[:, None]
    out, _ = master(x, CFG, SR)
    assert dsp.rms(dsp.butter(out, "lowpass", 12.0, SR, order=4)[SR:]) < 0.01


def test_a_silent_mix_cannot_be_mastered():
    with pytest.raises(ProductionError, match="silent"):
        master(np.zeros((SR, 2)), CFG, SR)


def test_lead_in_prepends_120_ms_of_silence():
    x = np.ones((1000, 2))
    out = with_lead_in(x, CFG, SR)
    lead = round(0.12 * SR)
    assert len(out) == 1000 + lead
    assert not out[:lead].any() and np.all(out[lead:] == 1.0)


def test_wav_subtypes(tmp_path):
    x = _music(seconds=1.0)
    master_path = write_wav(tmp_path / "a" / "m.wav", x, SR, subtype="PCM_24")
    raw_path = write_wav(tmp_path / "raw.wav", x, SR, subtype="FLOAT")
    assert sf.info(str(master_path)).subtype == "PCM_24"
    assert sf.info(str(raw_path)).subtype == "FLOAT"
    assert sf.info(str(master_path)).samplerate == SR and sf.info(str(master_path)).channels == 2


def test_mp3_command_matches_the_build_spec(tmp_path):
    tags = {"title": "Hisaab (Instrumental)", "TBPM": "80", "TKEY": "F#m", "genre": "Hip-Hop"}
    cmd = mp3_command(tmp_path / "ffmpeg", tmp_path / "in.wav", tmp_path / "out.mp3", tags, CFG)
    joined = " ".join(cmd)
    assert "-codec:a libmp3lame -b:a 320k -id3v2_version 3" in joined
    for pair in ("title=Hisaab (Instrumental)", "TBPM=80", "TKEY=F#m", "genre=Hip-Hop"):
        assert cmd[cmd.index(pair) - 1] == "-metadata"
    assert cmd[-1].endswith("out.mp3")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_encode_mp3(tmp_path):
    wav = write_wav(tmp_path / "m.wav", _music(seconds=2.0), SR, subtype="PCM_24")
    mp3 = encode_mp3(wav, tmp_path / "m.mp3", {"title": "T"}, CFG)
    info = sf.info(str(mp3))
    assert info.samplerate == SR and info.channels == 2 and info.format == "MP3"


def test_encode_mp3_reports_a_failed_encode(tmp_path):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    with pytest.raises(ProductionError, match="could not encode"):
        encode_mp3(tmp_path / "missing.wav", tmp_path / "m.mp3", {}, CFG)
