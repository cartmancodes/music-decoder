import numpy as np
import scipy.io.wavfile as wavfile

from music_decoder.dsp.tempwav import temp_wav


def test_temp_wav_roundtrip_and_cleanup() -> None:
    x = np.linspace(-1, 1, 2205, dtype=np.float32)
    with temp_wav(x, 22050) as p:
        sr, data = wavfile.read(p)
        assert sr == 22050
        assert data.dtype == np.int16
        assert data.shape == (2205,)
    assert not p.exists()


def test_temp_wav_cleans_up_on_error() -> None:
    x = np.zeros(100, dtype=np.float32)
    try:
        with temp_wav(x, 22050) as p:
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert not p.exists()
