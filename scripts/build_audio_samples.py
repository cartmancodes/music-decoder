# scripts/build_audio_samples.py
from pathlib import Path
import numpy as np
import scipy.io.wavfile as wavfile

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "audio_samples"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sr = 22050
    t = np.arange(sr) / sr
    sine = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    wavfile.write(str(OUT / "sine_440.wav"), sr, (sine * 32767).astype(np.int16))
    silence = np.zeros(sr * 2, dtype=np.int16)
    wavfile.write(str(OUT / "silence_2s.wav"), sr, silence)


if __name__ == "__main__":
    main()
