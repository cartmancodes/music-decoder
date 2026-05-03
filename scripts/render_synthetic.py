"""Render the .mid fixtures in tests/fixtures/synthetic to .wav."""
from pathlib import Path
from music_decoder.evaluation.fixtures.synthetic import SyntheticFixtures


def main() -> None:
    root = Path("tests/fixtures/synthetic")
    for fx in SyntheticFixtures(root=root).load():
        print(f"rendered {fx.audio_path}")


if __name__ == "__main__":
    main()
