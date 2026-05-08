"""Render committed .mid fixture files to .wav for the regression suite.

Phase B-5 added a fluidsynth path. The loader prefers fluidsynth + a soundfont
when a .sf2 file is available under ``<root>/soundfont/`` — that produces
realistic instrument timbres (transients, harmonic content, decay) that are
representative of what basic-pitch will see on real audio. When no soundfont
is present, the loader falls back to the original sine-wave synthesis (purely
deterministic; useful as a sanity-floor).

A user can install a soundfont with:

    python scripts/download_soundfont.py

The download is opt-in so a fresh checkout doesn't try to fetch ~6MB on its
first regression run.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pretty_midi
import scipy.io.wavfile as wavfile

from music_decoder.logging_setup import get_logger
from music_decoder.types import ChordSegment, KeyEstimate, TabbedNote

from .base import Fixture, GroundTruth

_SR = 22050
_log = get_logger("evaluation.synthetic")


def _safe_tempo(pm: pretty_midi.PrettyMIDI) -> float:
    """Return tempo estimate, falling back to first tempo change if estimation fails."""
    try:
        return float(pm.estimate_tempo())
    except ValueError:
        _, tempos = pm.get_tempo_changes()
        return float(tempos[0]) if len(tempos) > 0 else 120.0


def _find_soundfont(root: Path) -> Path | None:
    """Return the first .sf2 under <root>/soundfont/ or None if absent."""
    sf_dir = root / "soundfont"
    if not sf_dir.exists():
        return None
    for path in sorted(sf_dir.glob("*.sf2")):
        return path
    return None


class SyntheticFixtures:
    """Renders committed .mid files to .wav.

    Prefers fluidsynth + soundfont when a .sf2 is available under
    ``<root>/soundfont/``; falls back to sine-wave synthesis otherwise.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self._sf2_path = _find_soundfont(root)
        if self._sf2_path is not None:
            _log.info(
                "synthetic_fixtures_using_fluidsynth",
                extra={"sf2": str(self._sf2_path)},
            )
        else:
            _log.info("synthetic_fixtures_using_sine_fallback")

    def _ensure_wav(self, midi_path: Path) -> Path:
        wav_path = midi_path.with_suffix(".wav")
        # Cache key includes the soundfont path so swapping SF2s invalidates.
        cache_marker = midi_path.parent / f".{midi_path.stem}.synth_marker"
        current_marker = (
            f"sf2={self._sf2_path}" if self._sf2_path else "sine"
        )
        cache_valid = (
            wav_path.exists()
            and wav_path.stat().st_mtime > midi_path.stat().st_mtime
            and cache_marker.exists()
            and cache_marker.read_text() == current_marker
        )
        if cache_valid:
            return wav_path

        pm = pretty_midi.PrettyMIDI(str(midi_path))
        audio = self._synthesize(pm)
        peak = float(np.max(np.abs(audio)) or 1.0)
        audio = (audio / peak * 0.9).astype(np.float32)
        wavfile.write(str(wav_path), _SR, (audio * 32767).astype(np.int16))
        cache_marker.write_text(current_marker)
        return wav_path

    def _synthesize(
        self, pm: pretty_midi.PrettyMIDI,
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        """Render a PrettyMIDI object to mono float32 audio at _SR."""
        if self._sf2_path is not None:
            try:
                rendered: np.ndarray[Any, np.dtype[Any]] = pm.fluidsynth(
                    fs=_SR, sf2_path=str(self._sf2_path),
                )
                return np.asarray(rendered, dtype=np.float32)
            except Exception as e:
                _log.warning(
                    "fluidsynth_render_failed_falling_back",
                    extra={"error": str(e)},
                )
        sine: np.ndarray[Any, np.dtype[Any]] = pm.synthesize(fs=_SR)
        return np.asarray(sine, dtype=np.float32)

    def _ground_truth(self, midi_path: Path) -> GroundTruth:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        notes = [n for inst in pm.instruments for n in inst.notes]
        notes.sort(key=lambda n: (n.start, n.pitch))
        intervals = np.array([(n.start, n.end) for n in notes], dtype=float)
        pitches = np.array([n.pitch for n in notes], dtype=float)
        return GroundTruth(
            intervals=intervals if len(notes) else np.zeros((0, 2)),
            pitches_midi=pitches if len(notes) else np.zeros(0),
            key=None,
            tempo_bpm=_safe_tempo(pm) if notes else None,
            tab=None,
        )

    def load(self) -> Iterator[Fixture]:
        for midi_path in sorted(self.root.glob("*.mid")):
            wav = self._ensure_wav(midi_path)
            yield Fixture(
                name=midi_path.stem, source="synthetic",
                audio_path=wav, ground_truth=self._ground_truth(midi_path),
            )


# ---- v2 public-API fixture shape ------------------------------------------
#
# The v2 regression suite (`tests/integration/test_regression.py`) consumes
# fixtures whose ground truth is expressed in the public types that
# ``analyze()`` returns: a :class:`KeyEstimate`, a tuple of
# :class:`ChordSegment`, and a tuple of :class:`TabbedNote`. The v1 shape in
# ``base.py`` is preserved for backward compatibility — the v2 shape lives
# alongside it.


@dataclass(frozen=True)
class SyntheticGroundTruth:
    key: KeyEstimate
    chord_progression: tuple[ChordSegment, ...]
    tab: tuple[TabbedNote, ...]


@dataclass(frozen=True)
class SyntheticFixture:
    name: str
    audio_path: Path
    gt: SyntheticGroundTruth


# Hand-curated ground truth for the two committed synthetic clips. Both are
# short, deterministic MIDI files; deriving GT directly from the file would
# work but pinning it here makes the regression behavior obvious from
# inspection and resilient to small MIDI edits.
_SYNTHETIC_GT: dict[str, SyntheticGroundTruth] = {
    "c_major_scale": SyntheticGroundTruth(
        key=KeyEstimate(
            tonic="C", mode="major",
            profile="krumhansl_kessler", correlation=1.0, margin=0.0,
        ),
        chord_progression=(
            ChordSegment(start_s=0.0, end_s=4.0, root="C", quality="maj", confidence=1.0),
        ),
        tab=(),
    ),
    "g_major_chord": SyntheticGroundTruth(
        key=KeyEstimate(
            tonic="G", mode="major",
            profile="krumhansl_kessler", correlation=1.0, margin=0.0,
        ),
        chord_progression=(
            ChordSegment(start_s=0.0, end_s=2.0, root="G", quality="maj", confidence=1.0),
        ),
        tab=(),
    ),
}


def _default_root() -> Path:
    """Default fixture root: ``tests/fixtures/synthetic`` under the repo."""
    return Path(__file__).resolve().parents[4] / "tests" / "fixtures" / "synthetic"


def iter_synthetic_fixtures(
    root: Path | None = None,
) -> Iterator[SyntheticFixture]:
    """Yield v2 :class:`SyntheticFixture` records for the committed .mid clips.

    Each yield corresponds to one ``.mid`` under *root* (default:
    ``tests/fixtures/synthetic/``). The companion ``.wav`` is rendered on
    demand via :class:`SyntheticFixtures` (cached on disk so a second run is
    cheap). Ground truth is hand-pinned per-clip in ``_SYNTHETIC_GT``; clips
    with no entry are skipped silently so adding a new MIDI without updating
    the GT table does not break the regression run.
    """
    root = root if root is not None else _default_root()
    if not root.exists():
        return
    loader = SyntheticFixtures(root=root)
    for midi_path in sorted(root.glob("*.mid")):
        gt = _SYNTHETIC_GT.get(midi_path.stem)
        if gt is None:
            _log.info(
                "synthetic_fixture_no_ground_truth_skipping",
                extra={"name": midi_path.stem},
            )
            continue
        wav = loader._ensure_wav(midi_path)
        yield SyntheticFixture(name=midi_path.stem, audio_path=wav, gt=gt)
