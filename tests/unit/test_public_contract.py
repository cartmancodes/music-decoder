# tests/unit/test_public_contract.py
"""Frozen public-contract guard for the aggressive refactor.

Spec: docs/superpowers/specs/2026-05-17-aggressive-refactor-design.md
Any change here means the externally observable contract moved — which
the refactor forbids. If a rewrite trips this, the rewrite is wrong,
not this test.
"""

import inspect

import music_decoder
from music_decoder import api, errors


def test_package_exports_are_frozen():
    assert sorted(music_decoder.__all__) == sorted([
        "DADGAD", "DROP_C", "DROP_D", "D_STANDARD", "EB_HALF_STEP_DOWN",
        "STANDARD_EADGBE", "AnalysisResult", "ChordSegment", "ChordSymbol",
        "Composition", "KeyEstimate", "Note", "Scale", "TabPosition",
        "TabbedNote", "Tuning", "VoicedChord", "analyze", "compose",
    ])


def test_analyze_signature_is_frozen():
    sig = inspect.signature(api.analyze)
    params = list(sig.parameters)
    assert params == [
        "source", "declared_kind", "tuning", "use_separation", "progress",
    ]
    p = sig.parameters
    assert p["source"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert p["declared_kind"].kind is inspect.Parameter.KEYWORD_ONLY
    assert p["declared_kind"].default == "full_mix"
    assert p["use_separation"].default is True
    assert p["progress"].default is None
    assert p["tuning"].default == music_decoder.STANDARD_EADGBE


def test_compose_signature_is_frozen():
    sig = inspect.signature(api.compose)
    params = list(sig.parameters)
    assert params == [
        "scale", "progression", "bars_per_chord", "tempo_bpm",
        "style", "tuning", "seed", "out_dir",
    ]
    p = sig.parameters
    assert p["scale"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert p["bars_per_chord"].kind is inspect.Parameter.KEYWORD_ONLY
    assert p["bars_per_chord"].default == 1
    assert p["tempo_bpm"].default == 100.0
    assert p["style"].default == "fingerstyle"
    assert p["seed"].default is None
    assert p["out_dir"].default is inspect.Parameter.empty
    assert p["tuning"].default == music_decoder.STANDARD_EADGBE


def test_error_hierarchy_is_frozen():
    expected = {
        "MusicDecoderError": Exception,
        "IngestError": errors.MusicDecoderError,
        "YouTubeError": errors.IngestError,
        "CorruptAudioError": errors.IngestError,
        "SilentAudioError": errors.IngestError,
        "ClipTooShortError": errors.IngestError,
        "SeparationError": errors.MusicDecoderError,
        "TranscriptionError": errors.MusicDecoderError,
        "ChordRecognitionError": errors.MusicDecoderError,
        "TabAssignmentError": errors.MusicDecoderError,
        "KeyDetectionError": errors.MusicDecoderError,
        "BeatTrackingError": errors.MusicDecoderError,
        "CompositionError": errors.MusicDecoderError,
        "InvalidScaleError": errors.CompositionError,
        "InvalidProgressionError": errors.CompositionError,
        "SynthesisError": errors.MusicDecoderError,
    }
    for name, base in expected.items():
        cls = getattr(errors, name)
        assert issubclass(cls, base), f"{name} must subclass {base.__name__}"
        assert issubclass(cls, errors.MusicDecoderError) or cls is errors.MusicDecoderError


def test_cli_subcommands_and_flags_are_frozen():
    from click.testing import CliRunner

    from music_decoder.cli.main import main

    r = CliRunner().invoke(main, ["--help"])
    assert r.exit_code == 0
    for cmd in ("analyze", "compose", "ui", "doctor"):
        assert cmd in r.output
    ra = CliRunner().invoke(main, ["analyze", "--help"])
    assert ra.exit_code == 0
    for flag in ("--tuning", "--solo-guitar", "--full-mix",
                 "--no-separation", "--format"):
        assert flag in ra.output
    rc = CliRunner().invoke(main, ["compose", "--help"])
    assert rc.exit_code == 0
    for flag in ("--scale", "--progression", "--style", "--tempo",
                 "--bars", "--seed", "--tuning", "--out", "--format"):
        assert flag in rc.output
