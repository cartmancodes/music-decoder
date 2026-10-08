from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_setup_regenerates_madmom_c_extensions_under_project_constraints():
    setup_script = (ROOT / "setup.sh").read_text()
    makefile = (ROOT / "Makefile").read_text()

    assert '"numpy<2" "cython<3" "scipy>=0.16" "mido>=1.2.8"' in setup_script
    assert "pip download --no-build-isolation --no-deps --no-binary :all:" in setup_script
    assert "touch" in setup_script
    assert "madmom/ml/nn/layers.py" in setup_script
    assert "pip install --no-build-isolation --no-deps" in setup_script

    assert '"numpy<2" "cython<3" "scipy>=0.16" "mido>=1.2.8"' in makefile
    assert "pip download --no-build-isolation --no-deps --no-binary :all:" in makefile
    assert "touch" in makefile
    assert "madmom/ml/nn/layers.py" in makefile
    assert "pip install --no-build-isolation --no-deps" in makefile


def test_setup_prefetches_models_by_default_with_opt_outs():
    setup_script = (ROOT / "setup.sh").read_text()
    assert "music-decoder fetch-models" in setup_script
    assert "--no-models)" in setup_script
    assert "--no-soundfont)" in setup_script
    # the old fixture-dir download (dead mirrors) is gone
    assert "tests/fixtures/synthetic/soundfont/TimGM6mb.sf2" not in setup_script


def test_setup_help_documents_model_flags():
    import subprocess

    out = subprocess.run(
        ["bash", str(ROOT / "setup.sh"), "--help"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "--no-models" in out
    assert "fetch-models" in out
