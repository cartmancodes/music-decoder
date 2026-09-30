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
