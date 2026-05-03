import shutil
import subprocess

import pytest


@pytest.mark.integration
@pytest.mark.slow
def test_dockerfile_builds_successfully():
    """Verifies the Dockerfile parses and the image builds (does not run it)."""
    if shutil.which("docker") is None:
        pytest.skip("docker not available")
    # Also skip if the docker daemon is not running.
    daemon_check = subprocess.run(
        ["docker", "info"], capture_output=True, timeout=10,
    )
    if daemon_check.returncode != 0:
        pytest.skip("docker daemon not running")
    result = subprocess.run(
        ["docker", "build", "-t", "music-decoder-smoke", "--no-cache=false", "."],
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        print(result.stderr)
    assert result.returncode == 0, "docker build failed"
