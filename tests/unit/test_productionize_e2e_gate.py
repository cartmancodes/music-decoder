"""The full-song render tests download ~180 MB and need ~2.7 GB of memory
each, so `make test` must not run them unless asked."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
VAR = "MUSIC_DECODER_PRODUCTION_E2E"


def _marks(monkeypatch: pytest.MonkeyPatch, value: str | None) -> list[pytest.Mark]:
    if value is None:
        monkeypatch.delenv(VAR, raising=False)
    else:
        monkeypatch.setenv(VAR, value)
    path = ROOT / "tests" / "integration" / "test_productionize_e2e.py"
    spec = importlib.util.spec_from_file_location("e2e_gate_probe", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return [m.mark for m in module.pytestmark if m.mark.name == "skipif"]


def test_render_tests_are_skipped_unless_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    gated = [m for m in _marks(monkeypatch, None) if VAR in m.kwargs.get("reason", "")]
    assert gated and gated[0].args[0] is True


def test_render_tests_run_when_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    gated = [m for m in _marks(monkeypatch, "1") if VAR in m.kwargs.get("reason", "")]
    assert gated and gated[0].args[0] is False
