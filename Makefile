.PHONY: dev test test-fast lint typecheck regression run

dev:
	pip install -e ".[dev]"

test:
	pytest -q tests/unit tests/integration

test-fast:
	pytest -q tests/unit

regression:
	pytest -q -m regression tests/integration/test_regression.py

lint:
	ruff check src tests

typecheck:
	mypy --strict src

run:
	music-decoder ui
