.PHONY: install dev install-crepe test test-fast lint typecheck eval format clean

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

install-crepe:
	pip install "setuptools<70"
	pip install --no-build-isolation -e ".[crepe]"

test:
	pytest

test-fast:
	pytest -m "not slow and not regression" -n auto

lint:
	ruff check src tests

typecheck:
	mypy src

eval:
	pytest tests/regression -v

format:
	ruff check --fix src tests
	ruff format src tests

fixtures:
	python -c "import mirdata; mirdata.initialize('guitarset', data_home='tests/fixtures/guitarset').download(force_overwrite=False)"

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
