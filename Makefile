.PHONY: dev test test-fast lint typecheck regression run

dev:
	pip install --upgrade hatchling "numpy<2" "cython<3" "scipy>=0.16" "mido>=1.2.8"
	madmom_build_dir="$$(mktemp -d)" && pip download --no-build-isolation --no-deps --no-binary :all: --dest "$${madmom_build_dir}" "madmom>=0.16,<0.17" && madmom_src_dir="$${madmom_build_dir}/src" && mkdir "$${madmom_src_dir}" && tar -xzf "$${madmom_build_dir}"/madmom-*.tar.gz -C "$${madmom_src_dir}" --strip-components 1 && touch "$${madmom_src_dir}/madmom/audio/comb_filters.pyx" "$${madmom_src_dir}/madmom/features/beats_crf.pyx" "$${madmom_src_dir}/madmom/ml/hmm.pyx" "$${madmom_src_dir}/madmom/ml/nn/layers.py" && pip install --no-build-isolation --no-deps "$${madmom_src_dir}"; rc="$$?"; rm -rf "$${madmom_build_dir}"; exit "$$rc"
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
