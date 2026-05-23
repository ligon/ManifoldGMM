POETRY = poetry

FILES ?=

ifeq ($(strip $(FILES)),)
RUFF_TARGET = .
# Match CI's ``black --check .`` (the ``black`` target above) rather than
# the narrower ``src`` form, so ``quick-check`` catches tests-only
# formatting drift before it hits CI.
BLACK_TARGET = .
MYPY_TARGET = src tests
PYTEST_TARGET =
PYTEST_FLAGS = -m "not slow"
else
RUFF_TARGET = $(FILES)
BLACK_TARGET = $(FILES)
MYPY_TARGET = $(FILES)
PYTEST_TARGET = $(FILES)
PYTEST_FLAGS =
endif

ifdef PYTEST_TARGET
PYTEST_CMD = $(POETRY) run pytest $(PYTEST_TARGET)
else
PYTEST_CMD = $(POETRY) run pytest $(PYTEST_FLAGS)
endif

.PHONY: lint black mypy test test-parallel check quick-check slow-tests docstring-check use-local-datamat poetry-venv build publish release

lint:
	$(POETRY) run ruff check .

black:
	$(POETRY) run black --check .

mypy:
	$(POETRY) run mypy src tests

test:
	$(POETRY) run pytest

# Parallel runner via pytest-xdist.  Use locally for fast iteration on
# the slow MC tests.  Avoid inside cgroup-restricted Slurm jobs (where
# `auto` over-allocates) -- set an explicit ``-n $$SLURM_CPUS_ON_NODE``
# in the sbatch script instead.
test-parallel:
	$(POETRY) run pytest -n auto

check: lint black mypy test

quick-check:
	$(POETRY) run ruff check $(RUFF_TARGET)
	$(POETRY) run black --check $(BLACK_TARGET)
	$(POETRY) run mypy $(MYPY_TARGET)
	$(PYTEST_CMD)

slow-tests:
	$(POETRY) run pytest -m slow

docstring-check:
	$(POETRY) run python tools/check_docstrings.py $(if $(strip $(FILES)),$(FILES),)

use-local-datamat:
	$(POETRY) run pip install -e ../DataMat

build:
	$(POETRY) build

publish: build
	$(POETRY) publish

# Usage: make release BUMP=patch  (or minor, major, prepatch, etc.)
BUMP ?= patch
release: check build
	$(eval NEW_VER := $(shell $(POETRY) version $(BUMP) -s))
	git add pyproject.toml
	git commit -m "Bump version to $(NEW_VER)"
	git tag v$(NEW_VER)
	@echo "Tagged v$(NEW_VER). Run 'git push && git push --tags && make publish' to publish."

poetry-venv:
	$(POETRY) config virtualenvs.in-project true --local
	$(POETRY) install --with dev
