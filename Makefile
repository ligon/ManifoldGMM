POETRY = poetry

FILES ?=

ifeq ($(strip $(FILES)),)
RUFF_TARGET = .
BLACK_TARGET = src
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

.PHONY: lint black mypy test check quick-check slow-tests use-local-datamat

lint:
	$(POETRY) run ruff check .

black:
	$(POETRY) run black --check .

mypy:
	$(POETRY) run mypy src tests

test:
	$(POETRY) run pytest

check: lint black mypy test

quick-check:
	$(POETRY) run ruff check $(RUFF_TARGET)
	$(POETRY) run black --check $(BLACK_TARGET)
	$(POETRY) run mypy $(MYPY_TARGET)
	$(PYTEST_CMD)

slow-tests:
	$(POETRY) run pytest -m slow

use-local-datamat:
	$(POETRY) run pip install -e ../DataMat
