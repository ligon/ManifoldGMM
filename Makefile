POETRY = poetry

FILES ?=

ifeq ($(strip $(FILES)),)
RUFF_TARGET = .
BLACK_TARGET = .
MYPY_TARGET = src tests
PYTEST_TARGET =
else
RUFF_TARGET = $(FILES)
BLACK_TARGET = $(FILES)
MYPY_TARGET = $(FILES)
PYTEST_TARGET = $(FILES)
endif

ifdef PYTEST_TARGET
PYTEST_CMD = $(POETRY) run pytest $(PYTEST_TARGET)
else
PYTEST_CMD = $(POETRY) run pytest
endif

.PHONY: lint black mypy test check quick-check

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
	$(POETRY) run black $(BLACK_TARGET)
	$(POETRY) run mypy $(MYPY_TARGET)
	$(PYTEST_CMD)
