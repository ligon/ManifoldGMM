POETRY = poetry

.PHONY: lint black mypy test check

lint:
	$(POETRY) run ruff check .

black:
	$(POETRY) run black --check .

mypy:
	$(POETRY) run mypy src tests

test:
	$(POETRY) run pytest

check: lint black mypy test
