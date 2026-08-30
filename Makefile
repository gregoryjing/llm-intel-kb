.PHONY: install format lint test test-integration db-up db-down doctor init-db

install:
	python -m pip install -e ".[dev]"

format:
	ruff format src tests
	ruff check --fix src tests

lint:
	ruff check src tests
	ruff format --check src tests
	mypy

test:
	pytest

test-integration:
	pytest -m integration

db-up:
	docker compose up -d db

db-down:
	docker compose down

doctor:
	llm-intel-kb doctor

init-db:
	llm-intel-kb init-db
