SHELL := /bin/bash

.PHONY: format format_check lint test ci

format:
	ruff check --select I --fix .
	ruff format .

format_check:
	ruff format --check .

lint:
	ruff check .

test:
	pytest tests/ -v

ci: test format_check lint
	@echo "CI pipeline passed"
