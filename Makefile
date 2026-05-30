# Supply Chain Resilience Co-Pilot — developer entrypoints.
# Phase 0–1 targets are live. Service/data targets are stubbed until the
# corresponding phases land (they echo what they will do).

.DEFAULT_GOAL := help
SHELL := /bin/bash
UV := uv

.PHONY: help setup fmt lint type test check lint-imports up down data-init clean

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup: ## Create the virtualenv and install the project + dev tools
	$(UV) venv --python 3.11
	$(UV) pip install -e ".[dev]"

fmt: ## Auto-format with ruff
	$(UV) run ruff format packages tests
	$(UV) run ruff check --fix packages tests

lint: ## Lint with ruff (no changes)
	$(UV) run ruff check packages tests
	$(UV) run ruff format --check packages tests

type: ## Static type-check with mypy
	$(UV) run mypy packages

lint-imports: ## Enforce layer/dependency boundaries (ADR-0003)
	$(UV) run lint-imports

test: ## Run the test suite
	$(UV) run pytest

check: lint type lint-imports test ## Run all quality gates (CI parity)

up: ## Bring up the full stack (Docker Compose)
	@echo "[stub] docker compose -f deploy/docker-compose.yml up — wired in later phases"

down: ## Tear down the stack
	@echo "[stub] docker compose -f deploy/docker-compose.yml down"

data-init: ## Initialise data sources / feature store
	@echo "[stub] data initialisation — implemented in Module 1 (Airflow + Feast)"

clean: ## Remove caches and build artefacts
	rm -rf .mypy_cache .ruff_cache .pytest_cache *.egg-info build dist
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
