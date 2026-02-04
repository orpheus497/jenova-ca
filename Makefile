# Makefile — JENOVA
# ##Update: Minimal build/lint/test targets (Configurator C8, 2026-02-04)
# Purpose: One-command install, lint, and test for local development. CI uses .github/workflows/ci.yml.

.PHONY: install lint format test test-unit test-integration test-security

install:
	pip install -e ".[dev]"

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/
	mypy src/ --show-error-codes

format:
	ruff format src/ tests/

test:
	pytest tests/ -v --tb=short -m "not slow"

test-unit:
	pytest tests/unit/ -v --tb=short -m "not slow"

test-integration:
	pytest tests/integration/ -v --tb=short -m integration

test-security:
	pytest tests/security/ -v --tb=short
