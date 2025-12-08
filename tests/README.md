# Test Suite for The JENOVA Cognitive Architecture

## Overview

This directory contains the comprehensive test suite for The JENOVA Cognitive Architecture. Tests are organized by component and cover unit tests, integration tests, and system tests.

## Structure

```
tests/
├── __init__.py          # Test package initialization
├── conftest.py          # Pytest fixtures and configuration
├── test_basic.py        # Basic smoke tests (imports, structure)
├── test_memory.py       # Memory system tests
├── test_cortex.py       # Cortex graph operation tests
├── test_cognitive_engine.py  # Cognitive engine tests
└── README.md            # This file
```

## Quick Start

Run basic smoke tests first to verify setup:
```bash
pytest tests/test_basic.py -v
```

If these pass, the module import path is configured correctly.

## Running Tests

### Prerequisites

Install pytest and optional coverage tools:
```bash
pip install pytest pytest-cov
```

### Run all tests:
```bash
pytest
```

or

```bash
pytest tests/
```

### Run specific test file:
```bash
pytest tests/test_memory.py
```

### Run with coverage (requires pytest-cov):
```bash
pip install pytest-cov
pytest --cov=src/jenova --cov-report=html
```

### Run with verbose output:
```bash
pytest -v
```

## Configuration

Pytest is configured in `pyproject.toml`:
- `pythonpath = ["src"]` - Adds src directory to Python path for imports
- `testpaths = ["tests"]` - Sets default test directory
- `addopts = "-v --tb=short"` - Default options for verbose output and short tracebacks

## Test Categories

### Unit Tests
- Test individual components in isolation
- Use mocks for dependencies
- Fast execution

### Integration Tests
- Test component interactions
- Use real dependencies where possible
- Slower execution

### System Tests
- Test end-to-end workflows
- Use real components
- Slowest execution

## Fixtures

Common fixtures are defined in `conftest.py`:
- `temp_user_data_dir`: Temporary directory for test data
- `mock_config`: Mock configuration dictionary
- `mock_ui_logger`: Mock UI logger
- `mock_file_logger`: Mock file logger
- `mock_llm_interface`: Mock LLM interface

## Writing New Tests

1. Create a new test file following the naming convention `test_*.py`
2. Import necessary modules and fixtures
3. Write test functions prefixed with `test_`
4. Use fixtures from `conftest.py` for common setup
5. Add inline documentation per project standards

## Notes

- Tests use temporary directories to avoid polluting user data
- Mock LLM interface prevents actual model loading during tests
- All tests should be deterministic and isolated

