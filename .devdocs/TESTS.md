# JENOVA Cognitive Architecture - Test Documentation

## Purpose
This document tracks test coverage, test results, and testing procedures for the JENOVA project.

---

## Test Infrastructure

### Test Framework
- **Framework:** pytest
- **Location:** `tests/` directory
- **Configuration:** `tests/conftest.py`

### Test Files
| File | Purpose | Status |
|------|---------|--------|
| `tests/__init__.py` | Package initialization | ✅ Exists |
| `tests/conftest.py` | Pytest fixtures and configuration | ✅ Exists |
| `tests/test_basic.py` | Basic functionality tests | ✅ Exists |
| `tests/test_cognitive_engine.py` | Cognitive engine tests | ✅ Exists |
| `tests/test_cortex.py` | Cortex module tests | ✅ Exists |
| `tests/test_memory.py` | Memory systems tests | ✅ Exists |

### Additional Test Files
| File | Purpose | Status |
|------|---------|--------|
| `test_tui.py` | TUI integration tests | ✅ Exists (root) |
| `demo_ui.py` | UI demonstration script | ✅ Exists (root) |

---

## Running Tests

### Prerequisites
```bash
# Activate virtual environment (if using venv)
source venv/bin/activate

# Install test dependencies
pip install pytest pytest-cov
```

### Commands
```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_basic.py

# Run with coverage report
pytest tests/ --cov=src/jenova

# Run specific test function
pytest tests/test_basic.py::test_function_name
```

---

## Test Coverage

### Current Coverage Status
*To be updated after test runs*

| Module | Coverage | Notes |
|--------|----------|-------|
| cognitive_engine | TBD | Pending analysis |
| cortex | TBD | Pending analysis |
| memory | TBD | Pending analysis |
| ui | TBD | Pending analysis |
| utils | TBD | Pending analysis |

---

## Test Categories

### Unit Tests
- Individual function testing
- Mocked dependencies
- Fast execution

### Integration Tests
- Component interaction testing
- Real dependencies where feasible
- Moderate execution time

### End-to-End Tests
- Full system testing
- Requires model loaded
- Slow execution (not run in CI)

---

## Documentation Review Testing

### Commenting Verification Procedure
For the multi-session code review plan, each file should be verified:

1. **Script Header Present**
   ```python
   ##Script function and purpose: [Description]
   ```

2. **Function Comments Present**
   ```python
   ##Function purpose: [Description]
   def function_name():
   ```

3. **Block Comments Present**
   ```python
   ##Block purpose: [Description]
   code_block
   ```

### Manual Verification Checklist
- [ ] Check file has script header
- [ ] Count functions vs function comments
- [ ] Verify logical blocks have comments
- [ ] Confirm no code changes made

---

## Known Test Issues

### Current Issues
- None documented yet

### Resolved Issues
- None documented yet

---

## Test Results History

### Session 1
*Documentation creation session - no tests run*

### Future Sessions
*Test results to be logged here after each session*

---

## CI/CD Integration

### GitHub Actions
- Workflow files: `.github/workflows/` (if exists)
- Status: TBD

### Test Automation
- Pre-commit hooks: TBD
- Automated coverage reporting: TBD

---

## Contributing to Tests

### Test Writing Guidelines
1. Follow pytest conventions
2. Use descriptive test names
3. Add comments explaining test purpose
4. Mock external dependencies
5. Keep tests focused and atomic

### Test Documentation Format
```python
##Script function and purpose: Tests for [module name]
##This test file verifies [specific functionality]

import pytest
from jenova.module import function_to_test

##Function purpose: Test [specific behavior]
def test_function_name():
    ##Block purpose: Arrange - Set up test data
    test_input = "sample"
    
    ##Block purpose: Act - Execute the function
    result = function_to_test(test_input)
    
    ##Block purpose: Assert - Verify expected behavior
    assert result == expected_output
```
