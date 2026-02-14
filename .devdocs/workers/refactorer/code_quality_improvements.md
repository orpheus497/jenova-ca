# Code Quality Improvements - 2026-02-14

## Summary
Refactored core components to improve type safety, security (regex), and maintainability (DRY).

## Improvements

### Type Safety
*   **src/jenova/config/models.py**: Explicit return type for `to_proactive_config` using `TYPE_CHECKING` to avoid runtime circular imports while providing correct static analysis info.

### Security
*   **src/jenova/core/engine.py**: Fixed email regex character class `[A-Z|a-z]` which incorrectly included the pipe character. Changed to `[A-Za-z]`.

### Maintainability (DRY)
*   **src/jenova/core/task_executor.py**: Extracted `_generate_from_history` helper method to unify `_generate_insight` and `_generate_assumption` logic. This reduces duplication and ensures consistent logging and exception handling.

### Robustness
*   **src/jenova/core/task_executor.py**: Moved `_build_history_summary` call inside the `try` block to catch potential `ValueError` from `sanitize_for_prompt`.
*   **src/jenova/main.py**: Narrowed exception handling for scheduler initialization from `Exception` to `(RuntimeError, ValueError, KeyError, ImportError)` to avoid masking unexpected programming errors.

## Improvements - 2026-02-14T15:00:00Z

### Robustness
*   **src/jenova/core/engine.py**: Broadened scheduler exception handling in `think()` to `Exception` to ensure that any error in the post-turn scheduler logic (including attribute or type errors) is caught and logged, preventing it from breaking the conversation flow.

### Performance & Efficiency
*   **src/jenova/core/task_executor.py**: Optimized `execute_task` by moving the task dispatch mapping from a local dictionary created on every call to a single instance-level attribute `self._task_dispatch`. This reduces object allocation overhead and simplifies the dispatch logic.

### Standards
*   **CHANGELOG.md**: Fixed MD022 linting errors by adding blank lines after headings.
*   **src/jenova/main.py**: Removed unused `ProactiveConfig` import.