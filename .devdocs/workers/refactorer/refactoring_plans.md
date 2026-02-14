# Refactoring Plan - 2026-02-14

## Objective
Refactor specific files to fix linting issues, type hints, regex errors, exception handling, and code duplication as requested by the user.

## Tasks

1.  **CHANGELOG.md**:
    *   Fix MD022 (Missing blank line after headings).
    *   Target: `### Added`, `### Changed`, `### Fixed`.

2.  **src/jenova/config/models.py**:
    *   Improve type hinting for `to_proactive_config`.
    *   Use `TYPE_CHECKING` to import `ProactiveConfig` dataclass.
    *   Update return type from `object` to `ProactiveConfigDataclass`.

3.  **src/jenova/core/engine.py**:
    *   Fix email regex pattern `[A-Z|a-z]` -> `[A-Za-z]`.

4.  **src/jenova/core/task_executor.py**:
    *   Fix unhandled `ValueError` from `sanitize_for_prompt`.
    *   Refactor `_generate_insight` and `_generate_assumption` to use a common helper `_generate_from_history`.

5.  **src/jenova/main.py**:
    *   Narrow exception handling for scheduler initialization.
    *   Remove unused `ProactiveConfig` import.

## Verification
*   Verify `CHANGELOG.md` formatting.
*   Verify `src/jenova/config/models.py` imports and signature.
*   Verify `src/jenova/core/engine.py` regex.
*   Verify `src/jenova/core/task_executor.py` logic and exception handling.
*   Verify `src/jenova/main.py` exception handling and imports.

## New Tasks - 2026-02-14T15:00:00Z

1.  **src/jenova/core/engine.py**:
    *   Broaden scheduler exception handling in `think()` from `(RuntimeError, ValueError, OSError)` to `Exception`.

2.  **src/jenova/core/task_executor.py**:
    *   Refactor `execute_task` to use a pre-populated instance-level dispatch dictionary `self._task_dispatch`.