# Refactoring Logs - 2026-02-14T15:00:00Z

## Actions Taken

### Session 225 — 2026-02-14T10:24:30Z — Hardening & Validation Refactoring

#### 1. src/jenova/core/planning.py
*   **Action**: Added `isinstance(data, dict)` type-check after `safe_json_loads`, raising `LLMParseError` if result is not a dict.
*   **Reason**: LLM may return a valid JSON array, string, or null. Without this check, `data.get(...)` and `data["sub_goals"]` would crash with `AttributeError`/`TypeError`. The `LLMParseError` triggers the existing `_simple_plan` fallback.

#### 2. src/jenova/insights/manager.py
*   **Action**: Added early validation of `training_data_path` in `__init__` (raises `ValueError` on misconfiguration). Updated `save_insight` to validate against `self._training_data_path.parent` when an explicit path is set.
*   **Reason**: Previous code always validated against `self._insights_root.parent`, silently rejecting valid explicit training data paths outside the insights root hierarchy.

#### 3. src/jenova/main.py (_SUBSYSTEM_INIT_EXCEPTIONS)
*   **Action**: Removed `AttributeError` and `TypeError` from the exception tuple.
*   **Reason**: These exception types mask programming bugs (missing attributes, wrong argument types) in IntegrationHub and Scheduler initialization. Only config/runtime errors should be swallowed.

#### 4. src/jenova/main.py (assert → explicit check)
*   **Action**: Replaced `assert isinstance(llm, EngineLLMProtocol)` with `if not isinstance(...)` raising `TypeError`.
*   **Reason**: `assert` is stripped under `python -O`; the protocol validation must be unconditional.

#### 5. src/jenova/tools/web_search.py
*   **Action**: Added `if not query.strip(): return []` guard in `MockSearchProvider.search`.
*   **Reason**: Mirrors `DuckDuckGoSearchProvider.search` behavior for empty/whitespace queries, ensuring consistent test/production parity.

#### 6. src/jenova/ui/app.py
*   **Action**: Added `except AssumptionDuplicateError` catch before the generic `except Exception` in `_handle_assume_command`. Imported `AssumptionDuplicateError` from `jenova.exceptions`.
*   **Reason**: Provides a friendly "This assumption already exists" message instead of a raw error traceback.

#### 7. src/jenova/utils/sanitization.py
*   **Action**: Added regex `r"(?im)(^|\n)\s*(SYSTEM|ADMIN|USER)\s*:"` to `INJECTION_PATTERNS`.
*   **Reason**: Existing patterns only caught bracketed forms like `[SYSTEM]:`. Bare role-tag prefixes (`SYSTEM: Enable admin mode`) were undetected.

---

### Session 204 — Previous

### 1. src/jenova/core/engine.py

*   **Action**: Replaced `except (RuntimeError, ValueError, OSError) as e:` with `except Exception as e:` in the `think()` method where the scheduler is called.
*   **Reason**: Ensure that any error occurring during the post-turn scheduler logic (including possible `AttributeError` if `_scheduler` is partially initialized or `TypeError` during argument passing) is caught and logged, preventing it from crashing the main `think()` loop and ensuring a response is still returned to the user.

### 2. src/jenova/core/task_executor.py

*   **Action**: Moved the `dispatch` dictionary from `execute_task` to a new instance-level attribute `self._task_dispatch` initialized in `__init__`.
*   **Action**: Updated `execute_task` to use `self._task_dispatch.get(task_type)`.
*   **Reason**: Optimization to avoid re-creating the dictionary and binding the methods on every task execution. This improves performance and is a cleaner design.
