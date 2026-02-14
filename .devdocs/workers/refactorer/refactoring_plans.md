# Refactoring Plan: Async Cognitive Architecture (ISSUE-004)

**Date:** 2026-02-14T16:00:00Z  
**Author:** Refactorer (D3)  
**Status:** DRAFT / APPROVED  
**Related Issue:** ISSUE-004 (Async CognitiveEngine)

## 1. Objective

Transition the `CognitiveEngine` and its core dependencies from synchronous execution to an asynchronous architecture using Python's `asyncio`.

## 2. Motivation

*   **Responsiveness:** The current synchronous model blocks the main thread during LLM generation (which can take seconds) and vector database searches. This freezes the UI/CLI.
*   **Concurrency:** Enables parallel execution of independent tasks (e.g., retrieving context from multiple sources, background memory updates, concurrent assumption verification).
*   **Future-Proofing:** Modern Python tooling and AI frameworks are increasingly async-first.

## 3. Architecture Transition Plan

### Phase 1: Foundation (Async Interfaces)

Create asynchronous counterparts for core interfaces. Maintain synchronous wrappers for backward compatibility where necessary during transition.

*   **`src/jenova/llm/interface.py`**
    *   Add `async def generate_async(...) -> Completion`
    *   Refactor `generate` to use `asyncio.run(self.generate_async(...))` or keep separate sync implementation if efficient.
    *   *Note:* Ensure underlying LLM client supports async (e.g., `openai`, `aiohttp`) or wrap blocking calls in `loop.run_in_executor`.

*   **`src/jenova/core/knowledge.py`**
    *   Add `async def search_async(...)`
    *   Add `async def add_async(...)`
    *   *Note:* Check ChromaDB async client support. Fallback to `run_in_executor` for file I/O operations.

*   **`src/jenova/core/planning.py`**
    *   Update `Planner.plan` to `async def plan(...)`.
    *   Update logic to `await self.llm.generate_async(...)`.

### Phase 2: Engine Asyncification

Refactor the core engine loop to be non-blocking.

*   **`src/jenova/core/engine.py`**
    *   Rename `think` to `think_async` (or change signature to `async def think`).
    *   **Context Retrieval:** `context = await self._retrieve_context_async(...)`
    *   **Planning:** `plan = await self.planner.plan(...)`
    *   **Generation:** `completion = await self.llm.generate_async(...)`
    *   **Memory:** `await self._store_interaction_async(...)` (or spawn as background task for fire-and-forget).

*   **`src/jenova/core/task_executor.py`**
    *   Update `CognitiveTaskExecutor` to handle async execution of background tasks.

### Phase 3: Wiring & Entry Points

Update application entry points to drive the async engine.

*   **`src/jenova/ui/app.py` (Textual)**
    *   Textual is natively async.
    *   Update `on_input_submitted` handler to `await self.engine.think(...)`.
    *   Remove any thread-worker workarounds currently used for blocking calls.

*   **`src/jenova/main.py` (Headless/CLI)**
    *   Update `run_headless` to be `async def`.
    *   Use `asyncio.run(run_headless(...))` in `main()`.

## 4. Risks & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Colorless Async Functions** | Mixing sync/async code causes blocking. | Use `pytest-asyncio` strict mode; audit all I/O paths. |
| **Testing Overhead** | 400+ unit tests assume sync execution. | Massive refactoring of test suite required. Use `pytest-asyncio` fixtures. |
| **Race Conditions** | Shared state (Memory/Graph) accessed concurrently. | Ensure `KnowledgeStore` and `Graph` are thread-safe or use asyncio locks for critical sections. |
| **Error Handling** | Unawaited coroutines swallow exceptions. | Enforce strict `await` usage; use `asyncio.TaskGroup` (Py3.11+) or `gather` with error handling. |

## 5. Execution Order

1.  **Refactor:** LLM Interface (High value), Knowledge Store (I/O bound), Planner (Depends on LLM), Engine (Orchestrator), UI/CLI (Consumer), Tests (Parallel/per-component).

## 6. Verification

*   All unit tests pass (migrated to async where needed).
*   UI remains responsive during long LLM generation.
*   Background tasks (scheduler) do not block conversation.
