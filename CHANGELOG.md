# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **Comment schema preserved:** Restored mandatory `##Script function and purpose:` (and project comment schema) in `tests/__init__.py` and `tests/integration/__init__.py` after mistaken removal. CONTRIBUTING Code Review section now explicitly requires **rejecting** any CodeRabbit (or other tool) suggestion to remove or change the ##-prefixed comment convention; the schema is non-negotiable.

### Changed

- **Code review documentation (Cycles 1–10):** CONTRIBUTING.md: added "Code Review (CodeRabbit and Local Checks)" section documenting `coderabbit --plain`, rate-limit note, and local equivalents (ruff, mypy, pytest) aligned with CI. PR checklist: added code-review step and "after every run update documentation as needed." Documentation Requirements: added "after every code review run" fix-and-update-docs step. README §8: added pointer to CONTRIBUTING for code review and contribution standards. SESSION_HANDOFF: added CodeRabbit and local checks to Next Steps and Quick Reference; fixed numbering. AUDIT_REPORT: recorded CodeRabbit/local-check documentation as resolved. .gitignore: added `.coderabbit/` for optional CodeRabbit tool artifacts. Note: `coderabbit --plain` was run repeatedly; cloud rate limits applied, so local checks (ruff, mypy, pytest) are documented as alternatives and "run, fix, then update docs" is required after every review run.
- **Documentation and config (audit remediation):**
  - **README:** Clone URL kept as `orpheus497/jenova-ca` (canonical user repo). Updated Python to 3.10–3.13. Corrected command table: `/reset` and `/debug` now TUI & Headless; cognitive commands (`/insight`, `/reflect`, `/memory-insight`, `/meta`, `/verify`, `/develop_insight`, `/learn_procedure`, `/train`) moved to “Cognitive Commands (TUI)” as implemented. Updated test counts (19 unit files, 430+ unit tests, 3 integration files, 37 integration tests, 490+ total). Updated dependency count (6 dev dependencies). Added CONTRIBUTING.md to project structure. Revised “Planned Features” (removed implemented command handlers; added headless cognitive support as planned).
  - **CONTRIBUTING:** Clarified Python 3.10–3.13 alignment with CI matrix and pyproject classifiers.
  - **config.example.yaml:** Replaced Python-style comment schema with standard YAML `#` comments.
  - **CI:** Upgraded `actions/setup-python` from v5 to v6. Added Python 3.13 to test matrix. Added `ruff format --check` step.
  - **pyproject.toml:** Added Python 3.13 classifier.
  - **AUDIT_REPORT.md:** Marked audit items as resolved.
- **Session handoff:** Added `SESSION_HANDOFF.md` at repo root. Summarizes session work, current state, next steps, and quick reference for the next session or contributor.

## [4.0.1] - 2026-02-01

### Fixed

- **BUG-001:** `Memory.clear()` now preserves custom embedding and clears search cache (`src/jenova/memory/memory.py`).
- **BUG-002:** `GrammarLoader.load_from_file()` catches `UnicodeDecodeError` and re-raises as `GrammarError` for non-UTF-8 files (`src/jenova/utils/grammar.py`).
- **BUG-003:** `ScoredContext.top(n)` and `as_strings(n)` clamp negative `n` to avoid surprising slice semantics (`src/jenova/core/context_scorer.py`).
- **BUG-004:** App username init now logs failure cause before fallback to `"default"` so operators can debug (e.g. `USER` unset, validation failure) (`src/jenova/ui/app.py`).
- **BUG-005:** Integration `get_centrality_score` guards non-numeric centrality in metadata (e.g. `"high"`) with try/except; uses 0.0 on failure (`src/jenova/core/integration.py`).
- **BUG-006:** Finetune `load_training_data` skips invalid JSON lines with warning instead of failing entire load (`finetune/train.py`).
- **BUG-007:** `test_insights` opens insight files with `encoding="utf-8"` for cross-platform consistency (`tests/unit/test_insights.py`).
- **BUG-008:** `validate_username` and `validate_topic` reject `None` or non-str with `ValueError` instead of allowing `AttributeError` (`src/jenova/utils/validation.py`).
- **Commenting schema (Marshal B7):** Replaced 6 incorrect `##Class purpose: Define logger for ...` comments (above module-level `logger = ...`) with `##Step purpose: Initialize module logger` in: `src/jenova/utils/grammar.py`, `src/jenova/utils/performance.py`, `src/jenova/utils/cache.py`, `src/jenova/tools.py`, `src/jenova/graph/proactive.py`, `src/jenova/core/scheduler.py`.

### Changed

- **Test and CI (Test Extender D5):**
  - 23 unit-test fixes: grammar patch target (`llama_cpp.LlamaGrammar`), context_scorer `ScoringBreakdown.__lt__`, assumptions mock responses, memory metadata fallback and `_TestEmbedding`, query_analyzer enum (`QueryIntent.CONVERSATION`) and complexity assertion, integration mock query string.
  - Conftest: Python 3.14 guard with clear fail-fast message; optional `llama_cpp` and `onnxruntime` mocks for CI/FreeBSD compatibility.
  - CI: integration marker on `test_cognitive_flow`; install step timeout 15m; pytest timeouts 5m/10m for unit/integration and test matrix.
- **Documentation (Doc Updater C7 and user-directed):**
  - README and CONTRIBUTING paths and wording corrected for public documentation.
  - CHANGELOG updated with 2026-02-01 fixes.
  - **README refocused as program README:** README now presents the program (what JENOVA is, install, use, config) and the creator’s personal story: built over six months using only AI; creator did not touch a line of code and does not know how to write or read code—six months in, still doesn’t. Used as a personal project to learn about software development, project planning, program design, engineering software, and technology. First-person narrative throughout; CONTRIBUTING.md removed from project structure in README.
- **Code quality (2026-01-26):**
  - Fixed 3,831 linting violations across entire codebase; resolved 69 formatting issues; improved exception handling (proper `from` clauses), type annotations (TYPE_CHECKING), simplified conditionals, PEP 8 imports, no trailing whitespace, modernized imports, noqa for intentional patterns; enhanced maintainability and readability.

## [4.0.0] - 2026-01-26

### Added

#### Core Cognitive Architecture
- **Complete Architectural Rebuild**: Full Phase 7 remediation with production-ready codebase
- **Assumption Manager**: Comprehensive assumption tracking and verification system (`src/jenova/assumptions/manager.py`)
- **Insight Manager**: Advanced insight generation and management (`src/jenova/insights/manager.py`)
- **Concern Manager**: Topic-based concern organization (`src/jenova/insights/concerns.py`)
- **Query Analyzer**: Multi-level query analysis with intent detection, complexity scoring, and topic modeling (`src/jenova/core/query_analyzer.py` - ~1147 lines)
- **Context Scorer**: Configurable context retrieval with query-aware ranking (`src/jenova/core/context_scorer.py` - ~496 lines)
- **Context Organizer**: Intelligent context organization and prioritization (`src/jenova/core/context_organizer.py` - ~716 lines)
- **Integration Hub**: Unified knowledge representation bridging Memory and Cortex (`src/jenova/core/integration.py` - ~808 lines)
- **Multi-Level Planning**: Structured planning system for complex queries (`src/jenova/core/engine.py` - ~734 lines)
- **Cortex Intelligence**: Dict-based cognitive graph with advanced features (`src/jenova/graph/graph.py` - ~1420 lines)
  - Emotion analysis with Pydantic validation
  - Clustering and meta-insight generation
  - Orphan linking and contradiction detection
  - Connection suggestions

#### Utility Systems
- **Cognitive Scheduler**: Task scheduling system for background cognitive operations (`src/jenova/core/scheduler.py` - ~316 lines)
- **Proactive Engine**: Autonomous suggestion generation (`src/jenova/graph/proactive.py` - ~485 lines)
- **TTLCache/CacheManager**: Thread-safe caching system (`src/jenova/utils/cache.py` - ~430 lines)
- **Performance Monitor**: Performance profiling and timing utilities (`src/jenova/utils/performance.py` - ~325 lines)
- **Grammar Loader**: Centralized JSON grammar loading (`src/jenova/utils/grammar.py` - ~260 lines)
- **Tools Module**: Shell command execution and datetime utilities (`src/jenova/tools.py` - ~310 lines)

#### Security & Validation
- **LLM Output Validation**: Pydantic schemas for validating all LLM JSON responses (`src/jenova/graph/llm_schemas.py`)
- **Prompt Injection Sanitization**: Comprehensive sanitization utilities with ReDoS protection (`src/jenova/utils/sanitization.py`)
- **Safe JSON Parsing**: Robust JSON parsing with size limits, depth validation, and timeout protection (`src/jenova/utils/json_safe.py`)
- **Path Validation**: Secure path validation with sandboxing (`src/jenova/utils/validation.py`)
- **Error Message Sanitization**: Safe error handling without information leakage (`src/jenova/utils/errors.py`)
- **Username Validation**: Comprehensive username validation across all entry points including graph operations

#### Testing Infrastructure
- **Comprehensive Test Suite**: 365+ unit tests across 17 test files
- **Integration Tests**: 36 tests across 4 integration test files
- **Security Tests**: 23 adversarial input tests
- **Performance Benchmarks**: Benchmark suite for utility modules
- **CI/CD Pipeline**: GitHub Actions workflow with automated testing, coverage reporting, and security scanning

#### Documentation
- **Developer Documentation**: Code standards, architecture notes, and contribution guidelines
- **Contributing Guide**: Detailed contribution guidelines (CONTRIBUTING.md) with code standards

### Changed

#### Architecture
- **Complete Codebase Rebuild**: Legacy codebase archived; new production-ready codebase in `src/jenova/`
- **Dict-Based Graph**: Replaced networkx dependency with lightweight dict-based graph implementation
- **Graph Search Algorithm Rebuild**: Rebuilt `CognitiveGraph.search()` with hybrid search (embedding-based semantic search + inverted index keyword matching) replacing O(n) linear scan (P0-002)
- **Unified Memory System**: Consolidated episodic, semantic, and procedural memory into unified ChromaDB-based system
- **Textual TUI**: Modern terminal UI using Textual framework (replaced Bubble Tea Go TUI)
- **Type Safety**: Full type hints across entire codebase with strict mypy checking
- **Comment Schema**: Mandatory `##Comment:` schema enforced across all code files

#### Performance
- **Graph Search Optimization**: Implemented hybrid search with embedding-based semantic similarity and inverted index for keyword matching (P0-002)
- **Search Result Caching**: Added TTLCache (5-minute TTL) for graph search results to reduce redundant queries
- **Inverted Index**: Built keyword index for fast node lookup by content keywords
- **Context Scoring Optimization**: Implemented embedding cache, batch operations, and heap-based top-k selection for efficient context scoring (P1-004)

#### Security
- **All P0/P1 Issues Resolved**: 
  - P0-001: MemoryError shadowing fixed (renamed to `JenovaMemoryError`)
  - P1-001: LLM JSON validation with Pydantic schemas
  - P1-002: Thread-safe response cache with `threading.Lock`
  - P1-003: O(n²) edge cleanup fixed with adjacency index
- **Daedelus Security Patches Applied**:
  - P1-001: Input length validation before regex matching to prevent ReDoS attacks
  - P1-002: Username validation in graph operations (`get_nodes_by_user()`)
  - P1-003: JSON parsing timeout protection (5-second default timeout)
- **Exception Handling**: All bare exception handlers replaced with specific exception types
- **Input Validation**: Comprehensive input validation across all public APIs
- **Error Handling**: Explicit error handling following AP-003 (no silent failures)

#### Code Quality
- **Import Ordering**: PEP 8 compliant import ordering across all files
- **Type Refactoring**: Replaced `Any` types with explicit types in Pydantic validators
- **Code Uniformity**: Consistent coding style enforced by Marshal (B7)
- **Anti-Pattern Compliance**: All 12 original + 17 Haymaker audit violations resolved

#### Package Management
- **Modern Packaging**: pyproject.toml-based packaging (no requirements.txt)
- **Dependency Verification**: All 12 dependencies (6 core + 4 dev + 2 finetune) verified
- **Installation**: Standard pip-based installation workflow
- **Configuration Protection**: Enhanced .gitignore with config.yaml protection

### Fixed

#### Critical Bugs
- **Bare Exception Handling**: Fixed in `migrations.py` and `integration.py` (P1-001, P1-002)
- **Return None Violations**: Fixed 4 instances of silent failures (P2-002)
- **Import Ordering**: Fixed inconsistencies across 8 files (P3-001)
- **Input Validation**: Added validation to `prune_graph()` method (CRITIC-003)
- **Inline Imports**: Moved inline imports to module level (CRITIC-001)

#### Code Quality
- **Unused Imports**: Removed all unused imports
- **Commented Code**: Removed all commented-out legacy code
- **Debug Statements**: Removed all debugging print statements (legitimate ones remain in headless mode)
- **Repository Hygiene**: Removed 77MB core dump artifact, added `.core` to .gitignore

### Removed

- **Legacy Codebase**: Archived for reference only
- **Networkx Dependency**: Replaced with lightweight dict-based graph
- **Bubble Tea Go TUI**: Replaced with Textual Python TUI
- **Requirements.txt**: Replaced with pyproject.toml
- **Core Dump Artifacts**: Removed zellij.core and added prevention pattern

### Security

- **Comprehensive Security Audit**: All vulnerabilities identified and patched
- **Daedelus Security Patches**: Applied all three Daedelus-assigned P1 security patches:
  - Input length validation before regex matching (prevents ReDoS attacks)
  - Username validation in graph operations (prevents injection attacks)
  - JSON parsing timeout protection (prevents DoS via malicious JSON)
- **Security Test Suite**: 23 adversarial input tests
- **CI/CD Security Scanning**: Automated security scanning with pip-audit and bandit
- **Security Posture**: LOW RISK - Ready for release from security perspective

---

## [3.2.0] - 2026-01-15

### Changed
- **Bubble Tea as Sole UI:** Removed legacy prompt-toolkit/Textual terminal UI. Bubble Tea is now the only supported interface, providing a cleaner architecture with Go handling rendering and Python handling cognition.
- **Unified Entry Point:** Consolidated `main.py` and `main_bubbletea.py` into a single `main.py` entry point. The `jenova` command now directly launches the Bubble Tea UI.
- **Installation Scripts:** Completely overhauled all installation scripts:
  - `install.sh`: Now checks for Go dependency, validates Go version (1.21+), and automatically builds the TUI binary during installation
  - `uninstall.sh`: Now handles TUI binary removal, virtual environment cleanup, and models directory
  - `setup_venv.sh`: Now validates Go installation and automatically builds TUI after Python setup
  - `build_tui.sh`: Removed outdated environment variable references (JENOVA_UI no longer needed)
- **Branding:** Consistent use of "JENOVA Cognitive Architecture" throughout all scripts and documentation

### Added
- **Comprehensive Inline Documentation:** Added detailed documentation comments throughout the entire Python codebase following a consistent standard:
  - `##Script function and purpose:` at file headers
  - `##Class purpose:` for class definitions
  - `##Function purpose:` for method definitions
  - `##Block purpose:` for significant code blocks
- **Developer Documentation:** Expanded development documentation and contribution guidelines
- **Enhanced Query Analysis:** Added multi-level planning support with structured sub-goals and reasoning chains for complex queries
- **Integration Layer:** New Cortex-Memory integration layer for unified knowledge representation and cross-referencing
- **Context Scoring:** Enhanced context retrieval with configurable scoring weights and query-aware ranking

### Removed
- **Legacy Terminal UI:** Removed `src/jenova/ui/terminal.py` (prompt-toolkit based UI)
- **Dual UI Mode:** Removed `JENOVA_UI` environment variable - Bubble Tea is now the only UI
- **Separate Bubble Tea Entry Point:** Removed `src/jenova/main_bubbletea.py` - functionality merged into `main.py`

### Fixed
- **Interactive Mode State:** Fixed interactive mode handling in BubbleTeaUI to properly reset state on errors and exit commands
- **Procedure Learning Flow:** Fixed the multi-step procedure learning flow to validate input at each stage
- **Assumption Verification:** Fixed yes/no response validation to accept 'y' and 'n' shortcuts

---

*For earlier versions, see the project's version control history.*
