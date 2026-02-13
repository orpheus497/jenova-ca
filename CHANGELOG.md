# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **CognitiveScheduler Wiring (2026-02-13T11:26:36Z):**
  - Created `CognitiveTaskExecutor` dispatching 6 autonomous task types (GENERATE_INSIGHT, GENERATE_ASSUMPTION, VERIFY_ASSUMPTION, REFLECT, PRUNE_GRAPH, LINK_ORPHANS)
  - Wired `CognitiveScheduler` into engine pipeline — fires after every successful `think()` call
  - Autonomous insight and assumption generation from conversation history
  - **User Impact:** CognitiveScheduler now enables proactive suggestions and summarized insights in conversations, improving relevance without manual prompting.

- **IntegrationHub Wiring (2026-02-13T13:05:14Z):**
  - Wired `IntegrationHub` into engine pipeline with semantic memory and Cortex graph
  - Context expansion via graph relationships now active on every `think()` call
  - Memory→Cortex bidirectional feedback loop now operational
  - Unified Knowledge Map, consistency checking, and centrality scoring now available
  - **User Impact:** IntegrationHub context expansion via `think()` delivers more context-aware responses and fewer clarifying questions, while the Unified Knowledge Map ensures consistent recommendations across long conversations.

### Fixed

- **Code Quality Improvements (2026-02-11T08:39:11Z):**
  - Reduced verbose logging in Pydantic V1 compatibility patch (warning → debug level)
  - ChromaDB type inference fallbacks now log at debug level (expected behavior on Python 3.14)
  - Improved diagnostic messages for type inference fallbacks

## [4.1.0] - 2026-02-11

### Added

- **Documentation Reorganization (2026-02-11T02:36:00Z):**
  - Created `docs/` folder for project documentation
  - Added `docs/ROADMAP.md` with future enhancement plans
  - Separated contributing guidelines to `docs/CONTRIBUTING.md`
  - Updated README structure with additional documentation section

### Changed

- **Python 3.14 Compatibility (2026-02-11T02:24:00Z):**
  - Fixed Python 3.14 compatibility issues in ChromaDB integration
  - Updated Pydantic v2 migration compatibility layer
  - Enhanced Python 3.14 guard in test configuration
  - Added ChromaDB Python 3.14 compatibility patch (`fix_chromadb_py314_compat.py`)

- **Bug Fixes and Maintenance (2026-02-10T00:00:00Z - 2026-02-11T23:59:59Z):**
  - Fixed 18 bugs across 53 files (94.7% fix rate) via Bug Hunter C1 comprehensive audit
  - Resolved `zip(strict=False)` → `strict=True` issues (4 instances)
  - Fixed broad exception handling → specific exceptions (6 instances)
  - Added missing `dict.get()` None handling (2 instances)
  - Improved cache invalidation logic
  - Enhanced logging statements across codebase

- **Security Hardening (2026-02-10T18:30:00Z):**
  - P0 (CRITICAL): User input sanitization in planning prompts
  - P2 (HIGH): History context sanitization in context organizer
  - Enhanced prompt injection protection
  - Added security tracking tags (`##Sec:`)

- **Code Quality and Consistency (2026-02-10T00:00:00Z):**
  - Fixed all E501 line length violations (100+ instances)
  - Resolved import sorting inconsistencies
  - Achieved 100% Ruff linting compliance
  - Stabilized ConsistencyError re-export for mypy
  - Enhanced code uniformity across entire codebase

- **Testing Infrastructure (2026-02-10T00:00:00Z):**
  - 23 unit-test fixes applied
  - Added Python 3.14 compatibility guards in conftest.py
  - Enhanced llama_cpp and onnxruntime test mocks
  - Configured CI integration markers and timeouts
  - Achieved 100% coverage for validation.py and sanitization.py (51 tests)

- **Configuration and Optimization (2026-02-10T00:00:00Z):**
  - Enhanced ProactiveEngine seeding configuration
  - Improved headless mode `--user` support
  - Updated CI Makefile with new patterns
  - P1 connectivity-based DEVELOP selection optimization
  - Performance benchmarking across utility modules

- **Code Cleanup (2026-02-10T00:00:00Z):**
  - Extracted magic numbers to constants (8 instances)
  - Verified zero dead code in codebase
  - Enhanced maintainability metrics

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
- **Developer Documentation**: Complete `.devdocs/` structure with agent-specific documentation
- **Code Standards**: Comprehensive coding standards and anti-patterns documentation
- **Architecture Documentation**: System architecture, decision logs, and planning documents
- **Contributing Guide**: Detailed contribution guidelines with code standards

### Changed

#### Architecture
- **Complete Codebase Rebuild**: Legacy codebase archived to `.devdocs/resources/`, new production-ready codebase in `src/jenova/`
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

- **Legacy Codebase**: Archived to `.devdocs/resources/` for reference only
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
- **Developer Documentation:** Added `.devdocs/` directory with comprehensive development documentation:
  - `ARCHITECTURE.md`: System architecture overview
  - `BRIEFING.md`: Quick start guide for developers
  - `DECISIONS_LOG.md`: Architectural decision records
  - `PLANS.md`: Future development roadmap
  - `PROGRESS.md`: Development progress tracking
  - `SESSION_HANDOFF.md`: Context for development sessions
  - `SUMMARIES.md`: Code review summaries
  - `TESTS.md`: Testing documentation
  - `TODOS.md`: Outstanding tasks and improvements
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

*For earlier versions, see `.devdocs/resources/CHANGELOG.md`*
