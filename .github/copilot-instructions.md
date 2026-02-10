# JENOVA Cognitive Architecture - Copilot Instructions

This project is a self-aware AI cognitive architecture with graph-based memory and RAG-based response generation. **Every line of code was created via AI-assisted development** - not a single line was written by human hands directly.

## Agent Workflow System

This project uses a **specialized agent system** for development. Agents are AI assistants with specific roles and authority. Before making changes:

1. **Check `.devdocs/BRIEFING.md`** for current project status and active focus
2. **Review `.devdocs/AGENTS.md`** to understand agent roles and workflows
3. **Follow agent scope rules** - different agents have specific responsibilities (Builders create, Guardians review, Maintainers fix, Workers extend)
4. **Update `.devdocs/PROGRESS.md`** after each session with a timestamped entry (ISO 8601 format)

### Key Agent Workflows

- **New Features:** Architect (#1) → Logic Engineer/Interface Designer/Test Engineer (#2/#3/#4) in parallel → Scribe (#5)
- **Maintenance:** Operational Control Manager (E12) assigns to appropriate Maintainer (C) or Worker (D)
- **Pre-Release:** Marshal (#7) formats → Sentinel/Profiler/Critic (#6/#8/#9) audit in parallel → Gatekeeper (#10) releases

**Never assign maintenance work to Builders (Group A) or Guardians (Group B)** - only Maintainers (C) or Workers (D).

## Mandatory Comment Schema

All Python files **MUST** use standardized comment prefixes. This is non-negotiable and enforced across the entire codebase:

```python
##Script function and purpose: <file-level description>
##Class purpose: <class description>
##Method purpose: <method description>
##Function purpose: <standalone function description>
##Step purpose: <logical block description>
##Action purpose: <specific action description>
##Condition purpose: <if/elif/else condition description>
##Loop purpose: <for/while loop description>
##Error purpose: <error handling description>
##Note purpose: <contextual information>
##Fix: <bug fix description>
##Update: <feature extension description>
##Refactor: <logic cleanup description>
##Sec: <security patch description>
```

**Every file must start with `##Script function and purpose:`** and every class/method/function must have its corresponding prefix comment.

## Code Quality Standards

### Type Hints (Required)

All code must be fully typed with Python 3.10+ syntax:

```python
# ✅ Correct
def search(self, query: str, n_results: int = 5) -> list[MemoryResult]:
    ...

# ❌ Wrong - no type hints
def search(self, query, n_results=5):
    ...
```

- Use `|` for unions instead of `Union[]` (Python 3.10+)
- No `Any` types - always specify concrete types
- All parameters and return types must be annotated

### Error Handling

Always use specific exceptions from `src/jenova/exceptions.py`:

```python
from jenova.exceptions import MemorySearchError

try:
    results = self._collection.query(...)
except Exception as e:
    raise MemorySearchError(query, str(e)) from e
```

### Protocol-Based Architecture

JENOVA uses Protocol-based dependency injection. When implementing features:

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class GraphProtocol(Protocol):
    """Protocol for cognitive graph dependency.
    
    Implementations: CognitiveGraph (src/jenova/graph/graph.py)
    
    Contract:
        - add_node: Must persist node to graph storage
        - has_node: Must return True if node exists by ID
    """
    
    def add_node(self, node: Node) -> None:
        """Add a node to the graph."""
        ...
```

- Define protocols for dependencies
- Use `@runtime_checkable` decorator
- Document implementations and contracts in docstrings

### No Magic Values

Use named constants or configuration values:

```python
# ❌ Wrong
threads: -1
model_path: ""

# ✅ Correct
threads: "auto"
model_path: "auto"
```

## Build, Test, and Lint Commands

### Testing

```bash
# Run all tests
pytest tests/

# Run unit tests only (exclude integration and slow tests)
pytest tests/unit/ -m "not integration and not slow"

# Run a single test file
pytest tests/unit/test_memory.py

# Run a specific test
pytest tests/unit/test_memory.py::TestMemory::test_search

# Run with coverage
pytest tests/ --cov=src/jenova --cov-report=term-missing

# Run in parallel
pytest tests/ -n auto
```

Test markers:
- `integration`: Integration tests (use `-m "not integration"` to skip)
- `slow`: Slow-running tests (use `-m "not slow"` to skip)

### Linting and Formatting

```bash
# Format code (run before committing)
ruff format .

# Lint code
ruff check .

# Auto-fix linting issues
ruff check --fix .

# Type checking
mypy src/jenova
```

Standards:
- Line length: 100 characters
- Target: Python 3.10+
- Imports: Standard Library → Third-Party → Local Project
- Quotes: Double quotes preferred
- Indentation: 4 spaces

### Running the Application

```bash
# Interactive TUI mode (default)
jenova

# Headless mode with custom username
jenova --headless --user alice

# With custom config
jenova --config my-config.yaml

# Debug mode
jenova --debug
```

## Architecture Overview

### Core Components

1. **CognitiveEngine** (`src/jenova/core/engine.py`)
   - Central orchestrator for the "Retrieve → Plan → Execute → Reflect" cognitive cycle
   - Manages interaction between KnowledgeStore, LLM, and ResponseGenerator
   - Multi-level planning (simple → very complex) based on query complexity

2. **KnowledgeStore** (`src/jenova/core/knowledge.py`)
   - Unified interface for multi-layered memory system
   - Hybrid retrieval across episodic, semantic, and procedural memory
   - Context scoring and result ranking

3. **CognitiveGraph** (`src/jenova/graph/graph.py`)
   - Graph-based knowledge management with typed relationships
   - Semantic similarity search via vector embeddings
   - Clustering, meta-insights, contradiction detection, orphan linking

4. **InsightManager** (`src/jenova/insights/manager.py`)
   - Continuous learning from conversation reflection
   - Concern-based organization of insights
   - Graph integration for interconnected knowledge

5. **AssumptionManager** (`src/jenova/assumptions/manager.py`)
   - Forms and tracks assumptions about users
   - Status tracking: `unverified`, `true`, `false`
   - Graph integration with verification workflow

6. **Memory System** (`src/jenova/memory/`)
   - ChromaDB-backed vector storage (per-user isolation)
   - Three layers: Episodic, Semantic, Procedural
   - Embedding model fine-tuning support

7. **ProactiveEngine** (`src/jenova/graph/proactive.py`)
   - Autonomous suggestion generation
   - Category-based: explore, verify, develop, connect, reflect
   - Cooldown and priority scoring system

8. **CognitiveScheduler** (`src/jenova/core/scheduler.py`)
   - Turn-based background task scheduling
   - Priority system for task execution
   - Configurable intervals with acceleration logic

### Directory Structure

```
src/jenova/
├── core/           # Cognitive engine, knowledge store, response generation
├── memory/         # Multi-layered memory system (ChromaDB)
├── graph/          # Cognitive graph with relationships and clustering
├── insights/       # Insight generation and management
├── assumptions/    # Assumption formation and verification
├── llm/            # LLM interface (llama-cpp-python)
├── embeddings/     # Vector embedding generation
├── config/         # Configuration management (YAML + Pydantic)
├── ui/             # Textual TUI interface
└── utils/          # Utilities (caching, sanitization, validation)

tests/
├── unit/           # Unit tests (fast, isolated)
├── integration/    # Integration tests (require full setup)
└── benchmarks/     # Performance benchmarks

finetune/           # Embedding model fine-tuning
├── data.py         # Training data collection
└── train.py        # Fine-tuning with MultipleNegativesRankingLoss
```

### Key Patterns

**RAG Priority Hierarchy:** The RAG system prioritizes information sources in this order:
1. Retrieved context from cognitive graph and memory
2. Recent conversation history
3. Generated plan for complex queries
4. LLM base knowledge (lowest priority)

**Per-User Isolation:** All data is stored in `~/.jenova-ai/users/<username>/` with complete isolation between users.

**Protocol-Based DI:** Dependencies are injected via Protocol types, not concrete implementations. See `insights/manager.py` for reference implementations.

**Thread Safety:** All shared state operations use thread-safe caching (TTLCache) and atomic file operations.

## Security Considerations

JENOVA includes comprehensive security hardening:

- **Prompt injection protection** via `jenova.utils.sanitization.sanitize_user_query()`
- **Path traversal prevention** via `jenova.utils.validation.validate_path()`
- **Username validation** via `jenova.utils.validation.validate_username()`
- **Safe JSON parsing** with size/depth limits in `jenova.utils.json_safe`
- **LLM output validation** using Pydantic schemas
- **Error message sanitization** to prevent information leakage

When handling user input:
1. Always sanitize queries before passing to LLM
2. Validate file paths before any filesystem operations
3. Use Pydantic models to validate LLM JSON responses
4. Never expose internal paths or implementation details in errors

## Common Pitfalls

1. **Don't bypass the comment schema** - every file/class/method needs proper prefixes
2. **Don't use `Any` types** - always specify concrete types
3. **Don't import from `jenova.core.integration` for ConsistencyError** - it's re-exported from there but defined in `jenova.exceptions`
4. **Don't use magic numbers** - extract to named constants at module level
5. **Don't modify agent documentation folders you don't own** - coordinate via shared files (BRIEFING.md, PROGRESS.md)
6. **Don't forget ISO 8601 timestamps** - all documentation entries need timestamps with timezone

## Testing Requirements

- All new code must have corresponding tests in `tests/unit/` or `tests/integration/`
- Follow existing test patterns (use fixtures from `conftest.py`)
- Mark integration tests with `@pytest.mark.integration`
- Mark slow tests with `@pytest.mark.slow`
- Mock external dependencies (ChromaDB, llama-cpp-python, onnxruntime)
- Run tests before submitting: `pytest tests/`

## Documentation Requirements

When making changes:

1. Update docstrings to match new behavior (Google-style docstrings)
2. Update `.devdocs/PROGRESS.md` with timestamped session summary
3. Update relevant agent-specific docs in your agent folder
4. Never modify another agent's documentation folder
5. Update BRIEFING.md if changing active focus

## Platform Support

- **Primary targets:** FreeBSD 13.x/14.x and Linux (Ubuntu 22.04+, Debian 12+, Fedora 38+)
- **POSIX conventions:** Use `/` for path separators, LF line endings
- **No platform conditionals:** Code must work identically on FreeBSD and Linux
- **Atomic operations:** Use POSIX `rename()` semantics for file updates

## Configuration

Configuration is managed via YAML files parsed with Pydantic models (`src/jenova/config/models.py`):

- **Hardware:** CPU threads, GPU layers
- **Model:** Path, context length, temperature, sampling parameters
- **Memory:** Storage path, embedding model, result limits
- **Persona:** Name, system prompt, directives
- **Scheduler:** Task intervals, priority weights
- **Proactive:** Cooldowns, thresholds, suggestion limits

Default config: `config.example.yaml` - copy and customize as needed.

## Additional Resources

- **Current Status:** `.devdocs/BRIEFING.md`
- **Session History:** `.devdocs/PROGRESS.md`
- **Architecture Decisions:** `.devdocs/DECISIONS_LOG.md`
- **Test Results:** `.devdocs/TESTS.md`
- **Agent Roles:** `.devdocs/AGENTS.md`
- **Code Standards:** `.devdocs/guardians/marshal/formatting_standards.md`
- **Bug Tracking:** `.devdocs/maintainers/bug_hunter/bug_reports.md`

When in doubt, follow the patterns you see in existing code. This codebase has strict conventions that must be followed.
