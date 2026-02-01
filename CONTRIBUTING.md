# Contributing to JENOVA

Welcome to the JENOVA Cognitive Architecture project. This guide covers the coding standards and conventions required for all contributions.

---

## Getting Started

1. **Read the documentation**: Start with `.devdocs/BRIEFING.md` for current project status
2. **Understand the architecture**: Review `.devdocs/builders/architect/` for design decisions
3. **Follow the standards**: All code must comply with `.devdocs/guardians/marshal/CODE_STANDARDS.md`

### Supported Python Versions (Development & CI)

Use **Python 3.10, 3.11, or 3.12** for local development and running tests. **Python 3.14 is not supported**: the dependency `chromadb` requires `onnxruntime`, which does not ship wheels for Python 3.14, so `pip install -e ".[dev]"` will fail. CI uses Python 3.11. If your system default is 3.14, use `pyenv`, a venv from a supported Python, or a container.

---

## Code Standards Quick Reference

### Mandatory Comment Schema

All code MUST use standardized comment prefixes. This is non-negotiable.

| Prefix | Usage | Example |
|--------|-------|---------|
| `##Script function and purpose:` | Top of every Python file | `##Script function and purpose: Provides unified Memory class` |
| `##Class purpose:` | Before every class | `##Class purpose: Unified memory interface` |
| `##Method purpose:` | Before every method | `##Method purpose: Search memory for relevant content` |
| `##Function purpose:` | Before standalone functions | `##Function purpose: Load configuration from file` |
| `##Step purpose:` | Before logical code blocks | `##Step purpose: Generate unique ID` |
| `##Action purpose:` | Before specific actions | `##Action purpose: Initialize ChromaDB client` |
| `##Condition purpose:` | Before if statements | `##Condition purpose: Check if node exists` |
| `##Loop purpose:` | Before for/while loops | `##Loop purpose: Convert results to typed objects` |
| `##Error purpose:` | Before error handling | `##Error purpose: Wrap ChromaDB errors` |
| `##Note purpose:` | For contextual information | `##Note purpose: This is O(n) but acceptable for small graphs` |

### Maintenance Comment Tags

When making fixes or updates, use these tags:

| Tag | Usage |
|-----|-------|
| `##Fix:` | Bug fixes |
| `##Update:` | Feature extensions |
| `##Refactor:` | Logic cleanup |
| `##Sec:` | Security patches |
| `##Note:` | Contextual information |

---

## Code Quality Requirements

### Type Hints

All code must be fully typed:

```python
# ✅ Correct - Fully typed
def search(
    self,
    query: str,
    n_results: int = 5,
) -> list[MemoryResult]:
    ...

# ❌ Wrong - Untyped
def search(self, query, n_results=5):
    ...
```

### No Magic Values

```python
# ❌ Wrong
threads: -1
model_path: ""

# ✅ Correct
threads: "auto"
model_path: "auto"
```

### Error Handling

Always use specific exceptions from `src/jenova/exceptions.py`:

```python
# ✅ Correct
from jenova.exceptions import MemorySearchError

try:
    results = self._collection.query(...)
except Exception as e:
    raise MemorySearchError(query, str(e)) from e
```

---

## Protocol-Based Architecture

JENOVA uses Protocol-based dependency injection. When implementing new features:

1. **Define protocols** for dependencies (see `insights/manager.py` for examples)
2. **Document contracts** in protocol docstrings
3. **Use `@runtime_checkable`** for Protocol classes

Example:
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

---

## Testing Requirements

1. **All new code must have tests**
2. **Tests live in `tests/unit/` or `tests/integration/`**
3. **Follow existing test patterns**
4. **Run tests before submitting**: `pytest tests/`

---

## Documentation Requirements

When making changes:

1. **Update docstrings** to match new behavior
2. **Update `.devdocs/PROGRESS.md`** with session summary
3. **Maintain agent-specific docs** in your agent folder (if applicable)
4. **Never modify another agent's documentation folder**

---

## Pull Request Checklist

Before submitting:

- [ ] All code uses mandatory comment schema
- [ ] All functions and methods are fully typed
- [ ] All new code has corresponding tests
- [ ] All tests pass (`pytest tests/`)
- [ ] No `Any` types used
- [ ] Docstrings match implementation
- [ ] `.devdocs/PROGRESS.md` updated

---

## Quick Links

- **Full Code Standards**: `.devdocs/guardians/marshal/CODE_STANDARDS.md`
- **Anti-Patterns to Avoid**: `.devdocs/guardians/critic/ANTI_PATTERNS.md`
- **Current Status**: `.devdocs/BRIEFING.md`
- **Session Log**: `.devdocs/PROGRESS.md`

---

## Questions?

Review the existing codebase in `src/jenova/` for examples of correct patterns. When in doubt, follow the conventions you see in similar files.
