# JENOVA Cognitive Architecture - Planning Document

## Purpose
This document contains comprehensive plans for current and future development phases.

---

## Phase C: Cognitive Architecture Enhancement 🔄 READY

### Objective
Enhance the cognitive capabilities of JENOVA by improving key components: ProactiveEngine, QueryAnalyzer, Memory-Cortex integration, Cortex graph operations, and reflection capabilities.

### Timeline
- **Start:** Session 6 (2026-01-15)
- **Estimated Completion:** Session 6-8
- **Status:** 🔄 READY FOR IMPLEMENTATION

---

### Step C.1: Enhance ProactiveEngine

**Purpose:** Improve suggestion quality and context-awareness

**Current State Analysis:**
- 67 lines of code
- Basic suggestion generation based on unverified assumptions and underdeveloped nodes
- History parameter unused (marked with underscore)
- No sophisticated trigger conditions

**Planned Enhancements:**
1. **Conversation History Analysis**: Analyze recent conversation patterns
2. **Trigger Conditions**: Add configurable conditions for when to suggest
3. **Suggestion Variety**: Categorize suggestions by type (explore, verify, develop)
4. **User Engagement Tracking**: Track which suggestions were acted upon

**Implementation:**
```python
##Function purpose: Analyze conversation patterns for context-aware suggestions
def _analyze_conversation_patterns(self, history: List[str]) -> Dict[str, Any]:
    """Analyze recent conversation for recurring topics, questions, and interests."""
    # Implementation details
```

**Files to Modify:**
- `src/jenova/cortex/proactive_engine.py`

**Post-Change Requirements:**
- [ ] Update CHANGELOG.md
- [ ] Add inline comments
- [ ] Preserve all existing functionality

---

### Step C.2: Improve QueryAnalyzer

**Purpose:** Enhanced intent classification and entity extraction

**Current State Analysis:**
- 160 lines of code
- Basic intent classification (5 types)
- Simple complexity estimation
- Keyword extraction working

**Planned Enhancements:**
1. **Topic Modeling**: Extract main topics from query
2. **Entity Linking**: Link entities to Cortex nodes
3. **Query Reformulation**: Generate alternative query forms
4. **Confidence Scoring**: Add confidence to classifications

**Implementation:**
```python
##Function purpose: Extract and link entities to existing Cortex nodes
def extract_and_link_entities(self, query: str, cortex: Any, username: str) -> List[Dict[str, Any]]:
    """Extract entities and find related Cortex nodes."""
    # Implementation details
```

**Files to Modify:**
- `src/jenova/cognitive_engine/query_analyzer.py`

---

### Step C.3: Strengthen Memory-Cortex Integration

**Purpose:** Bidirectional feedback loops and cross-memory linking

**Current State Analysis:**
- MemorySearch already integrates with Cortex for centrality scoring
- Integration layer exists but could be enhanced
- One-way flow from memory to Cortex

**Planned Enhancements:**
1. **Bidirectional Feedback**: Cortex insights update memory relevance
2. **Cross-Memory Linking**: Link related items across memory types
3. **Automatic Consolidation**: Merge similar memories automatically
4. **Memory Reinforcement**: Strengthen frequently accessed memories

**Files to Modify:**
- `src/jenova/cognitive_engine/memory_search.py`
- `src/jenova/cortex/cortex.py` (minor)

---

### Step C.4: Optimize Cortex Graph Operations

**Purpose:** Performance and quality improvements for large graphs

**Current State Analysis:**
- Graph operations work but can be slow for large graphs
- Simple keyword matching for similarity
- Batch processing implemented but limited

**Planned Enhancements:**
1. **Semantic Similarity Cache**: Cache embeddings for faster matching
2. **Incremental Centrality**: Update centrality incrementally vs full recalc
3. **Smart Pruning**: Preserve high-value low-centrality nodes
4. **Graph Indexing**: Add index structures for faster lookups

**Files to Modify:**
- `src/jenova/cortex/cortex.py`
- `src/jenova/cortex/graph_metrics.py`

---

### Step C.5: Add Enhanced Reflection Capabilities

**Purpose:** Deeper pattern recognition and knowledge gap detection

**Current State Analysis:**
- Iterative deepening implemented
- Meta-insight generation working
- Basic orphan linking

**Planned Enhancements:**
1. **Pattern Recognition**: Detect recurring themes across time
2. **Knowledge Gap Detection**: Identify areas with sparse knowledge
3. **Temporal Trend Analysis**: Track how topics evolve over time
4. **Contradiction Detection**: Find conflicting insights

**Files to Modify:**
- `src/jenova/cortex/cortex.py`
- `src/jenova/cortex/clustering.py`

---

## Quality Checklist for Phase C

For each step, verify:
- [ ] All code has inline comments
- [ ] No features were removed
- [ ] CHANGELOG.md updated
- [ ] Type hints added to new code
- [ ] Existing tests still pass

---

## Phase B: Code Organization ✅ COMPLETE

### Timeline
- **Start:** Session 5 (2026-01-15)
- **Estimated Completion:** Session 5-6
- **Status:** ✅ COMPLETE

### Detailed Implementation Plan

---

### Step B.1: Create `utils/grammar_loader.py`

**Purpose:** Consolidate JSON grammar loading (currently duplicated 3x)

**Justification:**
- Same grammar loading code exists in `cortex/cortex.py` and `cognitive_engine/context_organizer.py`
- Reduces code duplication by ~30 lines
- Single point of maintenance for grammar loading

**Implementation:**
```python
##Script function and purpose: Provides centralized JSON grammar loading utility
##This module consolidates grammar loading to avoid code duplication

from typing import Optional
from llama_cpp import LlamaGrammar
import os

##Function purpose: Load JSON grammar from standard location
def load_json_grammar(ui_logger=None, file_logger=None) -> Optional[LlamaGrammar]:
    """Load JSON grammar for structured LLM output."""
    # Implementation here
```

**Files to Modify After Creation:**
- `src/jenova/cortex/cortex.py` - Import and use utility
- `src/jenova/cognitive_engine/context_organizer.py` - Import and use utility

**Post-Change Requirements:**
- [ ] Update CHANGELOG.md
- [ ] Verify no new dependencies added
- [ ] Test grammar loading works correctly

---

### Step B.2: Create `utils/file_io.py`

**Purpose:** Consolidate JSON file load/save operations (currently duplicated 4x)

**Justification:**
- Same file I/O pattern exists in cortex.py, insights/manager.py, assumptions/manager.py, concerns.py
- Provides consistent error handling for file operations
- Reduces ~40 lines of duplicated code

**Implementation:**
```python
##Script function and purpose: Provides centralized file I/O utilities
##This module consolidates JSON file operations to avoid code duplication

import json
import os
from typing import Any, Optional

##Function purpose: Load JSON from file with error handling
def load_json_file(filepath: str, default: Any = None) -> Any:
    """Load JSON from file, returning default on failure."""
    # Implementation here

##Function purpose: Save data to JSON file with error handling
def save_json_file(filepath: str, data: Any, indent: int = 2) -> bool:
    """Save data to JSON file, returning success status."""
    # Implementation here
```

**Files to Modify After Creation:**
- `src/jenova/cortex/cortex.py`
- `src/jenova/insights/manager.py`
- `src/jenova/insights/concerns.py`
- `src/jenova/assumptions/manager.py`

**Post-Change Requirements:**
- [ ] Update CHANGELOG.md
- [ ] Verify no new dependencies added
- [ ] Test file operations work correctly

---

### Step B.3: Use `extract_json()` Everywhere

**Purpose:** Replace manual JSON parsing with existing utility (4+ files)

**Justification:**
- `utils/json_parser.py` already provides `extract_json()` function
- Currently 4+ files do manual JSON parsing instead of using utility
- Consistent JSON extraction with error handling

**Files to Modify:**
1. `src/jenova/cognitive_engine/engine.py` - 5+ instances
2. `src/jenova/cortex/cortex.py` - 6+ instances
3. `src/jenova/cognitive_engine/context_organizer.py` - 1+ instance
4. `src/jenova/cognitive_engine/query_analyzer.py` - 1+ instance

**Pattern to Replace:**
```python
# Before (manual)
try:
    result = json.loads(response)
except json.JSONDecodeError:
    result = {}

# After (using utility)
from jenova.utils.json_parser import extract_json
result = extract_json(response, default={})
```

**Post-Change Requirements:**
- [ ] Update CHANGELOG.md
- [ ] Test JSON parsing works correctly in all contexts

---

### Step B.4: Add Type Hints to Constructors

**Purpose:** Add missing type hints to 15+ files

**Justification:**
- Improves code readability and IDE support
- Enables static type checking with mypy
- Documents expected parameter types

**Files to Modify:**
| File | Missing Types |
|------|---------------|
| `cortex/cortex.py` | config, ui_logger, file_logger, llm, cortex_root |
| `cortex/proactive_engine.py` | cortex, llm, ui_logger |
| `assumptions/manager.py` | Most parameters |
| `memory/semantic.py` | config, ui_logger, file_logger, db_path, llm, embedding_model |
| `memory/episodic.py` | Same as semantic.py |
| `memory/procedural.py` | Same as semantic.py |
| `insights/manager.py` | Multiple parameters |
| `ui/logger.py` | message_queue |

**Type Hint Pattern:**
```python
from typing import Any, Dict, Optional
from jenova.ui.logger import UILogger
from jenova.utils.file_logger import FileLogger

def __init__(
    self,
    config: Dict[str, Any],
    ui_logger: UILogger,
    file_logger: FileLogger,
    ...
) -> None:
```

**Post-Change Requirements:**
- [ ] Update CHANGELOG.md
- [ ] No functional changes - only type annotations

---

### Step B.5: Improve Error Handling Patterns

**Purpose:** Replace bare `except:` clauses with specific exception types (37+ instances)

**Justification:**
- Bare `except:` catches all exceptions including KeyboardInterrupt and SystemExit
- Makes debugging difficult by hiding exception types
- Should use specific exceptions and log appropriately

**Priority Files:**
1. `pydantic_compat.py` - 40+ instances (compatibility code, may need careful review)
2. `cortex/graph_metrics.py` - 3 instances
3. `cortex/clustering.py` - 3 instances
4. `ui/bubbletea.py` - 3 instances
5. `cognitive_engine/memory_search.py` - Multiple instances

**Pattern to Apply:**
```python
# Before
try:
    result = some_operation()
except Exception:
    pass  # Silent failure

# After
try:
    result = some_operation()
except (ValueError, KeyError) as e:
    self.file_logger.log_error(f"Operation failed: {e}")
    result = default_value
```

**Note:** `pydantic_compat.py` requires special handling as it's compatibility code for ChromaDB/Pydantic version issues. May need to preserve some bare except clauses with detailed comments.

**Post-Change Requirements:**
- [ ] Update CHANGELOG.md
- [ ] Test error scenarios work correctly
- [ ] Ensure no silent failures in critical paths

---

### Step B.6: Enhance FileLogger with DEBUG Level

**Purpose:** Add DEBUG logging level with config toggle

**Justification:**
- Currently only supports: log_info, log_warning, log_error
- No DEBUG level for development troubleshooting
- Config toggle prevents debug noise in production

**Implementation:**
```python
##Function purpose: Log debug messages when debug mode is enabled
def log_debug(self, message: str) -> None:
    """Log debug message if debug mode is enabled in config."""
    if self.debug_enabled:
        self._write_log("DEBUG", message)
```

**Config Addition (main_config.yaml):**
```yaml
logging:
  debug_enabled: false  # Enable for development
```

**Post-Change Requirements:**
- [ ] Update CHANGELOG.md
- [ ] Update main_config.yaml with new setting
- [ ] Test debug logging toggle works

---

### Step B.7: Update CHANGELOG.md

**Purpose:** Document all Phase B changes

**Requirements:**
- Update after EVERY code change
- Use proper semantic versioning
- Include Added, Changed, Fixed sections

**Template:**
```markdown
## [3.3.0] - 2026-01-15

### Added
- New `utils/grammar_loader.py` for centralized grammar loading
- New `utils/file_io.py` for centralized file operations
- DEBUG logging level in FileLogger with config toggle

### Changed
- Consolidated JSON parsing to use `extract_json()` utility
- Added type hints to 15+ constructor methods
- Improved error handling patterns (replaced bare except clauses)

### Fixed
- Silent exception swallowing in cortex/graph_metrics.py
- Missing error logging in memory_search.py
```

---

### Step B.8: Verify Requirements and Setup Scripts

**Purpose:** Ensure all installation files are up to date

**Files to Check:**
- `requirements.txt` - No changes expected for Phase B
- `pyproject.toml` - No changes expected for Phase B
- `install.sh` - No changes expected for Phase B
- `setup_venv.sh` - No changes expected for Phase B
- `build_tui.sh` - No changes expected for Phase B

**Verification Checklist:**
- [ ] requirements.txt matches pyproject.toml dependencies
- [ ] All scripts have proper documentation comments
- [ ] Version numbers are consistent

---

### Step B.9: Run CodeRabbit --prompt-only

**Purpose:** Generate review prompt for code quality validation

**Command:**
```bash
# Generate CodeRabbit review prompt
coderabbit --prompt-only
```

**Review Focus:**
- Type hint completeness
- Error handling patterns
- Code duplication elimination
- Documentation quality

---

## Commenting Standard (Reference)

All new code must follow:

```python
##Script function and purpose: [Explanation]

##Class purpose: [Explanation]
class ClassName:
    ##Function purpose: [Explanation]
    def method_name(self):
        ##Block purpose: [Explanation]
        code_here
```

---

## Quality Checklist

For each Phase B step, verify:
- [ ] All code has inline comments
- [ ] No features were removed
- [ ] CHANGELOG.md updated
- [ ] requirements.txt verified
- [ ] Setup scripts verified
- [ ] Tests still pass

---

## Session Workflow

1. Read `.devdocs/BRIEFING.md`
2. Read `.devdocs/SESSION_HANDOFF.md`
3. Review current phase in PLANS.md
4. Follow NON-NEGOTIABLE RULES
5. Ask for permission before actions
6. Update all .devdocs/ after changes
7. Update CHANGELOG.md after code changes
8. Verify requirements.txt and setup scripts
9. Run CodeRabbit before completing
