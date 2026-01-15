# JENOVA Cognitive Architecture - Decisions Log

## Purpose
This document records all architectural and process decisions made during development. Each decision includes the context, rationale, alternatives considered, and outcome.

---

## Decision Log

### DEC-006: BubbleTea as Sole UI
**Date:** 2026-01-15
**Session:** 4
**Author:** Claude AI Assistant

**Context:**
The codebase has two UI implementations:
1. Python-based TerminalUI using prompt_toolkit (terminal.py)
2. Go-based BubbleTea TUI (tui/main.go + bubbletea.py bridge)

This creates maintenance burden, code duplication (two nearly identical main.py files), and unnecessary dependencies.

**Decision:**
Consolidate to BubbleTea as the ONLY user interface:
- Remove `terminal.py` (469 lines)
- Remove duplicate `main.py` entry point
- Rename `main_bubbletea.py` to `main.py`
- Remove `prompt-toolkit` from requirements.txt
- Enhance `bubbletea.py` to support all features

**Rationale:**
- BubbleTea provides modern, responsive TUI with Go performance
- Reduces codebase complexity and maintenance burden
- Eliminates duplicate initialization code
- Follows user's explicit requirement for BubbleTea-only UI
- Go handles UI rendering while Python handles cognitive logic (clean separation)

**Alternatives Considered:**
1. Keep both UIs (rejected: maintenance burden, user explicitly requested single UI)
2. Keep only Python terminal UI (rejected: BubbleTea is more modern, user preference)
3. Rewrite everything in Go (rejected: massive effort, Python excellent for ML/AI)

**Outcome:**
Phase A initiated to implement this decision.

---

### DEC-007: Unified Entry Point Pattern
**Date:** 2026-01-15
**Session:** 4
**Author:** Claude AI Assistant

**Context:**
`main.py` and `main_bubbletea.py` are 99% identical (134 lines each), differing only in:
- Line 20: UI import statement
- Line 116: UI class instantiation

This violates DRY (Don't Repeat Yourself) principle.

**Decision:**
Create a unified `main.py` that:
1. Contains shared initialization logic in a reusable function
2. Uses only BubbleTeaUI (after Phase A completion)
3. Simplifies the `jenova` executable script

**Rationale:**
- Eliminates code duplication
- Single source of truth for initialization
- Easier maintenance and updates
- Cleaner architecture

**Alternatives Considered:**
1. Keep both files (rejected: violates DRY, maintenance nightmare)
2. Create shared module for initialization (rejected: over-engineering for this case)

**Outcome:**
Will be implemented in Phase A, Step 3.

---

### DEC-008: Feature Parity Before Removal
**Date:** 2026-01-15
**Session:** 4
**Author:** Claude AI Assistant

**Context:**
Before removing `terminal.py`, we must ensure all features are available in BubbleTeaUI.

**Decision:**
Perform complete feature audit and enhance `bubbletea.py` BEFORE removing `terminal.py`:
1. Audit all commands and features
2. Implement missing features in BubbleTea
3. Test all functionality
4. Only then remove terminal.py

**Rationale:**
- Follows NON-NEGOTIABLE RULE: ALL features must be preserved
- Reduces risk of feature loss
- Enables proper testing before removal
- Documents exactly what needs implementation

**Features Requiring Enhancement:**
- `/learn_procedure` - Interactive multi-step input
- `/verify` - Full interactive verification flow
- `/help` - Enhanced formatting

**Outcome:**
Audit completed, enhancements pending.

---

## Previous Decisions (Sessions 1-3)

### DEC-001: Documentation Folder Structure
**Date:** Previous
**Session:** 1
**Decision:** Create `.devdocs/` folder with 8 documentation files
**Outcome:** Implemented and maintained.

### DEC-002: Code Commenting Format Standard
**Date:** Previous
**Session:** 1
**Decision:** Use `##Script/Function/Block purpose:` format
**Outcome:** Applied to all 52 files.

### DEC-003: Multi-Session Review Approach
**Date:** Previous
**Session:** 1
**Decision:** Divide review into multiple sessions by component priority
**Outcome:** Completed in 3 sessions.

### DEC-004: Existing Comments Preservation
**Date:** Previous
**Session:** 1
**Decision:** Preserve existing comments, add missing ones
**Outcome:** Successfully maintained.

### DEC-005: Package __init__.py Documentation Standard
**Date:** 2026-01-14
**Session:** 3
**Decision:** All `__init__.py` files receive descriptive headers
**Outcome:** Applied to all 9 package files.

---

## Resolved Pending Decisions

### RESOLVED: PENDING-001 → DEC-009: Interactive Command Handling in BubbleTea
**Date:** 2026-01-15
**Session:** 4
**Author:** Claude AI Assistant

**Context:** 
The `/learn_procedure` command requires multi-step interactive input:
1. Prompt for procedure name
2. Loop for steps (until "done")
3. Prompt for expected outcome

**Options Considered:**
1. Implement state machine in Go TUI for multi-step prompts
2. Implement state machine in Python bubbletea.py bridge
3. Use special message types for interactive flows
4. Redesign command to use single structured input

**Decision:** Option 2 - Implement state machine in Python bubbletea.py bridge

**Implementation:**
- Added `interactive_mode` state variable to `BubbleTeaUI` class
- Modes: `'normal'`, `'verify'`, `'learn_procedure_name'`, `'learn_procedure_steps'`, `'learn_procedure_outcome'`
- Added `procedure_data` dict for accumulating multi-step input
- Created handler methods: `_handle_procedure_name()`, `_handle_procedure_step()`, `_handle_procedure_outcome()`
- Flow: `/learn_procedure` → name prompt → steps loop (until "done") → outcome prompt → `engine.learn_procedure()`

**Rationale:**
- Keeps Go TUI simple and focused on rendering
- All cognitive logic stays in Python
- Easier to maintain and debug
- Consistent with existing command handling patterns

**Outcome:** Successfully implemented in `src/jenova/ui/bubbletea.py`. Full interactive multi-step procedure learning now works.

---

## Pending Decisions

*No pending decisions at this time.*

---

## Phase B Decisions (Session 5)

### DEC-010: Create Centralized Grammar Loader Utility
**Date:** 2026-01-15
**Session:** 5
**Author:** Claude AI Assistant

**Context:**
JSON grammar loading code is duplicated in 3 locations:
- `cortex/cortex.py`
- `cognitive_engine/context_organizer.py`
- Scattered in other places

**Decision:**
Create `utils/grammar_loader.py` with a single `load_json_grammar()` function.

**Rationale:**
- Reduces code duplication by ~30 lines
- Single point of maintenance
- Consistent error handling for grammar loading

**Alternatives Considered:**
1. Leave as-is (rejected: violates DRY, maintenance burden)
2. Use decorator pattern (rejected: over-engineering for simple function)

**Outcome:**
Pending implementation.

---

### DEC-011: Create Centralized File I/O Utility
**Date:** 2026-01-15
**Session:** 5
**Author:** Claude AI Assistant

**Context:**
JSON file load/save operations are duplicated in 4+ locations:
- `cortex/cortex.py`
- `insights/manager.py`
- `insights/concerns.py`
- `assumptions/manager.py`

**Decision:**
Create `utils/file_io.py` with `load_json_file()` and `save_json_file()` functions.

**Rationale:**
- Reduces code duplication by ~40 lines
- Consistent error handling for file operations
- Centralized default value handling

**Alternatives Considered:**
1. Leave as-is (rejected: violates DRY)
2. Use Python's pathlib only (rejected: doesn't solve error handling duplication)

**Outcome:**
Pending implementation.

---

### DEC-012: Mandate CHANGELOG Updates After Every Code Change
**Date:** 2026-01-15
**Session:** 5
**Author:** Claude AI Assistant

**Context:**
Need to ensure all code changes are properly documented and tracked.

**Decision:**
Require CHANGELOG.md update after every code modification, no exceptions.

**Rationale:**
- Maintains accurate change history
- Helps with version management
- Required for proper release documentation
- Enables developers to understand what changed

**Alternatives Considered:**
1. Update only for major changes (rejected: loses detail, inconsistent)
2. Auto-generate from git commits (rejected: not available, less descriptive)

**Outcome:**
Added to post-change checklist in TODOS.md and SESSION_HANDOFF.md.

---

### DEC-013: Verify Requirements and Setup Scripts After Changes
**Date:** 2026-01-15
**Session:** 5
**Author:** Claude AI Assistant

**Context:**
Need to ensure installation files remain consistent with codebase.

**Decision:**
Add verification step for requirements.txt and setup scripts after every code change.

**Rationale:**
- Prevents installation issues
- Ensures consistency between code and dependencies
- Catches missing dependencies early

**Alternatives Considered:**
1. Verify only when dependencies change (rejected: may miss issues)
2. Automated CI checks (rejected: not available in current setup)

**Outcome:**
Added to post-change checklist in TODOS.md and SESSION_HANDOFF.md.

---

### DEC-014: ProactiveEngine Enhancement with Category System
**Date:** 2026-01-15
**Session:** 6
**Author:** Claude AI Assistant

**Context:**
The ProactiveEngine had basic suggestion generation but did not utilize conversation history, lacked suggestion variety, and had no configurable trigger conditions. The `_history` parameter was unused.

**Decision:**
Implement comprehensive ProactiveEngine enhancement with:
1. `SuggestionCategory` enum for 5 suggestion types
2. Conversation history analysis for context-aware suggestions
3. Priority-based category selection with rotation
4. Configurable trigger conditions via main_config.yaml
5. User engagement tracking for analytics

**Rationale:**
- Conversation history provides valuable context for relevant suggestions
- Category system ensures suggestion variety and prevents repetition
- Configurable triggers give users control over suggestion frequency
- Engagement tracking enables future optimization of suggestion types
- All features backward-compatible (config optional)

**Alternatives Considered:**
1. Simple enhancement with history keywords only (rejected: insufficient improvement)
2. ML-based topic modeling (rejected: adds complexity, requires additional dependencies)
3. User preference learning (deferred: requires engagement data first)

**Outcome:**
Successfully implemented. ProactiveEngine grew from 67 to 280+ lines with 6 new methods. Syntax verified. Configuration added to main_config.yaml.

---

### DEC-015: QueryAnalyzer Enhancement with Topic Modeling and Entity Linking
**Date:** 2026-01-15
**Session:** 7
**Author:** Claude AI Assistant

**Context:**
The QueryAnalyzer provided basic intent classification, entity extraction, complexity assessment, and keyword extraction. However, it lacked:
- Topic modeling with category classification
- Entity linking to Cortex nodes for knowledge graph integration
- Query reformulation for alternative search strategies
- Confidence scores for classifications

**Decision:**
Implement comprehensive QueryAnalyzer enhancement with:
1. `TopicCategory` enum for 8 topic types (technical, personal, creative, analytical, procedural, conceptual, temporal, unknown)
2. `EntityLink` and `TopicResult` dataclasses for structured results
3. Topic modeling via `_parse_topics()` and heuristic `_extract_default_topics()`
4. Entity linking to Cortex nodes via `_link_entities_to_cortex()` and `_search_cortex_for_entity()`
5. Query reformulation via `generate_reformulations()`
6. Confidence scoring for intent, complexity, type, and overall analysis
7. Cortex reference injection via `set_cortex()` for dynamic linking
8. Human-readable output via `get_analysis_summary()`

**Rationale:**
- Topic modeling provides structured understanding of query domain
- Entity linking connects queries to existing knowledge graph for enhanced retrieval
- Reformulation enables alternative search strategies for complex queries
- Confidence scores help downstream components assess analysis reliability
- Per-request username setting enables proper Cortex context for entity linking
- All features backward-compatible (config toggles, graceful degradation)

**Alternatives Considered:**
1. External NLP library for NER (rejected: adds dependencies, LLM already capable)
2. Semantic embeddings for entity linking (deferred: current keyword matching sufficient for MVP)
3. Caching entity links (deferred: evaluate performance needs first)

**Outcome:**
Successfully implemented. QueryAnalyzer grew from 160 to 380+ lines with 13 new methods. Syntax verified. Configuration options added to main_config.yaml. Engine updated for per-request username setting.

---

## Decision Template

```markdown
### DEC-XXX: [Title]
**Date:** [Date]
**Session:** [Session Number]
**Author:** [Author]

**Context:**
[Background and problem being addressed]

**Decision:**
[What was decided]

**Rationale:**
[Why this decision was made]

**Alternatives Considered:**
[Other options and why they were rejected]

**Outcome:**
[Result of implementing the decision]
```
