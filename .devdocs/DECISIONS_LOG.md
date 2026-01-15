# JENOVA Cognitive Architecture - Decisions Log

## Purpose
This document records all architectural and process decisions made during development. Each decision includes the context, rationale, alternatives considered, and outcome.

---

## Decision Log

### DEC-006: BubbleTea as Sole UI
**Date:** 2025-01-15
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
**Date:** 2025-01-15
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
**Date:** 2025-01-15
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
**Date:** 2025-01-14
**Session:** 3
**Decision:** All `__init__.py` files receive descriptive headers
**Outcome:** Applied to all 9 package files.

---

## Resolved Pending Decisions

### RESOLVED: PENDING-001 → DEC-009: Interactive Command Handling in BubbleTea
**Date:** 2025-01-15
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
